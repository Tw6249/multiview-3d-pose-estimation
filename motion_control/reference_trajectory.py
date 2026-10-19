"""Causal, model-bounded reference trajectories; no motor or network interface.

Input is TWIST's 35-vector (root local velocity, height, tilt, yaw rate, SDK q),
target model joint limits and monotonic source ns. Output is a reference with
continuous position/rate/acceleration and bounded jerk, not a balanced robot
command. Ruckig 0.12.2 provides synchronized state-to-state generation. Current
signal derivatives use only three accepted past source samples. Explicitly tagged
offline references may provide future-based derivatives; they remain offline.
Missing/stale data must request a stop/stand transition from the caller. Solver
or limit failure raises; never clip a generated trajectory and claim continuity.
"""
from collections import deque
import json
from pathlib import Path
import numpy as np


def load_config():
    return json.loads((Path(__file__).resolve().parents[1]/'config/g1_reference_trajectory.json').read_text())


class SourceReferenceDerivative:
    def __init__(self, config=None):
        self.cfg=config or load_config();self.history=deque(maxlen=self.cfg['source_derivative_window'])

    def reset(self):self.history.clear()

    def update(self, row):
        if row is None or not row.get('accepted',False):self.reset();return np.zeros(35)
        t=row['timestamp_ns'];reference=np.asarray(row['reference'],float)
        if type(t) is not int or reference.shape!=(35,) or not np.isfinite(reference).all():raise ValueError('Invalid source reference')
        if self.history and t<self.history[-1][0]:raise ValueError('Reordered source reference')
        if self.history and (t-self.history[-1][0])/1e9>self.cfg['source_max_gap_s']:self.reset()
        if not self.history or t!=self.history[-1][0]:self.history.append((t,reference.copy()))
        if row.get('offline_trajectory',{}).get('uses_future_estimates'):
            offline_rate=np.asarray(row.get('offline_reference_rate_per_s',np.zeros(35)),float)
            if offline_rate.shape!=(35,) or not np.isfinite(offline_rate).all():raise ValueError('Invalid offline derivative')
            return offline_rate
        if len(self.history)<self.history.maxlen:return np.zeros(35)
        times=np.array([(r[0]-t)/1e9 for r in self.history]);times-=times.mean()
        rate=times@np.array([r[1] for r in self.history])/(times@times)
        # Root feedback is handled separately. Arm loss must not generate a
        # velocity through the neutral-wrist or missing-arm prior.
        rate[:6]=0.
        arms=row.get('arm_observed',[False,False])
        if not arms[0]:rate[6+15:6+22]=0.
        if not arms[1]:rate[6+22:6+29]=0.
        return rate


class BoundedReferenceTrajectory:
    def __init__(self, initial, joint_limits, model_sha256, config=None, dt=None):
        from ruckig import Ruckig,InputParameter,Trajectory,Synchronization
        self.cfg=config or load_config();self.dt=float(dt if dt is not None else self.cfg['control_dt_s'])
        if not 0<self.dt<=.04 or self.cfg.get('hardware_enabled') is not False:raise ValueError('Offline simulation trajectory profile required')
        self.model_sha256=model_sha256
        limits=np.asarray(joint_limits,float)
        if limits.shape!=(29,2) or not np.isfinite(limits).all() or np.any(limits[:,0]>=limits[:,1]):raise ValueError('Invalid model joint limits')
        self.lower=np.r_[self.cfg['root_min'],limits[:,0]];self.upper=np.r_[self.cfg['root_max'],limits[:,1]]
        self.max_rate=np.r_[self.cfg['root_max_rate'],np.full(29,self.cfg['joint_max_velocity_rad_s'])]
        self.max_acceleration=np.r_[self.cfg['root_max_acceleration'],np.full(29,self.cfg['joint_max_acceleration_rad_s2'])]
        self.max_jerk=np.r_[self.cfg['root_max_jerk'],np.full(29,self.cfg['joint_max_jerk_rad_s3'])]
        if any(a.shape!=(35,) or not np.isfinite(a).all() for a in (self.lower,self.upper,self.max_rate,self.max_acceleration,self.max_jerk)):
            raise ValueError('Invalid trajectory profile shape')
        if any(np.any(a<=0) for a in (self.max_rate,self.max_acceleration,self.max_jerk)):raise ValueError('Nonpositive trajectory limit')
        self.position=np.asarray(initial,float).copy();self.rate=np.zeros(35);self.acceleration=np.zeros(35)
        if self.position.shape!=(35,) or not np.isfinite(self.position).all() or np.any(self.position<self.lower) or np.any(self.position>self.upper):raise ValueError('Initial reference outside model bounds')
        groups=self.cfg.get('synchronization_groups',[list(range(35))])
        if sorted(j for group in groups for j in group)!=list(range(35)) or any(not g for g in groups):raise ValueError('Each reference axis must belong to exactly one synchronization group')
        self.groups=[]
        for group in groups:
            ids=np.array(group);inp=InputParameter(len(ids));curve=Trajectory(len(ids))
            inp.max_velocity=self.max_rate[ids].tolist();inp.max_acceleration=self.max_acceleration[ids].tolist();inp.max_jerk=self.max_jerk[ids].tolist()
            inp.synchronization=Synchronization.Time
            self.groups.append((ids,Ruckig(len(ids)),inp,curve))

    def step(self, target, target_rate=None, prediction_s=0.):
        target=np.asarray(target,float);rate=np.zeros(35) if target_rate is None else np.asarray(target_rate,float)
        if target.shape!=(35,) or rate.shape!=(35,) or not np.isfinite(target).all() or not np.isfinite(rate).all():raise ValueError('Nonfinite or invalid trajectory target')
        if not np.isfinite(prediction_s) or prediction_s<0:raise ValueError('Invalid prediction horizon')
        rate=np.clip(rate,-.8*self.max_rate,.8*self.max_rate)
        target=target+rate*min(prediction_s,self.cfg['maximum_prediction_s'])
        bounded=np.clip(target,self.lower,self.upper);clipped=bool(np.any(bounded!=target))
        # The open-source generator has no position limits. A nonzero terminal
        # speed at a position bound would make the next replan infeasible. Bound
        # that speed by a conservative braking-distance envelope (terminal a=0),
        # then verify the complete generated curve below.
        distance=np.maximum(0.,np.where(rate>=0,self.upper-bounded,bounded-self.lower))
        ramp=self.max_acceleration**2/self.max_jerk
        terminal_speed=np.sqrt(ramp**2+self.max_acceleration*distance)-ramp
        rate=np.sign(rate)*np.minimum(np.abs(rate),.95*terminal_speed)
        fallback=False;position=np.empty(35);velocity=np.empty(35);acceleration=np.empty(35)
        for ids,generator,inp,curve in self.groups:
            inp.current_position=self.position[ids].tolist();inp.current_velocity=self.rate[ids].tolist();inp.current_acceleration=self.acceleration[ids].tolist()
            inp.target_position=bounded[ids].tolist();inp.target_acceleration=[0.]*len(ids)
            for attempt in range(2):
                inp.target_velocity=(rate[ids] if attempt==0 else np.zeros(len(ids))).tolist()
                result=generator.calculate(inp,curve)
                if int(result)<0:raise ValueError('Trajectory solver failed; independent stop required: '+str(result))
                extrema=curve.position_extrema;next_position=np.asarray(curve.at_time(self.dt)[0])
                if (all(e.min>=lo-1e-8 and e.max<=hi+1e-8 for e,lo,hi in zip(extrema,self.lower[ids],self.upper[ids]))
                        and np.all(next_position>=self.lower[ids]-1e-8) and np.all(next_position<=self.upper[ids]+1e-8)):break
                fallback=True
            else:raise ValueError('Trajectory leaves model position limits; independent stop required')
            position[ids],velocity[ids],acceleration[ids]=map(np.asarray,curve.at_time(self.dt))
        # Audit the sampled interval as well as the analytical trajectory bounds.
        jerk=(acceleration-self.acceleration)/self.dt
        if np.any(np.abs(velocity)>self.max_rate+1e-7) or np.any(np.abs(acceleration)>self.max_acceleration+1e-7) or np.any(np.abs(jerk)>self.max_jerk+1e-5):
            raise ValueError('Trajectory numerical limit violation; independent stop required')
        self.position,self.rate,self.acceleration=position,velocity,acceleration
        return position.copy(),dict(reference_rate_per_s=velocity.tolist(),reference_acceleration_per_s2=acceleration.tolist(),
            interval_mean_jerk_per_s3=jerk.tolist(),position_target_clipped=clipped,zero_target_rate_fallback=fallback,
            target_model_sha256=self.model_sha256,hardware_enabled=False)
