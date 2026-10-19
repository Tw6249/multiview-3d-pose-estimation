"""Offline finite-horizon withdrawal search using robot-only branch simulation.

Current G1 integration state + policy history + applied 35D reference -> one
bounded-duration withdrawal schedule and diagnostics. World Z up, m/rad/s;
integer robot simulation-clock ns. No future human observations, no source
timestamp generation. This module must not run against a live device clock.
It predicts contact changes with full dynamics, not a constant-height model.
Exact model/state is a simulation assumption, not an observed human pose.
"""
from collections import deque
import copy
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .twist import TwistPlant,TwistPolicy,DEFAULT_REF,DEFAULT_Q,CONFIG
from .stop_transition import smoothstep
from pose_sim.robot_assets import require_asset


def load_config():
    return json.loads((Path(__file__).resolve().parents[1]/'config/g1_recovery_rollout.json').read_text())


def snapshot(plant):
    spec=plant.mj.mjtState.mjSTATE_INTEGRATION
    state=np.empty(plant.mj.mj_stateSize(plant.model,spec))
    plant.mj.mj_getState(plant.model,plant.data,state,spec)
    return state


def branch(plant,policy,state):
    """Separate mjData/history; share immutable model and inference session."""
    p=object.__new__(TwistPlant)
    for name in ('mj','model','sha256','joint_ids','limits','effort'):setattr(p,name,getattr(plant,name))
    p.data=p.mj.MjData(p.model);p.mj.mj_setState(p.model,p.data,state,p.mj.mjtState.mjSTATE_INTEGRATION)
    # mj_step will refresh dynamics exactly as on the original next tick.
    q=object.__new__(TwistPolicy);q.session=policy.session;q.input=policy.input
    q.history=deque([v.copy() for v in policy.history],maxlen=10);q.last_action=policy.last_action.copy()
    return p,q


class BatchedPolicy:
    def __init__(self,device='cuda'):
        import onnxruntime as ort
        if device not in ('cpu','cuda'):raise ValueError('Explicit policy device required')
        if device=='cuda':
            from pose_sim.runtime import preload_cuda
            preload_cuda(ort)
        expected='CUDAExecutionProvider' if device=='cuda' else 'CPUExecutionProvider'
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        providers=[(expected,{'use_tf32':'0'})] if device=='cuda' else [expected]
        self.session=ort.InferenceSession(str(require_asset(CONFIG['policy_asset'])),sess_options=options,providers=providers)
        if self.session.get_providers()[0]!=expected:raise RuntimeError('Refuse silent policy device fallback')
        self.session.disable_fallback();self.input=self.session.get_inputs()[0].name;self.device=device

    def actions(self,observations):
        observations=np.asarray(observations,np.float32)
        if observations.ndim!=2 or observations.shape[1]!=1432 or not np.isfinite(observations).all():raise ValueError('Invalid batched policy observation')
        action=self.session.run(None,{self.input:observations})[0]
        if action.shape!=(len(observations),29) or not np.isfinite(action).all():raise ValueError('Invalid rollout policy output')
        return action


def boundary_reference(plant,applied,mix):
    measured=np.asarray(applied,float).copy();measured[6:]=plant.data.qpos[7:]
    yaw=Rotation.from_quat(plant.data.qpos[[4,5,6,3]]).as_euler('xyz')[2]
    measured[:2]=(Rotation.from_euler('z',yaw).as_matrix().T@plant.data.qvel[:3])[:2]
    measured[2]=plant.data.qpos[2]
    return (1-mix)*np.asarray(applied)+mix*measured


def scheduled_reference(elapsed,entry,parameters):
    delay,blend_seconds,brake_seconds,_,gain=parameters
    blend=smoothstep((elapsed-delay)/blend_seconds)
    result=(1-blend)*entry+blend*DEFAULT_REF
    result[[0,1,5]]=gain*entry[[0,1,5]]*(1-smoothstep(elapsed/brake_seconds))
    return result


def loaded_feet(plant,foot_ids,threshold):
    """Loaded floor contact, not an inferred human foot pose."""
    forces={s:0. for s in foot_ids}
    for index,c in enumerate(plant.data.contact):
        bodies=[int(plant.model.geom_bodyid[g]) for g in (c.geom1,c.geom2)]
        if 0 not in bodies:continue
        force=np.zeros(6);plant.mj.mj_contactForce(plant.model,plant.data,index,force)
        for side,body in foot_ids.items():
            if body in bodies:forces[side]+=max(0.,float(force[0]))
    return {s for s,f in forces.items() if f>threshold}


def rollout_metrics(plant):
    angles=Rotation.from_quat(plant.data.qpos[[4,5,6,3]]).as_euler('xyz')[:2]
    return (float(plant.data.qpos[2]),float(max(abs(angles))),
            float(np.linalg.norm(plant.data.qvel[:3])),float(np.linalg.norm(plant.data.qvel[3:6])))


def assess_history(history,cfg):
    h=np.asarray(history);tail=round(cfg['terminal_window_seconds']/cfg['control_dt_seconds'])
    if h.ndim!=2 or h.shape[1]!=4 or len(h)<tail or not np.isfinite(h).all():raise ValueError('Invalid dynamics rollout history')
    end=h[-tail:];height=float(h[:,0].min());tilt=float(h[:,1].max())
    speed=float(end[:,2].mean());angular=float(end[:,3].mean())
    feasible=bool(height>cfg['minimum_height_m'] and tilt<cfg['maximum_tilt_rad'] and
        end[:,0].min()>cfg['terminal_minimum_height_m'] and end[:,1].max()<cfg['terminal_maximum_tilt_rad'] and
        speed<cfg['terminal_maximum_speed_m_s'] and angular<cfg['terminal_maximum_angular_speed_rad_s'])
    return dict(feasible=feasible,minimum_height_m=height,maximum_tilt_rad=tilt,
        terminal_minimum_height_m=float(end[:,0].min()),terminal_maximum_tilt_rad=float(end[:,1].max()),
        terminal_speed_m_s=speed,terminal_angular_speed_rad_s=angular,
        cost=float(1000*(not feasible)+100*max(0,cfg['minimum_height_m']-height)+10*tilt**2+50*speed**2+10*end[:,1].max()**2))


class RecoveryRollout:
    def __init__(self,config=None):
        self.cfg=copy.deepcopy(config or load_config())
        if self.cfg.get('hardware_enabled') is not False or self.cfg.get('online_eligible') is not False or self.cfg.get('target')!='unitree_g1_29dof':raise ValueError('Offline G1 profile required')
        if self.cfg['control_dt_seconds']!=.02:raise ValueError('Pinned policy requires 50Hz reference updates')
        if self.cfg['device'] not in ('cpu','cuda'):raise ValueError('Explicit CPU or CUDA device required')
        for key in ('horizon_seconds','terminal_window_seconds','minimum_height_m','maximum_tilt_rad',
                    'terminal_minimum_height_m','terminal_maximum_tilt_rad','terminal_maximum_speed_m_s',
                    'terminal_maximum_angular_speed_rad_s','loaded_contact_threshold_n'):
            if not np.isfinite(self.cfg[key]) or self.cfg[key]<=0:raise ValueError('Invalid positive rollout bound: '+key)
        for key in ('delay_range_seconds','pose_blend_range_seconds','brake_range_seconds','entry_feedback_mix_range','velocity_gain_range'):
            v=np.asarray(self.cfg[key],float)
            if v.shape!=(2,) or not np.isfinite(v).all() or v[0]<0 or v[1]<v[0]:raise ValueError('Invalid candidate range: '+key)
        if min(self.cfg['pose_blend_range_seconds'])<=0 or min(self.cfg['brake_range_seconds'])<=0 or max(self.cfg['entry_feedback_mix_range'])>1:raise ValueError('Invalid blend bounds')
        if type(self.cfg['candidates']) is not int or not 4<=self.cfg['candidates']<=256:raise ValueError('Require bounded candidate count')
        if type(self.cfg['maximum_cpu_verifications']) is not int or not 1<=self.cfg['maximum_cpu_verifications']<=self.cfg['candidates']:raise ValueError('Invalid verification budget')
        rates=np.asarray(self.cfg['reference_rate_per_second'],float)
        if rates.shape!=(35,) or not np.isfinite(rates).all() or np.any(rates<=0):raise ValueError('Invalid reference rate bounds')
        perturbations=np.asarray(self.cfg['diagnostic_velocity_perturbations_world_m_s'],float)
        if perturbations.ndim!=2 or perturbations.shape[1]!=3 or len(perturbations)>16 or not np.isfinite(perturbations).all():raise ValueError('Invalid velocity sensitivity cases')
        if self.cfg['terminal_window_seconds']<.02:raise ValueError('Require at least one terminal sample')
        if self.cfg['horizon_seconds']<max(self.cfg['delay_range_seconds'])+max(self.cfg['pose_blend_range_seconds'])+self.cfg['terminal_window_seconds']:
            raise ValueError('Horizon must cover the complete transition and standing tail')
        self.policy=None

    def candidate_parameters(self):
        from scipy.stats import qmc
        limits=np.array([self.cfg[k] for k in ('delay_range_seconds','pose_blend_range_seconds','brake_range_seconds','entry_feedback_mix_range','velocity_gain_range')])
        n=self.cfg['candidates']
        if n<4:raise ValueError('At least four withdrawal candidates required')
        samples=qmc.Halton(5,scramble=True,seed=self.cfg['seed']).random(n-4)
        anchors=np.array([[0,.12,.1,0,1],[.12,1.5,.3,1,1],[.6,1.5,.3,1,1],[0,.4,.6,1,1]])
        return np.r_[np.clip(anchors,limits[:,0],limits[:,1]),limits[:,0]+samples*(limits[:,1]-limits[:,0])]

    def verify(self,plant,policy,saved,applied,entry,parameters,velocity_offset=None):
        """Replay using the actual CPU controller; GPU prediction alone cannot pass."""
        p,pol=branch(plant,policy,saved);ref=applied.copy();history=[];dt=self.cfg['control_dt_seconds']
        if velocity_offset is not None:p.data.qvel[:3]+=np.asarray(velocity_offset)
        delta=np.asarray(self.cfg['reference_rate_per_second'])*dt
        for tick in range(round(self.cfg['horizon_seconds']/dt)):
            requested=scheduled_reference(tick*20_000_000/1e9,entry,parameters)
            ref+=np.clip(requested-ref,-delta,delta)
            p.step(pol.step(ref,*p.state()),dt);history.append(rollout_metrics(p))
        return assess_history(history,self.cfg)

    def plan(self,plant,policy,applied,timestamp_ns,clock_domain='offline_simulation'):
        if clock_domain!='offline_simulation' or type(timestamp_ns) is not int or timestamp_ns<0:raise ValueError('Recovery search cannot freeze a physical device clock')
        applied=np.asarray(applied,float)
        if applied.shape!=(35,) or not np.isfinite(applied).all():raise ValueError('Invalid applied reference')
        if abs(plant.model.opt.timestep-.001)>1e-12:raise ValueError('Require pinned 1kHz simulation physics')
        begin=time.perf_counter();rpy=Rotation.from_quat(plant.data.qpos[[4,5,6,3]]).as_euler('xyz')
        if plant.data.qpos[2]<self.cfg['minimum_height_m'] or max(abs(rpy[:2]))>self.cfg['maximum_tilt_rad']:
            return None,dict(accepted=False,reason='Entry already outside tested recovery envelope',wall_ms=(time.perf_counter()-begin)*1000,candidate_count=0)
        if self.policy is None:self.policy=BatchedPolicy(self.cfg['device'])
        parameters=self.candidate_parameters();saved=snapshot(plant)
        branches=[branch(plant,policy,saved) for _ in parameters]
        entries=[boundary_reference(plant,applied,p[3]) for p in parameters]
        refs=np.repeat(applied[None],len(parameters),axis=0)
        dt=self.cfg['control_dt_seconds'];delta=np.asarray(self.cfg['reference_rate_per_second'])*dt
        count=round(self.cfg['horizon_seconds']/dt);hist=[[] for _ in parameters]
        contact_history=[[] for _ in parameters];landings=[[] for _ in parameters];foot_ids={s:plant.model.body(n).id for s,n in self.cfg['feet'].items()}
        probe,_=branch(plant,policy,saved);probe.mj.mj_forward(probe.model,probe.data)
        initial_contacts=loaded_feet(probe,foot_ids,self.cfg['loaded_contact_threshold_n'])
        inference_ms=0.
        for tick in range(count):
            observations=[]
            for i,(p,pol) in enumerate(branches):
                requested=scheduled_reference(tick*20_000_000/1e9,entries[i],parameters[i]);refs[i]+=np.clip(requested-refs[i],-delta,delta)
                observations.append(pol.observation(refs[i],*p.state())[0])
            start=time.perf_counter();actions=self.policy.actions(observations);inference_ms+=(time.perf_counter()-start)*1000
            for i,((p,pol),action) in enumerate(zip(branches,actions)):
                pol.last_action=action.copy();p.step(DEFAULT_Q+np.clip(action,-10,10)*CONFIG['action_scale'],dt)
                hist[i].append(rollout_metrics(p))
                contacts=loaded_feet(p,foot_ids,self.cfg['loaded_contact_threshold_n'])
                previous=contact_history[i][-1] if contact_history[i] else initial_contacts
                for side in contacts-previous:landings[i].append(dict(time_s=(tick+1)*dt,side=side,foot_body_world_m=p.data.xpos[foot_ids[side]].tolist()))
                contact_history[i].append(contacts)
        results=[]
        for i,p in enumerate(parameters):
            results.append(dict(id=i,parameters=p.tolist(),**assess_history(hist[i],self.cfg),
                contact_change_count=sum(a!=b for a,b in zip([initial_contacts]+contact_history[i],contact_history[i])),landings=landings[i]))
        feasible=sorted([r for r in results if r['feasible']],key=lambda r:r['cost']);winner=None;verified=[]
        for candidate in feasible[:self.cfg['maximum_cpu_verifications']]:
            i=candidate['id'];result=self.verify(plant,policy,saved,applied,entries[i],parameters[i]);verified.append(dict(id=i,**result))
            if result['feasible']:winner=candidate;break
        sensitivity=[]
        if winner is not None:
            i=winner['id']
            for offset in self.cfg['diagnostic_velocity_perturbations_world_m_s']:
                sensitivity.append(dict(initial_velocity_offset_world_m_s=offset,
                    **self.verify(plant,policy,saved,applied,entries[i],parameters[i],offset)))
        diagnostic=dict(accepted=winner is not None,reason='Nominal branch passed actual-controller verification' if winner else 'No verified candidate satisfies the horizon checks',
            wall_ms=(time.perf_counter()-begin)*1000,batched_policy_ms=inference_ms,device=self.policy.device,candidate_count=len(parameters),candidates=results,
            cpu_verifications=verified,selected=None if winner is None else winner['id'],hardware_enabled=False,online_eligible=False,source_future_used=False,
            velocity_sensitivity=sensitivity,sensitivity_scope='Diagnostic only, not used to reselect candidates. Fixed +/-0.05 m/s initial world velocity perturbations are not calibrated uncertainty or model mismatch coverage.',
            contact_scope='Loaded contact above threshold; touchdown position is the robot foot body origin, not a measured human foot orientation')
        plan=None if winner is None else dict(timestamp_ns=timestamp_ns,model_sha256=plant.sha256,entry=entries[winner['id']],parameters=parameters[winner['id']])
        return plan,diagnostic
