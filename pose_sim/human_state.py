"""Causal source-human state estimation. No robot model or ground-truth inputs.

World positions are metres, velocities m/s, timestamps integer nanoseconds.
Raw triangulation remains immutable. Prediction is bounded and explicitly marked;
it never receives fabricated image inliers or a fresh observation timestamp.
A new or expired track is 'tentative' until it has been accepted on
track_confirmation_frames consecutive updates; tentative points are output for
inspection but are never labelled 'filtered', so downstream gates cannot act on them.
"""
import argparse
from collections import deque
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares

from .geometry import project
from .landmarks import BONES
from .reacquisition import Reacquisition, current_structure_support

DEFAULTS = dict(measurement_floor_m=.018, acceleration_noise_m_s2=10.,
                innovation_gate=25., max_prediction_ms=100., reset_gap_ms=500.,
                bone_calibration_samples=12, bone_sigma_m=.006,
                bone_gate_fraction=.3, bone_gate_m=.07, ground_height_m=None,
                track_confirmation_frames=1)


def _orientation(forward, lateral):
    if np.linalg.norm(forward) < .025 or np.linalg.norm(lateral) < .02:
        return None
    x = forward/np.linalg.norm(forward)
    y = lateral-x*np.dot(lateral,x)
    if np.linalg.norm(y) < .015:
        return None
    y /= np.linalg.norm(y)
    return np.column_stack((x,y,np.cross(x,y))).tolist()


class HumanStateEstimator:
    def __init__(self, config=None):
        self.cfg = {**DEFAULTS, **(config or {})}
        for k in DEFAULTS:
            v = self.cfg[k]
            if k == 'ground_height_m' and v is None:
                continue
            if not np.isfinite(v) or (k != 'ground_height_m' and v <= 0):
                raise ValueError('Invalid human state parameter: '+k)
        if type(self.cfg['track_confirmation_frames']) is not int:
            raise ValueError('track_confirmation_frames must be a positive integer')
        self.last_time = self.signature = None
        self.filters, self.samples, self.lengths = {}, {}, {}
        self.reacquisition = Reacquisition()

    def update(self, row, cameras=None, detection=None):
        names = row['joint_names']
        t = row['timestamp_ns']
        if type(t) is not int or (self.last_time is not None and t <= self.last_time):
            raise ValueError('Source timestamps must be strictly increasing integer nanoseconds')
        if (len(names) != len(set(names)) or len(row['joints_world_m']) != len(names)
                or len(row['quality']) != len(names)):
            raise ValueError('Invalid named landmark record')
        if row.get('landmark_schema_id') not in ('coco17', 'human33'):
            raise ValueError('Unsupported human landmark schema')
        signature = (row.get('source_id'), row.get('landmark_schema_id'),
                     row.get('calibration_sha256'),row.get('perception_signature'),tuple(names))
        dt = (t-self.last_time)/1e9 if self.last_time is not None else 0.
        reset = self.signature is not None and (signature != self.signature or dt*1000 > self.cfg['reset_gap_ms'])
        if reset:
            self.filters, self.samples, self.lengths = {}, {}, {}
            self.reacquisition = Reacquisition()
            dt = 0.
        self.last_time, self.signature = t, signature
        measurements, noise, reasons = {}, {}, {}
        for name, point, q in zip(names, row['joints_world_m'], row['quality']):
            if point is None or q is None:
                reasons[name] = 'missing triangulation'
                continue
            point = np.asarray(point,float)
            sigma = q.get('linearized_max_std_m', np.inf)
            if (point.shape != (3,) or not np.isfinite(point).all() or not np.isfinite(sigma)
                    or sigma > .08 or q.get('ray_angle_deg',0) < 2
                    or q.get('mean_reprojection_px',np.inf) > 8 or len(set(q.get('inlier_cameras',[]))) < 2):
                reasons[name] = 'unreliable triangulation'
                continue
            measurements[name] = point
            # Explicit heuristic floor; not an empirically calibrated confidence interval.
            noise[name] = max(float(sigma), self.cfg['measurement_floor_m'])
            if len(q['inlier_cameras']) == 2:
                noise[name] *= 1.7
        for state in self.filters.values():
            f = np.eye(6); f[:3,3:] = np.eye(3)*dt
            g = np.vstack((np.eye(3)*(.5*dt*dt),np.eye(3)*dt))
            state['x'] = f@state['x']
            state['p'] = f@state['p']@f.T + g@g.T*self.cfg['acceleration_noise_m_s2']**2
        independent_measurements = dict(measurements)
        recovery_velocity = {}
        if self.cfg.get('state_reacquisition', False):
            quality_by_name = dict(zip(names, row['quality']))
            self.reacquisition.retain(measurements)
            for name, z in measurements.items():
                q = quality_by_name[name]
                supported = (len(set(q['inlier_cameras'])) >= 3 or
                             current_structure_support(name, independent_measurements, quality_by_name))
                for a, b in BONES:
                    neighbour = b if a == name else a if b == name else None
                    key = a+'--'+b
                    if neighbour not in independent_measurements or key not in self.lengths:
                        continue
                    tolerance = self._bone_gate_tolerance(key, self.lengths[key])
                    nq = quality_by_name[neighbour]
                    if (np.isfinite(tolerance) and len(set(nq['inlier_cameras'])) >= 3
                            and abs(np.linalg.norm(z-independent_measurements[neighbour])-self.lengths[key]) <= tolerance):
                        supported = True
                recovery_velocity[name] = self.reacquisition.observe(name, z, t, noise[name], supported)
        # Mature bone lengths act as an outlier gate. Select the endpoint with
        # poorer temporal agreement, rather than withdrawing an entire person.
        for a,b in BONES:
            key = a+'--'+b
            if a not in measurements or b not in measurements or key not in self.lengths:
                continue
            length = np.linalg.norm(measurements[b]-measurements[a])
            expected = self.lengths[key]
            if abs(length-expected) <= self._bone_gate_tolerance(key,expected):
                continue
            def deviation(n):
                return (np.linalg.norm(measurements[n]-self.filters[n]['x'][:3])
                        if n in self.filters else noise[n]*3)
            bad = max((a,b),key=deviation)
            del measurements[bad]
            reasons[bad] = 'bone length outlier'
        accepted = set(); reacquired = set()
        for name, z in measurements.items():
            r = np.eye(3)*noise[name]**2
            state = self.filters.get(name)
            expired = state is not None and (t-state['last_observed'])/1e6 > self.cfg['max_prediction_ms']
            if state is None or expired:
                velocity = recovery_velocity.get(name) if expired else None
                state = dict(x=np.r_[z,velocity if velocity is not None else [0.,0.,0.]],
                             p=np.diag([noise[name]**2]*3+[4.]*3),last_observed=t,observations=1,
                             streak=0,confirmed=False)
                self.filters[name] = state
                if expired and self.cfg.get('state_reacquisition', False):
                    # Fresh image geometry remains available after a gap. Do not
                    # call it a continuing observed motion or allow it to drive.
                    reacquired.add(name)
                    reasons[name] = 'fresh geometry after expired track; control transition required'
                    continue
            else:
                innovation = z-state['x'][:3]
                s = state['p'][:3,:3]+r
                if innovation@np.linalg.solve(s,innovation) > self.cfg['innovation_gate']:
                    reasons[name] = 'temporal innovation outlier'
                    velocity = recovery_velocity.get(name)
                    if velocity is not None:
                        self.filters[name] = dict(x=np.r_[z,velocity],
                            p=np.diag([max(.05,noise[name])**2]*3+[4.]*3),last_observed=t,observations=1,
                            streak=0,confirmed=False)
                        reacquired.add(name)
                        reasons[name] = 'continuous current observations overrule history; control transition required'
                    continue
                gain = np.linalg.solve(s,state['p'][:3,:]).T
                state['x'] += gain@innovation
                h = np.zeros((3,6)); h[:,:3] = np.eye(3)
                a = np.eye(6)-gain@h
                state['p'] = a@state['p']@a.T+gain@r@gain.T
                state['last_observed'] = t
                state['observations'] += 1
            accepted.add(name)
        for name, state in self.filters.items():
            # Confirmation needs consecutive acceptances; any miss or rejection
            # before confirmation restarts the count. Confirmed tracks keep their
            # status until they expire or the estimator resets.
            if state.get('confirmed', True):
                continue
            state['streak'] = state['streak']+1 if name in accepted else 0
            state['confirmed'] = state['streak'] >= self.cfg['track_confirmation_frames']
        tentative = {n for n,s in self.filters.items() if not s.get('confirmed', True)}
        confirmed_accepted = accepted-tentative
        self._learn_lengths(measurements, confirmed_accepted, noise)
        visible = {n: s['x'][:3].copy() for n,s in self.filters.items()
                   if (t-s['last_observed'])/1e6 <= self.cfg['max_prediction_ms']}
        before = {n:p.copy() for n,p in visible.items()}
        self._constrain(visible)
        points, qualities, states, velocities = [], [], [], []
        for i,name in enumerate(names):
            state, point = self.filters.get(name), visible.get(name)
            age = t-state['last_observed'] if state else None
            if point is None:
                points.append(None); qualities.append(None); velocities.append(None)
                states.append(dict(status='missing',last_observed_timestamp_ns=state['last_observed'] if state else None,
                                   observation_age_ns=age,reason=reasons.get(name,'prediction expired')))
                continue
            correction = point-before[name]
            state['x'][:3] = point
            covariance = state['p'][:3,:3]+np.outer(correction,correction)
            sigma = float(np.sqrt(np.linalg.eigvalsh(covariance).max()))
            points.append(point.tolist()); velocities.append(state['x'][3:].tolist())
            status = ('reacquiring' if name in reacquired else 'tentative' if name in tentative
                      else 'filtered' if name in accepted else 'predicted')
            states.append(dict(status=status,last_observed_timestamp_ns=state['last_observed'],
                               observation_age_ns=age,position_covariance_m2=covariance.tolist(),
                               max_std_m=sigma,structure_correction_m=float(np.linalg.norm(correction)),
                               reason=reasons.get(name) or ('track awaiting confirmation' if name in tentative else None),
                               track_reinitialized=name in reacquired,
                               confirmation_streak=state.get('streak') if name in tentative else None))
            q = copy.deepcopy(row['quality'][i]) if name in accepted else None
            if q is not None:
                q['raw_measurement_reprojection_px'] = q['mean_reprojection_px']
                q['linearized_max_std_m'] = max(q['linearized_max_std_m'],sigma)
                q['estimator_covariance_m2'] = covariance.tolist()
                if cameras is not None and detection is not None:
                    residuals = [np.linalg.norm(project(point,cameras[c])[0]-detection['views'][c]['keypoints_px'][i])
                                 for c in q['inlier_cameras']]
                    q['mean_reprojection_px'] = float(np.mean(residuals))
                else:
                    # Cannot reuse a raw-point reprojection score for a corrected point.
                    q = None
            qualities.append(q)
        result = {k:copy.deepcopy(row[k]) for k in ('frame_id','timestamp_ns','source_id','landmark_schema_id',
                                                   'joint_names','calibration_sha256','perception_signature',
                                                   'landmark_model_sources') if k in row}
        result.update(state_schema_id='human_state_v1',joints_world_m=points,velocities_world_m_s=velocities,
                      quality=qualities,joint_states=states,bone_lengths_m=dict(self.lengths),reset=bool(reset),
                      estimator='causal constant-velocity Kalman and source bone constraints',
                      uncertainty_note='Heuristic observation floor and motion model; excludes calibration and correspondence uncertainty.',
                      orientations=self._orientations(visible,confirmed_accepted),
                      orientation_convention='World-from-landmark-frame; foot X heel to toes, Y anatomical left; palm X wrist to middle MCP, Y pinky to index. These are landmark frames, not anatomical joint rotations.',
                      contacts=self._contacts(visible,confirmed_accepted),
                      unobserved=['anatomical axial twist','pelvis versus waist tilt separation','contact forces'])
        return result

    def _learn_lengths(self, measurements, accepted, noise):
        for a,b in BONES:
            key = a+'--'+b
            if a in accepted and b in accepted and key not in self.lengths:
                length = float(np.linalg.norm(measurements[b]-measurements[a]))
                if .08 < length < .85:
                    sample = self.samples.setdefault(key,deque(maxlen=int(self.cfg['bone_calibration_samples'])))
                    sample.append(length)
                    if len(sample) == sample.maxlen:
                        self.lengths[key] = float(np.median(sample))

    def _bone_gate_tolerance(self, key, length):
        return max(self.cfg['bone_gate_m'],length*self.cfg['bone_gate_fraction'])

    def _bone_constraint_sigma(self, key):
        return self.cfg['bone_sigma_m']

    def _constrain(self, points):
        constraints = [(a,b,self.lengths[a+'--'+b]) for a,b in BONES
                       if a in points and b in points and a+'--'+b in self.lengths]
        if not constraints:
            return
        names = sorted({n for a,b,_ in constraints for n in (a,b)})
        index = {n:i for i,n in enumerate(names)}
        target = np.array([points[n] for n in names])
        sigmas = np.array([max(self.cfg['measurement_floor_m'],
                        np.sqrt(np.linalg.eigvalsh(self.filters[n]['p'][:3,:3]).max())) for n in names])
        bone_sigmas = {a+'--'+b:self._bone_constraint_sigma(a+'--'+b) for a,b,_ in constraints}
        def residual(x):
            p = x.reshape(-1,3)
            return np.r_[((p-target)/sigmas[:,None]).ravel(),
                         [(np.linalg.norm(p[index[b]]-p[index[a]])-length)/bone_sigmas[a+'--'+b]
                          for a,b,length in constraints]]
        def jacobian(x):
            p = x.reshape(-1,3)
            j = np.zeros((target.size+len(constraints),target.size))
            j[:target.size] = np.diag(np.repeat(1/sigmas,3))
            for r,(a,b,_) in enumerate(constraints,target.size):
                ia,ib = index[a],index[b]; delta=p[ib]-p[ia]
                grad = delta/max(np.linalg.norm(delta),1e-9)/bone_sigmas[a+'--'+b]
                j[r,3*ia:3*ia+3]=-grad; j[r,3*ib:3*ib+3]=grad
            return j
        fit = least_squares(residual,target.ravel(),jac=jacobian,method='lm',max_nfev=8)
        if np.isfinite(fit.x).all():
            for name,p in zip(names,fit.x.reshape(-1,3)):
                points[name]=p

    def _orientations(self, p, accepted):
        result = {}
        for side in ('left','right'):
            foot = [f'{side}_{n}' for n in ('heel','big_toe','small_toe')]
            hand = [f'{side}_hand_{n}' for n in ('wrist','middle_mcp','index_mcp','pinky_mcp')]
            if all(n in accepted for n in foot):
                heel,big,small = [p[n] for n in foot]
                lateral = small-big if side=='left' else big-small
                result[side+'_foot'] = _orientation((big+small)/2-heel,lateral)
            else: result[side+'_foot'] = None
            if all(n in accepted for n in hand):
                wrist,middle,index,pinky = [p[n] for n in hand]
                result[side+'_palm'] = _orientation(middle-wrist,index-pinky)
            else: result[side+'_palm'] = None
        return result

    def _contacts(self, p, accepted):
        result = {}
        for side in ('left','right'):
            names = [f'{side}_{n}' for n in ('heel','big_toe','small_toe')]
            if (self.cfg['ground_height_m'] is None or not all(n in accepted for n in names)
                    or any(self.filters[n]['observations']<3 for n in names)):
                result[side] = None
                continue
            scores=[]
            for n in names:
                height=abs(p[n][2]-self.cfg['ground_height_m'])
                speed=np.linalg.norm(self.filters[n]['x'][3:])
                scores.append(float(np.exp(-.5*(height/.035)**2-.5*(speed/.2)**2)))
            result[side] = dict(stationary_contact_score=max(scores),
                                ground_height_m=self.cfg['ground_height_m'],calibrated_probability=False)
        return result


def estimate_session(session, output=None, config=None):
    from .pipeline import read_jsonl,write_jsonl
    session=Path(session); output=Path(output) if output else session
    cfg=json.loads(Path(config).read_text()) if config else {}
    engine=HumanStateEstimator(cfg)
    cameras=json.loads((session/'cameras.json').read_text())['cameras']
    detections={r['frame_id']:r for r in read_jsonl(output/'detections.jsonl')}
    states,times=[],[]
    source_sha=hashlib.sha256((output/'poses3d.jsonl').read_bytes()).hexdigest()
    for row in read_jsonl(output/'poses3d.jsonl'):
        begin=time.perf_counter()
        states.append(engine.update(row,cameras,detections[row['frame_id']]))
        states[-1]['source_poses_sha256']=source_sha
        times.append(time.perf_counter()-begin)
    write_jsonl(output/'states3d.jsonl',states)
    (output/'state_timing.json').write_text(json.dumps(dict(mean_seconds=float(np.mean(times)),
        p95_seconds=float(np.percentile(times,95)),configuration=engine.cfg),indent=2))
    return states


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('session',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--config',type=Path)
    args=parser.parse_args();estimate_session(args.session,args.output,args.config)
