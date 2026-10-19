"""Causal heel/toe support evidence, independent of robots and ankle proxies.

Inputs: CURRENT named foot points in Z-up world metres, velocities m/s, filter
position std (m) and velocity covariance (m²/s²), attachment validity, a supplied
flat floor height (m), and increasing exposure ns. Output: contact / swing /
unknown hypotheses, witness landmarks and unchanged exposure time. 'contact'
is the existing human-state schema label, NOT measured force or a probability.

Near-floor motion may be sliding or foot rolling: low stationarity does not
prove flight. Swing requires every heel/toe point clearly above the supplied
plane. Missing, uncertain, conflicting or gapped evidence withdraws old labels.
The floor and covariance scales remain uncalibrated; no floor is learned here.
"""
import json
from pathlib import Path
import numpy as np


def load_config():
    return json.loads((Path(__file__).resolve().parents[1]/'config/human_support.json').read_text())


class FootSupportEstimator:
    def __init__(self, config=None):
        self.cfg=load_config() if config is None else dict(config)
        for k,v in self.cfg.items():
            if k in ('schema_version','status'):continue
            if not np.isfinite(v) or v<=0:raise ValueError('Invalid support bound: '+k)
        if type(self.cfg['confirmation_frames']) is not int or self.cfg['confirmation_frames']<2:
            raise ValueError('Support requires repeated current evidence')
        if self.cfg['minimum_stationary_score']>=1:raise ValueError('Invalid stationary score bound')
        self.last_time=None;self.memory={}

    def update(self,timestamp_ns,positions,velocities,states,attachments,ground_height_m):
        if type(timestamp_ns) is not int or (self.last_time is not None and timestamp_ns<=self.last_time):
            raise ValueError('Increasing exposure ns required')
        if self.last_time is not None and (timestamp_ns-self.last_time)/1e9>self.cfg['maximum_gap_s']:
            self.memory.clear()
        self.last_time=timestamp_ns;feet={}
        for side in ('left','right'):
            def withdraw(reason,score=None):
                self.memory.pop(side,None)
                return dict(state='unknown',score=score,reason=reason,calibrated_probability=False,
                            witness_landmarks=[],confirmation_frames=0)
            names=[side+'_'+n for n in ('heel','big_toe','small_toe')]
            if not isinstance(ground_height_m,(int,float)) or not np.isfinite(ground_height_m):
                feet[side]=withdraw('No supplied finite ground plane');continue
            if not attachments.get(side,False):
                feet[side]=withdraw('Foot attachment unsupported');continue
            try:
                p=np.array([positions[n] for n in names],float)
                sigmas=np.array([states[n]['max_std_m'] for n in names],float)
                valid=(p.shape==(3,3) and sigmas.shape==(3,) and
                       all(states[n].get('status')=='filtered' and states[n].get('last_observed_timestamp_ns')==timestamp_ns for n in names) and
                       np.isfinite(np.r_[p.ravel(),sigmas]).all() and np.all(sigmas>=0))
            except (KeyError,ValueError,TypeError):valid=False
            if not valid:
                feet[side]=withdraw('Missing or malformed current landmark kinematics');continue
            if max(sigmas)>self.cfg['maximum_position_std_m']:
                feet[side]=withdraw('Landmark position uncertainty too large');continue
            height=p[:,2]-ground_height_m
            lifted=np.all(height-self.cfg['swing_uncertainty_multiplier']*sigmas>self.cfg['minimum_swing_height_m'])
            # Clearance is observable from position alone. A poor derivative must
            # not erase that evidence; velocity confidence is needed to infer
            # stationarity near the floor, not to establish geometric clearance.
            witnesses=[];score=None
            if not lifted:
                try:
                    v=np.array([velocities[n] for n in names],float)
                    cov=np.array([states[n]['velocity_covariance_m2_s2'] for n in names],float)
                    valid=v.shape==(3,3) and cov.shape==(3,3,3) and np.isfinite(np.r_[v.ravel(),cov.ravel()]).all() and np.allclose(cov,cov.transpose(0,2,1),atol=1e-10,rtol=0)
                    eigen=np.linalg.eigvalsh(cov) if valid else np.array([-1.])
                    valid=valid and eigen.min()>=-1e-10
                except (KeyError,ValueError,TypeError):valid=False
                if not valid:
                    feet[side]=withdraw('Missing or malformed velocity covariance');continue
                if np.sqrt(max(0.,float(eigen.max())))>self.cfg['maximum_velocity_std_m_s']:
                    feet[side]=withdraw('Landmark velocity uncertainty too large for stationarity');continue
                speed=np.linalg.norm(v,axis=1)
                scores=np.exp(-.5*(height/self.cfg['height_score_scale_m'])**2-.5*(speed/self.cfg['speed_score_scale_m_s'])**2)
                witnesses=[n for n,s in zip(names,scores) if s>self.cfg['minimum_stationary_score']]
                score=float(max(scores))
            target='contact' if witnesses else 'swing' if lifted else 'unknown'
            if target=='unknown':
                feet[side]=withdraw('Neither stationary near-plane point nor whole-foot clearance',score);continue
            old=self.memory.get(side,{})
            count=old.get('count',0)+1 if old.get('candidate')==target else 1
            self.memory[side]=dict(candidate=target,count=count)
            feet[side]=dict(state=target if count>=self.cfg['confirmation_frames'] else 'unknown',
                score=score,reason='Repeated current evidence' if count>=self.cfg['confirmation_frames'] else 'Confirming new evidence',
                witness_landmarks=witnesses if target=='contact' else names,
                confirmation_frames=count,calibrated_probability=False)
        labels=[feet[s]['state'] for s in ('left','right')]
        phase='unknown' if 'unknown' in labels else 'double' if labels==['contact','contact'] else 'left' if labels[0]=='contact' else 'right' if labels[1]=='contact' else 'flight'
        return dict(timestamp_ns=timestamp_ns,feet=feet,phase=phase,ground_height_m=ground_height_m,
                    physical_support_verified=False,uncertainty_calibrated=False,
                    basis='Current heel/toe kinematics against supplied flat plane; no force observation')
