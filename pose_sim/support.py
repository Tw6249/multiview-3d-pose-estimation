"""Causal ankle support hypotheses from markerless body geometry, never force truth.

Input: named current world positions in metres (Z up), integer exposure ns and
source leg length in metres. Output: support_candidate / swing / unknown per
foot with diagnostics and unchanged exposure time. No robot model or commands.
The low ankle envelope is not a sole plane. Velocity/height gates are heuristics,
not calibrated probabilities. Missing, stale, gapped or noisy data reset contact
confirmation; unknown remains unknown. No fixed alternating gait is generated.
"""
from collections import deque
import numpy as np


class AnkleSupportEstimator:
    def __init__(self,config):
        self.cfg=dict(config);self.last_time=None;self.floor_history=deque()
        self.motion=deque(maxlen=self.cfg['velocity_samples'])
        self.confirmed=[False,False];self.counts=[0,0]
        self.anchor_source=[None,None]

    def reset_motion(self):
        self.motion.clear();self.confirmed=[False,False];self.counts=[0,0]
        self.anchor_source=[None,None]

    def update(self,timestamp_ns,positions,leg_length_m,current=True):
        if type(timestamp_ns) is not int or (self.last_time is not None and timestamp_ns<=self.last_time):
            raise ValueError('Support estimator requires increasing exposure ns')
        if self.last_time is not None and (timestamp_ns-self.last_time)/1e9>self.cfg['maximum_gap_s']:
            self.reset_motion()
        self.last_time=timestamp_ns
        result=dict(timestamp_ns=timestamp_ns,measured_contact=False,calibrated_probability=False,
                    source='current markerless ankle motion; flat-ground hypothesis',feet={})
        def unknown(reason):
            result['feet']={s:dict(status='unknown',reason=reason) for s in ('left','right')}
            return result
        try:
            ankles=np.array([positions[s+'_ankle'] for s in ('left','right')],float)
            leg_length_m=float(leg_length_m)
        except (KeyError,TypeError,ValueError):ankles=np.full((2,3),np.nan);leg_length_m=np.nan
        if not current or ankles.shape!=(2,3) or not np.isfinite(ankles).all() or not np.isfinite(leg_length_m) or leg_length_m<=0:
            self.reset_motion();return unknown('missing or noncurrent body geometry')
        self.floor_history.append((timestamp_ns,float(np.min(ankles[:,2]))))
        while self.floor_history and (timestamp_ns-self.floor_history[0][0])/1e9>self.cfg['history_seconds']:
            self.floor_history.popleft()
        self.motion.append((timestamp_ns,ankles.copy()))
        if len(self.floor_history)<self.cfg['minimum_floor_samples'] or len(self.motion)<self.motion.maxlen:
            return unknown('insufficient continuous history')
        floor=float(np.percentile([v for _,v in self.floor_history],10))
        times=np.array([(t-timestamp_ns)/1e9 for t,_ in self.motion]);times-=times.mean()
        values=np.array([p for _,p in self.motion]);mean=values.mean(axis=0)
        velocity=np.einsum('t,tsd->sd',times,values)/(times@times)
        residual=np.sqrt(np.mean(np.sum((values-mean-times[:,None,None]*velocity)**2,axis=2),axis=0))/leg_length_m
        for i,side in enumerate(('left','right')):
            speed=float(np.linalg.norm(velocity[i])/leg_length_m)
            height=float((ankles[i,2]-floor)/leg_length_m)
            reliable=residual[i]<=self.cfg['maximum_fit_residual_leg_fraction']
            slow=speed<=self.cfg['enter_speed_leg_s']
            low=abs(height)<=self.cfg['support_height_leg_fraction']
            if reliable and slow and low:self.counts[i]+=1
            else:self.counts[i]=0
            if self.confirmed[i] and (not reliable or not low or speed>self.cfg['exit_speed_leg_s']):self.confirmed[i]=False
            drift=float(np.linalg.norm(ankles[i]-self.anchor_source[i])/leg_length_m) if self.anchor_source[i] is not None else 0.
            if self.confirmed[i] and drift>self.cfg['maximum_anchor_drift_leg_fraction']:
                self.confirmed[i]=False;self.counts[i]=0
            if self.counts[i]>=self.cfg['confirmation_frames'] and not self.confirmed[i]:
                self.confirmed[i]=True;self.anchor_source[i]=mean[i].copy()
            if not self.confirmed[i]:self.anchor_source[i]=None
            status='support_candidate' if self.confirmed[i] else 'swing' if reliable and height>=self.cfg['swing_height_leg_fraction'] else 'unknown'
            result['feet'][side]=dict(status=status,ankle_height_above_support_proxy_m=height*leg_length_m,
                speed_m_s=speed*leg_length_m,fit_residual_m=float(residual[i]*leg_length_m),confirmation_frames=self.counts[i])
            result['feet'][side]['anchor_displacement_m']=drift*leg_length_m
        result['ankle_support_level_proxy_m']=floor
        return result
