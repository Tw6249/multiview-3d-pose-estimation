"""Causal descending-swing landing window from markerless ankles, no gait clock.

Named source XYZ metres, leg length metres, support hypotheses and integer
exposure ns -> optional same-side touchdown time WINDOW in the source clock.
Height is relative to an ankle support proxy, not a measured sole plane.
Linear descent extrapolation is an uncalibrated hypothesis, not contact truth.
Missing/gapped/conflicting observations clear the window. No robot state here.
"""
from collections import deque
import numpy as np


class SourceLandingWindow:
    def __init__(self,config):
        self.cfg=dict(config)
        if type(self.cfg['history_samples']) is not int or not 3<=self.cfg['history_samples']<=20:raise ValueError('Require bounded multi-frame source history')
        for key in ('maximum_gap_s','minimum_descent_leg_s','minimum_peak_clearance_leg_fraction',
                    'minimum_current_clearance_leg_fraction','clearance_residual_multiplier','maximum_fit_residual_leg_fraction',
                    'landing_time_uncertainty_s','minimum_remaining_s','maximum_remaining_s'):
            if not np.isfinite(self.cfg[key]) or self.cfg[key]<=0:raise ValueError('Invalid phase bound: '+key)
        if self.cfg['minimum_remaining_s']>=self.cfg['maximum_remaining_s']:raise ValueError('Invalid source time window')
        self.history=deque(maxlen=self.cfg['history_samples']);self.last_time=None

    def update(self,timestamp_ns,positions,leg_length_m,support,current=True):
        if type(timestamp_ns) is not int or (self.last_time is not None and timestamp_ns<=self.last_time):raise ValueError('Increasing source exposure ns required')
        if self.last_time is not None and (timestamp_ns-self.last_time)/1e9>self.cfg['maximum_gap_s']:self.history.clear()
        self.last_time=timestamp_ns
        def unknown(reason):return dict(status='unknown',source_timestamp_ns=timestamp_ns,reason=reason,event=None)
        try:
            ankles=np.array([positions[s+'_ankle'] for s in ('left','right')],float)
            valid=current and ankles.shape==(2,3) and np.isfinite(ankles).all() and np.isfinite(leg_length_m) and leg_length_m>0 and support['timestamp_ns']==timestamp_ns
        except (KeyError,ValueError,TypeError):valid=False
        if not valid:self.history.clear();return unknown('Missing, noncurrent or mismatched source observations')
        self.history.append((timestamp_ns,ankles.copy()))
        if len(self.history)<self.history.maxlen:return unknown('Insufficient continuous source motion')
        sides=('left','right');states=[support['feet'][s]['status'] for s in sides]
        if states.count('support_candidate')!=1:return unknown('Require exactly one source support hypothesis')
        side=1-states.index('support_candidate');times=np.array([(t-timestamp_ns)/1e9 for t,_ in self.history]);times-=times.mean()
        values=np.array([p[side] for _,p in self.history]);velocity=times@values/(times@times)
        residual=float(np.sqrt(np.mean(np.sum((values-values.mean(axis=0)-times[:,None]*velocity)**2,axis=1))))
        if residual>self.cfg['maximum_fit_residual_leg_fraction']*leg_length_m:return unknown('Unreliable source descent fit')
        height=support['feet'][sides[side]]['ankle_height_above_support_proxy_m']
        floor=ankles[side,2]-height
        # The high-clearance 'swing' detector becomes unknown close to touchdown.
        # Retain descent only when current clearance exceeds observed fit noise,
        # and this SAME ankle had clear elevation in the continuous history.
        clearance=max(self.cfg['minimum_current_clearance_leg_fraction']*leg_length_m,self.cfg['clearance_residual_multiplier']*residual)
        if height<=clearance or np.max(values[:,2]-floor)<self.cfg['minimum_peak_clearance_leg_fraction']*leg_length_m:return unknown('Insufficient observed swing clearance above noise')
        if not np.isfinite(height) or height<=0 or velocity[2]>=-self.cfg['minimum_descent_leg_s']*leg_length_m:return unknown('Swing is not observably descending')
        remaining=float(height/-velocity[2]);spread=self.cfg['landing_time_uncertainty_s']
        lo=max(self.cfg['minimum_remaining_s'],remaining-spread);hi=min(self.cfg['maximum_remaining_s'],remaining+spread)
        if lo>hi:return unknown('Landing outside bounded extrapolation horizon')
        return dict(status='hypothesis',source_timestamp_ns=timestamp_ns,calibrated_uncertainty=False,
            event=dict(side=sides[side],stance_side=sides[1-side],phase='observed_descending_swing',
                earliest_source_ns=timestamp_ns+round(lo*1e9),latest_source_ns=timestamp_ns+round(hi*1e9),
                nominal_source_ns=timestamp_ns+round(float(np.clip(remaining,lo,hi))*1e9),ankle_velocity_source_world_m_s=velocity.tolist(),fit_residual_m=residual))
