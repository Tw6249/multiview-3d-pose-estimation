"""Quality/freshness state machine for a planner, independent of robot hardware.

This does not certify balance or send commands. 'transition' requests a bounded
planner blend; 'stop' requests its independent stop path. Only current observed
legs with domain-matched uncertainty may initiate 'follow'.
"""
import numpy as np


class PerceptionGate:
    def __init__(self,max_age_ms=150.,stop_age_ms=300.,max_radius_m=.1,recovery_frames=3):
        self.max_age_ns=int(max_age_ms*1e6);self.stop_age_ns=int(stop_age_ms*1e6)
        self.max_radius=max_radius_m;self.recovery_frames=recovery_frames
        self.good=0;self.last_usable=None;self.last_stamp=None;self.signature=None

    def update(self,state,now_ns):
        if type(now_ns) is not int:raise ValueError('Monotonic common-clock time must be integer ns')
        if state is None:
            self.good=0
            mode='stop' if self.last_usable is None or now_ns-self.last_usable>self.stop_age_ns else 'transition'
            return dict(mode=mode,reason='no fresh state',allow_new_motion=False)
        t=state['timestamp_ns'];age=now_ns-t
        if type(t) is not int or age<0:raise ValueError('Exposure and output clock mismatch')
        signature=tuple(state.get(k) for k in ('source_id','calibration_sha256','perception_signature'))
        if self.signature is not None and signature!=self.signature:
            self.good=0;self.last_usable=None
        self.signature=signature
        if self.last_stamp is not None and t-self.last_stamp>self.max_age_ns:self.good=0
        reason=None
        if self.last_stamp is not None and t<=self.last_stamp:reason='reused or out-of-order observation'
        self.last_stamp=max(t,self.last_stamp) if self.last_stamp is not None else t
        if age>self.max_age_ns:reason='stale exposure'
        if state.get('delivery',{}).get('valid_for_new_motion') is False:reason='delivery deadline missed'
        if not state.get('uncertainty_calibration',{}).get('valid_for_domain'):reason='uncertainty uncalibrated for this domain'
        samples=dict(zip(state['joint_names'],state['joint_states']))
        for name in (s+'_'+j for s in ('left','right') for j in ('hip','knee','ankle')):
            sample=samples.get(name,{})
            radius=sample.get('error_radius_m')
            if (sample.get('status')!='filtered' or sample.get('last_observed_timestamp_ns')!=t
                    or radius is None or not np.isfinite(radius) or radius>self.max_radius):
                reason='unobserved or uncertain '+name;break
        if reason:
            self.good=0
            mode='stop' if self.last_usable is None or now_ns-self.last_usable>self.stop_age_ns else 'transition'
        else:
            self.good+=1;self.last_usable=t
            mode='follow' if self.good>=self.recovery_frames else 'transition'
            reason='consecutive observed states' if mode=='follow' else 'recovery hysteresis'
        return dict(mode=mode,reason=reason,allow_new_motion=mode=='follow',observation_age_ms=age/1e6)
