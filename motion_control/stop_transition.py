"""Bounded-duration, experimental withdrawal of imitation references.

Input: control-clock integer ns, desired/applied 35D reference (body vx/vy m/s,
height m, roll/pitch rad, yaw rate rad/s, 29 joints rad), source availability,
and separately clocked robot feedback. Output: a reference to the existing rate
limiter and diagnostics. No motor output, source timestamp or new gait produced.
Fresh source passes unchanged. Missing data uses a finite stop, never extrapolated
human motion. Support timeout is explicit, not a claim that fallback is safe.
"""
import copy
import json
from pathlib import Path
import numpy as np


def load_config():
    return json.loads((Path(__file__).resolve().parents[1]/'config/g1_stop_transition.json').read_text())


def smoothstep(x):
    x=np.clip(x,0.,1.);return x*x*x*(10+x*(-15+6*x))


class StopTransition:
    def __init__(self,standing_reference,model_sha256,mode='support_aware',config=None):
        self.cfg=copy.deepcopy(config or load_config());self.standing=np.array(standing_reference,float)
        if self.standing.shape!=(35,) or not np.isfinite(self.standing).all():raise ValueError('Invalid standing reference')
        if self.cfg.get('hardware_enabled') is not False or self.cfg.get('target')!='unitree_g1_29dof':raise ValueError('Simulation stop profile required')
        if mode not in ('timed','support_aware'):raise ValueError('Unknown stop mode')
        if self.cfg.get('entry_state') not in ('applied_reference','robot_feedback'):raise ValueError('Unknown stop boundary state')
        if not 0<self.cfg['minimum_wait_seconds']<=self.cfg['maximum_wait_seconds']:raise ValueError('Invalid stop waiting bounds')
        if min(self.cfg['brake_seconds'],self.cfg['pose_blend_seconds'],self.cfg['support_confirmation_seconds'])<=0:raise ValueError('Invalid transition duration')
        self.model_sha256=model_sha256;self.mode=mode;self.last_time=None;self.clear()

    def clear(self):
        self.entered=None;self.entry=None;self.ready_since=None;self.blend_start=None;self.timeout=False

    def step(self,timestamp_ns,requested,applied,state,feedback=None):
        if type(timestamp_ns) is not int or (self.last_time is not None and timestamp_ns<=self.last_time):
            raise ValueError('Stop transition requires increasing control-clock ns')
        requested=np.asarray(requested,float);applied=np.asarray(applied,float)
        if requested.shape!=(35,) or applied.shape!=(35,) or not np.isfinite(np.r_[requested,applied]).all():raise ValueError('Invalid stop reference')
        self.last_time=timestamp_ns
        active=state=='return_to_stand' or (state=='vision_unavailable' and self.cfg['apply_to_missing'])
        if not active:
            self.clear();return requested.copy(),dict(phase='source_or_legacy',active=False)
        ready=False;valid=False
        if feedback is not None:
            stamp=feedback.get('timestamp_ns');age=None if type(stamp) is not int else (timestamp_ns-stamp)/1e9
            valid=(age is not None and 0<=age<=self.cfg['feedback_max_age_seconds'] and
                   feedback.get('clock_domain')=='offline_simulation' and feedback.get('target_model_sha256')==self.model_sha256)
            margin=feedback.get('capture_margin_m');velocity=np.asarray(feedback.get('com_velocity_world_m_s',[]),float)
            ready=bool(valid and feedback.get('double_support') and margin is not None and np.isfinite(margin) and
                margin>=self.cfg['minimum_capture_margin_m'] and velocity.shape==(3,) and np.isfinite(velocity).all() and
                np.linalg.norm(velocity[:2])<=self.cfg['maximum_com_speed_m_s'])
        if self.entered is None:
            self.entered=timestamp_ns;self.entry=applied.copy()
            if self.cfg['entry_state']=='robot_feedback' and valid:
                q=np.asarray(feedback.get('joint_positions_rad',[]),float)
                v=np.asarray(feedback.get('root_velocity_heading_local_m_s',[]),float)
                height=feedback.get('root_height_m')
                if q.shape==(29,) and v.shape==(3,) and height is not None and np.isfinite(np.r_[q,v,height]).all():
                    self.entry[6:]=q;self.entry[:2]=v[:2];self.entry[2]=height
        elapsed=(timestamp_ns-self.entered)/1e9
        if ready:
            if self.ready_since is None:self.ready_since=timestamp_ns
        else:self.ready_since=None
        confirmed=self.ready_since is not None and (timestamp_ns-self.ready_since)/1e9>=self.cfg['support_confirmation_seconds']
        if self.blend_start is None and elapsed>=self.cfg['minimum_wait_seconds']:
            if self.mode=='timed' or confirmed or elapsed>=self.cfg['maximum_wait_seconds']:
                self.blend_start=timestamp_ns;self.timeout=self.mode=='support_aware' and not confirmed
        blend=0. if self.blend_start is None else float(smoothstep((timestamp_ns-self.blend_start)/1e9/self.cfg['pose_blend_seconds']))
        out=(1-blend)*self.entry+blend*self.standing
        # Brake velocity/yaw references even while waiting for support. Retained
        # joint pose is a bounded transition boundary, not a new source sample.
        out[[0,1,5]]=self.entry[[0,1,5]]*(1-smoothstep(elapsed/self.cfg['brake_seconds']))
        phase='waiting_support' if self.blend_start is None else 'settled_reference' if blend>=1 else 'blending_stand'
        return out,dict(phase=phase,active=True,elapsed_s=elapsed,feedback_valid=valid,support_ready=ready,
            support_confirmed=confirmed,timeout_fallback=self.timeout,blend_fraction=blend,hardware_enabled=False,
            source_motion_generated=False)
