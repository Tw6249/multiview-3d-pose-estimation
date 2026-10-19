"""Experimental causal polynomial velocity from raw multi-view observations.

World m -> endpoint velocity m/s, covariance m²/s² at unchanged exposure ns.
Nine-frame quadratic weighted fit, no future samples or filtered-position reuse.
Missing/weak observations break each joint's history. Covariance includes a
measurement floor and residual variance, but not correlated detector/calibration
bias. This is a separate candidate, never overwrites Kalman velocity/covariance.
"""
from collections import deque
import numpy as np


class LandmarkVelocity:
    def __init__(self,frames=9,minimum_samples=5,noise_floor_m=.018,maximum_gap_s=.12,maximum_residual_m=.04):
        if type(frames) is not int or type(minimum_samples) is not int or not 5<=minimum_samples<=frames<=15 or min(noise_floor_m,maximum_gap_s,maximum_residual_m)<=0 or not np.isfinite([noise_floor_m,maximum_gap_s,maximum_residual_m]).all():raise ValueError('Invalid velocity configuration')
        self.frames=frames;self.minimum=minimum_samples;self.floor=noise_floor_m;self.gap=maximum_gap_s;self.residual=maximum_residual_m
        self.history={};self.last=None;self.signature=None

    def update(self,row):
        t=row['timestamp_ns']
        if type(t) is not int or (self.last is not None and t<=self.last):raise ValueError('Increasing exposure ns required')
        signature=tuple(row.get(k) for k in ('source_id','calibration_sha256','perception_signature','landmark_schema_id'))
        if signature!=self.signature or (self.last is not None and (t-self.last)/1e9>self.gap):self.history.clear()
        self.last=t;self.signature=signature;result={}
        for n,p,q in zip(row['joint_names'],row['joints_world_m'],row['quality']):
            result[n]=dict(status='unknown',velocity_world_m_s=None,velocity_covariance_m2_s2=None)
            if p is None or q is None or not np.isfinite(p).all() or len(set(q.get('inlier_cameras',[])))<2 or q.get('ray_angle_deg',0)<2 or q.get('mean_reprojection_px',np.inf)>8 or not 0<=q.get('linearized_max_std_m',np.inf)<.08:
                self.history.pop(n,None);continue
            history=self.history.setdefault(n,deque(maxlen=self.frames));history.append((t,np.array(p),max(self.floor,q['linearized_max_std_m'])))
            if len(history)<self.minimum:continue
            ts=np.array([(h[0]-t)/1e9 for h in history]);X=np.column_stack((np.ones(len(ts)),ts,.5*ts**2));y=np.array([h[1] for h in history]);sigma=np.array([h[2] for h in history]);W=1/sigma**2
            H=X.T@(W[:,None]*X)
            if np.linalg.cond(H)>1e8:continue
            inverse=np.linalg.inv(H);beta=inverse@(X.T@(W[:,None]*y));residual=y-X@beta;rms=float(np.sqrt(np.mean(residual**2)))
            if rms>self.residual:continue
            # Model misspecification can inflate but never shrink the noise floor.
            inflation=max(1.,float(np.sum((residual/sigma[:,None])**2)/(3*(len(ts)-3))))
            C=np.eye(3)*inverse[1,1]*inflation
            result[n]=dict(status='estimated',velocity_world_m_s=beta[1].tolist(),velocity_covariance_m2_s2=C.tolist(),
                maximum_std_m_s=float(np.sqrt(C[0,0])),fit_rms_m=rms,samples=len(history),effective_timestamp_ns=t,calibrated=False)
        return dict(timestamp_ns=t,joints=result,method='Causal raw-observation quadratic fit; endpoint derivative; experimental')
