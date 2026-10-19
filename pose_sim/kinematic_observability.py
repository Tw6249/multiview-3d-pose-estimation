"""Diagnostic landmark rotation/hinge uncertainty, never anatomical completion.

Input: current human state, world Z-up m and position covariance m², exposure
ns. Output: landmark-frame rotations and local SO(3) covariance rad², hinge
flexion and error scale rad. Numerical Jacobians propagate CURRENT landmark
uncertainty. Joint independence and local linearity are assumptions, not an
empirical coverage guarantee. Missing/predicted/degenerate points yield unknown.
Angular velocity uses exposure dt and a conservative std bound with unknown
inter-frame correlation. No anatomical axial twist, pelvis/waist decomposition,
contact force or robot target is inferred. Input state is never modified.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .human_state import _orientation


def torso_frame(points):
    lh,rh,ls,rs=points;up=(ls+rs-lh-rh)/2;y=lh-rh
    if np.linalg.norm(y)<.1 or np.linalg.norm(up)<.15:return None
    y=y/np.linalg.norm(y);z=up-y*(up@y)
    if np.linalg.norm(z)<.1:return None
    z/=np.linalg.norm(z);return np.column_stack((np.cross(y,z),y,z))


def hinge_flexion(points):
    a,b,c=points;u=a-b;v=c-b
    if min(np.linalg.norm(u),np.linalg.norm(v))<.08:return None
    cross=np.linalg.norm(np.cross(u,v));dot=u@v
    # At exactly straight/folded geometry the bend plane and linearized scalar
    # uncertainty are nonsmooth. Do not return a falsely precise zero derivative.
    if cross/(np.linalg.norm(u)*np.linalg.norm(v))<1e-3:return None
    return float(np.pi-np.arctan2(cross,dot))


def propagate_landmarks(points,covariances,function,rotation=False):
    points=np.asarray(points,float);cov=np.asarray(covariances,float)
    if points.ndim!=2 or points.shape[1]!=3 or cov.shape!=(len(points),3,3) or not np.isfinite(np.r_[points.ravel(),cov.ravel()]).all():return None
    if not np.allclose(cov,cov.transpose(0,2,1),atol=1e-10,rtol=0) or np.linalg.eigvalsh(cov).min()<-1e-10:return None
    value=function(points)
    if value is None:return None
    value=np.asarray(value);n=3 if rotation else 1;J=np.zeros((n,points.size));h=1e-5
    for i in range(points.size):
        plus=points.copy();minus=points.copy();plus.flat[i]+=h;minus.flat[i]-=h
        a,b=function(plus),function(minus)
        if a is None or b is None:return None
        J[:,i]=(Rotation.from_matrix(np.array(a)@np.array(b).T).as_rotvec() if rotation else np.atleast_1d(a-b))/(2*h)
    result=np.zeros((n,n))
    for k,C in enumerate(cov):result+=J[:,3*k:3*k+3]@C@J[:,3*k:3*k+3].T
    return value,result


class KinematicObservability:
    def __init__(self):self.last=None;self.signature=None;self.previous={}

    def update(self,row):
        t=row['timestamp_ns']
        if type(t) is not int or (self.last is not None and t<=self.last):raise ValueError('Increasing exposure ns required')
        signature=tuple(row.get(k) for k in ('source_id','calibration_sha256','perception_signature','landmark_schema_id'))
        if signature!=self.signature or (self.last is not None and t-self.last>150_000_000):self.previous.clear()
        self.last=t;self.signature=signature
        points=dict(zip(row['joint_names'],row['joints_world_m']));states=dict(zip(row['joint_names'],row['joint_states']))
        def current(names):
            if not all(points.get(n) is not None and states[n].get('status')=='filtered' and states[n].get('last_observed_timestamp_ns')==t and states[n].get('position_covariance_m2') is not None for n in names):return None
            return np.array([points[n] for n in names]),np.array([states[n]['position_covariance_m2'] for n in names])
        frames={};hinges={}
        groups=[('torso',('left_hip','right_hip','left_shoulder','right_shoulder'),torso_frame)]
        for side in ('left','right'):
            def foot(p,side=side):
                heel,big,small=p;return _orientation((big+small)/2-heel,small-big if side=='left' else big-small)
            groups.append((side+'_foot',tuple(side+'_'+n for n in ('heel','big_toe','small_toe')),foot))
            def palm(p):
                wrist,middle,index,pinky=p;return _orientation(middle-wrist,index-pinky)
            groups.append((side+'_palm',tuple(side+'_hand_'+n for n in ('wrist','middle_mcp','index_mcp','pinky_mcp')),palm))
        for name,names,fn in groups:
            data=current(names);fit=propagate_landmarks(*data,fn,rotation=True) if data is not None else None
            if fit is None:
                frames[name]=dict(status='unknown',rotation_world=None,rotation_covariance_world_rad2=None,angular_velocity_world_rad_s=None)
                self.previous.pop(name,None);continue
            R,C=fit;std=float(np.sqrt(max(0,np.linalg.eigvalsh(C).max())));angular=None;angular_scale=None
            old=self.previous.get(name)
            if old and 0<t-old[0]<=150_000_000:
                dt=(t-old[0])/1e9;angular=(Rotation.from_matrix(R@old[1].T).as_rotvec()/dt).tolist()
                angular_scale=(std+old[2])/dt
            self.previous[name]=(t,R,std)
            frames[name]=dict(status='local_uncertainty_available',rotation_world=R.tolist(),rotation_covariance_world_rad2=C.tolist(),
                maximum_rotation_std_rad=std,angular_velocity_world_rad_s=angular,angular_velocity_error_scale_rad_s=angular_scale,
                anatomical_rotation_verified=False,calibrated=False)
        for side in ('left','right'):
            for joint,parts in [('knee',('hip','knee','ankle')),('elbow',('shoulder','elbow','wrist'))]:
                data=current(tuple(side+'_'+p for p in parts));fit=propagate_landmarks(*data,hinge_flexion) if data is not None else None
                hinges[side+'_'+joint]=dict(flexion_rad=float(fit[0]) if fit else None,std_rad=float(np.sqrt(max(0,fit[1][0,0]))) if fit else None,
                    observed_dof=1 if fit else 0,axial_twist_rad=None,calibrated=False)
        return dict(timestamp_ns=t,landmark_frames=frames,hinges=hinges,
            assumptions='Local first-order propagation; joint cross-covariances unknown; angular velocity uses worst-correlation scalar std bound.',
            unobserved=['anatomical axial twist','pelvis versus waist tilt','ground reaction forces'])
