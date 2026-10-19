"""Bounded causal multi-view/multi-frame body fit with revisable dimensions.

Input: immutable current triangulation + assigned image pixels, calibrated
world metres / OpenCV projection, strictly increasing exposure ns. Output: a
copy of current observations and diagnostics. Five observed frames are jointly
fit to their ORIGINAL pixels, acceleration and shared soft bone lengths. Raw
multi-view evidence, never the optimized result, updates the dimension model.
Missing current joints remain missing; history is not an additional camera.
No future observations, robot model, contact assumption or truth labels.

Solver/correction/reprojection failures return original observations. Pixel,
motion and shape scales are heuristic; smaller residual is not proof of better
accuracy. Original covariance is retained and inflated by correction, never
shrunk by treating correlated history as independent measurements.
"""
from collections import deque
import copy
import json
from pathlib import Path
import numpy as np
from scipy.linalg import solve
from .geometry import CameraGeometry
from .body_shape import BodyShape
from .landmarks import BONES


def load_config():
    return json.loads((Path(__file__).resolve().parents[1]/'config/vision_joint_window.json').read_text())


class JointWindow:
    def __init__(self,cameras,config=None):
        self.cfg=load_config() if config is None else dict(config)
        for k,v in self.cfg.items():
            if k=='status':continue
            if not np.isfinite(v) or v<=0:raise ValueError('Invalid window bound: '+k)
        if type(self.cfg['frames']) is not int or not 3<=self.cfg['frames']<=8:raise ValueError('Bounded 3-8 frame window required')
        if type(self.cfg['maximum_iterations']) is not int or self.cfg['maximum_iterations']>12:raise ValueError('Bounded iteration count required')
        self.cameras=cameras;self.geometry={c:CameraGeometry([cam]) for c,cam in cameras.items()}
        self.history=deque(maxlen=self.cfg['frames']);self.shape=BodyShape();self.signature=None;self.last=None

    def update(self,row,detection):
        t=row['timestamp_ns']
        if type(t) is not int or (self.last is not None and t<=self.last):raise ValueError('Increasing exposure ns required')
        if detection['timestamp_ns']!=t or detection['joint_names']!=row['joint_names']:raise ValueError('Observation correspondence mismatch')
        signature=tuple(row.get(k) for k in ('source_id','landmark_schema_id','perception_signature','calibration_sha256'))+(tuple(row['joint_names']),)
        if signature!=self.signature or (self.last is not None and (t-self.last)/1e9>self.cfg['maximum_gap_s']):
            self.history.clear();self.shape=BodyShape()
        self.signature=signature;self.last=t;out=copy.deepcopy(row)
        out['shape_observation']={k:copy.deepcopy(row[k]) for k in ('timestamp_ns','joint_names','joints_world_m','quality')}
        body={n for pair in BONES for n in pair};names=row['joint_names']
        supported={n for n,p,q in zip(names,row['joints_world_m'],row['quality']) if n in body and p is not None and q is not None and np.isfinite(p).all() and q.get('mean_reprojection_px',np.inf)<5.}
        shape=self.shape.update(row,supported)
        self.history.append((copy.deepcopy(row),copy.deepcopy(detection)))
        diagnostic=dict(history_frames=len(self.history),uses_future=False,current_observation_coverage_preserved=True,
            fitted_joints=0,accepted_joints=0,rejected_joints=0,shape_source='independent raw multi-view history',calibrated_uncertainty=False)
        out['joint_window']=diagnostic
        if len(self.history)<3:diagnostic['status']='warming';return out
        nodes=[];observations=[];index={};times=[]
        for f,(r,d) in enumerate(self.history):
            times.append((r['timestamp_ns']-t)/1e9)
            for j,(n,p,q) in enumerate(zip(names,r['joints_world_m'],r['quality'])):
                if n not in body or p is None or q is None or not np.isfinite(p).all():continue
                ids=[c for c in q['inlier_cameras'] if c in self.geometry]
                if len(ids)<2:continue
                k=len(nodes);nodes.append(p);index[f,n]=k
                for c in ids:
                    v=d['views'][c];xy=np.array(v['keypoints_px'][j],float)
                    observations.append((k,self.geometry[c],xy,max(.2,float(v['scores'][j]))**.5/self.cfg['pixel_scale_px']))
        if not nodes:diagnostic['status']='no_observations';return out
        nn=len(nodes);lengths={};bones=[]
        for a,b in BONES:
            key=a+'--'+b;s=shape.get(key,{})
            if s.get('status')!='stable':continue
            pairs=[(index[f,a],index[f,b]) for f in range(len(times)) if (f,a) in index and (f,b) in index]
            if not pairs:continue
            li=3*nn+len(lengths);lengths[key]=(li,s['length_m'],max(self.cfg['minimum_length_prior_scale_m'],s['uncertainty_m']))
            bones.extend((a,b,li) for a,b in pairs)
        temporal=[]
        for f in range(2,len(times)):
            dt1=times[f-1]-times[f-2];dt2=times[f]-times[f-1]
            for n in body:
                if all((i,n) in index for i in (f-2,f-1,f)):
                    coeff=np.array([1/dt1,-1/dt1-1/dt2,1/dt2])*2/(dt1+dt2)/self.cfg['acceleration_scale_m_s2']
                    temporal.append(([index[i,n] for i in (f-2,f-1,f)],coeff))
        x=np.r_[np.array(nodes).ravel(),[v[1] for v in lengths.values()]];initial=x.copy()
        nr=2*len(observations)+len(bones)+len(lengths)+3*len(temporal)
        def system(x,jacobian=True):
            points=x[:nn*3].reshape(-1,3);r=np.empty(nr);J=np.zeros((nr,len(x))) if jacobian else None;at=0
            for k,g,xy,w in observations:
                r[at:at+2]=(g.project(points[k])[0]-xy)*w
                if jacobian:J[at:at+2,3*k:3*k+3]=g.jacobian(points[k])[0]*w
                at+=2
            for a,b,li in bones:
                delta=points[a]-points[b];norm=np.linalg.norm(delta);scale=self.cfg['bone_residual_scale_m']
                r[at]=(norm-x[li])/scale
                if jacobian:
                    grad=delta/max(norm,1e-9)/scale;J[at,3*a:3*a+3]=grad;J[at,3*b:3*b+3]=-grad;J[at,li]=-1/scale
                at+=1
            for li,target,scale in lengths.values():
                r[at]=(x[li]-target)/scale
                if jacobian:J[at,li]=1/scale
                at+=1
            for ids,coeff in temporal:
                r[at:at+3]=coeff@points[ids]
                if jacobian:
                    for k,w in zip(ids,coeff):J[at:at+3,3*k:3*k+3]=np.eye(3)*w
                at+=3
            return r,J
        scale=self.cfg['robust_scale']
        def cost(r):return float(np.sum(2*scale**2*(np.sqrt(1+(r/scale)**2)-1)))
        try:
            first,_=system(x,False);initial_cost=cost(first)
            for _ in range(self.cfg['maximum_iterations']):
                r,J=system(x);w=1/np.sqrt(1+(r/scale)**2);H=J.T@(w[:,None]*J)
                step=solve(H+np.eye(len(x))*1e-7,J.T@(w*r),assume_a='pos',check_finite=False)
                if not np.isfinite(step).all():raise ValueError('Nonfinite fit')
                before=cost(r);accepted=False
                for alpha in (1.,.5,.25,.125):
                    candidate=x-alpha*step;after,_=system(candidate,False)
                    if np.isfinite(after).all() and cost(after)<=before:
                        x=candidate;accepted=True;break
                if not accepted or np.linalg.norm(alpha*step)<1e-5:break
            last,_=system(x,False)
            if not np.isfinite(x).all() or cost(last)>initial_cost+1e-7:raise ValueError('Invalid cost')
        except (ValueError,np.linalg.LinAlgError):diagnostic['status']='fit_failed';return out
        diagnostic.update(status='fitted',objective_before=initial_cost,objective_after=cost(last),lengths_m={k:float(x[v[0]]) for k,v in lengths.items()})
        f=len(times)-1
        for j,n in enumerate(names):
            if (f,n) not in index:continue
            k=index[f,n];p=x[3*k:3*k+3];raw=initial[3*k:3*k+3];delta=p-raw;correction=float(np.linalg.norm(delta))
            q=out['quality'][j];errors=[float(np.linalg.norm(self.geometry[c].project(p)[0]-detection['views'][c]['keypoints_px'][j])) for c in q['inlier_cameras']]
            diagnostic['fitted_joints']+=1
            if correction>self.cfg['maximum_current_correction_m'] or max(errors)>self.cfg['maximum_reprojection_px']:
                diagnostic['rejected_joints']+=1;continue
            out['joints_world_m'][j]=p.tolist();q['mean_reprojection_px']=float(np.mean(errors));q['joint_window_correction_m']=correction
            # Do not claim confidence gain from reused/correlated observations.
            q['linearized_max_std_m']=max(q['linearized_max_std_m'],correction)
            if q.get('position_covariance_m2') is not None:q['position_covariance_m2']=(np.array(q['position_covariance_m2'])+np.outer(delta,delta)).tolist()
            diagnostic['accepted_joints']+=1
        return out
