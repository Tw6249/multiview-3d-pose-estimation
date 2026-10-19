"""Markerless COCO body -> size-independent G1 full-body motion references.

Only observations are read here. Ground truth, animation rigs and example policy
motions are never inputs. COCO17 leaves foot/wrist rotations and pelvis-vs-waist
tilt ambiguous; declared upright-pelvis/flat-foot/neutral-wrist priors resolve IK.
"""
import json
import copy
import hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .pose import OneEuro, unit
from .twist import TwistPlant, DEFAULT_Q
from .unitree import SDK_JOINT_NAMES
from .root_path import ScaledRootPath

NAMES=tuple(f'{side}_{joint}' for joint in ('shoulder','elbow','wrist','hip','knee','ankle') for side in ('left','right'))
BONES=((0,2),(2,4),(1,3),(3,5),(6,8),(8,10),(7,9),(9,11))
LINKS=tuple((f'{side}_{a}_link',f'{side}_{b}_link') for side,a,b in (
    ('left','shoulder_roll','elbow'),('left','elbow','wrist_yaw'),
    ('right','shoulder_roll','elbow'),('right','elbow','wrist_yaw'),
    ('left','hip_pitch','knee'),('left','knee','ankle_pitch'),
    ('right','hip_pitch','knee'),('right','knee','ankle_pitch')))


def read_body(row):
    if row.get('landmark_schema_id','coco17') not in ('coco17','human33'):
        raise ValueError('Require named human observations')
    if row.get('state_schema_id') not in (None,'human_state_v1','human_state_v2'):
        raise ValueError('Unsupported human state schema')
    names=row['joint_names']
    if len(names)!=len(set(names)):raise ValueError('Duplicate body point names')
    points=[]
    for name in NAMES:
        i=names.index(name);q=row['quality'][i];p=row['joints_world_m'][i]
        optional=name.endswith(('shoulder','elbow','wrist'))
        if row.get('state_schema_id') in ('human_state_v1','human_state_v2'):
            state=row['joint_states'][i]
            if state['status']!='filtered' or state['last_observed_timestamp_ns']!=row['timestamp_ns']:
                # Prediction is available to planners, but not automatically
                # promoted to a fresh image observation for this controller.
                p=q=None
        if p is None or q is None:
            if optional:points.append([np.nan]*3);continue
            raise ValueError('Missing '+name)
        if (not np.isfinite(p).all() or q['mean_reprojection_px']>8 or q['ray_angle_deg']<2 or
            not np.isfinite(q['linearized_max_std_m']) or q['linearized_max_std_m']>.05 or
            len(set(q['inlier_cameras']))<2):
            if optional:points.append([np.nan]*3);continue
            raise ValueError('Unreliable '+name)
        points.append(p)
    return np.array(points,float)


class FullBodyRetargeter:
    def __init__(self, input_filtered=False):
        self.plant=TwistPlant();self.model=self.plant.model;self.data=self.plant.data
        self.mj=self.plant.mj
        self.ids=[(self.model.body(a).id,self.model.body(b).id) for a,b in LINKS]
        self.active=np.r_[np.arange(19),np.arange(22,26)]
        self.q=DEFAULT_Q.copy();self.q[[19,20,21,26,27,28]]=0
        self.data.qpos[:7]=[0,0,0,1,0,0,0]
        self.data.qpos[7:]=self.q;self.mj.mj_kinematics(self.model,self.data)
        self.target_leg_length=np.mean([np.linalg.norm(self.data.xpos[b]-self.data.xpos[a]) for a,b in self.ids[4:]])*2
        self.last_points=self.last_time=self.last_origin=self.last_heading=None
        self.lengths=None
        self.root_path=ScaledRootPath()
        self.filter=OneEuro(dict(min_cutoff_hz=2.5,beta=.2,derivative_cutoff_hz=1))
        self.root_filter=OneEuro(dict(min_cutoff_hz=.8,beta=.05,derivative_cutoff_hz=1))
        self.input_filtered=input_filtered

    def solve(self, points, timestamp_ns):
        # A rejected fit must not poison filters, scale calibration or IK history.
        keys=('filter','root_filter','q','last_points','last_time','last_origin','last_heading','lengths','root_path')
        saved={k:copy.deepcopy(getattr(self,k)) for k in keys}
        try:return self._solve(points,timestamp_ns)
        except (ValueError,KeyError,IndexError,TypeError):
            for k,value in saved.items():setattr(self,k,value)
            raise

    def _solve(self, points, timestamp_ns):
        p=np.asarray(points,float)
        if p.shape!=(12,3) or not np.isfinite(p[6:]).all():raise ValueError('Invalid body geometry')
        observed=np.isfinite(p).all(axis=1)
        t=int(timestamp_ns)
        dt=(t-self.last_time)/1e9 if self.last_time is not None else 0
        if self.last_time is not None and dt<=0:raise ValueError('Nonmonotonic observation')
        lengths=np.array([np.linalg.norm(p[b]-p[a]) for a,b in BONES])
        if np.any(lengths<.1) or np.any(lengths>.85):raise ValueError('Implausible human limb length')
        if self.lengths is not None:
            bad=np.maximum(lengths/self.lengths,self.lengths/lengths)>1.65
            if bad[4:].any():raise ValueError('Leg bone length discontinuity')
            # Arm geometry errors withdraw only the corresponding arm reference.
            for k in np.where(bad[:4])[0]:
                side=0 if k<2 else 1
                p[[side,side+2,side+4]]=np.nan
                observed[[side,side+2,side+4]]=False
            lengths=np.array([np.linalg.norm(p[b]-p[a]) for a,b in BONES])
        if self.last_points is not None and dt<.5 and np.nanmax(np.linalg.norm(p-self.last_points,axis=1))/dt>7:
            raise ValueError('Human point jump')
        # Left hip defines +Y; with gravity this determines heading without face points.
        torso_observed=observed[:2].all()
        lateral=.3*(p[6]-p[7])+.7*(p[0]-p[1]) if torso_observed else p[6]-p[7]
        lateral[2]=0
        if np.linalg.norm(lateral)<.08:raise ValueError('Unobservable pelvis heading')
        if self.last_time is not None and dt>.5:
            self.filter=OneEuro(dict(min_cutoff_hz=2.5,beta=.2,derivative_cutoff_hz=1))
            self.root_filter=OneEuro(dict(min_cutoff_hz=.8,beta=.05,derivative_cutoff_hz=1))
        raw=p.copy()
        # Internal filter placeholders carry no observational validity. Missing
        # arms contribute no IK residual and are explicitly returned to neutral.
        filled=p.copy()
        for k in np.where(~observed)[0]:
            filled[k]=self.filter.filtered[k] if self.filter.filtered is not None else p[6+k%2]+[0,0,.5]
        p=filled if self.input_filtered else self.filter.update(filled,dt)
        lateral=.3*(p[6]-p[7])+.7*(p[0]-p[1]) if torso_observed else p[6]-p[7]
        lateral[2]=0
        root=self.root_filter.update(np.array([(p[6]+p[7])/2,unit(lateral)]),dt)
        lateral=unit(root[1])
        forward=np.cross(lateral,[0,0,1]);R=np.column_stack((forward,lateral,[0,0,1]))
        heading=np.arctan2(forward[1],forward[0]);origin=root[0]
        bone_observed=np.array([observed[a] and observed[b] for a,b in BONES])
        directions=np.array([unit(p[b]-p[a])@R if good else np.zeros(3)
                             for (a,b),good in zip(BONES,bone_observed)])
        torso_up=unit((p[0]+p[1])/2-origin)@R if torso_observed else np.array([0,0,1])
        shoulder_left=unit(p[0]-p[1])@R if torso_observed else np.array([0,1,0])
        previous=self.q.copy();active=self.active
        neutral_target=previous.copy()
        for indices,points_idx in ((np.arange(15,19),(0,2,4)),(np.arange(22,26),(1,3,5))):
            if not observed[list(points_idx)].all():
                neutral_target[indices]+=np.clip(DEFAULT_Q[indices]-previous[indices],-.12,.12)
        def residual(x):
            self.data.qpos[7:]=previous;self.data.qpos[7+active]=x
            self.mj.mj_kinematics(self.model,self.data)
            actual=np.array([unit(self.data.xpos[b]-self.data.xpos[a]) for a,b in self.ids])
            torso=self.data.body('torso_link').xmat.reshape(3,3)
            foot_rot=[]
            for side in ('left','right'):
                foot=self.data.body(side+'_ankle_roll_link').xmat.reshape(3,3)
                foot_rot.extend(Rotation.from_matrix(foot).as_rotvec()*.45)
            return np.r_[((actual-directions)*bone_observed[:,None]).ravel(),(torso[:,2]-torso_up)*1.3,
                         (torso[:,1]-shoulder_left)*.5,foot_rot,
                         .10*(x-neutral_target[active]),.025*(x-DEFAULT_Q[active])]
        fit=least_squares(residual,previous[active],bounds=(self.plant.limits[active,0]+1e-5,
            self.plant.limits[active,1]-1e-5),max_nfev=45,ftol=1e-5,xtol=1e-5,gtol=1e-5)
        candidate=previous.copy();candidate[active]=fit.x
        residual(fit.x)
        actual=np.array([unit(self.data.xpos[b]-self.data.xpos[a]) for a,b in self.ids])
        errors=np.degrees(np.arccos(np.clip(np.sum(actual*directions,axis=1),-1,1)))
        if errors[bone_observed].max()>35:raise ValueError('Full-body IK cannot match observed limb directions')
        # Foot sole offset is a model prior; infer base height from the lowest foot.
        # This retains the source's crouch/leg extension rather than fixing height.
        feet=np.array([self.data.body(s+'_ankle_roll_link').xpos.copy() for s in ('left','right')])
        height=-np.min(feet[:,2])+.035
        if not .45<height<.95:raise ValueError('Unreachable base height')
        source_leg=np.mean((self.lengths if self.lengths is not None else lengths)[4:])*2
        scale=self.target_leg_length/source_leg
        velocity=np.zeros(2);yaw_rate=0
        if self.last_origin is not None and 0<dt<=.5:
            velocity=((origin-self.last_origin)/dt*scale)@R
            velocity=velocity[:2]
            yaw_rate=np.arctan2(np.sin(heading-self.last_heading),np.cos(heading-self.last_heading))/dt
        if np.linalg.norm(velocity)>1.5 or abs(yaw_rate)>2.5:raise ValueError('Root motion exceeds imitation profile')
        path=self.root_path.update(origin[:2],scale,t)
        reference=np.r_[velocity,height,0,0,yaw_rate,candidate]
        self.q=candidate;self.last_points=raw;self.last_time=t;self.last_origin=origin;self.last_heading=heading
        if self.lengths is None:self.lengths=lengths.copy()
        else:self.lengths=np.where(np.isfinite(lengths),np.where(np.isfinite(self.lengths),.98*self.lengths+.02*lengths,lengths),self.lengths)
        return dict(timestamp_ns=t,reference=reference.tolist(),joint_names=SDK_JOINT_NAMES,
            root_position_m=np.r_[path,height].tolist(),root_heading_rad=float(heading),
            root_path_frame='source_world_xy_anchored_at_first_accepted_pelvis; target_height_z',
            limb_direction_error_deg=[float(e) if valid else None for e,valid in zip(errors,bone_observed)],source_leg_length_m=float(source_leg),
            scale_to_target=float(scale),source_body_points_m=[point.tolist() if valid else None for point,valid in zip(p,observed)],
            observed_body_points=observed.tolist(),arm_observed=[bool(observed[[0,2,4]].all()),bool(observed[[1,3,5]].all())],
            torso_observed=bool(torso_observed),
            foot_height_above_lowest_m=(feet[:,2]-feet[:,2].min()).tolist(),
            priors=['upright pelvis; torso tilt assigned to waist','heading from fused hip/shoulder lateral axes',
                    'flat foot heading','neutral wrists'],
            unobserved=['pelvis vs waist tilt separation','foot yaw/roll/pitch','wrist/hand orientation','finger joints'])


def retarget_session(session, pose_file='poses3d.jsonl'):
    session=Path(session);filtered=pose_file=='states3d.jsonl'
    engine=FullBodyRetargeter(input_filtered=filtered);out=[];signature=None;last_seen=None
    source_sha=hashlib.sha256((session/'poses3d.jsonl').read_bytes()).hexdigest() if filtered else None
    for line in (session/pose_file).read_text(encoding='utf-8').splitlines():
        row=json.loads(line)
        if filtered and (row.get('state_schema_id') not in ('human_state_v1','human_state_v2') or row.get('source_poses_sha256')!=source_sha):
            raise ValueError('Human state is stale or does not match the current triangulation')
        current=tuple(row.get(k) for k in ('source_id','landmark_schema_id','calibration_sha256','perception_signature'))
        if signature is None:signature=current
        if current!=signature:raise ValueError('Source identity/schema/calibration changed during imitation')
        if type(row['timestamp_ns']) is not int or (last_seen is not None and row['timestamp_ns']<=last_seen):
            raise ValueError('Nonmonotonic source clock')
        last_seen=row['timestamp_ns']
        result=dict(frame_id=row['frame_id'],timestamp_ns=row['timestamp_ns'],accepted=False)
        try:
            result.update(engine.solve(read_body(row),row['timestamp_ns']));result['accepted']=True
        except (ValueError,KeyError,IndexError,TypeError) as e:result['reason']=str(e)
        out.append(result)
    (session/'humanoid_reference.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in out),encoding='utf-8')
    return out
