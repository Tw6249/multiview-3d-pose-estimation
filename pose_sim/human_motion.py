"""Robot-independent human motion state with explicit observability and support.

Adds revisable dimensions, causal local velocity fits, anatomical hinge angles,
landmark-frame rotations and contact hysteresis. Never manufactures axial twist.
All clocks and world axes inherit the calibrated input. No truth files are read.
"""
from collections import deque
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from .body_shape import BodyShape
from .human_state import HumanStateEstimator
from .auxiliary import filter_measurements, attachment_checks, groups
from .foot_support import FootSupportEstimator


class HumanMotionEstimator(HumanStateEstimator):
    def __init__(self, config=None):
        super().__init__(config)
        self.shape=BodyShape();self.motion_history=deque(maxlen=4);self.previous_rotations={}
        self.foot_support=FootSupportEstimator(self.cfg.get('foot_support'));self._current_row=None

    def _learn_lengths(self, measurements, accepted, noise):
        # Dimension learning must not depend on rejection by the old dimensions:
        # otherwise a biased initializer makes its own correction impossible.
        # BodyShape applies independent multi-view quality and robust-window gates.
        shape_row=self._current_row.get('shape_observation',self._current_row)
        raw_supported={n for n,p,q in zip(shape_row['joint_names'],
            shape_row['joints_world_m'],shape_row['quality'])
            if p is not None and q is not None and np.isfinite(p).all()
            and q.get('mean_reprojection_px',np.inf)<5.}
        shape=self.shape.update(shape_row,raw_supported)
        # A provisional, uncertain dimension can exert a weak constraint before
        # maturity. It cannot veto a current measurement. This avoids an abrupt
        # unconstrained -> rigid transition without fixing initialization bias.
        self.lengths={k:v['length_m'] for k,v in shape.items() if v['samples']>=6}

    def _bone_gate_tolerance(self,key,length):
        dim=self.shape.result[key]
        if dim['status']!='stable':return np.inf
        return max(super()._bone_gate_tolerance(key,length),3*dim['uncertainty_m'])

    def _bone_constraint_sigma(self,key):
        dim=self.shape.result[key]
        return max(self.cfg['bone_sigma_m'],dim['uncertainty_m'],
                   .025 if dim['status']!='stable' else 0.)

    def update(self,row,cameras=None,detection=None):
        signature=(row.get('source_id'),row.get('landmark_schema_id'),row.get('calibration_sha256'),
                   row.get('perception_signature'),tuple(row['joint_names']))
        if self.signature is not None and (signature!=self.signature or row['timestamp_ns']-self.last_time>self.cfg['reset_gap_ms']*1e6):
            self.shape=BodyShape();self.motion_history.clear();self.previous_rotations={}
            self.foot_support=FootSupportEstimator(self.cfg.get('foot_support'))
        if self.cfg.get('auxiliary_consistency',True):row=filter_measurements(row,self.cfg.get('attachment'))
        self._current_row=row
        out=super().update(row,cameras,detection)
        if self.cfg.get('auxiliary_consistency',True):self._check_output_attachments(out,row)
        out['state_schema_id']='human_state_v2'
        out['fusion_calibration_sha256']=row.get('fusion_calibration_sha256')
        out['association']=copy.deepcopy(row.get('association',{}))
        out['auxiliary_association']=copy.deepcopy(row.get('auxiliary_association',{}))
        out['observation_consistency']=copy.deepcopy(row.get('observation_consistency',{}))
        out['shape']=copy.deepcopy(self.shape.result)
        out['estimator']='causal observation filter, revisable dimensions, landmark kinematics and support hysteresis'
        p={n:np.array(v) for n,v,s in zip(out['joint_names'],out['joints_world_m'],out['joint_states'])
           if v is not None and s['status']=='filtered'}
        states=dict(zip(out['joint_names'],out['joint_states']));t=out['timestamp_ns']
        self.motion_history.append((t,p))
        velocity={};velocity_status={};window_velocity={}
        for i,n in enumerate(out['joint_names']):
            samples=[(stamp,ps[n]) for stamp,ps in self.motion_history if n in ps and t-stamp<=150_000_000]
            if n in p and len(samples)>=3:
                ts=np.array([(s[0]-t)/1e9 for s in samples]); xyz=np.array([s[1] for s in samples])
                # A causal least-squares slope rejects position differencing noise.
                ts-=ts.mean();denom=ts@ts
                if denom>1e-8:
                    window_velocity[n]=(ts@xyz/denom).tolist()
            # The LS slope is correlated with the filter positions and is not
            # an independent velocity measurement. Never overwrite the Kalman
            # velocity while retaining its old covariance.
            if n in p and self.filters[n]['observations']>=3:
                velocity[n]=np.array(out['velocities_world_m_s'][i])
            velocity_status[n]='joint_filter' if n in velocity else 'model_prediction' if out['joints_world_m'][i] is not None else 'unknown'
            if out['joints_world_m'][i] is not None:
                states[n]['velocity_covariance_m2_s2']=self.filters[n]['p'][3:,3:].tolist()
                states[n]['velocity_uncertainty_calibrated']=False
        out['velocity_status']=velocity_status
        out['window_velocity_diagnostic_m_s']=window_velocity
        root_names=('left_hip','right_hip','left_shoulder','right_shoulder')
        rotation=None;root=None
        if all(n in p for n in root_names):
            root=(p['left_hip']+p['right_hip'])/2
            up=(p['left_shoulder']+p['right_shoulder'])/2-root
            left=p['left_hip']-p['right_hip']
            if np.linalg.norm(up)>.15 and np.linalg.norm(left)>.1:
                y=left/np.linalg.norm(left);z=up-y*(up@y)
                if np.linalg.norm(z)>.1:
                    z/=np.linalg.norm(z);rotation=np.column_stack((np.cross(y,z),y,z))
        out['root']=dict(position_world_m=root.tolist() if root is not None else None,
            velocity_world_m_s=((velocity['left_hip']+velocity['right_hip'])/2).tolist() if all(n in velocity for n in root_names[:2]) else None,
            landmark_rotation_world=rotation.tolist() if rotation is not None else None,
            anatomical_pelvis_rotation=None,reason='Hip line and shoulder midpoint define a torso landmark frame; pelvis/waist tilt is not separately observed')
        orientations=dict(out['orientations']);orientations['torso']=rotation.tolist() if rotation is not None else None
        orientation_quality={};angular={}
        for name,r in orientations.items():
            angular[name]=None
            if r is None:
                orientation_quality[name]=dict(observable=False,reason='missing or degenerate landmarks')
                self.previous_rotations.pop(name,None)
                continue
            if name=='torso': needed=root_names
            elif name.endswith('_foot'):needed=tuple(name.split('_')[0]+'_'+s for s in ('heel','big_toe','small_toe'))
            else:needed=tuple(name.split('_')[0]+'_hand_'+s for s in ('wrist','middle_mcp','index_mcp','pinky_mcp'))
            cloud=np.array([p[n] for n in needed]);values=np.linalg.svd(cloud-cloud.mean(axis=0),compute_uv=False)
            sigma=max(states[n]['max_std_m'] for n in needed)
            sensitivity=float(2*sigma/max(values[1],1e-9))
            reliable=sensitivity<.5
            orientation_quality[name]=dict(observable=reliable,linearized_error_scale_rad=sensitivity,
                calibrated=False,reason='landmark geometry sensitivity; not a calibrated rotation interval')
            previous=self.previous_rotations.get(name)
            if previous and reliable and 0<t-previous[0]<=150_000_000:
                angular[name]=(Rotation.from_matrix(np.array(r)@previous[1].T).as_rotvec()/((t-previous[0])/1e9)).tolist()
            if reliable:self.previous_rotations[name]=(t,np.array(r))
            else:self.previous_rotations.pop(name,None)
        out['orientations']=orientations;out['orientation_quality']=orientation_quality
        out['angular_velocities_world_rad_s']=angular
        out['joint_rotations']={}
        for side in ('left','right'):
            for joint,parts in [('elbow',('shoulder','elbow','wrist')),('knee',('hip','knee','ankle'))]:
                ns=[side+'_'+s for s in parts];angle=None
                if all(n in p for n in ns):
                    a=p[ns[0]]-p[ns[1]];b=p[ns[2]]-p[ns[1]]
                    if min(np.linalg.norm(a),np.linalg.norm(b))>.08:
                        angle=float(np.pi-np.arccos(np.clip(a@b/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1)))
                out['joint_rotations'][side+'_'+joint]=dict(flexion_rad=angle,axial_twist_rad=None,
                    rotation_world=None,observed_dof=1 if angle is not None else 0)
        out['support']=self._support(p,velocity,states,t)
        out['unobserved']+=['anatomical full 3-DOF joint rotation','ground reaction force','centre of pressure']
        return out

    def _check_output_attachments(self,out,row):
        points={n:np.array(p) for n,p,s in zip(out['joint_names'],out['joints_world_m'],out['joint_states'])
                if p is not None and s['status']=='filtered'}
        sigmas={n:s['max_std_m'] for n,s in zip(out['joint_names'],out['joint_states']) if n in points}
        diagnostic,bad=attachment_checks(points,sigmas,self.cfg.get('attachment'))
        out['attachment_measurements']=row['attachment_measurements'];out['attachment_state']=diagnostic
        for i,n in enumerate(out['joint_names']):
            pre_rejected=any(n in d['rejected'] for d in row['attachment_measurements'].values())
            if pre_rejected:out['joint_states'][i]['reason']='detail attachment measurement rejected'
            if n in bad:
                # A filtered output can become inconsistent with its body anchor
                # after independent Kalman updates. Withdraw it, never snap it.
                s=out['joint_states'][i]
                out['joint_states'][i]=dict(status='missing',last_observed_timestamp_ns=None,
                    observation_age_ns=None,reason='detail attachment state rejected')
                out['joints_world_m'][i]=out['velocities_world_m_s'][i]=out['quality'][i]=None
                self.filters.pop(n,None)
        for group,anchor,proximal,detail in groups():
            if diagnostic[group]['status']!='supported' or any(n in bad for n in detail):
                out['orientations'][group]=None
                if group.endswith('_foot'):out['contacts'][group.split('_')[0]]=None
        self._attachment_state=diagnostic

    def _support(self,p,velocity,states,t):
        attachments={side:not self.cfg.get('auxiliary_consistency',True) or
            self._attachment_state[side+'_foot']['status']=='supported' for side in ('left','right')}
        return self.foot_support.update(t,p,velocity,states,attachments,self.cfg['ground_height_m'])
