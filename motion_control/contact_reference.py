"""Experimental contact-consistent G1 reference projection, never a motor command.

Input: accepted source-derived TWIST reference, anchored world pelvis path (m),
heading (rad), exposure ns, explicit ankle support hypotheses. Output: bounded
leg q / base pose with source time, arms and original scoring targets preserved.
Root XY may adjust in a configured small corridor; its velocity is differentiated
consistently. These corrections serve source leg/pelvis motion, not a new gait.
Model floor nonpenetration is enforced; support hypotheses anchor a lowest sole
collision point. Its identity is a model prior, not an observed toe/heel label.
Point mode allows foot rotation; flat-sole mode is a separate experimental prior.
Unknown support never becomes a locked foot. Solver/limit failures withdraw the
candidate and clear anchors. This is kinematic feasibility, not force balance.
"""
import copy
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from .twist import TwistPlant,DEFAULT_Q
from .unitree import SDK_JOINT_NAMES


def load_config():
    return json.loads((Path(__file__).resolve().parents[1]/'config/g1_contact_reference.json').read_text())


def wrap(x):return np.arctan2(np.sin(x),np.cos(x))


class ContactReferenceProjector:
    def __init__(self,config=None):
        self.cfg=copy.deepcopy(config or load_config())
        if self.cfg.get('hardware_enabled') is not False or self.cfg.get('target')!='unitree_g1_29dof':raise ValueError('Simulation G1 contact profile required')
        if self.cfg.get('contact_geometry') not in ('point','flat'):raise ValueError('Unknown contact geometry')
        self.plant=TwistPlant();self.model=self.plant.model;self.data=self.plant.data;self.mj=self.plant.mj
        self.ids=np.array([SDK_JOINT_NAMES.index(n) for n in self.cfg['leg_joint_names']])
        if not np.array_equal(self.ids,np.arange(12)):raise ValueError('Require ordered G1 twelve leg joints')
        self.body_ids={s:self.model.body(n).id for s,n in self.cfg['feet'].items()}
        self.geoms={}
        for side,body in self.body_ids.items():
            ids=[i for i in range(self.model.ngeom) if self.model.geom_bodyid[i]==body
                 and self.model.geom_type[i]==self.mj.mjtGeom.mjGEOM_SPHERE and self.model.geom_contype[i]!=0]
            if len(ids)!=4:raise ValueError('Expected pinned G1 four-sphere sole model')
            self.geoms[side]=ids
        self.last_time=None;self.previous=None;self.previous_source=None;self.anchors={}

    def withdraw(self):self.previous=None;self.previous_source=None;self.anchors={}

    def geometry(self,row,q=None,height=None,jacobian=False,root_offset_xy=None):
        ref=np.asarray(row['reference'],float)
        q=ref[6:] if q is None else q;z=ref[2] if height is None else height
        quat=Rotation.from_euler('xyz',[ref[3],ref[4],row['root_heading_rad']]).as_quat()[[3,0,1,2]]
        xy=np.asarray(row['root_position_m'][:2])+np.asarray(root_offset_xy if root_offset_xy is not None else [0.,0.])
        self.data.qpos[:]=np.r_[xy,z,quat,q]
        self.mj.mj_kinematics(self.model,self.data)
        if jacobian:self.mj.mj_comPos(self.model,self.data)
        result={}
        for side,body in self.body_ids.items():
            ids=self.geoms[side];points=self.data.geom_xpos[ids].copy()
            points[:,2]-=self.model.geom_size[ids,0]
            rotation=self.data.xmat[body].reshape(3,3).copy();centre=points.mean(axis=0)
            item=dict(centre=centre,points=points,rotation=rotation,yaw=np.arctan2(rotation[1,0],rotation[0,0]))
            if jacobian:
                # Sphere bottom differs from its centre by a fixed world-Z radius.
                jpos=np.zeros((3,self.model.nv));jrot=np.zeros_like(jpos)
                self.mj.mj_jac(self.model,self.data,jpos,jrot,self.data.geom_xpos[ids].mean(axis=0),body)
                item['position_jac']=np.c_[jpos[:,6+self.ids],jpos[:,2],jpos[:,:2]]
                item['rotation_jac']=np.c_[jrot[:,6+self.ids],jrot[:,2],jrot[:,:2]]
                point_jacs=[]
                for i in ids:
                    self.mj.mj_jac(self.model,self.data,jpos,jrot,self.data.geom_xpos[i],body)
                    point_jacs.append(np.c_[jpos[:,6+self.ids],jpos[:,2],jpos[:,:2]])
                item['point_jacs']=np.array(point_jacs);item['floor_jac']=item['point_jacs'][:,2]
            result[side]=item
        return result

    def update(self,row,support):
        out=copy.deepcopy(row);t=row['timestamp_ns']
        if type(t) is not int or (self.last_time is not None and t<=self.last_time):
            self.withdraw();raise ValueError('Nonmonotonic contact reference')
        dt=None if self.last_time is None else (t-self.last_time)/1e9
        if not row.get('accepted',False):
            self.last_time=t;self.withdraw();return out
        # Validate a complete packet before advancing its clock. A malformed
        # packet still withdraws all anchors, so corrected input cannot revive
        # a contact inferred before the fault.
        try:
            if support['timestamp_ns']!=t:raise ValueError('Contact evidence exposure mismatch')
            ref=np.asarray(row['reference'],float)
            if ref.shape!=(35,) or not np.isfinite(ref).all():raise ValueError('Invalid source reference')
            root=np.asarray(row['root_position_m'],float)
            if root.shape!=(3,) or not np.isfinite(root).all() or not np.isfinite(float(row['root_heading_rad'])):
                raise ValueError('Invalid source root pose')
            if row.get('root_path_frame')!='source_world_xy_anchored_at_first_accepted_pelvis; target_height_z':
                raise ValueError('Require corrected anchored pelvis path')
            if any(support['feet'][s]['status'] not in ('unknown','swing','support_candidate') for s in self.body_ids):
                raise ValueError('Invalid support hypothesis')
        except (KeyError,TypeError,ValueError) as exc:
            self.withdraw();raise ValueError('Invalid contact packet: '+str(exc)) from exc
        self.last_time=t
        if dt is None or dt>self.cfg['support']['maximum_gap_s']:self.withdraw()
        out['source_reference']=copy.deepcopy(row.get('source_reference',row['reference']))
        out['source_root_position_m']=copy.deepcopy(row.get('source_root_position_m',row['root_position_m']))
        out['source_root_heading_rad']=row.get('source_root_heading_rad',row['root_heading_rad'])
        base=self.geometry(row);active=[];ground=self.cfg['ground_height_m']
        for side in self.body_ids:
            status=support['feet'][side]['status']
            if status=='support_candidate' and self.cfg['enable_stance_anchors']:
                active.append(side)
                if side not in self.anchors:
                    point=int(np.argmin(base[side]['points'][:,2]));xy=base[side]['points'][point,:2] if self.cfg['contact_geometry']=='point' else base[side]['centre'][:2]
                    self.anchors[side]=dict(position=np.r_[xy,ground],yaw=base[side]['yaw'],point=point,since_ns=t)
            else:self.anchors.pop(side,None)
        source=ref[6+self.ids];sigma=np.array(self.cfg['joint_fidelity_sigma_rad'])
        deviation=np.array(self.cfg['joint_max_deviation_rad'])
        lower=np.maximum(self.plant.limits[self.ids,0],source-deviation)
        upper=np.minimum(self.plant.limits[self.ids,1],source+deviation)
        if self.previous is not None:
            travel=self.cfg['joint_velocity_limit_rad_s']*dt
            lower=np.maximum(lower,self.previous[:12]-travel);upper=np.minimum(upper,self.previous[:12]+travel)
        lower=np.r_[lower,max(self.cfg['height_range_m'][0],ref[2]-self.cfg['height_max_deviation_m'])]
        upper=np.r_[upper,min(self.cfg['height_range_m'][1],ref[2]+self.cfg['height_max_deviation_m'])]
        radius=self.cfg.get('root_correction_max_m',0.)
        correction=self.previous[13:15] if self.previous is not None else np.zeros(2)
        travel=self.cfg.get('root_correction_rate_m_s',0.)*dt if self.previous is not None else 0.
        # Tiny bounds keep a fixed zero correction compatible with scipy bounds.
        lower=np.r_[lower,np.maximum(-radius,correction-travel)-1e-12]
        upper=np.r_[upper,np.minimum(radius,correction+travel)+1e-12]
        diagnostic=dict(schema='contact_reference_v1',support=copy.deepcopy(support),stance_anchors=list(active),
            target_model_sha256=self.plant.sha256,hardware_enabled=False,contact_measured=False,
            priors=['Flat target floor','Flat sole and retained landing yaw only for support candidates','Neutral swing ankle; rotation unobserved'],
            source_path_unchanged=not bool(radius),uses_future=False)
        diagnostic['contact_geometry']=self.cfg['contact_geometry']
        diagnostic['anchor_geometry']={s:dict(point=a['point'],since_ns=a['since_ns'],position_m=a['position'].tolist()) for s,a in self.anchors.items()}
        diagnostic['priors']=['Flat target floor',('Lowest model sole point as support pivot; rotation not constrained' if self.cfg['contact_geometry']=='point' else 'Flat sole and retained landing yaw'),
                              'Neutral swing ankle' if self.cfg['enable_swing_ankle_prior'] else 'Original unobserved ankle prior retained']
        out['contact_projection']=diagnostic
        def reject(reason):
            out['accepted']=False;out['reason']=reason;diagnostic['projected']=False;self.withdraw();return out
        if np.any(lower>=upper):return reject('Contact reference position/rate/fidelity bounds incompatible')
        start=np.clip(np.r_[source,ref[2],correction],lower,upper);cached_x=None;cached=None
        def fk(x):
            nonlocal cached_x,cached
            if cached_x is None or not np.array_equal(x,cached_x):
                q=ref[6:].copy();q[self.ids]=x[:12];cached=self.geometry(row,q,x[12],True,x[13:15]);cached_x=x.copy()
            return cached
        def cost(x):
            residual=(x[:12]-source)/sigma;value=float(residual@residual);grad=np.r_[2*residual/sigma,np.zeros(3)]
            height=(x[12]-ref[2])/self.cfg['height_sigma_m'];value+=height**2;grad[12]+=2*height/self.cfg['height_sigma_m']
            root_sigma=self.cfg.get('root_correction_sigma_m',.1);value+=float(x[13:15]@x[13:15]/root_sigma**2);grad[13:15]+=2*x[13:15]/root_sigma**2
            if self.previous is not None:
                correction=((x[:12]-source)-(self.previous[:12]-self.previous_source))/self.cfg['correction_change_sigma_rad']
                value+=float(correction@correction);grad[:12]+=2*correction/self.cfg['correction_change_sigma_rad']
            for side,ids in (('left',(4,5)),('right',(10,11))):
                if support['feet'][side]['status']=='swing' and self.cfg['enable_swing_ankle_prior']:
                    ids=np.array(ids);d=(x[ids]-DEFAULT_Q[self.ids[ids]])/self.cfg['swing_ankle_prior_sigma_rad']
                    value+=float(d@d);grad[ids]+=2*d/self.cfg['swing_ankle_prior_sigma_rad']
            return value,grad
        free_points={s:[i for i in range(4) if s not in active or (self.cfg['contact_geometry']=='point' and i!=self.anchors[s]['point'])] for s in self.body_ids}
        def floor(x):return np.concatenate([fk(x)[s]['points'][ids,2]-ground for s,ids in free_points.items() if ids])
        def floor_jac(x):return np.concatenate([fk(x)[s]['floor_jac'][ids] for s,ids in free_points.items() if ids])
        def stance(x,jacobian=False):
            values=[];jacs=[]
            for side in active:
                g=fk(x)[side];a=self.anchors[side];anchor,yaw=a['position'],a['yaw'];normal=g['rotation'][:,2];forward=g['rotation'][:,0]
                if self.cfg['contact_geometry']=='point':
                    values.extend(g['points'][a['point']]-anchor);jacs.extend(g['point_jacs'][a['point']]);continue
                values.extend(np.r_[g['centre']-anchor,normal[:2],wrap(g['yaw']-yaw)])
                skew=np.array([[0,-normal[2],normal[1]],[normal[2],0,-normal[0]],[-normal[1],normal[0],0]])
                normal_jac=-skew@g['rotation_jac']
                fskew=np.array([[0,-forward[2],forward[1]],[forward[2],0,-forward[0]],[-forward[1],forward[0],0]])
                fjac=-fskew@g['rotation_jac'];yaw_jac=(forward[0]*fjac[1]-forward[1]*fjac[0])/max(1e-8,forward[0]**2+forward[1]**2)
                jacs.extend(np.r_[g['position_jac'],normal_jac[:2],yaw_jac[None]])
            return np.array(jacs if jacobian else values)
        # A flat, anchored stance sole already imposes floor equality. Adding
        # redundant active inequalities makes SLSQP unnecessarily degenerate.
        constraints=[dict(type='ineq',fun=floor,jac=floor_jac)] if any(free_points.values()) else []
        if active:constraints.append(dict(type='eq',fun=stance,jac=lambda x:stance(x,True)))
        fit=minimize(cost,start,jac=True,bounds=list(zip(lower,upper)),constraints=constraints,method='SLSQP',
            options=dict(maxiter=self.cfg['solver_max_iterations'],ftol=1e-8))
        width=3 if self.cfg['contact_geometry']=='point' else 6
        final=fk(fit.x);eq=stance(fit.x).reshape(-1,width) if active else np.empty((0,width))
        minimum=float(min(np.min(g['points'][:,2]-ground) for g in final.values()))
        diagnostic.update(solver_success=bool(fit.success),solver_message=str(fit.message),iterations=int(fit.nit),
            minimum_sole_height_m=minimum,maximum_position_residual_m=float(np.max(np.abs(eq[:,:3]))) if len(eq) else 0.,
            maximum_orientation_residual_rad=float(np.max(np.abs(eq[:,3:]))) if len(eq) and width==6 else None)
        if (not fit.success or not np.isfinite(fit.x).all() or minimum < -self.cfg['position_tolerance_m']
                or np.any(fit.x<lower-1e-7) or np.any(fit.x>upper+1e-7)
                or (width==6 and any(final[s]['rotation'][2,2]<.99 for s in active))
                or diagnostic['maximum_position_residual_m']>self.cfg['position_tolerance_m']
                or (width==6 and (diagnostic['maximum_orientation_residual_rad'] or 0)>self.cfg['orientation_tolerance_rad'])):
            return reject('Contact projection infeasible or solver failed')
        target=ref.copy();target[6+self.ids]=fit.x[:12];target[2]=fit.x[12]
        if self.previous is not None:
            world_rate=(fit.x[13:15]-self.previous[13:15])/dt
            rotation=Rotation.from_euler('z',row['root_heading_rad']).as_matrix()[:2,:2]
            target[:2]+=rotation.T@world_rate
        if np.linalg.norm(target[:2])>1.5:return reject('Contact-corrected root velocity exceeds imitation profile')
        out['reference']=target.tolist();out['root_position_m'][2]=float(fit.x[12])
        out['root_position_m'][:2]=(np.asarray(row['root_position_m'][:2])+fit.x[13:15]).tolist()
        diagnostic.update(projected=True,maximum_leg_change_rad=float(np.max(np.abs(fit.x[:12]-source))),
            height_change_m=float(fit.x[12]-ref[2]),root_correction_xy_m=fit.x[13:15].tolist())
        self.previous=fit.x.copy();self.previous_source=source.copy()
        return out
