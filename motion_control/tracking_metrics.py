"""Evaluation-only robot feedback metrics; never used as source pose observations."""
import numpy as np
from scipy.spatial.transform import Rotation


def standing_stop_audit(states,dt):
    """Evaluate the final 0.2 s of return-to-standing; not a hardware stop test.

    Keep the historical speed criterion visible, but a motionless fallen robot
    must not pass stopping. The final target is standing, so the existing 0.5 m /
    0.5 rad bounds apply here without classifying an earlier crouch as a fall.
    """
    if not np.isfinite(dt) or not 0<dt<=.1:raise ValueError('Invalid feedback interval')
    count=round(.2/dt)+1
    if len(states)<count:raise ValueError('Insufficient stopping feedback')
    samples=states[-count:]
    xyz=np.asarray([s['root_position_m'] for s in samples],float)
    rpy=np.asarray([s['root_rpy_rad'] for s in samples],float)
    if xyz.shape!=(count,3) or rpy.shape!=(count,3) or not np.isfinite(np.r_[xyz,rpy]).all():
        raise ValueError('Invalid stopping feedback')
    speed=float(np.mean(np.linalg.norm(np.diff(xyz,axis=0)/dt,axis=1)))
    upright=bool(np.min(xyz[:,2])>.5 and np.max(np.abs(rpy[:,:2]))<.5)
    return dict(window_s=(count-1)*dt,mean_base_speed_m_s=speed,
                minimum_height_m=float(np.min(xyz[:,2])),maximum_abs_roll_pitch_rad=float(np.max(np.abs(rpy[:,:2]))),
                speed_only_passed=bool(speed<.05),upright_passed=upright,passed=bool(speed<.05 and upright),
                scope='End-of-replay standing stop in simulation; no physical emergency-stop certification')


class SourceLegAudit:
    """Compare observed thigh/shank directions; do not score invented foot twist.

    Source world Z is up. Remove each body's heading only; robot tilt remains an
    error. Root heading/path are evaluated separately. Own FK data never feeds
    simulation or source perception. These are estimated-source, not MoCap errors.
    """
    def __init__(self,plant):
        self.plant=plant;self.data=plant.mj.MjData(plant.model)
        self.links=[(plant.model.body(s+'_'+a+'_link').id,plant.model.body(s+'_'+b+'_link').id)
                    for s in ('left','right') for a,b in (('hip_pitch','knee'),('knee','ankle_pitch'))]
        self.source_links=((6,8),(8,10),(7,9),(9,11))

    def update(self,row):
        if row is None or not row.get('accepted',False):return None
        values=row.get('source_body_points_m')
        if not isinstance(values,list) or len(values)!=12:return None
        points=np.array([[np.nan]*3 if p is None else p for p in values],float)
        if points.shape!=(12,3) or not np.isfinite(points[6:]).all():return None
        self.data.qpos[:]=self.plant.data.qpos
        self.plant.mj.mj_kinematics(self.plant.model,self.data)
        yaw=Rotation.from_quat(self.data.qpos[[4,5,6,3]]).as_euler('xyz')[2]
        source_rotation=Rotation.from_euler('z',row.get('source_root_heading_rad',row['root_heading_rad'])).as_matrix()
        target_rotation=Rotation.from_euler('z',yaw).as_matrix();errors=[]
        for (a,b),(i,j) in zip(self.source_links,self.links):
            source=(points[b]-points[a])@source_rotation
            target=(self.data.xpos[j]-self.data.xpos[i])@target_rotation
            norms=np.linalg.norm(source)*np.linalg.norm(target)
            if norms<1e-8:return None
            errors.append(float(np.degrees(np.arccos(np.clip(source@target/norms,-1,1)))))
        return errors


def phase_lag(actual,source,dt,indices):
    actual=np.asarray(actual);source=np.asarray(source);result={}
    for j in indices:
        choices=[]
        for lag in range(-15,16):
            a=actual[max(0,lag):min(len(actual),len(actual)+lag),j]
            b=source[max(0,-lag):min(len(source),len(source)-lag),j]
            valid=np.isfinite(a)&np.isfinite(b)
            if valid.sum()<30 or min(np.std(a[valid]),np.std(b[valid]))<.03:continue
            choices.append((float(np.corrcoef(a[valid],b[valid])[0,1]),lag,int(valid.sum())))
        if choices:
            corr,lag,count=max(choices);result[str(j)]=dict(lag_ms=lag*dt*1000,correlation=corr,samples=count)
        else:result[str(j)]=None
    return result


def foot_contacts(plant):
    """Contact-point tangential speed in simulated stance, not human foot truth."""
    result={side:dict(normal_force_n=0.,tangent_speeds_m_s=[]) for side in ('left','right')}
    bodies={plant.model.body(side+'_ankle_roll_link').id:side for side in result}
    for contact_index,contact in enumerate(plant.data.contact):
        ids=[int(plant.model.geom_bodyid[g]) for g in (contact.geom1,contact.geom2)]
        if 0 not in ids:continue
        body=next((b for b in ids if b in bodies),None)
        if body is None:continue
        force=np.zeros(6);plant.mj.mj_contactForce(plant.model,plant.data,contact_index,force)
        side=bodies[body];result[side]['normal_force_n']+=abs(float(force[0]))
        if force[0]<=5.:continue
        jac=np.zeros((3,plant.model.nv));rotation_jac=np.zeros_like(jac)
        plant.mj.mj_jac(plant.model,plant.data,jac,rotation_jac,contact.pos,body)
        velocity=jac@plant.data.qvel;normal=np.asarray(contact.frame).reshape(3,3)[0]
        tangent=velocity-(velocity@normal)*normal
        result[side]['tangent_speeds_m_s'].append(float(np.linalg.norm(tangent)))
    return result
