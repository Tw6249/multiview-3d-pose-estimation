"""Robot-only centroidal/contact feedback; never a source-pose observation.

World XYZ m, m/s, contact N, Z up, robot control-clock ns. Capture point assumes
constant COM height over a flat floor with negligible angular-momentum effects.
Unknown/line support has no polygon margin. This is not force feasibility.
The simulation adapter recomputes a private data copy, never the plant state.
Physical use requires calibrated odometry/contact estimation and clock mapping.
"""
import numpy as np
from scipy.spatial import ConvexHull,QhullError
from scipy.spatial.transform import Rotation


def capture_support(com,velocity,contact_points,ground_z=0.,gravity=9.81):
    com=np.asarray(com,float);velocity=np.asarray(velocity,float)
    points=np.asarray(contact_points,float).reshape(-1,3)
    if com.shape!=(3,) or velocity.shape!=(3,) or not np.isfinite(np.r_[com,velocity,points.ravel(),ground_z,gravity]).all() or gravity<=0:
        raise ValueError('Nonfinite robot support feedback')
    height=float(com[2]-ground_z)
    if height<=.1:return dict(capture_xy_m=None,capture_margin_m=None,com_margin_m=None,support_polygon_xy_m=[],capture_distance_to_support_m=None)
    capture=com[:2]+velocity[:2]*np.sqrt(height/gravity)
    result=dict(capture_xy_m=capture.tolist(),capture_margin_m=None,com_margin_m=None,support_polygon_xy_m=[],capture_distance_to_support_m=None)
    xy=np.unique(points[:,:2],axis=0)
    if not len(xy):return result
    def edge_distance(vertices):
        ends=np.roll(vertices,-1,axis=0);segments=ends-vertices
        fraction=np.clip(np.sum((capture-vertices)*segments,axis=1)/np.maximum(np.sum(segments**2,axis=1),1e-20),0,1)
        return float(np.min(np.linalg.norm(vertices+fraction[:,None]*segments-capture,axis=1)))
    result['capture_distance_to_support_m']=edge_distance(xy[[0,-1]])
    if len(xy)<3:return result
    try:hull=ConvexHull(xy)
    except QhullError:return result
    if hull.volume<1e-8:return result
    # Hull normals are unit length; positive margin is inside all half planes.
    normal=hull.equations[:,:2];offset=hull.equations[:,2]
    result.update(capture_margin_m=float(np.min(-(normal@capture+offset))),
                  com_margin_m=float(np.min(-(normal@com[:2]+offset))),support_polygon_xy_m=xy[hull.vertices].tolist())
    result['capture_distance_to_support_m']=0. if result['capture_margin_m']>=0 else edge_distance(xy[hull.vertices])
    return result


class SimulationSupportFeedback:
    def __init__(self,plant,config):
        self.plant=plant;self.cfg=config;self.data=plant.mj.MjData(plant.model)
        self.root=plant.model.body(config['root_body']).id
        self.bodies={plant.model.body(name).id:side for side,name in config['feet'].items()}
        self.jac=np.zeros((3,plant.model.nv))
        self.angular_jac=np.zeros((3,plant.model.nv))

    def read(self,timestamp_ns):
        p=self.plant;d=self.data;source=p.data
        for name in ('qpos','qvel','ctrl','qacc_warmstart','qfrc_applied','xfrc_applied'):
            getattr(d,name)[:]=getattr(source,name)
        d.time=source.time;p.mj.mj_forward(p.model,d)
        p.mj.mj_jacSubtreeCom(p.model,d,self.jac,self.root)
        com=d.subtree_com[self.root].copy();velocity=self.jac@d.qvel
        p.mj.mj_angmomMat(p.model,d,self.angular_jac,self.root);angular_momentum=self.angular_jac@d.qvel
        forces={s:0. for s in self.bodies.values()};points=[];by_foot={s:[] for s in self.bodies.values()}
        for i,c in enumerate(d.contact):
            bodies=[int(p.model.geom_bodyid[g]) for g in (c.geom1,c.geom2)]
            body=next((b for b in bodies if b in self.bodies),None)
            if 0 not in bodies or body is None:continue
            normal=np.asarray(c.frame).reshape(3,3)[0]
            if abs(normal[2])<.95:continue
            force=np.zeros(6);p.mj.mj_contactForce(p.model,d,i,force)
            force_n=max(0.,float(force[0]));forces[self.bodies[body]]+=force_n
            if force_n>self.cfg['contact_force_threshold_n']:
                points.append(c.pos.copy());by_foot[self.bodies[body]].append(c.pos.tolist())
        gravity=float(-p.model.opt.gravity[2]);weight=float(p.model.body_subtreemass[self.root]*gravity)
        floor=float(np.median(np.array(points)[:,2])) if points else 0.
        result=capture_support(com,velocity,points,floor,gravity)
        heading=Rotation.from_quat(d.qpos[[4,5,6,3]]).as_euler('xyz')[2]
        heading_rotation=Rotation.from_euler('z',heading).as_matrix()
        result.update(timestamp_ns=timestamp_ns,clock_domain='offline_simulation',target_model_sha256=p.sha256,
            mass_kg=float(p.model.body_subtreemass[self.root]),gravity_m_s2=gravity,
            com_world_m=com.tolist(),com_velocity_world_m_s=velocity.tolist(),normal_force_n=forces,
            angular_momentum_world_kg_m2_s=angular_momentum.tolist(),loaded_contact_points_by_foot_world_m=by_foot,
            loaded_contact_points_world_m=[p.tolist() for p in points],joint_positions_rad=d.qpos[7:].tolist(),
            root_height_m=float(d.qpos[2]),root_velocity_heading_local_m_s=(heading_rotation.T@d.qvel[:3]).tolist(),
            double_support=bool(all(f>self.cfg['minimum_foot_load_fraction']*weight for f in forces.values())),
            hardware_validated=False,origin='ideal simulated robot feedback; not human observation')
        return result
