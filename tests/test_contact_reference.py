import copy
import importlib.util
import unittest
import numpy as np
from motion_control.twist import DEFAULT_REF


@unittest.skipUnless(importlib.util.find_spec('mujoco'),'Requires frozen robot environment')
class ContactReferenceTests(unittest.TestCase):
    def row(self,t=0):
        return dict(timestamp_ns=t,accepted=True,reference=DEFAULT_REF.tolist(),root_position_m=[0.,0.,float(DEFAULT_REF[2])],
                    root_heading_rad=0.,root_path_frame='source_world_xy_anchored_at_first_accepted_pelvis; target_height_z')

    def support(self,t=0,left='unknown',right='unknown'):
        return dict(timestamp_ns=t,measured_contact=False,feet={s:dict(status=v) for s,v in (('left',left),('right',right))})

    def test_sphere_jacobians_match_finite_difference(self):
        from motion_control.contact_reference import ContactReferenceProjector
        p=ContactReferenceProjector();r=self.row();g=p.geometry(r,jacobian=True)
        q=np.array(r['reference'][6:])
        for j in (0,3,4,6,10):
            qp=q.copy();qp[j]+=1e-6;other=p.geometry(r,q=qp)
            for s in ('left','right'):
                np.testing.assert_allclose((other[s]['centre']-g[s]['centre'])/1e-6,g[s]['position_jac'][:,j],atol=1e-6)
                np.testing.assert_allclose((other[s]['points'][:,2]-g[s]['points'][:,2])/1e-6,g[s]['floor_jac'][:,j],atol=1e-6)

    def test_unknown_support_keeps_anchors_empty_and_original_targets(self):
        from motion_control.contact_reference import ContactReferenceProjector
        p=ContactReferenceProjector();r=self.row();before=copy.deepcopy(r)
        out=p.update(r,self.support())
        self.assertTrue(out['accepted']);self.assertEqual(p.anchors,{})
        self.assertEqual(r,before);self.assertEqual(out['source_reference'],r['reference'])
        self.assertEqual(out['reference'][18:],r['reference'][18:])
        self.assertGreaterEqual(out['contact_projection']['minimum_sole_height_m'],-.001)

    def test_stance_constraint_preserves_landing_point_as_pelvis_moves(self):
        from motion_control.contact_reference import ContactReferenceProjector
        p=ContactReferenceProjector();a=p.update(self.row(),self.support(left='support_candidate'))
        self.assertTrue(a['accepted']);point=a['contact_projection']['anchor_geometry']['left']['point'];first=p.geometry(a)['left']['points'][point]
        r=self.row(40_000_000);r['root_position_m'][0]=.005
        b=p.update(r,self.support(40_000_000,left='support_candidate'))
        self.assertTrue(b['accepted']);second=p.geometry(b)['left']['points'][point]
        np.testing.assert_allclose(first,second,atol=.001)
        self.assertEqual(b['source_root_position_m'][:2],[.005,0.])
        offset=np.array(b['root_position_m'][:2])-b['source_root_position_m'][:2]
        self.assertLess(np.linalg.norm(offset),.015)
        prior=np.array(a['root_position_m'][:2])-a['source_root_position_m'][:2]
        np.testing.assert_allclose(np.array(b['reference'][:2])-r['reference'][:2],(offset-prior)/.04,atol=1e-10)

    def test_missing_withdraws_and_infeasible_pose_is_not_accepted(self):
        from motion_control.contact_reference import ContactReferenceProjector
        p=ContactReferenceProjector();p.update(self.row(),self.support(left='support_candidate'))
        missing=dict(timestamp_ns=40_000_000,accepted=False,reason='occluded')
        self.assertEqual(p.update(missing,self.support(40_000_000)),missing);self.assertEqual(p.anchors,{})
        r=self.row(80_000_000);r['reference'][2]=.1;r['root_position_m'][2]=.1
        out=p.update(r,self.support(80_000_000));self.assertFalse(out['accepted'])
        self.assertIsNone(p.previous)

    def test_leg_audit_preserves_unknown_arms_and_does_not_mutate_physics(self):
        from motion_control.contact_reference import ContactReferenceProjector
        from motion_control.tracking_metrics import SourceLegAudit
        p=ContactReferenceProjector();plant=p.plant;audit=SourceLegAudit(plant)
        row=self.row();plant.data.qpos[7:]=DEFAULT_REF[6:];plant.mj.mj_kinematics(plant.model,plant.data)
        points=[None]*12
        for a,b,ia,ib in ((6,8,'left_hip_pitch','left_knee'),(8,10,'left_knee','left_ankle_pitch'),
                          (7,9,'right_hip_pitch','right_knee'),(9,11,'right_knee','right_ankle_pitch')):
            points[a]=plant.data.body(ia+'_link').xpos.tolist();points[b]=plant.data.body(ib+'_link').xpos.tolist()
        row['source_body_points_m']=points;before=plant.data.qpos.copy()
        np.testing.assert_allclose(audit.update(row),0.,atol=2e-6)
        np.testing.assert_array_equal(plant.data.qpos,before)

    def test_malformed_packet_clears_anchors_without_advancing_clock(self):
        from motion_control.contact_reference import ContactReferenceProjector
        for kind in ('clock','root','frame','support'):
            p=ContactReferenceProjector();p.update(self.row(),self.support(left='support_candidate'))
            row=self.row(40_000_000);support=self.support(40_000_000)
            if kind=='clock':support['timestamp_ns']+=1
            elif kind=='root':row['root_position_m'][0]=float('nan')
            elif kind=='frame':row['root_path_frame']='unanchored'
            else:support['feet']['left']['status']='confident'
            with self.subTest(kind=kind),self.assertRaises(ValueError):p.update(row,support)
            self.assertEqual(p.last_time,0);self.assertEqual(p.anchors,{})
            out=p.update(self.row(40_000_000),self.support(40_000_000))
            self.assertTrue(out['accepted']);self.assertEqual(p.anchors,{})

    def test_contact_profile_rejects_hardware_and_wrong_leg_order(self):
        from motion_control.contact_reference import ContactReferenceProjector,load_config
        cfg=load_config();cfg['hardware_enabled']=True
        with self.assertRaises(ValueError):ContactReferenceProjector(cfg)
        cfg=load_config();cfg['leg_joint_names'].reverse()
        with self.assertRaises(ValueError):ContactReferenceProjector(cfg)


if __name__=='__main__':unittest.main()
