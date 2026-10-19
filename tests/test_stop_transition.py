import copy
import importlib.util
import unittest
import numpy as np
from motion_control.stop_transition import StopTransition,load_config
from motion_control.support_feedback import capture_support,SimulationSupportFeedback
from motion_control.twist import DEFAULT_REF


class StopTransitionTests(unittest.TestCase):
    def feedback(self,t,**extra):
        return dict(timestamp_ns=t,clock_domain='offline_simulation',target_model_sha256='model',
                    double_support=True,capture_margin_m=.02,com_velocity_world_m_s=[.1,0,0],**extra)

    def test_current_source_passes_unchanged_and_cancels_pending_stop(self):
        p=StopTransition(DEFAULT_REF,'model');r=DEFAULT_REF.copy();r[0]=.4;r[9]=1.
        out,_=p.step(0,r,r,'tracking');np.testing.assert_array_equal(out,r)
        p.step(20_000_000,DEFAULT_REF,r,'return_to_stand',self.feedback(20_000_000))
        out,d=p.step(40_000_000,r,r,'tracking');np.testing.assert_array_equal(out,r)
        self.assertFalse(d['active']);self.assertIsNone(p.entered)

    def test_braking_does_not_wait_for_support_or_create_steps(self):
        p=StopTransition(DEFAULT_REF,'model');r=DEFAULT_REF.copy();r[[0,1,5]]=[.4,-.2,.3];r[9]=1.
        for i in range(17):out,d=p.step(i*20_000_000,DEFAULT_REF,r,'return_to_stand')
        np.testing.assert_array_equal(out[[0,1,5]],0.)
        np.testing.assert_array_equal(out[6:],r[6:]);self.assertEqual(d['phase'],'waiting_support')
        self.assertFalse(d['source_motion_generated'])

    def test_support_requires_fresh_model_bound_confirmed_feedback(self):
        for defect in ('stale','future','model','clock','unknown','single'):
            p=StopTransition(DEFAULT_REF,'model')
            for i in range(20):
                t=i*20_000_000;f=self.feedback(t)
                if defect=='stale':f['timestamp_ns']-=100_000_000
                elif defect=='future':f['timestamp_ns']+=1
                elif defect=='model':f['target_model_sha256']='wrong'
                elif defect=='clock':f['clock_domain']='camera'
                elif defect=='unknown':f['capture_margin_m']=None
                else:f['double_support']=False
                out,d=p.step(t,DEFAULT_REF,DEFAULT_REF,'return_to_stand',f)
            self.assertEqual(d['phase'],'waiting_support',defect)
        p=StopTransition(DEFAULT_REF,'model')
        for i in range(9):out,d=p.step(i*20_000_000,DEFAULT_REF,DEFAULT_REF,'return_to_stand',self.feedback(i*20_000_000))
        self.assertEqual(d['phase'],'blending_stand');self.assertFalse(d['timeout_fallback'])

    def test_missing_feedback_has_finite_explicit_timeout(self):
        p=StopTransition(DEFAULT_REF,'model');r=DEFAULT_REF.copy();r[9]=1.;r[0]=.4
        for i in range(120):out,d=p.step(i*20_000_000,DEFAULT_REF,r,'return_to_stand')
        np.testing.assert_allclose(out,DEFAULT_REF);self.assertTrue(d['timeout_fallback'])
        self.assertEqual(d['phase'],'settled_reference')

    def test_missing_source_option_is_separate_from_terminal_transition(self):
        p=StopTransition(DEFAULT_REF,'model');_,d=p.step(0,DEFAULT_REF,DEFAULT_REF,'vision_unavailable')
        self.assertFalse(d['active'])
        cfg=load_config();cfg['apply_to_missing']=True;p=StopTransition(DEFAULT_REF,'model',config=cfg)
        _,d=p.step(0,DEFAULT_REF,DEFAULT_REF,'vision_unavailable');self.assertTrue(d['active'])
        with self.assertRaises(ValueError):p.step(0,DEFAULT_REF,DEFAULT_REF,'vision_unavailable')
        cfg['hardware_enabled']=True
        with self.assertRaises(ValueError):StopTransition(DEFAULT_REF,'model',config=cfg)

    def test_robot_feedback_boundary_only_uses_fresh_robot_state(self):
        cfg=load_config();cfg['entry_state']='robot_feedback'
        for stamp,expected in ((0,.5),(-100_000_000,0.)):
            p=StopTransition(DEFAULT_REF,'model',config=cfg);f=self.feedback(stamp)
            f.update(joint_positions_rad=(DEFAULT_REF[6:]+.1).tolist(),root_height_m=.7,root_velocity_heading_local_m_s=[.5,0,0])
            out,_=p.step(0,DEFAULT_REF,DEFAULT_REF,'return_to_stand',f)
            self.assertEqual(out[0],expected)
            np.testing.assert_array_equal(out[6:],DEFAULT_REF[6:]+(.1 if stamp==0 else 0.))


class SupportFeedbackTests(unittest.TestCase):
    def test_capture_point_uses_velocity_and_preserves_translation(self):
        points=np.array([[-.1,-.2,0],[.1,-.2,0],[.1,.2,0],[-.1,.2,0]])
        a=capture_support([0,0,.8],[0,0,0],points)
        self.assertAlmostEqual(a['capture_margin_m'],.1)
        b=capture_support([0,0,.8],[1,0,0],points);self.assertLess(b['capture_margin_m'],0)
        c=capture_support([2,3,4.8],[1,0,0],points+[2,3,4],4.)
        self.assertAlmostEqual(b['capture_margin_m'],c['capture_margin_m'])

    def test_absent_and_degenerate_support_are_unknown(self):
        for points in ([],[[0,0,0]],[[0,0,0],[.1,0,0],[.2,0,0]]):
            r=capture_support([0,0,.8],[0,0,0],points);self.assertIsNone(r['capture_margin_m'])
        with self.assertRaises(ValueError):capture_support([0,0,float('nan')],[0,0,0],[])
        line=capture_support([.1,.3,.8],[0,0,0],[[0,0,0],[.2,0,0]])
        self.assertIsNone(line['capture_margin_m']);self.assertAlmostEqual(line['capture_distance_to_support_m'],.3)

    @unittest.skipUnless(importlib.util.find_spec('mujoco'),'Requires robot environment')
    def test_feedback_does_not_mutate_plant_and_matches_finite_difference_com_velocity(self):
        from motion_control.twist import TwistPlant
        p=TwistPlant();p.data.qvel[:3]=[.3,-.2,.1];p.data.qvel[3:6]=[.1,-.2,.15];p.data.qvel[9]=.3
        before={k:getattr(p.data,k).copy() for k in ('qpos','qvel','ctrl','qacc')}
        adapter=SimulationSupportFeedback(p,load_config());a=adapter.read(0)
        for k,v in before.items():np.testing.assert_array_equal(getattr(p.data,k),v)
        p.mj.mj_integratePos(p.model,adapter.data.qpos,adapter.data.qvel,1e-6)
        p.mj.mj_kinematics(p.model,adapter.data);p.mj.mj_comPos(p.model,adapter.data)
        np.testing.assert_allclose((adapter.data.subtree_com[adapter.root]-a['com_world_m'])/1e-6,a['com_velocity_world_m_s'],atol=1e-7)


if __name__=='__main__':unittest.main()
