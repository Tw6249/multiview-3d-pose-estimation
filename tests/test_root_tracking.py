import unittest
import numpy as np
from motion_control.root_tracking import RootTracker,planar_rotation,load_config
from motion_control.twist import DEFAULT_REF


class RootTrackingFrameTests(unittest.TestCase):
    def test_lagging_heading_rotates_forward_velocity_into_robot_frame(self):
        row=dict(root_position_m=[0,0,.8],root_heading_rad=0.)
        p=RootTracker(row,[0,0,.8],0.);ref=DEFAULT_REF.copy();ref[:2]=[.5,0.]
        out,d=p.correct(row,ref,[0,0,.8],np.pi/2)
        np.testing.assert_allclose(out[:2],[0,-.5],atol=1e-12)
        np.testing.assert_allclose(d['applied_feedforward_world_m_s'],[.5,0],atol=1e-12)
        self.assertLess(d['feedforward_world_frame_error_m_s'],1e-12)
        np.testing.assert_array_equal(out[6:],ref[6:]);np.testing.assert_array_equal(ref[:2],[.5,0])

    def test_legacy_ablation_exposes_direction_error_without_changing_joints(self):
        cfg=load_config();cfg['velocity_frame']='legacy_unrotated';row=dict(root_position_m=[0,0,.8],root_heading_rad=0.)
        p=RootTracker(row,[0,0,.8],0.,cfg);ref=DEFAULT_REF.copy();ref[0]=.5
        out,d=p.correct(row,ref,[0,0,.8],np.pi/2)
        np.testing.assert_array_equal(out[:2],ref[:2]);self.assertAlmostEqual(d['feedforward_world_frame_error_m_s'],np.sqrt(.5))

    def test_independent_source_and_robot_world_rotations_preserve_local_command(self):
        ref=DEFAULT_REF.copy();ref[:2]=[.4,-.2];results=[]
        for source_angle,robot_angle in [(0,0),(.7,-1.2),(-2.3,2.8)]:
            s=planar_rotation(source_angle);r=planar_rotation(robot_angle)
            first=dict(root_position_m=np.r_[s@[3.,4.],.8],root_heading_rad=.2+source_angle)
            row=dict(root_position_m=np.r_[s@[3.02,4.01],.8],root_heading_rad=.8+source_angle)
            p=RootTracker(first,np.r_[r@[1.,2.],.8],-.1+robot_angle)
            out,_=p.correct(row,ref,np.r_[r@[1.01,2.02],.8],.3+robot_angle);results.append(out)
        for out in results[1:]:np.testing.assert_allclose(out,results[0],atol=1e-12)

    def test_heading_wrap_and_matching_heading_preserve_velocity(self):
        row=dict(root_position_m=[0,0,.8],root_heading_rad=np.pi-.01)
        p=RootTracker(row,[0,0,.8],np.pi-.01);ref=DEFAULT_REF.copy();ref[:2]=[.3,-.2]
        out,_=p.correct(row,ref,[0,0,.8],-np.pi-.01)
        np.testing.assert_allclose(out[:2],ref[:2],atol=1e-12)

    def test_invalid_packets_and_hardware_profile_fail_closed(self):
        row=dict(root_position_m=[0,0,.8],root_heading_rad=0.)
        p=RootTracker(row,[0,0,.8],0.)
        with self.assertRaises(ValueError):p.correct(row,DEFAULT_REF,[0,np.nan,.8],0.)
        cfg=load_config();cfg['hardware_enabled']=True
        with self.assertRaises(ValueError):RootTracker(row,[0,0,.8],0.,cfg)


if __name__=='__main__':unittest.main()
