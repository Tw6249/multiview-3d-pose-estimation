import copy
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from pose_sim.kinematic_observability import propagate_landmarks,torso_frame,hinge_flexion,KinematicObservability
from pose_sim.landmark_velocity import LandmarkVelocity


class KinematicObservabilityTests(unittest.TestCase):
    def test_rotation_covariance_tracks_noise_and_world_rotation(self):
        p=np.array([[0,.1,1],[0,-.1,1],[0,.2,1.5],[0,-.2,1.5]]);C=np.tile(np.eye(3)*1e-4,(4,1,1))
        R,cov=propagate_landmarks(p,C,torso_frame,True);Q=Rotation.from_euler('z',.8).as_matrix()
        r,rotcov=propagate_landmarks(p@Q.T+[2,3,4],C,torso_frame,True)
        np.testing.assert_allclose(r,Q@R,atol=1e-8);np.testing.assert_allclose(rotcov,Q@cov@Q.T,atol=1e-8)
        _,large=propagate_landmarks(p,C*4,torso_frame,True);np.testing.assert_allclose(large,4*cov)

    def test_degeneracy_and_anatomical_twist_stay_unknown(self):
        p=np.array([[0,0,0],[0,0,1],[0,0,2.]])
        self.assertIsNone(propagate_landmarks(p,np.tile(np.eye(3)*.01,(3,1,1)),hinge_flexion))
        self.assertIsNone(torso_frame(np.zeros((4,3))))

    def test_missing_and_predicted_points_do_not_supply_rotation(self):
        e=KinematicObservability();names=['left_hip','right_hip','left_shoulder','right_shoulder']
        row=dict(timestamp_ns=0,joint_names=names,joints_world_m=[[0,.1,1],[0,-.1,1],[0,.2,1.5],[0,-.2,1.5]],
            joint_states=[dict(status='filtered',last_observed_timestamp_ns=0,position_covariance_m2=(np.eye(3)*.001).tolist()) for _ in names])
        saved=copy.deepcopy(row);r=e.update(row);self.assertEqual(saved,row);self.assertIsNotNone(r['landmark_frames']['torso']['rotation_world'])
        row['timestamp_ns']=40_000_000
        for s in row['joint_states']:s['last_observed_timestamp_ns']=row['timestamp_ns']
        row['joint_states'][0]['status']='predicted';r=e.update(row)
        self.assertIsNone(r['landmark_frames']['torso']['rotation_world']);self.assertIn('anatomical axial twist',r['unobserved'])

    def test_quadratic_endpoint_velocity_is_not_window_center_velocity(self):
        e=LandmarkVelocity();q=dict(inlier_cameras=['a','b','c'],ray_angle_deg=30,mean_reprojection_px=0,linearized_max_std_m=.005)
        for i in range(15):
            t=i*.04;row=dict(timestamp_ns=i*40_000_000,joint_names=['p'],joints_world_m=[[t*t,2*t,1.]],quality=[q])
            r=e.update(row)
        np.testing.assert_allclose(r['joints']['p']['velocity_world_m_s'],[2*t,2,0],atol=1e-9)
        self.assertEqual(r['joints']['p']['effective_timestamp_ns'],row['timestamp_ns'])
        self.assertGreater(r['joints']['p']['maximum_std_m_s'],0)

    def test_velocity_missing_and_source_switch_clear_history(self):
        e=LandmarkVelocity();q=dict(inlier_cameras=['a','b'],ray_angle_deg=30,mean_reprojection_px=0,linearized_max_std_m=.005)
        for i in range(10):r=e.update(dict(timestamp_ns=i*40_000_000,joint_names=['p'],joints_world_m=[[i*.04,0,1]],quality=[q]))
        self.assertEqual(r['joints']['p']['status'],'estimated')
        r=e.update(dict(timestamp_ns=400_000_000,joint_names=['p'],joints_world_m=[None],quality=[None]));self.assertEqual(r['joints']['p']['status'],'unknown')
        r=e.update(dict(timestamp_ns=440_000_000,source_id='new',joint_names=['p'],joints_world_m=[[1,0,1]],quality=[q]));self.assertEqual(r['joints']['p']['status'],'unknown')


if __name__=='__main__':unittest.main()
