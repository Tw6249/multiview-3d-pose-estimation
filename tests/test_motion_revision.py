import copy
import unittest
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tests.test_motion_frontend import scene
from pose_sim.human_motion import HumanMotionEstimator
from pose_sim.multiview_temporal import MultiViewTemporal
from pose_sim.motion_evaluation import compare_methods
from pose_sim.inference import submit_single_pose_calls


class MotionRevisionTests(unittest.TestCase):
    def test_knee_only_swap_does_not_relabel_correct_hips_or_ankles(self):
        cameras,cfg,xyz,rows=scene();fusion=MultiViewTemporal(dict(cameras=cameras),cfg,structured=False)
        state=HumanMotionEstimator();previous=None
        for row in rows[:15]:
            p,d=fusion.update(row,previous);previous=state.update(p,cameras,d)
        row=copy.deepcopy(rows[15])
        for cid in ('cam00','cam01'):
            xy=row['views'][cid]['keypoints_px'];xy[13],xy[14]=xy[14],xy[13]
        p,d=fusion.update(row,previous)
        self.assertEqual({c['group'] for c in p['association']['corrections']},{'knee'})
        for cid in cameras:
            for j in (11,12,15,16):
                self.assertEqual(d['views'][cid]['keypoints_px'][j],rows[15]['views'][cid]['keypoints_px'][j])
        np.testing.assert_allclose(np.array(p['joints_world_m'])[[13,14]],xyz[15,[13,14]],atol=.001)
        self.assertFalse(p['association']['candidate_scores_calibrated'])

    def test_global_wrist_ambiguity_is_local_not_whole_arm_rejection(self):
        cameras,cfg,xyz,rows=scene();fusion=MultiViewTemporal(dict(cameras=cameras),cfg,structured=False)
        state=HumanMotionEstimator();previous=None
        for row in rows[:10]:
            p,d=fusion.update(row,previous);previous=state.update(p,cameras,d)
        row=copy.deepcopy(rows[10])
        for view in row['views'].values():
            xy=view['keypoints_px'];xy[9],xy[10]=xy[10],xy[9]
        p,d=fusion.update(row,previous)
        self.assertIsNone(p['joints_world_m'][9])
        self.assertIsNotNone(p['joints_world_m'][7]);self.assertIsNotNone(p['joints_world_m'][11])

    def test_reported_velocity_predicts_next_missing_observation(self):
        state=HumanMotionEstimator();q=dict(inlier_cameras=['a','b','c'],ray_angle_deg=40,
            linearized_max_std_m=.005,mean_reprojection_px=1.)
        for f in range(20):
            row=dict(frame_id=f,timestamp_ns=f*40_000_000,joint_names=['left_wrist'],
                landmark_schema_id='coco17',source_id='s',perception_signature='m',calibration_sha256='c',
                joints_world_m=[[.01*f*f,0.,1.]],quality=[q])
            out=state.update(row)
        expected=np.array(out['joints_world_m'][0])+.04*np.array(out['velocities_world_m_s'][0])
        before=np.array(out['joint_states'][0]['velocity_covariance_m2_s2'])
        row.update(frame_id=20,timestamp_ns=800_000_000,joints_world_m=[None],quality=[None])
        missing=state.update(row)
        np.testing.assert_allclose(missing['joints_world_m'][0],expected)
        after=np.array(missing['joint_states'][0]['velocity_covariance_m2_s2'])
        self.assertGreaterEqual(np.linalg.eigvalsh(after-before).min(),-1e-10)
        self.assertEqual(missing['joint_states'][0]['status'],'predicted')
        self.assertFalse(missing['joint_states'][0]['velocity_uncertainty_calibrated'])

    def test_provisional_dimensions_do_not_hard_reject_measurements(self):
        e=HumanMotionEstimator();e.shape.result={'left_hip--left_knee':dict(status='collecting',uncertainty_m=.06)}
        self.assertTrue(np.isinf(e._bone_gate_tolerance('left_hip--left_knee',.4)))
        self.assertGreaterEqual(e._bone_constraint_sigma('left_hip--left_knee'),.06)

    def test_common_metrics_penalize_missing_and_predicted_targets(self):
        gt=[dict(frame_id=0,timestamp_ns=0,joint_names=['a','b'],joints_world_m=[[0,0,0],[0,0,0]])]
        first={**gt[0],'joints_world_m':[[.01,0,0],[.3,0,0]],'joint_states':[dict(status='filtered')]*2}
        second={**gt[0],'joints_world_m':[[.02,0,0],None],'joint_states':[dict(status='predicted'),dict(status='missing')]}
        r=compare_methods({'a':[first],'b':[second]},gt)['methods']
        self.assertAlmostEqual(r['a']['common_position_error_mm']['mean'],10)
        self.assertAlmostEqual(r['b']['common_position_error_mm']['mean'],20)
        self.assertEqual(r['b']['coverage'],.5);self.assertEqual(r['b']['current_all_target_success'],0)
        self.assertIsNone(r['a']['common_current_error_mm'])
        self.assertEqual(compare_methods({'empty':[]},gt)['methods']['empty']['all_target_success'],0)

    def test_common_metrics_support_legacy_coco_reference_and_null_targets(self):
        from pose_sim.geometry import JOINTS
        gt=[dict(frame_id=0,timestamp_ns=0,joints_world_m=[None]*5+[[0.,0.,0.]]*12)]
        row=dict(**gt[0],joint_names=JOINTS)
        r=compare_methods({'baseline':[row]},gt)
        self.assertEqual(r['position_targets'],12)
        self.assertEqual(r['methods']['baseline']['common_position_error_mm']['mean'],0.)

    def test_concurrent_pose_preserves_batch_one_and_order(self):
        class FakeModel:
            backend='onnxruntime'
            def preprocess(self,image,box):return image,0,1
            def inference(self,image):return image+1
            def postprocess(self,value,center,scale):return value[None],np.ones((1,len(value)))
        images=[np.full((2,2),i,float) for i in range(4)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=submit_single_pose_calls(FakeModel(),images,[None]*4,pool)
            values=[f.result()[0] for f in futures]
        for i,v in enumerate(values):np.testing.assert_array_equal(v,images[i]+1)


if __name__=='__main__':unittest.main()
