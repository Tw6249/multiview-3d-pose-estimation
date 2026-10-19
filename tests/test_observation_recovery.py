import copy
import unittest
import numpy as np
from tests.test_motion_frontend import scene
from pose_sim.observation_consistency import ObservationConsistency
from pose_sim.human_motion import HumanMotionEstimator
from pose_sim.reacquisition import Reacquisition
from pose_sim.multiview_temporal import MultiViewTemporal
from motion_control.perception_gate import PerceptionGate


class ObservationRecoveryTests(unittest.TestCase):
    def test_accepted_measurements_are_kept_as_bounded_independent_history(self):
        e=HumanMotionEstimator({'state_reacquisition':True})
        q=dict(inlier_cameras=['a','b','c'],ray_angle_deg=40,linearized_max_std_m=.005,mean_reprojection_px=1.)
        for f in range(6):
            out=e.update(dict(frame_id=f,timestamp_ns=f*40_000_000,source_id='a',landmark_schema_id='coco17',
                calibration_sha256='cal',perception_signature='model',joint_names=['right_wrist'],
                joints_world_m=[[.04*f,0.,1.]],quality=[q]))
            self.assertEqual(out['joint_states'][0]['status'],'filtered')
        h=e.reacquisition.pending['right_wrist']
        self.assertEqual(len(h),3)
        np.testing.assert_allclose([r[1][0] for r in h],[.12,.16,.20])
        np.testing.assert_allclose(e.reacquisition.observe('right_wrist',np.array([.24,0.,1.]),240_000_000,.018,True),[1.,0.,0.],atol=1e-12)

    def test_one_isolated_outlier_cannot_reset_an_established_track(self):
        e=HumanMotionEstimator({'state_reacquisition':True})
        q=dict(inlier_cameras=['a','b','c'],ray_angle_deg=40,linearized_max_std_m=.005,mean_reprojection_px=1.)
        for f in range(12):
            out=e.update(dict(frame_id=f,timestamp_ns=f*40_000_000,source_id='a',landmark_schema_id='coco17',
                calibration_sha256='cal',perception_signature='model',joint_names=['right_wrist'],
                joints_world_m=[[1. if f==10 else 0.,0.,1.]],quality=[q]))
            self.assertFalse(out['joint_states'][0]['track_reinitialized'])
            self.assertLess(abs(out['joints_world_m'][0][0]),.01)
        self.assertEqual(out['joint_states'][0]['status'],'filtered')

    def test_source_change_discards_continuous_candidate_evidence(self):
        e=HumanMotionEstimator({'state_reacquisition':True})
        q=dict(inlier_cameras=['a','b','c'],ray_angle_deg=40,linearized_max_std_m=.005,mean_reprojection_px=1.)
        for f in range(4):
            out=e.update(dict(frame_id=f,timestamp_ns=f*40_000_000,source_id='b' if f==3 else 'a',landmark_schema_id='coco17',
                calibration_sha256='cal',perception_signature='model',joint_names=['right_wrist'],
                joints_world_m=[[0.,0.,1.]],quality=[q]))
        self.assertTrue(out['reset'])
        self.assertEqual(len(e.reacquisition.pending['right_wrist']),1)

    def test_reacquiring_does_not_seed_cross_view_identity(self):
        cameras,cfg,xyz,rows=scene();fusion=MultiViewTemporal(dict(cameras=cameras),cfg)
        estimator=HumanMotionEstimator();pose,assigned=fusion.update(rows[0])
        state=estimator.update(pose,cameras,assigned)
        self.assertIn('right_wrist',fusion.prior(state,rows[1]))
        state['joint_states'][state['joint_names'].index('right_wrist')]['status']='reacquiring'
        self.assertNotIn('right_wrist',fusion.prior(state,rows[1]))

    def test_jump_audit_includes_reset_frame_and_motion_across_missing_gap(self):
        from scripts.benchmark_recovery import output_steps
        rows=[dict(timestamp_ns=i*40_000_000,joint_names=['wrist'],joints_world_m=[p],
                   joint_states=[dict(status='reacquiring' if i==1 else 'filtered',track_reinitialized=i==1)])
              for i,p in enumerate(([0,0,0],[1,0,0],None,[1.2,0,0]))]
        truth=[dict(joint_names=['wrist'],joints_world_m=[[i*.1,0,0]]) for i in range(4)]
        audit=output_steps(rows,truth)
        self.assertAlmostEqual(audit['adjacent_output_step_error_mm']['maximum'],900.)
        self.assertAlmostEqual(audit['reset_output_step_error_mm']['maximum'],900.)
        self.assertAlmostEqual(audit['gap_output_step_error_mm']['maximum'],0.)

    def test_collapsed_arm_removes_only_independently_contradicted_side(self):
        cameras, cfg, xyz, rows = scene()
        row = copy.deepcopy(rows[0]); camera = 'cam01'
        for left, right in ((5,6), (7,8), (9,10)):
            row['views'][camera]['keypoints_px'][right] = row['views'][camera]['keypoints_px'][left].copy()
        frozen = copy.deepcopy(row)
        out = ObservationConsistency(cameras, cfg).update(row)
        self.assertEqual(row, frozen)
        for left, right in ((5,6), (7,8), (9,10)):
            self.assertEqual(out['views'][camera]['scores'][left], .9)
            self.assertEqual(out['views'][camera]['scores'][right], 0.)
        self.assertEqual(out['views'][camera]['scores'][17:], frozen['views'][camera]['scores'][17:])

    def test_overlap_without_other_view_evidence_is_not_relabelled(self):
        cameras, cfg, xyz, rows = scene(); row = copy.deepcopy(rows[0])
        row['views'] = {k: v for k, v in row['views'].items() if k in ('cam00','cam01')}
        row['views']['cam00']['keypoints_px'][10] = row['views']['cam00']['keypoints_px'][9].copy()
        out = ObservationConsistency(cameras, cfg).update(row)
        self.assertEqual(out['views'], row['views'])

    def test_genuine_projected_overlap_is_preserved(self):
        cameras,cfg,xyz,rows=scene();row=copy.deepcopy(rows[0])
        for view in row['views'].values():
            view['keypoints_px'][10]=view['keypoints_px'][9].copy()
        self.assertEqual(ObservationConsistency(cameras,cfg).update(row)['views'],row['views'])

    def test_weak_unrelated_instance_rejected_without_history(self):
        cameras, cfg, xyz, rows = scene(); row = copy.deepcopy(rows[0])
        bad = row['views']['cam00']
        bad['keypoints_px'] = [[10000.,10000.]]*33
        bad['scores'] = [.25]*33
        out = ObservationConsistency(cameras, cfg).update(row)
        self.assertFalse(any(out['views']['cam00']['scores']))
        self.assertEqual(out['views']['cam01'], row['views']['cam01'])

    def test_stale_other_cameras_cannot_certify_correspondence(self):
        cameras, cfg, xyz, rows = scene(); row = copy.deepcopy(rows[0])
        for j in (6,8,10):
            row['views']['cam01']['keypoints_px'][j] = row['views']['cam01']['keypoints_px'][j-1].copy()
        for c in ('cam02','cam03'):
            row['views'][c]['timestamp_ns'] -= 20_000_000
        self.assertEqual(ObservationConsistency(cameras,cfg).update(row)['views'], row['views'])

    def test_continuous_current_track_recovers_wrong_history_with_transition(self):
        e = HumanMotionEstimator({'state_reacquisition':True})
        q = dict(inlier_cameras=['a','b','c'],ray_angle_deg=40,linearized_max_std_m=.005,mean_reprojection_px=1.)
        def row(f, x, source='a', quality=q):
            return dict(frame_id=f,timestamp_ns=f*40_000_000,source_id=source,landmark_schema_id='coco17',
                calibration_sha256='cal',perception_signature='model',joint_names=['right_wrist'],
                joints_world_m=[[x,0.,1.]],quality=[quality])
        for f in range(10):
            e.update(row(f,0.))
        self.assertEqual(e.update(row(10,1.))['joint_states'][0]['status'],'predicted')
        self.assertEqual(e.update(row(11,1.01))['joint_states'][0]['status'],'predicted')
        recovered = e.update(row(12,1.02))
        self.assertEqual(recovered['joint_states'][0]['status'],'reacquiring')
        self.assertTrue(recovered['joint_states'][0]['track_reinitialized'])
        self.assertAlmostEqual(recovered['joints_world_m'][0][0],1.02)
        self.assertIsNone(recovered['quality'][0])
        self.assertEqual(e.update(row(13,1.03))['joint_states'][0]['status'],'filtered')

    def test_missing_unsupported_and_discontinuous_candidates_do_not_accumulate(self):
        r=Reacquisition(); p=np.array([0.,0.,1.])
        self.assertIsNone(r.observe('a',p,0,.01,True))
        self.assertIsNone(r.observe('a',p,40_000_000,.01,False))
        self.assertIsNone(r.observe('a',p,80_000_000,.01,True))
        r.retain(set())
        self.assertIsNone(r.observe('a',p,120_000_000,.01,True))
        self.assertIsNone(r.observe('a',p+5,160_000_000,.01,True))
        self.assertIsNone(r.observe('a',p+5,240_000_000,.01,True))

    def test_expired_track_exposes_fresh_geometry_as_transition_not_observed_motion(self):
        e=HumanMotionEstimator({'state_reacquisition':True})
        q=dict(inlier_cameras=['a','b'],ray_angle_deg=40,linearized_max_std_m=.005,mean_reprojection_px=1.)
        for f in range(12):
            row=dict(frame_id=f,timestamp_ns=f*40_000_000,source_id='a',landmark_schema_id='coco17',
                calibration_sha256='cal',perception_signature='model',joint_names=['right_wrist'],
                joints_world_m=[[1. if f>=10 else 0.,0.,1.]] if f==0 or f>=10 else [None],quality=[q] if f==0 or f>=10 else [None])
            out=e.update(row)
            if f==10:
                self.assertEqual(out['joint_states'][0]['status'],'reacquiring')
                self.assertEqual(out['joint_states'][0]['last_observed_timestamp_ns'],400_000_000)
                self.assertIsNone(out['quality'][0])
                np.testing.assert_allclose(out['velocities_world_m_s'][0],[0,0,0])

    def test_reacquiring_is_not_allowed_to_follow_robot(self):
        gate=PerceptionGate(); names=[s+'_'+j for s in ('left','right') for j in ('hip','knee','ankle')]
        for f in range(3):
            t=f*40_000_000
            row=dict(timestamp_ns=t,joint_names=names,uncertainty_calibration={'valid_for_domain':True},
                joint_states=[dict(status='reacquiring',last_observed_timestamp_ns=t,error_radius_m=.01) for _ in names])
            self.assertNotEqual(gate.update(row,t)['mode'],'follow')


if __name__=='__main__':unittest.main()
