import copy
import unittest
import numpy as np
from pose_sim.benchmark_data import camera_ring,articulated_motion,validate_splits
from pose_sim.geometry import project
from pose_sim.landmarks import HUMAN33
from pose_sim.multiview_temporal import MultiViewTemporal
from pose_sim.human_motion import HumanMotionEstimator
from pose_sim.body_shape import BodyShape
from pose_sim.motion_evaluation import fit_uncertainty,apply_uncertainty
from motion_control.perception_gate import PerceptionGate


def scene():
    cameras=camera_ring();cfg=dict(max_sync_error_ms=5,confidence_threshold=.35,ransac_threshold_px=8.,min_triangulation_angle_deg=1.)
    xyz,_=articulated_motion(dict(height_m=1.72,leg_ratio=.49),'walk',np.arange(45)/30,.2)
    rows=[]
    for f,points in enumerate(xyz):
        rows.append(dict(frame_id=f,timestamp_ns=f*33_333_333,source_id='a',perception_signature='model',
            joint_names=HUMAN33,landmark_schema_id='human33',views={c:dict(timestamp_ns=f*33_333_333,
                keypoints_px=project(points,cam).tolist(),scores=[.9]*33) for c,cam in cameras.items()}))
    return cameras,cfg,xyz,rows


class MotionFrontendTests(unittest.TestCase):
    def test_cross_view_swap_recovered_without_touching_independent_feet(self):
        cameras,cfg,xyz,rows=scene();fusion=MultiViewTemporal(dict(cameras=cameras),cfg);state=HumanMotionEstimator();last=None
        for r in rows[:15]:
            pose,assigned=fusion.update(r,last);last=state.update(pose,cameras,assigned)
        row=copy.deepcopy(rows[15]);before=copy.deepcopy(row)
        for c in ('cam00','cam01'):
            xy=row['views'][c]['keypoints_px']
            for a,b in ((5,6),(7,8),(9,10),(11,12),(13,14),(15,16)):xy[a],xy[b]=xy[b],xy[a]
        immutable=copy.deepcopy(row);pose,assigned=fusion.update(row,last)
        self.assertEqual(row,immutable);self.assertGreaterEqual(len(pose['association']['corrections']),4)
        np.testing.assert_allclose(assigned['views']['cam00']['keypoints_px'][17:],before['views']['cam00']['keypoints_px'][17:])
        error=np.linalg.norm(np.array(pose['joints_world_m'])-xyz[15],axis=1)
        self.assertLess(error.max(),.025)

    def test_calibration_or_source_change_disables_temporal_label_prior(self):
        cameras,cfg,xyz,rows=scene();f=MultiViewTemporal(dict(cameras=cameras),cfg);s=HumanMotionEstimator()
        p,d=f.update(rows[0]);out=s.update(p,cameras,d)
        changed=copy.deepcopy(rows[1]);changed['source_id']='new'
        self.assertEqual(f.prior(out,changed),{})
        out['fusion_calibration_sha256']='different';self.assertEqual(f.prior(out,rows[1]),{})

    def test_global_label_symmetry_is_flagged_instead_of_certified(self):
        cameras,cfg,xyz,rows=scene();f=MultiViewTemporal(dict(cameras=cameras),cfg);s=HumanMotionEstimator();last=None
        for row in rows[:10]:
            p,d=f.update(row,last);last=s.update(p,cameras,d)
        row=copy.deepcopy(rows[10])
        for v in row['views'].values():
            for a,b in ((5,6),(7,8),(9,10),(11,12),(13,14),(15,16)):
                v['keypoints_px'][a],v['keypoints_px'][b]=v['keypoints_px'][b],v['keypoints_px'][a]
        p,d=f.update(row,last)
        self.assertTrue(p['association']['ambiguities']);self.assertIsNone(p['joints_world_m'][13])

    def test_v2_prediction_cannot_bypass_robot_observation_gate(self):
        from motion_control.humanoid import read_body
        cameras,cfg,xyz,rows=scene();f=MultiViewTemporal(dict(cameras=cameras),cfg);s=HumanMotionEstimator()
        p,d=f.update(rows[0]);row=s.update(p,cameras,d)
        row['joint_states'][13]['status']='predicted'
        with self.assertRaisesRegex(ValueError,'Missing left_knee'):read_body(row)
        row['state_schema_id']='unknown'
        with self.assertRaisesRegex(ValueError,'schema'):read_body(row)

    def test_shape_revises_early_bias_and_never_claims_bias_identifiable(self):
        shape=BodyShape(window=45,minimum_samples=12)
        q=dict(inlier_cameras=['a','b','c'],ray_angle_deg=40,linearized_max_std_m=.005)
        for f in range(100):
            length=.5 if f<20 else .4;angle=.02*f
            row=dict(joint_names=['left_hip','left_knee'],timestamp_ns=f*40_000_000,
                joints_world_m=[[0,0,1],[length*np.sin(angle),0,1-length*np.cos(angle)]],quality=[q,q])
            result=shape.update(row,set(row['joint_names']))
        estimate=result['left_hip--left_knee']
        self.assertEqual(estimate['status'],'stable');self.assertAlmostEqual(estimate['length_m'],.4)
        self.assertFalse(estimate['systematic_bias_identifiable'])

    def test_uncertainty_never_transfers_synthetic_calibration_to_rgb(self):
        cal=fit_uncertainty(list(range(1,25)),'synthetic','hash')
        row={'joint_states':[dict(max_std_m=.01)]}
        apply_uncertainty(row,cal,'rgb','hash');self.assertIsNone(row['joint_states'][0]['error_radius_m'])
        apply_uncertainty(row,cal,'synthetic','changed');self.assertIsNone(row['joint_states'][0]['error_radius_m'])
        apply_uncertainty(row,cal,'synthetic','hash');self.assertGreater(row['joint_states'][0]['error_radius_m'],.01)
        with self.assertRaises(ValueError):fit_uncertainty([1,2,3],'x','y',.99)

    def test_state_shape_learning_is_not_locked_by_its_own_outlier_gate(self):
        e=HumanMotionEstimator();q=dict(inlier_cameras=['a','b','c'],ray_angle_deg=40,
            linearized_max_std_m=.005,mean_reprojection_px=1.)
        for f in range(150):
            length=.55 if f<45 else .32;a=.025*f
            row=dict(frame_id=f,timestamp_ns=f*40_000_000,joint_names=['left_hip','left_knee'],
                landmark_schema_id='coco17',source_id='s',perception_signature='m',calibration_sha256='c',
                joints_world_m=[[0.,0.,1.],[length*np.sin(a),0.,1.-length*np.cos(a)]],quality=[q,q])
            out=e.update(row)
        self.assertAlmostEqual(out['shape']['left_hip--left_knee']['length_m'],.32,places=6)

    def test_gate_hysteresis_stale_prediction_and_watchdog(self):
        gate=PerceptionGate();names=[s+'_'+j for s in ('left','right') for j in ('hip','knee','ankle')]
        for f in range(1,4):
            t=f*40_000_000;row=dict(timestamp_ns=t,joint_names=names,uncertainty_calibration={'valid_for_domain':True},
                joint_states=[dict(status='filtered',last_observed_timestamp_ns=t,error_radius_m=.03) for _ in names])
            result=gate.update(row,t+10_000_000)
        self.assertEqual(result['mode'],'follow')
        row['timestamp_ns']+=40_000_000;row['joint_states'][0]['status']='predicted'
        self.assertEqual(gate.update(row,row['timestamp_ns'])['mode'],'transition')
        self.assertEqual(gate.update(None,1_000_000_000)['mode'],'stop')

    def test_gate_recovery_requires_same_source_and_bounded_time_gaps(self):
        gate=PerceptionGate();names=[s+'_'+j for s in ('left','right') for j in ('hip','knee','ankle')]
        for f in range(1,6):
            t=f*400_000_000;row=dict(timestamp_ns=t,joint_names=names,source_id='a',uncertainty_calibration={'valid_for_domain':True},
                joint_states=[dict(status='filtered',last_observed_timestamp_ns=t,error_radius_m=.03) for _ in names])
            self.assertEqual(gate.update(row,t)['mode'],'transition')
        for f in range(1,3):
            t+=20_000_000;row['timestamp_ns']=t
            for sample in row['joint_states']:sample['last_observed_timestamp_ns']=t
            result=gate.update(row,t)
        self.assertEqual(result['mode'],'follow')
        t+=20_000_000;row['timestamp_ns']=t;row['source_id']='new'
        for sample in row['joint_states']:sample['last_observed_timestamp_ns']=t
        self.assertEqual(gate.update(row,t)['mode'],'transition')

    def test_subject_leak_is_rejected(self):
        with self.assertRaises(ValueError):validate_splits({'splits':{s:[{'subject':'same'}] for s in ('tune','calibration','test')}})

    def test_motion_schema_keeps_unobservable_dof_unknown(self):
        cameras,cfg,xyz,rows=scene();f=MultiViewTemporal(dict(cameras=cameras),cfg);state=HumanMotionEstimator()
        p,d=f.update(rows[0]);out=state.update(p,cameras,d)
        self.assertEqual(out['state_schema_id'],'human_state_v2')
        self.assertIsNone(out['joint_rotations']['left_knee']['axial_twist_rad'])
        self.assertIsNone(out['root']['anatomical_pelvis_rotation'])
        self.assertEqual(out['support']['phase'],'unknown')


if __name__=='__main__':unittest.main()
