import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from pose_sim.geometry import rigid_transform,project
from pose_sim.robot_source import RobotSource,fit_session
from pose_sim.pipeline import reconstruct,read_jsonl,write_jsonl
from motion_control.transfer import TaskMapper,LocomotionIntent,run
from motion_control.unitree import G1Description,SDK_JOINT_NAMES,locomotion_request

ROOT = Path(__file__).resolve().parents[1]


def source_config():
    # Algebraic fixture ONLY: no images, detector, real CAD or robot accuracy implied.
    cfg = json.loads((ROOT/'config/source_robot.json').read_text())
    cfg.update(source_id='source_robot_demo',landmark_schema_id='test/natural_landmarks_v1',links=[])
    for name in ('source_base','source_tool'):
        points = [[0,0,0],[0.2,0,0],[0.03,0.1,0],[0.01,0.02,0.08]]
        cfg['links'].append({'name':name,'orientation_identifiable':True,'landmarks':[
            {'name':f'feature{i}','feature_type':'natural_geometry','position_link_m':p} for i,p in enumerate(points)]})
    return cfg


def transfer_config():
    return json.loads((ROOT/'config/transfer_source_to_g1.json').read_text())


def link_observation(ts,x=0,valid=True):
    t = np.eye(4); t[0,3] = x
    return {'timestamp_ns':ts,'source_id':'source_robot_demo','units':'SI',
            'perception_mode':'markerless','detector_model_sha256':'fixture_detector',
            'source_model_sha256':'fixture_source','world_frame':'fixed_calibration_world',
            'calibration_sha256':'fixture_cameras',
            'links':{'source_base':{'valid':valid,'orientation_observed':True,'T_world_link':t.tolist()},
                     'source_tool':{'valid':valid,'orientation_observed':True,'T_world_link':t.tolist()}}}


class RobotSourceTests(unittest.TestCase):
    def pose(self, source, t=None):
        if t is None:
            t = np.eye(4)
            t[:3,:3] = Rotation.from_euler('xyz',[-0.6,0.4,0.8]).as_matrix()
            t[:3,3] = [1,2,3]
        points = np.vstack([source.local_points[link['name']]@t[:3,:3].T+t[:3,3] for link in source.links])
        return {'source_id':source.config['source_id'],'joint_names':source.names,'frame_id':0,'timestamp_ns':0,
                'landmark_schema_id':source.config['landmark_schema_id'],'source_model_sha256':source.sha256,
                'calibration_sha256':'test_calibration','perception_mode':'markerless',
                'joints_world_m':points.tolist(),'quality':[{'linearized_max_std_m':0.001}]*len(points)}

    def test_natural_landmarks_recover_link_pose_and_partial_occlusion(self):
        cfg = source_config()
        source = RobotSource(cfg)
        t = np.eye(4); t[:3,:3] = Rotation.from_euler('xyz',[-0.6,0.4,0.8]).as_matrix(); t[:3,3] = [1,2,3]
        pose = self.pose(source,t)
        result = source.fit_links(pose)
        np.testing.assert_allclose(result['links']['source_base']['T_world_link'],t,atol=1e-10)
        pose['joints_world_m'][0] = None
        self.assertTrue(source.fit_links(pose)['links']['source_base']['valid'])
        pose['joints_world_m'][1] = None
        self.assertFalse(source.fit_links(pose)['links']['source_base']['valid'])

    def test_projected_observations_to_offline_g1_contract_without_images_or_truth(self):
        # A downstream contract regression; projected observations are NOT image inference.
        with tempfile.TemporaryDirectory() as td:
            session = Path(td)
            source = RobotSource(source_config())
            cameras = {cid:{'K':[[750,0,320],[0,750,320],[0,0,1]],'R_world_to_camera':np.eye(3).tolist(),
                't_world_to_camera_m':[x,0,0],'width':640,'height':640,'distortion':[0]*5}
                for cid,x in [('left',-0.35),('right',0.35),('middle',0)]}
            (session/'cameras.json').write_text(json.dumps({'cameras':cameras}))
            (session/'config.json').write_bytes((ROOT/'config/demo.json').read_bytes())
            records = []
            for i in range(3):
                t = np.eye(4); t[:3,3] = [i*0.015,0,1.8]
                pose = self.pose(source,t)
                records.append({**{k:v for k,v in pose.items() if k not in ('quality','joints_world_m')},
                    'frame_id':i,'timestamp_ns':i*100_000_000,'views':{cid:{'timestamp_ns':i*100_000_000,
                    'keypoints_px':project(pose['joints_world_m'],cam).tolist(),'scores':[1]*len(source.names)} for cid,cam in cameras.items()}})
            write_jsonl(session/'detections.jsonl',records)
            reconstruct(session)
            rows = list(read_jsonl(session/'poses3d.jsonl'))
            self.assertEqual(len(rows[0]['joints_world_m']),8)
            self.assertNotIn('left_shoulder',rows[0]['joint_names'])
            self.assertEqual(rows[0]['source_id'],'source_robot_demo')
            for row in rows:
                self.assertTrue(all(x['valid'] for x in source.fit_links(row)['links'].values()))
            fit_session(session,source)
            report = run(session,ROOT/'config/transfer_source_to_g1.json',session/'transfer')
            self.assertGreater(report['nonzero_velocity_requests'],0)
            self.assertTrue(report['end_zero_velocity'])
            self.assertEqual(report['network_calls'],0)
            self.assertFalse((session/'ground_truth.jsonl').exists())

    def test_source_identity_and_reflection_rejected(self):
        t = np.eye(4); t[0,0] = -1
        with self.assertRaises(ValueError): rigid_transform(t)
        source = RobotSource(source_config())
        with self.assertRaises(ValueError): source.fit_links({'source_id':'wrong','joint_names':source.names})

    def test_incomplete_or_artificial_marker_profile_rejected(self):
        with self.assertRaisesRegex(ValueError,'incomplete'):
            RobotSource(json.loads((ROOT/'config/source_robot.json').read_text()))
        cfg = source_config(); cfg['links'][0]['landmarks'][0]['feature_type'] = 'aruco_corner'
        with self.assertRaises(ValueError): RobotSource(cfg)
        cfg = source_config(); cfg['links'][0]['landmarks'][1]['name'] = 'feature0'
        with self.assertRaises(ValueError): RobotSource(cfg)

    def test_symmetry_and_collinearity_cannot_become_observed_orientation(self):
        cfg = source_config(); cfg['links'][0]['orientation_identifiable'] = False
        source = RobotSource(cfg)
        row = source.fit_links(self.pose(source))
        self.assertFalse(row['links']['source_base']['orientation_observed'])
        cfg = source_config()
        for i,p in enumerate(cfg['links'][0]['landmarks']): p['position_link_m'] = [i*0.1,0,0]
        source = RobotSource(cfg)
        self.assertFalse(source.fit_links(self.pose(source))['links']['source_base']['valid'])

    def test_correspondence_outlier_and_uncertain_points_rejected(self):
        source = RobotSource(source_config()); pose = self.pose(source)
        pose['joints_world_m'][0][0] += 0.5
        self.assertFalse(source.fit_links(pose)['links']['source_base']['valid'])
        pose = self.pose(source)
        pose['quality'][0] = {'linearized_max_std_m':0.5}
        pose['quality'][1] = {'linearized_max_std_m':float('nan')}
        self.assertFalse(source.fit_links(pose)['links']['source_base']['valid'])

    def test_tiny_noncollinear_features_still_have_unreliable_orientation(self):
        cfg = source_config()
        for point in cfg['links'][0]['landmarks']:
            point['position_link_m'] = (np.array(point['position_link_m'])*0.01).tolist()
        source = RobotSource(cfg)
        result = source.fit_links(self.pose(source))['links']['source_base']
        self.assertFalse(result['valid'])
        self.assertGreater(result['linearized_max_orientation_std_rad'],cfg['max_orientation_std_rad'])

    def test_model_schema_and_calibration_binding_required(self):
        source = RobotSource(source_config())
        for key in ('source_model_sha256','landmark_schema_id','perception_mode','calibration_sha256'):
            with self.subTest(key=key):
                pose = self.pose(source); pose[key] = None
                with self.assertRaises(ValueError): source.fit_links(pose)


class TargetMappingTests(unittest.TestCase):
    def test_legacy_marker_source_and_detector_swap_rejected(self):
        for factory in (TaskMapper,LocomotionIntent):
            module = factory(transfer_config())
            operation = module.map if isinstance(module,TaskMapper) else module.observe
            legacy = link_observation(0); legacy.pop('perception_mode')
            with self.assertRaises(ValueError): operation(legacy)
            operation(link_observation(0))
            changed = link_observation(100_000_000); changed['detector_model_sha256'] = 'swapped_model'
            with self.assertRaises(ValueError): operation(changed)
    def test_source_motion_maps_to_target_tasks_without_joint_name_assumptions(self):
        mapper = TaskMapper(transfer_config())
        first = mapper.map(link_observation(0))
        second = mapper.map(link_observation(100_000_000,0.2))
        self.assertFalse(second['motor_command'])
        self.assertEqual(second['contact_state'],'unknown')
        self.assertAlmostEqual(second['tasks']['right_hand']['T_reference_task'][0][3]-first['tasks']['right_hand']['T_reference_task'][0][3],0.1)
        missing = mapper.map(link_observation(200_000_000,valid=False))
        self.assertFalse(missing['tasks']['right_hand']['valid'])

    def test_native_locomotion_expiry_missing_and_jump_request_stop(self):
        planner = LocomotionIntent(transfer_config())
        planner.observe(link_observation(0)); planner.tick(0)
        planner.observe(link_observation(100_000_000,0.015))
        cmd = planner.tick(100_000_000)
        self.assertGreater(cmd['request']['parameters']['velocity'][0],0)
        self.assertLessEqual(cmd['request']['parameters']['velocity'][0],0.05+1e-10)
        self.assertEqual(planner.tick(400_000_000)['request']['parameters']['velocity'],[0,0,0])
        planner.observe(link_observation(500_000_000)); planner.tick(500_000_000)
        planner.observe(link_observation(600_000_000,100))
        self.assertEqual(planner.tick(600_000_000)['request']['parameters']['velocity'],[0,0,0])

    def test_reordered_source_and_competing_target_maps_rejected(self):
        cfg = transfer_config()
        cfg['reference_mappings'].append(copy.deepcopy(cfg['reference_mappings'][0]))
        with self.assertRaises(ValueError): TaskMapper(cfg)
        mapper = TaskMapper(transfer_config())
        mapper.map(link_observation(10))
        with self.assertRaises(ValueError): mapper.map(link_observation(9))
        changed = link_observation(11)
        changed['world_frame'] = 'different_world'
        with self.assertRaises(ValueError): mapper.map(changed)

    def test_unobserved_orientation_cannot_drive_full_pose_or_walking(self):
        row = link_observation(0)
        row['links']['source_base']['orientation_observed'] = False
        mapper = TaskMapper(transfer_config())
        self.assertFalse(mapper.map(row)['tasks']['pelvis']['valid'])
        planner = LocomotionIntent(transfer_config())
        planner.observe(row)
        self.assertEqual(planner.tick(0)['request']['parameters']['velocity'],[0,0,0])

    def test_g1_official_variants_preserve_sdk_holes(self):
        a,b = G1Description('29dof'),G1Description('23dof')
        self.assertEqual(a.size,29); self.assertEqual(b.size,23)
        self.assertEqual(a.names,SDK_JOINT_NAMES)
        self.assertNotIn(13,b.sdk_indices)
        self.assertNotIn(20,b.sdk_indices)
        self.assertEqual(b.sdk_indices[b.names.index('right_shoulder_pitch_joint')],22)
        self.assertTrue(np.all(a.lower < a.upper))
        fk = a.forward(np.zeros(29))
        self.assertGreater(fk['left_shoulder_pitch_link'][1,3],0)
        self.assertLess(fk['right_shoulder_pitch_link'][1,3],0)
        for t in fk.values(): rigid_transform(t)

    def test_offline_sdk_request_cannot_enable_low_level_or_infinite_motion(self):
        result = locomotion_request([0.1,0,0],0.15)
        self.assertTrue(result['dry_run']); self.assertEqual(result['api_id'],7105)
        with self.assertRaises(ValueError): locomotion_request([0.1,0,0],864000)
        with self.assertRaises(ValueError): locomotion_request([0.1,0,0],0.15,'whole_body')
        with self.assertRaises(ValueError): locomotion_request([float('nan'),0,0],0.15)


if __name__ == '__main__': unittest.main()
