import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from motion_control.adapters import SimulatedRobot, ros2_trajectory
from motion_control.contracts import JointFeedback
from motion_control.controller import MotionController, StopRequired
from motion_control.model import RobotModel
from motion_control.pose import PoseProcessor, OneEuro
from motion_control.retarget import ArmRetargeter, JointTarget
from motion_control.replay import run, write_rows

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'config/robot_demo.json'


def body_row(model, ts=0, q=None):
    arms = model.forward(model.home if q is None else q)
    names, points = [], []
    for part in ('shoulder', 'elbow', 'wrist'):
        for side in ('left','right'):
            names.append(f'{side}_{part}')
            points.append(arms[side][part].tolist())
    names += ['left_hip','right_hip']
    points += [[0,0.16,0.7],[0,-0.16,0.7]]
    return {'frame_id': ts//20_000_000, 'timestamp_ns': ts, 'joint_names': names,
            'joints_world_m': points, 'quality': [dict(mean_reprojection_px=1,ray_angle_deg=45,linearized_max_std_m=0.005,
                                                     inlier_cameras=['a','b']) for _ in points]}


def feedback(model, ts, q=None, clock='test'):
    return JointFeedback(ts,clock,model.names,tuple(model.home if q is None else q),tuple(np.zeros(model.size)))


class PoseAndIKTests(unittest.TestCase):
    def setUp(self):
        self.model = RobotModel.load(CONFIG)

    def test_body_axes_rigid_transform_invariance_and_handedness(self):
        row = body_row(self.model)
        a = PoseProcessor(self.model.config['perception']).process(row)
        rot = Rotation.from_euler('xyz',[0.3,0.4,-1.1]).as_matrix()
        row['joints_world_m'] = (np.asarray(row['joints_world_m'])@rot.T+[2,-1,0.4]).tolist()
        b = PoseProcessor(self.model.config['perception']).process(row)
        self.assertIsNotNone(b)
        np.testing.assert_allclose(b.rotation_body_to_world.T@b.rotation_body_to_world,np.eye(3),atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(b.rotation_body_to_world),1)
        for side in ('left','right'):
            np.testing.assert_allclose(a.arm_directions_body[side],b.arm_directions_body[side],atol=1e-12)

    def test_ik_recovers_observable_directions_not_unobservable_twist(self):
        q = self.model.home.copy()+[0.25,-0.3,0.2,0.4,-0.3,0.2,-0.2,0.3]
        obs = PoseProcessor(self.model.config['perception']).process(body_row(self.model,q=q))
        result = ArmRetargeter(self.model).solve(obs)
        self.assertTrue(result.accepted)
        self.assertLess(result.max_direction_error_deg,3)
        self.assertTrue(np.all(result.positions_rad >= self.model.lower))

    def test_missing_nan_quality_degenerate_and_outlier_are_withdrawn(self):
        for kind in ('missing','nan','quality','angle','cameras','degenerate','bone','uncertainty'):
            with self.subTest(kind=kind):
                p = PoseProcessor(self.model.config['perception'])
                self.assertIsNotNone(p.process(body_row(self.model)))
                row = body_row(self.model,20_000_000)
                if kind == 'missing': row['joints_world_m'][4] = None
                if kind == 'nan': row['joints_world_m'][4][0] = float('nan')
                if kind == 'quality': row['quality'][4]['mean_reprojection_px'] = 50
                if kind == 'angle': row['quality'][4]['ray_angle_deg'] = 0.1
                if kind == 'cameras': row['quality'][4]['inlier_cameras'] = ['a','a']
                if kind == 'degenerate': row['joints_world_m'][0] = row['joints_world_m'][1]
                if kind == 'bone': row['joints_world_m'][4][0] += 1
                if kind == 'uncertainty': row['quality'][4]['linearized_max_std_m'] = 0.2
                self.assertIsNone(p.process(row))
                self.assertEqual(p.last_valid,0)

    def test_timestamp_precision_order_and_gap_reset(self):
        p = PoseProcessor(self.model.config['perception'])
        epoch = 1_700_000_000_000_000_001
        self.assertIsNotNone(p.process(body_row(self.model,epoch)))
        self.assertIsNone(p.process(body_row(self.model,epoch)))
        self.assertIsNone(p.process(body_row(self.model,epoch-1)))
        self.assertIsNotNone(p.process(body_row(self.model,epoch+1_000_000_000)))

    def test_filter_reduces_stationary_noise_and_is_causal(self):
        cfg = self.model.config['perception']
        filt = OneEuro(cfg)
        rng = np.random.default_rng(9)
        raw = rng.normal(0,0.008,(300,8,3))
        smooth = np.array([filt.update(x,1/60) for x in raw])
        self.assertLess(smooth[50:].std(),raw[50:].std()*0.5)
        # Running only a prefix produces exactly the same prefix.
        other = OneEuro(cfg)
        np.testing.assert_array_equal(smooth[:20],np.array([other.update(x,1/60) for x in raw[:20]]))

    def test_invalid_robot_model_rejected(self):
        for field, value in [('max_velocity_rad_s',0),('home_rad',99),('upper_rad',float('nan'))]:
            cfg = copy.deepcopy(self.model.config)
            cfg['joints'][0][field] = value
            with self.assertRaises(ValueError): RobotModel(cfg)
        cfg = copy.deepcopy(self.model.config)
        cfg['chains'][0]['axes'][0] = [0,0,2]
        with self.assertRaises(ValueError): RobotModel(cfg)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.model = RobotModel.load(CONFIG)

    def target(self, ts, q):
        return JointTarget(ts,np.asarray(q),0,True)

    def test_adversarial_targets_respect_bounds_velocity_acceleration(self):
        c = MotionController(self.model,'test')
        robot = SimulatedRobot(self.model,'test')
        previous = self.model.home.copy()
        previous_v = np.zeros(self.model.size)
        for i in range(1200):
            now = i*c.period_ns
            q = self.model.upper if (i//150)%2 == 0 else self.model.lower
            c.observe(self.target(now,q))
            command = c.tick(now,robot.read_feedback(now))
            robot.send(command,now)
            values, velocity = np.array(command.positions_rad),np.array(command.velocities_rad_s)
            self.assertTrue(np.all(values >= self.model.lower-1e-9))
            self.assertTrue(np.all(values <= self.model.upper+1e-9))
            self.assertTrue(np.all(np.abs(velocity) <= self.model.velocity+1e-9))
            self.assertTrue(np.all(np.abs(velocity-previous_v)/0.02 <= self.model.acceleration+1e-9))
            np.testing.assert_allclose((values-previous)/0.02,velocity,atol=1e-10)
            previous, previous_v = values, velocity

    def test_pose_loss_brakes_then_holds_and_recovers_with_new_frames(self):
        c, robot = MotionController(self.model,'test'), SimulatedRobot(self.model,'test')
        states = []
        for i in range(60):
            now = i*c.period_ns
            if i < 10: c.observe(self.target(now,self.model.home+0.4))
            if i == 10: c.observe(None)
            cmd = c.tick(now,robot.read_feedback(now))
            robot.send(cmd,now)
            states.append(cmd.state)
        self.assertIn('TRACKING',states[:10])
        self.assertIn('BRAKING',states[10:])
        self.assertEqual(states[-1],'HOLD')
        self.assertIsNone(cmd.source_timestamp_ns)
        c.observe(self.target(1_200_000_000,self.model.home))
        self.assertEqual(c.consecutive,1)

    def test_stale_and_future_pose_never_track(self):
        for offset in (-1_000_000_000,1_000_000_000):
            c = MotionController(self.model,'test')
            for i in range(2):
                now = 2_000_000_000+i*c.period_ns
                c.observe(self.target(now+offset,self.model.home+0.2))
                self.assertEqual(c.tick(now,feedback(self.model,now)).state,'HOLD')

    def test_feedback_faults_and_estop_latch(self):
        for fault in ('stale','future','clock','names','nan','tracking','moving','disabled','estop','deadline'):
            with self.subTest(fault=fault):
                c = MotionController(self.model,'test')
                c.tick(1_000_000_000,feedback(self.model,1_000_000_000))
                now = 1_020_000_000
                f = feedback(self.model,now)
                if fault == 'stale': f = replace(f,timestamp_ns=0)
                if fault == 'future': f = replace(f,timestamp_ns=now+1)
                if fault == 'clock': f = replace(f,clock_domain='device_clock')
                if fault == 'names': f = replace(f,joint_names=tuple(reversed(f.joint_names)))
                if fault == 'nan': f = replace(f,positions_rad=tuple([float('nan')]*8))
                if fault == 'tracking': f = replace(f,positions_rad=tuple(self.model.home+0.5))
                if fault == 'moving': f = replace(f,velocities_rad_s=tuple([99.]*8))
                if fault == 'disabled': f = replace(f,enabled=False)
                if fault == 'deadline': now += 30_000_000
                with self.assertRaises(StopRequired): c.tick(now,f,estop=fault=='estop')
                with self.assertRaises(StopRequired): c.tick(1_040_000_000,feedback(self.model,1_040_000_000))

    def test_adapter_rejects_replay_expiry_model_and_nonfinite(self):
        for fault in ('replay','expiry','model','nan','order','velocity','clock'):
            with self.subTest(fault=fault):
                robot = SimulatedRobot(self.model,'test')
                cmd = MotionController(self.model,'test').tick(0,feedback(self.model,0))
                if fault == 'replay': robot.send(cmd,0)
                if fault == 'model': cmd = replace(cmd,model_sha256='other')
                if fault == 'nan': cmd = replace(cmd,positions_rad=tuple([float('nan')]*8))
                if fault == 'order': cmd = replace(cmd,joint_names=tuple(reversed(cmd.joint_names)))
                if fault == 'velocity': cmd = replace(cmd,velocities_rad_s=tuple([99.]*8))
                if fault == 'clock': cmd = replace(cmd,clock_domain='other')
                with self.assertRaises(ValueError): robot.send(cmd,cmd.valid_until_ns if fault=='expiry' else 0)
                self.assertFalse(robot.enabled)

    def test_independent_device_watchdog_stops_without_host_ticks(self):
        robot = SimulatedRobot(self.model,'test')
        cmd = MotionController(self.model,'test').tick(0,robot.read_feedback(0))
        robot.send(cmd,0)
        self.assertFalse(robot.read_feedback(cmd.valid_until_ns).enabled)
        self.assertIn('watchdog',robot.stop_reason)

    def test_adapter_independently_rejects_discontinuous_position_commands(self):
        robot = SimulatedRobot(self.model,'test')
        c = MotionController(self.model,'test')
        first = c.tick(0,robot.read_feedback(0))
        robot.send(first,0)
        second = c.tick(c.period_ns,robot.read_feedback(c.period_ns))
        bad = replace(second,positions_rad=tuple(self.model.home+0.2))
        with self.assertRaisesRegex(ValueError,'discontinuity'): robot.send(bad,c.period_ns)

    def test_bounded_clock_jitter_uses_actual_timestep(self):
        c, robot = MotionController(self.model,'test'), SimulatedRobot(self.model,'test')
        now, previous_time = 0, None
        previous_velocity = np.zeros(self.model.size)
        for i in range(100):
            c.observe(self.target(now,self.model.home+0.3))
            cmd = c.tick(now,robot.read_feedback(now))
            robot.send(cmd,now)
            if previous_time is not None:
                acceleration = np.abs(np.array(cmd.velocities_rad_s)-previous_velocity)/((now-previous_time)/1e9)
                self.assertTrue(np.all(acceleration <= self.model.acceleration+1e-8))
            previous_time, previous_velocity = now,np.array(cmd.velocities_rad_s)
            now += [19_000_003,22_000_001,18_000_007][i%3]

    def test_malformed_target_withdraws_previous_motion(self):
        c = MotionController(self.model,'test')
        c.observe(self.target(0,self.model.home))
        c.observe(self.target(20_000_000,[float('nan')]*self.model.size))
        self.assertIsNone(c.target)
        self.assertEqual(c.consecutive,0)

    def test_command_export_preserves_named_order_and_integer_time(self):
        c = MotionController(self.model,'test')
        start = 1_700_000_000_000_000_001
        a = c.tick(start,feedback(self.model,start))
        b = c.tick(start+c.period_ns,feedback(self.model,start+c.period_ns))
        payload = ros2_trajectory([a,b])
        self.assertEqual(payload['joint_names'],list(self.model.names))
        self.assertEqual(payload['points'][1]['time_from_start'],{'sec':0,'nanosec':20_000_000})
        with self.assertRaises(ValueError): ros2_trajectory([b,a])


class ReplayTests(unittest.TestCase):
    def test_complete_replay_without_ground_truth_and_no_overwrite(self):
        model = RobotModel.load(CONFIG)
        with tempfile.TemporaryDirectory() as temp:
            session, out = Path(temp), Path(temp)/'control'
            rows = [body_row(model,i*100_000_000,model.home+np.array([0,-0.3,0.2,0.3,0,-0.3,-0.2,0.3])) for i in range(10)]
            write_rows(session/'poses3d.jsonl',rows)
            report = run(session,out,CONFIG,120)
            self.assertTrue(report['passed'])
            self.assertEqual(report['accepted_observations'],10)
            self.assertGreater(report['states']['TRACKING'],0)
            self.assertFalse((session/'ground_truth.jsonl').exists())
            self.assertTrue(json.loads((out/'ros2_trajectory.json').read_text())['simulation_only'])
            with self.assertRaises(FileExistsError): run(session,out,CONFIG)

    def test_all_stale_observations_cannot_pass_demo(self):
        model = RobotModel.load(CONFIG)
        with tempfile.TemporaryDirectory() as temp:
            session = Path(temp)
            write_rows(session/'poses3d.jsonl',[body_row(model,i*100_000_000) for i in range(5)])
            with self.assertRaises(RuntimeError): run(session,session/'control',CONFIG,500)
            report = json.loads((session/'control/control_report.json').read_text())
            self.assertFalse(report['checks']['motion_generated'])


if __name__ == '__main__':
    unittest.main()
