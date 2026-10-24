import unittest
import numpy as np

from motion_control.continuous_pose_reference import (
    build_c2_reference_stream,
    load_recovery_config,
    retarget_continuous_pose,
)
from motion_control.twist import TwistPlant
from pose_sim.continuous_state import BODY, ContinuousHumanState, load_config
from tests.test_human_state import observation


def blank_joints(row, names=BODY):
    for name in names:
        i = row['joint_names'].index(name)
        row['joints_world_m'][i] = row['quality'][i] = None
    return row


class ContinuousPoseReferenceTests(unittest.TestCase):
    def test_turn_and_crouch_clamping_without_lockout_or_standing_reset(self):
        cfg = load_config()
        estimator = ContinuousHumanState(window=True)
        states = []
        for f in range(30):
            obs = observation(f, shift=0.015 * f)
            # Inject a deep crouch on frames 10..15 by lowering hips/shoulders toward ankles
            if 10 <= f <= 15:
                for name in ('left_hip', 'right_hip', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'):
                    idx = obs['joint_names'].index(name)
                    obs['joints_world_m'][idx][2] -= 0.36
            states.append(estimator.update(obs))
        refs = retarget_continuous_pose(states, cfg, max_joint_vel_rad_s=6.0, min_height_m=0.48)
        self.assertTrue(all(r['accepted'] for r in refs))
        q = np.array([r['reference'][6:] for r in refs], float)
        max_step_rad = float(np.abs(np.diff(q, axis=0)).max())
        self.assertLessEqual(max_step_rad, 6.0 * 0.04 + 1e-5)
        heights = [r['reference'][2] for r in refs]
        self.assertTrue(all(0.48 - 1e-6 <= h <= 0.95 + 1e-6 for h in heights))

    def test_short_gap_hold_and_c2_stream_bounds(self):
        cfg = load_config()
        estimator = ContinuousHumanState(window=True)
        states = []
        for f in range(35):
            obs = observation(f, shift=0.01 * f)
            if 12 <= f < 20:
                blank_joints(obs)
            states.append(estimator.update(obs))
        # Force a 2-frame permission dip at frames 18..19 to verify short-gap pose hold
        for f in (18, 19):
            for s in states[f]['joint_states']:
                s['control_permission'] = False
        refs = retarget_continuous_pose(states, cfg, max_hold_s=0.40)
        self.assertTrue(all(r['accepted'] for r in refs))
        self.assertTrue(refs[18]['short_gap_held'])
        self.assertTrue(refs[19]['short_gap_held'])

        plant = TwistPlant()
        traj_cfg = load_recovery_config()
        stream = build_c2_reference_stream(refs, plant, traj_cfg, dt=0.02)
        q_50 = stream['bounded'][:, 7:]
        dq_disc = np.diff(q_50, axis=0) / 0.02
        ddq_disc = np.diff(dq_disc, axis=0) / 0.02
        jerk_disc = np.diff(ddq_disc, axis=0) / 0.02
        self.assertLessEqual(float(np.abs(dq_disc).max()), 6.0 + 1e-5)
        self.assertLessEqual(float(np.abs(ddq_disc).max()), 120.0 + 1e-4)
        self.assertLessEqual(float(np.abs(jerk_disc).max()), 3000.0 + 1e-2)

        stream_bal = build_c2_reference_stream(refs, plant, traj_cfg, dt=0.02, joint_mode='per_dof_balanced')
        q_bal = stream_bal['bounded'][:, 7:]
        dq_bal = np.diff(q_bal, axis=0) / 0.02
        ddq_bal = np.diff(dq_bal, axis=0) / 0.02
        jerk_bal = np.diff(ddq_bal, axis=0) / 0.02
        self.assertLessEqual(float(np.abs(dq_bal).max()), 6.0 + 1e-5)
        self.assertLessEqual(float(np.abs(ddq_bal).max()), 90.0 + 1e-4)
        self.assertLessEqual(float(np.abs(jerk_bal).max()), 1500.0 + 1e-2)
        # Held source frames must be excluded from active tracking score mask
        held_ticks = np.isin(stream['source_ids'], [18, 19])
        self.assertTrue(held_ticks.any())
        self.assertFalse(bool(stream['active'][held_ticks].any()))

    def test_gimbal_barrier_and_per_dof_rate_stream_bounds(self):
        cfg = load_config()
        estimator = ContinuousHumanState(window=True)
        states = [estimator.update(observation(f, shift=0.012 * f)) for f in range(30)]
        refs = retarget_continuous_pose(states, cfg, gimbal_barrier=True)
        self.assertTrue(all(r['accepted'] for r in refs))
        plant = TwistPlant()
        traj_cfg = load_recovery_config()
        stream = build_c2_reference_stream(refs, plant, traj_cfg, dt=0.02, joint_mode='per_dof_rate')
        q_50 = stream['bounded'][:, 7:]
        dq_disc = np.diff(q_50, axis=0) / 0.02
        ddq_disc = np.diff(dq_disc, axis=0) / 0.02
        jerk_disc = np.diff(ddq_disc, axis=0) / 0.02
        self.assertLessEqual(float(np.abs(dq_disc).max()), 6.0 + 1e-5)
        self.assertLessEqual(float(np.abs(ddq_disc).max()), 120.0 + 1e-4)
        self.assertLessEqual(float(np.abs(jerk_disc).max()), 3000.0 + 1e-2)

        stream_bal = build_c2_reference_stream(refs, plant, traj_cfg, dt=0.02, joint_mode='per_dof_balanced')
        q_bal = stream_bal['bounded'][:, 7:]
        dq_bal = np.diff(q_bal, axis=0) / 0.02
        ddq_bal = np.diff(dq_bal, axis=0) / 0.02
        jerk_bal = np.diff(ddq_bal, axis=0) / 0.02
        self.assertLessEqual(float(np.abs(dq_bal).max()), 6.0 + 1e-5)
        self.assertLessEqual(float(np.abs(ddq_bal).max()), 90.0 + 1e-4)
        self.assertLessEqual(float(np.abs(jerk_bal).max()), 1500.0 + 1e-2)

    def test_nonmonotonic_clock_raises(self):
        cfg = load_config()
        estimator = ContinuousHumanState(window=True)
        s0 = estimator.update(observation(0))
        s1 = estimator.update(observation(1))
        s1_bad = dict(s1, timestamp_ns=s0['timestamp_ns'], state_timestamp_ns=s0['timestamp_ns'])
        with self.assertRaises(ValueError):
            retarget_continuous_pose([s0, s1_bad], cfg)


if __name__ == '__main__':
    unittest.main()
