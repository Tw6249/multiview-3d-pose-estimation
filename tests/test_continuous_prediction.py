import copy
import unittest
import numpy as np
from tests.test_human_state import observation
from pose_sim.continuous_state import ContinuousHumanState, BODY, load_config
from motion_control.continuous_reference import permitted_body
from motion_control.continuous_reference import RecoveryReferenceTrajectory
from motion_control.humanoid_replay import ReferenceGate


def blank(row, names=BODY):
    for name in names:
        i = row['joint_names'].index(name)
        row['joints_world_m'][i] = row['quality'][i] = None
    return row


class ContinuousPredictionTests(unittest.TestCase):
    def test_persistent_prediction_age_and_expiry(self):
        e = ContinuousHumanState()
        for f in range(25):
            e.update(observation(f, shift=.01*f))
        first = None
        for f in range(25, 45):
            out = e.update(blank(observation(f)))
            j = out['joint_names'].index('left_hip')
            s = out['joint_states'][j]
            self.assertEqual(s['last_observed_timestamp_ns'], 24*40_000_000)
            self.assertIsNone(out['quality'][j])
            self.assertEqual(s['prediction_age_ns'], (f-24)*40_000_000)
            if first is None:
                first = out
        self.assertTrue(first['joint_states'][j]['control_permission'])
        self.assertIsNotNone(out['joints_world_m'][j])
        self.assertFalse(s['control_permission'])
        self.assertLess(np.linalg.norm(np.array(out['joints_world_m'][j]) -
                                      np.array(first['joints_world_m'][j])), .6)
        with self.assertRaises(ValueError):
            permitted_body(out)

    def test_window_noise_and_outlier(self):
        e = ContinuousHumanState(); errors = []; raw = []
        for f in range(60):
            row = observation(f, shift=.01*f, noise=.025)
            if f == 40:
                row['joints_world_m'][row['joint_names'].index('left_wrist')][0] += 2
            out = e.update(row)
            if f > 20:
                target = observation(f, shift=.01*f)
                for k, name in enumerate(BODY):
                    truth = np.array(target['joints_world_m'][target['joint_names'].index(name)])
                    errors.append(np.linalg.norm(np.array(out['joints_world_m'][k])-truth))
                    raw.append(np.linalg.norm(np.array(row['joints_world_m'][row['joint_names'].index(name)])-truth))
        self.assertLess(np.mean(errors), np.mean(raw))
        self.assertLess(max(errors), .3)

    def test_recovery_and_future_independence(self):
        a, b = ContinuousHumanState(), ContinuousHumanState()
        for f in range(50):
            row = observation(f, shift=.01*f)
            if 25 <= f < 35:
                blank(row)
            one, two = a.update(copy.deepcopy(row)), b.update(copy.deepcopy(row))
            self.assertEqual(one['joints_world_m'], two['joints_world_m'])
            self.assertFalse(one['uses_future_frames'])
            if f == 35:
                self.assertTrue(any(s['estimate_status'] == 'recovering' for s in one['joint_states']))
        self.assertTrue(all(s['control_permission'] for s in one['joint_states']))
        self.assertTrue(all(s['estimate_status'] == 'updated' for s in one['joint_states']))
        self.assertLess(abs(one['joints_world_m'][6][0]-.49), .04)

    def test_clock_schema_reset_and_no_invented_axes(self):
        e = ContinuousHumanState(); e.update(observation(1))
        with self.assertRaises(ValueError):
            e.update(observation(1))
        row = observation(2); row['source_id'] = 'different_person'
        result = e.update(row)
        self.assertTrue(result['reset'])
        self.assertIsNone(result['contacts'])
        self.assertIsNone(result['orientations']['feet'])
        with self.assertRaises(ValueError):
            permitted_body(observation(3))

    def test_single_frame_ablation_predicts_motion_not_just_hold(self):
        e = ContinuousHumanState(window=False)
        for f in range(35):
            before = e.update(observation(f, shift=.02*f))
        after = e.update(blank(observation(35)))
        self.assertGreater(after['joints_world_m'][6][0], before['joints_world_m'][6][0]+.005)
        self.assertGreater(after['velocities_world_m_s'][6][0], .1)

    def test_dropout_cannot_reduce_measurement_uncertainty(self):
        e = ContinuousHumanState()
        for f in range(20):
            row = observation(f)
            for q in row['quality']:
                if q is not None:
                    q['linearized_max_std_m'] = .12
            before = e.update(row)
        after = e.update(blank(observation(20)))
        self.assertGreater(after['joint_states'][6]['uncertainty_scale_m'],
                           before['joint_states'][6]['uncertainty_scale_m'])

    def test_input_not_modified_or_predictions_fed_back(self):
        e = ContinuousHumanState()
        row = observation(0); saved = copy.deepcopy(row)
        e.update(row)
        self.assertEqual(row, saved)
        for f in range(1, 5):
            e.update(blank(observation(f)))
        for track in e.tracks.values():
            self.assertEqual(sum(r['z'] is not None for r in track['history']), 1)

    def test_transport_age_counts_and_legacy_unchanged(self):
        row = dict(timestamp_ns=0, accepted=True, reference=[0.]*35)
        gate = ReferenceGate(); gate.receive(row, 0)
        self.assertIsNotNone(gate.get(200_000_000))
        row.update(reference_schema_id='continuous_simulation_reference_v1',
                   hardware_enabled=False, prediction_valid_until_ns=100_000_000)
        self.assertIsNone(gate.get(120_000_000))

    def test_recovery_curve_failure_retains_verified_motion(self):
        import json
        from pathlib import Path
        from motion_control.twist import TwistPlant, DEFAULT_REF
        plant = TwistPlant()
        cfg = json.loads((Path(__file__).resolve().parents[1]/'config/g1_continuous_recovery.json').read_text())
        curve = RecoveryReferenceTrajectory(DEFAULT_REF, plant.limits, plant.sha256, cfg)
        target = DEFAULT_REF.copy(); target[21] += .8
        for _ in range(6):
            curve.step(target)
        saved = [(plan, elapsed) if plan is not None else None for plan, elapsed in curve.plans]
        class Failure:
            def calculate(self, *args):
                return -1
        curve.groups = [(ids, Failure(), inp, old) for ids, _, inp, old in curve.groups]
        for _ in range(30):
            _, diag = curve.step(DEFAULT_REF)
            self.assertTrue(diag['retained_plan_groups'])
            self.assertLessEqual(np.max(np.abs(curve.rate)), 6.+1e-7)
        ids = curve.groups[-2][0]
        plan, elapsed = saved[-2]
        np.testing.assert_allclose(curve.position[ids], plan.at_time(elapsed+.6)[0], atol=1e-8)


if __name__ == '__main__':
    unittest.main()
