import copy
import unittest
import numpy as np
from pose_sim.foot_contact import MultiViewFootContact
from pose_sim.ground import fit_ground, validate_ground


def plane():
    return dict(normal_world=[0, 0, 1], offset_m=0., uncertainty_bound_m=.01,
                xy_min_m=[-2, -2], xy_max_m=[2, 2], units='metre', calibration_sha256='cal')


def row(i, height=.015, speed=0.):
    t = i*40_000_000
    names = [s+'_'+p for s in ('left', 'right') for p in ('heel', 'big_toe', 'small_toe')]
    return dict(timestamp_ns=t, source_id='human', landmark_schema_id='human33', calibration_sha256='cal',
        joint_names=names, joints_world_m=[[speed*t/1e9+j*.05, 0, height] for j in range(6)],
        joint_states=[dict(status='filtered', last_observed_timestamp_ns=t, max_std_m=.01) for _ in names],
        quality=[dict(inlier_cameras=['a', 'b', 'c'], ray_angle_deg=30., mean_reprojection_px=1.) for _ in names],
        attachment_state={s+'_foot':dict(status='supported') for s in ('left', 'right')})


class VisualContactTests(unittest.TestCase):
    def confirmed(self, engine=None):
        engine = engine or MultiViewFootContact()
        for i in range(8):
            out = engine.update(row(i), plane())
        return engine, out

    def test_stationary_contact_then_clearance_and_slide(self):
        engine, out = self.confirmed()
        self.assertEqual(out['phase'], 'double')
        self.assertFalse(out['physical_support_verified'])
        self.assertEqual(engine.update(row(8, .3), plane())['phase'], 'unknown')
        self.assertEqual(engine.update(row(9, .3), plane())['phase'], 'feet_clear')
        engine = MultiViewFootContact()
        for i in range(9):
            out = engine.update(row(i, speed=.8), plane())
        self.assertEqual(out['phase'], 'unknown')

    def test_one_region_can_support_but_swing_requires_whole_foot(self):
        engine = MultiViewFootContact()
        for i in range(9):
            r = row(i); r['joints_world_m'][0][2] = .3
            out = engine.update(r, plane())
        self.assertEqual(out['feet']['left']['state'], 'contact')
        self.assertNotIn('left_heel', out['feet']['left']['witness_landmarks'])

    def test_missing_stale_single_view_and_identity_change_withdraw(self):
        for mode in ('stale', 'single', 'attachment', 'identity', 'floor', 'gap'):
            engine, _ = self.confirmed(); r = row(8); ground = plane()
            if mode == 'stale':
                for s in r['joint_states']: s['last_observed_timestamp_ns'] = 0
            if mode == 'single':
                for q in r['quality']: q['inlier_cameras'] = ['a']
            if mode == 'attachment': r['attachment_state'] = {}
            if mode == 'identity': r['source_id'] = 'other'
            if mode == 'floor': ground['calibration_sha256'] = 'wrong'
            if mode == 'gap': r = row(20)
            self.assertEqual(engine.update(r, ground)['phase'], 'unknown', mode)

    def test_future_append_does_not_change_prefix(self):
        def run(count):
            e = MultiViewFootContact()
            return [e.update(row(i, .3 if i > 9 else .015), plane()) for i in range(count)]
        self.assertEqual(run(8), run(15)[:8])

    def test_ambiguous_motion_withdraws_previous_label(self):
        engine, _ = self.confirmed(); r = row(8)
        for p in r['joints_world_m']: p[0] += .08
        self.assertEqual(engine.update(r, plane())['phase'], 'unknown')

    def test_opt_in_state_integration_does_not_change_pose_or_velocity(self):
        from pose_sim.human_motion import HumanMotionEstimator
        base = HumanMotionEstimator()
        candidate = HumanMotionEstimator(dict(contact_baseline=True, ground_plane=plane()))
        for i in range(10):
            r = row(i)
            r['frame_id'] = i
            for q in r['quality']: q['linearized_max_std_m'] = .01
            a, b = base.update(copy.deepcopy(r)), candidate.update(copy.deepcopy(r))
            for key in ('joints_world_m','joint_states','velocities_world_m_s','orientations'):
                self.assertEqual(a[key], b[key])
            self.assertEqual(b['support']['schema'], 'multiview_foot_contact_v1')

    def test_robust_scene_plane_and_degenerate_geometry(self):
        rng = np.random.default_rng(2)
        xy = rng.uniform(-2, 2, (80, 2))
        xyz = np.c_[xy, .02*xy[:, 0]-.01*xy[:, 1]+.4+rng.normal(0, .002, 80)]
        result = fit_ground(xyz, 'cal')
        n, d, _, _, _ = validate_ground(result, 'cal')
        self.assertLess(np.max(np.abs(xyz@n+d)), .01)
        with self.assertRaises(ValueError): fit_ground(np.c_[np.arange(30), np.zeros((30, 2))], 'cal')
        with self.assertRaises(ValueError): validate_ground(result, 'other')

    def test_geometry_change_resets_contact_confirmation(self):
        engine, _ = self.confirmed(); ground = plane(); ground['offset_m'] = .001
        self.assertEqual(engine.update(row(8), ground)['phase'], 'unknown')


if __name__ == '__main__':
    unittest.main()
