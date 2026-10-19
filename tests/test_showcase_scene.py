import math
import unittest
from pose_sim.showcase_scene import sample_motion


class ShowcaseSceneTests(unittest.TestCase):
    def test_default_keeps_original_linear_motion(self):
        cfg = dict(fps=24, frames=96, animation_loop_seconds=2,
                   human_translation_m_s=[0, -.83, 0])
        for f in (0, 47, 48, 95):
            animation, translation, yaw = sample_motion(cfg, f)
            self.assertEqual(animation, f % 48)
            self.assertEqual(translation, [0, -.83*(f/24-95/48), 0])
            self.assertEqual(yaw, 0)

    def test_circle_heading_follows_tangent_both_directions(self):
        for speed in (-1.0, 1.0):
            cfg = dict(fps=24, frames=144,
                       root_motion=dict(kind='circle', radius_m=.8, angular_speed_rad_s=speed))
            for f in (0, 20, 70):
                _, p, yaw = sample_motion(cfg, f)
                angle = speed*f/24
                tangent = [-math.sin(angle)*speed, math.cos(angle)*speed]
                heading = [math.sin(yaw), -math.cos(yaw)]
                self.assertAlmostEqual(sum(a*b for a,b in zip(tangent, heading)), 1.0)
                self.assertAlmostEqual(math.hypot(*p[:2]), .8)

    def test_unknown_motion_fails_instead_of_silent_fallback(self):
        with self.assertRaises(ValueError):
            sample_motion(dict(fps=24, root_motion=dict(kind='unknown')), 0)

    def test_video_refuses_timestamp_mismatch_missing_and_duplicates(self):
        from scripts.vision_showcase_video import join_rows
        frames = [dict(frame_id=0,timestamp_ns=0), dict(frame_id=1,timestamp_ns=40_000_000)]
        self.assertEqual(len(list(join_rows(frames, frames))), 2)
        for bad in (frames[:1], frames[::-1], [frames[0], dict(frame_id=1,timestamp_ns=41_000_000)]):
            with self.assertRaises(ValueError):
                join_rows(frames, bad)
        with self.assertRaises(ValueError):
            join_rows([frames[0], frames[0]])


if __name__ == '__main__':
    unittest.main()
