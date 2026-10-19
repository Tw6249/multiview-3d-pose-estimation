import unittest
from pathlib import Path
import tempfile
import numpy as np

from scripts.report_mainline import arr, assert_alignment, error_summary, recovery_delay


class MainlineAuditTests(unittest.TestCase):
    def test_legacy_coco_reference_uses_named_body_subset(self):
        points = [[float(i), 0., 1.] for i in range(17)]
        points[10] = None
        mapped = arr([dict(joints_world_m=points)])
        self.assertEqual(mapped.shape, (1, 12, 3))
        self.assertEqual(mapped[0, 0, 0], 5.)
        self.assertTrue(np.isnan(mapped[0, 5]).all())

    def test_robot_preview_reads_unicode_absolute_path(self):
        import cv2
        from motion_control.humanoid_replay import read_preview_image
        image = np.full((12, 16, 3), [10, 70, 200], dtype=np.uint8)
        with tempfile.TemporaryDirectory(prefix='姿态预览_') as folder:
            path = Path(folder)/'相机.png'
            cv2.imencode('.png', image)[1].tofile(path)
            np.testing.assert_array_equal(read_preview_image(path), image)

    def test_missing_outputs_remain_in_success_denominator(self):
        result = error_summary(np.array([[10., np.nan, 200.]]), np.array([[True, False, False]]))
        self.assertAlmostEqual(result['coverage'], 2/3)
        self.assertAlmostEqual(result['all_target_within_100mm'], 1/3)
        self.assertEqual(result['error_mm']['count'], 2)
        self.assertEqual(result['current_over_150mm'], 0)

    def test_scoring_rejects_clock_misalignment(self):
        estimate = [dict(frame_id=0, timestamp_ns=40)]
        for reference in ([dict(frame_id=0, timestamp_ns=41)], []):
            with self.assertRaises(ValueError):
                assert_alignment(estimate, reference)

    def test_recovery_requires_three_consecutive_current_observations(self):
        errors = np.zeros((12, 2))
        current = np.ones_like(errors, dtype=bool)
        current[5, 0] = False
        self.assertEqual(recovery_delay(errors, current, 1, 4, [0, 1]), 80)
        errors[6:, 0] = 200
        self.assertIsNone(recovery_delay(errors, current, 1, 4, [0, 1]))


if __name__ == '__main__':
    unittest.main()
