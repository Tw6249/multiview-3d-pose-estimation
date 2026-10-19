import unittest
from types import SimpleNamespace as NS
import numpy as np

from pose_sim.probabilistic_pose import extract_prediction
from pose_sim.sensors import ExposureFrame


class ProbabilisticPoseTests(unittest.TestCase):
    def sample(self):
        p = NS(keypoints=np.ones((1, 23, 2)), keypoints_oks=np.full((1, 23), .8),
               keypoints_probs=np.full((1, 23), .6), keypoints_visible=np.full((1, 23), .3))
        return NS(pred_instances=p, pred_fields=NS(heatmaps=np.full((23, 64, 48), 1 / 3072)),
                  metainfo=dict(input_center=np.array([100., 120.]), input_scale=np.array([94., 126.])))

    def frame(self):
        return ExposureFrame('cam2', 123, 130, 'camera_clock', np.zeros((240, 320, 3), np.uint8))

    def test_distinct_scores_and_exposure_preserved(self):
        sample = self.sample(); p = extract_prediction(sample, self.frame())
        self.assertEqual((p.camera_id, p.exposure_timestamp_ns, p.clock_domain), ('cam2', 123, 'camera_clock'))
        np.testing.assert_allclose(p.predicted_oks, .8)
        np.testing.assert_allclose(p.presence, .6)
        np.testing.assert_allclose(p.visibility, .3)
        self.assertFalse(p.semantics['calibrated_for_domain'])
        sample.pred_fields.heatmaps[:] = 0
        np.testing.assert_allclose(p.spatial_mass.sum((1, 2)), 1, atol=1e-6)
        self.assertFalse(p.spatial_mass.flags.writeable)

    def test_missing_real_outputs_never_fallback_to_scores(self):
        for field in ('keypoints_oks', 'keypoints_probs', 'keypoints_visible'):
            sample = self.sample(); delattr(sample.pred_instances, field)
            sample.pred_instances.keypoint_scores = np.ones((1, 23))
            with self.assertRaises(ValueError): extract_prediction(sample, self.frame())

    def test_bad_outputs_and_mass_rejected(self):
        for value in (np.nan, -1., 1.1):
            sample = self.sample(); sample.pred_instances.keypoints_probs[0, 0] = value
            with self.assertRaises(ValueError): extract_prediction(sample, self.frame())
        sample = self.sample(); sample.pred_fields.heatmaps *= 2
        with self.assertRaises(ValueError): extract_prediction(sample, self.frame())

    def test_udp_heatmap_mapping_keeps_outside_unknown(self):
        p = extract_prediction(self.sample(), self.frame())
        np.testing.assert_allclose(p.pixels_to_heatmap([[53, 57], [147, 183], [100, 120]]), [[0, 0], [47, 63], [23.5, 31.5]])
        self.assertLess(p.pixels_to_heatmap([0, 0])[0], 0)
        with self.assertRaises(ValueError): p.pixels_to_heatmap([np.nan, 0])
