import unittest
import numpy as np
from pose_sim.geometry import project, triangulate, CameraGeometry
from pose_sim.pipeline import reconstruct_record


def camera(x, y=0, distortion=None):
    return {'K': [[900, 0, 320], [0, 900, 240], [0, 0, 1]],
            'R_world_to_camera': np.eye(3).tolist(),
            't_world_to_camera_m': [-x, -y, 0],
            'distortion': distortion or [0]*5, 'width': 640, 'height': 480}


class GeometryTests(unittest.TestCase):
    def test_vectorized_projection_and_jacobian_match_rotated_cameras(self):
        from scipy.spatial.transform import Rotation
        cs=[camera(-.7),camera(.7,distortion=[.1,-.03,.001,-.002,0])]
        for c in cs:c['R_world_to_camera']=Rotation.from_euler('xyz',[.1,.3,-.2]).as_matrix().tolist()
        for selected in ([cs[0]],cs):
            g=CameraGeometry(selected);x=np.array([.2,-.1,3.])
            np.testing.assert_allclose(g.project(x),[project(x,c)[0] for c in selected],atol=1e-8)
            numerical=np.stack([(g.project(x+v*1e-5)-g.project(x-v*1e-5))/2e-5 for v in np.eye(3)],axis=-1)
            np.testing.assert_allclose(g.jacobian(x),numerical,atol=1e-5)

    def test_exact_reconstruction_and_distortion(self):
        cs = [camera(-0.6, distortion=[0.1, -0.03, 0.001, -0.002, 0]), camera(0.6), camera(0, 0.5)]
        gt = np.array([0.13, 0.2, 3.4])
        result = triangulate(cs, [project(gt, c)[0] for c in cs], [1]*3)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result['xyz_m'], gt, atol=1e-6)

    def test_reject_bad_view(self):
        cs = [camera(-0.7), camera(0.7), camera(0, 0.6), camera(0, -0.6)]
        gt = np.array([0.1, -0.1, 3.0])
        xy = np.array([project(gt, c)[0] for c in cs])
        xy[-1] += [140, -90]
        result = triangulate(cs, xy, [0.9]*4)
        np.testing.assert_allclose(result['xyz_m'], gt, atol=1e-6)
        self.assertNotIn(3, result['inliers'])

    def test_low_parallax_is_invalid(self):
        cs = [camera(0), camera(0.0001)]
        gt = [0, 0, 5]
        self.assertIsNone(triangulate(cs, [project(gt, c)[0] for c in cs], [1, 1]))

    def test_one_camera_cannot_reconstruct(self):
        self.assertIsNone(triangulate([camera(0)], [[320, 240]], [1]))

    def test_fast_solver_matches_reference_under_noise_distortion_and_outliers(self):
        rng=np.random.default_rng(17)
        cs=[camera(-.7,distortion=[.08,-.02,.001,0,0]),camera(.7),camera(0,.6),camera(0,-.6)]
        for i in range(35):
            gt=np.r_[rng.uniform(-.4,.4,2),rng.uniform(2.,5.)]
            xy=np.array([project(gt,c)[0] for c in cs])+rng.normal(0,1.8,(4,2))
            if i%3==0:xy[-1]+=[50,-45]
            a=triangulate(cs,xy,[.9]*4,solver='scipy');b=triangulate(cs,xy,[.9]*4,solver='newton')
            self.assertIsNotNone(a);self.assertIsNotNone(b)
            self.assertEqual(a['inliers'],b['inliers'])
            np.testing.assert_allclose(a['xyz_m'],b['xyz_m'],atol=5e-5)
        close=[camera(0),camera(.0001)]
        self.assertIsNone(triangulate(close,[project([0,0,5],c)[0] for c in close],[1,1],solver='newton'))

    def test_timestamp_and_missing_detection_gate(self):
        cs = {'a': camera(-0.5), 'b': camera(0.5), 'c': camera(0, 0.5)}
        cfg = {'max_sync_error_ms': 5, 'confidence_threshold': 0.35,
               'ransac_threshold_px': 8, 'min_triangulation_angle_deg': 1}
        row = {'frame_id': 0, 'timestamp_ns': 0, 'views': {}}
        for name, cam in cs.items():
            row['views'][name] = {'timestamp_ns': 0, 'keypoints_px': [project([0,0,3], cam)[0].tolist()]*17, 'scores': [1]*17}
        self.assertTrue(all(x is not None for x in reconstruct_record(row, cs, cfg)['joints_world_m']))
        row['views']['b']['timestamp_ns'] = 20_000_000
        row['views']['c']['keypoints_px'] = None
        result = reconstruct_record(row, cs, cfg)
        self.assertTrue(all(x is None for x in result['joints_world_m']))
        self.assertIn('timestamp', result['excluded_views']['b'])


if __name__ == '__main__':
    unittest.main()
