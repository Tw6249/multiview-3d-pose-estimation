import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from pose_sim.sensors import ClockMapping, ExposureFrame, validate_calibration
from pose_sim.geometry import project, triangulate
from scripts.clean_outputs import checked_target


class SensorTests(unittest.TestCase):
    def test_integer_clock_mapping_drift_and_uncertainty_budget(self):
        origin = 1_700_000_000_000_000_001
        mapping = ClockMapping('device','host',origin,origin+100,1_000_010,1_000_000,500,1_000_000_000)
        self.assertEqual(mapping.map_exposure(origin+1_000_000,1000),origin+1_000_110)
        with self.assertRaises(ValueError): mapping.map_exposure(origin,499)
        with self.assertRaises(ValueError): mapping.map_exposure(origin+2_000_000_000,1000)

    def test_exposure_frame_rejects_wrong_clock_or_image_shape(self):
        ExposureFrame('a',1,2,'mapped',np.zeros((10,20,3),np.uint8)).validate()
        with self.assertRaises(ValueError): ExposureFrame('a',2,1,'mapped',np.zeros((10,20,3),np.uint8)).validate()
        with self.assertRaises(ValueError): ExposureFrame('a',1,2,'mapped',np.zeros((10,20))).validate()

    def test_calibration_requires_proper_rotation_and_finite_values(self):
        c = {'K':[[900,0,320],[0,900,240],[0,0,1]],'R_world_to_camera':np.eye(3).tolist(),
             't_world_to_camera_m':[0,0,0],'width':640,'height':480}
        validate_calibration({'cameras':{'a':c,'b':c}})
        bad = copy.deepcopy(c)
        bad['R_world_to_camera'][0][0] = -1
        with self.assertRaises(ValueError): validate_calibration({'cameras':{'a':c,'b':bad}})
        bad = copy.deepcopy(c)
        bad['K'][0][0] = float('nan')
        with self.assertRaises(ValueError): validate_calibration({'cameras':{'a':c,'b':bad}})

    def test_geometry_uncertainty_increases_with_distance(self):
        cams = [{'K':[[900,0,320],[0,900,240],[0,0,1]],'R_world_to_camera':np.eye(3).tolist(),
                 't_world_to_camera_m':[x,0,0]} for x in (-0.5,0.5)]
        results = [triangulate(cams,[project([0,0,z],c)[0] for c in cams],[1,1]) for z in (2,5)]
        self.assertGreater(results[1]['linearized_max_std_m'],results[0]['linearized_max_std_m']*5)
        self.assertTrue(np.all(np.linalg.eigvalsh(results[0]['position_covariance_m2']) > 0))

    def test_cleanup_rejects_root_parent_and_outside_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'outputs'
            root.mkdir()
            child = root/'failed'
            child.mkdir()
            self.assertEqual(checked_target(child,root),child.resolve())
            for path in (root,Path(temp)):
                with self.assertRaises(ValueError): checked_target(path,root)


if __name__ == '__main__':
    unittest.main()
