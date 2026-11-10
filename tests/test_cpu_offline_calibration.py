import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.local_cpu_offline import camera_projection


def test_camera_pose_inversion_and_intrinsic_scaling():
    rotation = Rotation.from_euler("z", 35, degrees=True)
    center = np.array([2., -3., 1.5])
    quat = rotation.as_quat()[[3,0,1,2]]
    camera = dict(intrinsics=[1200.,1000.,960.,540.], resolution=[1920,1080],
                  rotation_quaternion=quat, translation=center, distortion_coeffs=[0,0,0,0])
    point_camera = np.array([.4,-.2,2.])
    point_world = rotation.apply(point_camera) + center
    P = camera_projection(camera,960,540)
    projected = P @ np.r_[point_world,1.]
    assert np.allclose(projected[:2]/projected[2], [600*.4/2+480,500*(-.2)/2+270])
    assert np.allclose(P @ np.r_[center,1.], 0)
    camera["distortion_coeffs"] = [.1,0,0,0]
    with pytest.raises(ValueError,match="undistorted"):
        camera_projection(camera,960,540)
