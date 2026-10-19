"""Local triangulation conditioning under an explicit pixel-noise assumption."""
import numpy as np


def linearized_covariance(xyz, cameras, project, pixel_sigma=2.0):
    """(J.T J)^-1 sigma^2, for independent isotropic image noise.

    This excludes calibration, synchronization and correspondence errors. It is
    NOT an empirically calibrated probability of human-pose correctness.
    """
    xyz = np.asarray(xyz, float)
    step = 1e-5
    columns = []
    for axis in np.eye(3):
        columns.append(np.concatenate([(project(xyz+step*axis,cam)[0]-project(xyz-step*axis,cam)[0])/(2*step)
                                       for cam in cameras]))
    jacobian = np.column_stack(columns)
    return covariance_from_jacobian(jacobian,pixel_sigma)


def covariance_from_jacobian(jacobian,pixel_sigma=2.0):
    _, singular, vt = np.linalg.svd(jacobian, full_matrices=False)
    if len(singular) != 3 or singular[-1] <= 1e-9 or not np.isfinite(singular).all():
        return None
    covariance = (vt.T*(pixel_sigma/singular)**2)@vt
    return {'position_covariance_m2': covariance.tolist(),
            'linearized_max_std_m': float(pixel_sigma/singular[-1]),
            'assumed_pixel_sigma': pixel_sigma}
