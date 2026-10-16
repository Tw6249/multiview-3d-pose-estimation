from itertools import combinations
import numpy as np
import cv2
from scipy.optimize import least_squares

JOINTS = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
          'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
          'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
          'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
         (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
         (0, 1), (0, 2), (1, 3), (2, 4)]


def arrays(cam):
    return (np.asarray(cam['K'], float), np.asarray(cam['R_world_to_camera'], float),
            np.asarray(cam['t_world_to_camera_m'], float), np.asarray(cam.get('distortion', [0]*5), float))


def project(xyz, cam):
    k, r, t, d = arrays(cam)
    p, _ = cv2.projectPoints(np.asarray(xyz, float).reshape(-1, 3), cv2.Rodrigues(r)[0], t, k, d)
    return p.reshape(-1, 2)


def center(cam):
    _, r, t, _ = arrays(cam)
    return -r.T @ t


def dlt(cams, xy, weights):
    rows = []
    for cam, pixel, w in zip(cams, xy, weights):
        k, r, t, d = arrays(cam)
        u, v = cv2.undistortPoints(np.asarray(pixel, float).reshape(1, 1, 2), k, d)[0, 0]
        p = np.column_stack((r, t))
        rows.extend([np.sqrt(w) * (u*p[2]-p[0]), np.sqrt(w) * (v*p[2]-p[1])])
    _, _, vt = np.linalg.svd(rows)
    if abs(vt[-1, 3]) < 1e-10:
        raise ValueError('Point at infinity')
    return vt[-1, :3] / vt[-1, 3]


def triangulate(cams, xy, scores, threshold_px=8.0, min_angle_deg=1.0):
    """Pair-enumeration RANSAC, weighted DLT, robust pixel reprojection refinement."""
    xy, scores = np.asarray(xy, float), np.asarray(scores, float)
    if len(cams) < 2:
        return None
    def errors(x):
        return np.array([np.linalg.norm(project(x, c)[0]-p) for c, p in zip(cams, xy)])
    def positive(x, indices):
        return all((arrays(cams[i])[1] @ x + arrays(cams[i])[2])[2] > 0 for i in indices)
    best = None
    for pair in combinations(range(len(cams)), 2):
        try:
            x = dlt([cams[i] for i in pair], xy[list(pair)], scores[list(pair)])
        except (ValueError, np.linalg.LinAlgError):
            continue
        e = errors(x)
        ids = np.flatnonzero(e <= threshold_px)
        if len(ids) < 2 or not positive(x, ids):
            continue
        rank = (len(ids), float(scores[ids].sum()), -float(e[ids].mean()))
        if best is None or rank > best[0]:
            best = rank, ids, x
    if best is None:
        return None
    ids = best[1]
    for _ in range(2):
        x = dlt([cams[i] for i in ids], xy[ids], scores[ids])
        def residual(v):
            return np.concatenate([np.sqrt(scores[i])*(project(v, cams[i])[0]-xy[i]) for i in ids])
        x = least_squares(residual, x, loss='soft_l1', f_scale=3.0, max_nfev=40).x
        new_ids = np.flatnonzero(errors(x) <= threshold_px)
        if len(new_ids) < 2:
            return None
        if np.array_equal(new_ids, ids):
            break
        ids = new_ids
    if not np.isfinite(x).all() or not positive(x, ids):
        return None
    rays = np.array([x-center(cams[i]) for i in ids])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    angles = [np.degrees(np.arccos(np.clip(abs(np.dot(rays[a], rays[b])), 0, 1)))
              for a, b in combinations(range(len(rays)), 2)]
    angle = max(angles)
    if angle < min_angle_deg:
        return None
    return {'xyz_m': x.tolist(), 'inliers': ids.tolist(),
            'mean_reprojection_px': float(errors(x)[ids].mean()), 'ray_angle_deg': float(angle)}
