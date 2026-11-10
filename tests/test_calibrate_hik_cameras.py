"""Checkerboard multi-camera calibration: corner-order canonicalisation and recovery of a known synthetic rig."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.calibrate_hik_cameras import board_points, calibrate, detect_corners

PATTERN, SQUARE = (9, 6), 0.06
SIZE = (812, 620)
K = np.array([[560.0, 0, 406.0], [0, 560.0, 310.0], [0, 0, 1]])
DIST = np.array([-0.10, 0.03, 0.0004, -0.0003, 0.0])


def _render_board(px=40, margin=60):
    cols, rows = PATTERN[0] + 1, PATTERN[1] + 1
    img = np.full((rows * px + 2 * margin, cols * px + 2 * margin), 255, np.uint8)
    for j in range(rows):
        for i in range(cols):
            if (i + j) % 2 == 0:
                img[margin + j * px:margin + (j + 1) * px, margin + i * px:margin + (i + 1) * px] = 0
    return img


def test_corner_origin_is_the_same_physical_corner_after_180_degree_rotation():
    img = _render_board()
    a = detect_corners(img, PATTERN)
    b = detect_corners(np.ascontiguousarray(img[::-1, ::-1]), PATTERN)
    h, w = img.shape
    assert a is not None and b is not None
    assert np.allclose(b, np.c_[w - 1 - a[:, 0], h - 1 - a[:, 1]], atol=0.5)


def _look_at(centre, target):
    z = target - centre
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    R = np.vstack([x, np.cross(z, x), z])
    return R, -R @ centre


def test_recovers_known_four_camera_rig_and_floor():
    rng = np.random.default_rng(3)
    obj = board_points(PATTERN, SQUARE)
    cams = {}
    for k, az in enumerate(np.radians([20, 110, 200, 290])):
        centre = np.array([3.0 * np.cos(az), 3.0 * np.sin(az), 1.5])
        cams[f"CAM{k}"] = (centre, *_look_at(centre, np.array([0, 0, 0.9])))
    serials = sorted(cams)

    def observe(R_wb, t_wb):
        X = (R_wb @ obj.T).T + t_wb
        obs = {}
        for s, (centre, R, t) in cams.items():
            if R_wb[:, 2] @ (centre - X.mean(axis=0)) >= 0:
                continue
            uv, _ = cv2.projectPoints(X, cv2.Rodrigues(R)[0], t, K, DIST)
            uv = uv.reshape(-1, 2)
            if (uv > 5).all() and (uv[:, 0] < SIZE[0] - 5).all() and (uv[:, 1] < SIZE[1] - 5).all():
                obs[s] = uv + rng.normal(0, 0.15, uv.shape)
        return obs

    bundles = []
    while len(bundles) < 60:
        centre = np.r_[rng.uniform(-0.8, 0.8, 2), rng.uniform(0.5, 1.6)]
        viewer = cams[serials[rng.integers(4)]][0]
        z = -(viewer - centre) / np.linalg.norm(viewer - centre)
        x = np.cross([0, 0, 1.0], z)
        x /= np.linalg.norm(x)
        R_wb = Rotation.from_rotvec(rng.normal(0, 0.35, 3)).as_matrix() @ np.c_[x, np.cross(z, x), z]
        t_wb = centre - R_wb @ obj.mean(axis=0)
        obs = observe(R_wb, t_wb)
        if obs:
            bundles.append(dict(phase="extrinsics", obs=obs))
    thickness = 0.004
    R_floor = np.c_[[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]]
    for dx in (0.0, 0.3):
        bundles.append(dict(phase="floor", obs=observe(R_floor, np.array([dx, 0.0, thickness]))))
    assert len(bundles[-2]["obs"]) >= 2

    calib, report = calibrate(bundles, serials, SIZE, PATTERN, SQUARE, board_thickness_m=thickness)
    assert report["failures"] == []
    assert report["floor_check"]["z_abs_err_mm_max"] < 3.0
    for s in serials:
        got = np.array(calib["cameras"][s]["camera_centre_w"])
        assert np.linalg.norm(got - cams[s][0]) < 0.025, (s, got, cams[s][0])
        assert calib["cameras"][s]["reprojection_rmse_px"] < 0.3


def test_symmetric_board_is_rejected():
    with pytest.raises(ValueError, match="symmetric"):
        calibrate([], ["A"], SIZE, (8, 6), SQUARE)
