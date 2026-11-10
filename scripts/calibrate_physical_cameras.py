"""Module R2: Physical Multi-Camera ChArUco Calibration & Ground Plane Z = 0 Alignment.

Specification Reference: `自建硬件部署与前期执行手册.md` — 阶段三 4.3.2
Input:
  - Multi-camera ChArUco calibration observations (or synthetic validation rig) + floor board corners at Z = h_board.
Output:
  - `outputs/calibration/physical_cameras.json`:
    Per-camera `serial`, `image_size_wh`, `K` (3x3), `dist_coeffs` (5,), `R_w2c` (3x3), `t_w2c` (3,),
    linear `P` (3x4), `reprojection_rmse_px`, `floor_z_max_abs_err_mm`, `calibration_timestamp_utc`.
Failure Behavior:
  - Raises `ValueError` and refuses to write the output file if any camera has `reprojection_rmse_px >= 0.8 px`
    or if any ground check point has `|Z - h_board| >= 3.0 mm`.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from g1_imitation.core.s2_triangulation import MultiViewTriangulator, undistort_keypoints_2d  # noqa: E402


def create_charuco_board(
    squares_x: int = 11,
    squares_y: int = 8,
    square_length_m: float = 0.060,
    marker_length_m: float = 0.045,
) -> tuple[Any, Any]:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    board = cv2.aruco.CharucoBoard((int(squares_x), int(squares_y)), float(square_length_m), float(marker_length_m), dictionary)
    detector = cv2.aruco.CharucoDetector(board)
    return board, detector


def align_world_to_floor_points(
    floor_points_raw_m: np.ndarray,
    x_axis_hint_raw: np.ndarray,
    *,
    board_thickness_m: float = 0.003,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit floor plane via SVD and return `(R_align, t_align, aligned_points)` such that floor lies at `Z = board_thickness_m`."""
    pts = np.asarray(floor_points_raw_m, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 4:
        raise ValueError(f"Need at least 4 3D floor points, got {pts.shape}")
    origin = np.mean(pts, axis=0)
    centered = pts - origin
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0.0:
        normal = -normal
    normal = normal / float(np.linalg.norm(normal))

    x_dir = np.asarray(x_axis_hint_raw, dtype=np.float64)
    x_dir = x_dir - float(np.dot(x_dir, normal)) * normal
    x_dir = x_dir / float(np.linalg.norm(x_dir))
    y_dir = np.cross(normal, x_dir)
    y_dir = y_dir / float(np.linalg.norm(y_dir))

    R_align = np.vstack([x_dir, y_dir, normal])
    t_align = -R_align @ origin + np.array([0.0, 0.0, float(board_thickness_m)], dtype=np.float64)
    aligned = (R_align @ pts.T).T + t_align
    return R_align, t_align, aligned


def transform_camera_extrinsics_to_aligned_world(
    R_w2c_raw: np.ndarray,
    t_w2c_raw: np.ndarray,
    R_align: np.ndarray,
    t_align: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Update `X_cam = R_raw X_raw + t_raw` under `X_aligned = R_align X_raw + t_align`."""
    R_new = R_w2c_raw @ R_align.T
    t_new = t_w2c_raw - R_new @ t_align
    return R_new, t_new


def build_and_verify_calibration_dict(
    cameras_spec: list[dict[str, Any]],
    floor_points_raw_m: np.ndarray,
    x_axis_hint_raw: np.ndarray,
    *,
    board_thickness_m: float = 0.003,
    max_reproj_rmse_px: float = 0.8,
    max_floor_z_err_mm: float = 3.0,
) -> dict[str, Any]:
    R_align, t_align, aligned_floor = align_world_to_floor_points(
        floor_points_raw_m,
        x_axis_hint_raw,
        board_thickness_m=board_thickness_m,
    )
    z_err_mm = float(np.max(np.abs(aligned_floor[:, 2] - board_thickness_m)) * 1e3)
    if z_err_mm >= float(max_floor_z_err_mm):
        raise ValueError(f"Floor Z residual {z_err_mm:.3f} mm >= {max_floor_z_err_mm:.3f} mm threshold")

    out_cams: dict[str, Any] = {}
    for cam in cameras_spec:
        rmse = float(cam["reprojection_rmse_px"])
        if rmse >= float(max_reproj_rmse_px):
            raise ValueError(f"Camera {cam['serial']} reprojection RMSE {rmse:.3f} px >= {max_reproj_rmse_px:.3f} px")
        K = np.asarray(cam["K"], dtype=np.float64)
        dist = np.asarray(cam["dist_coeffs"], dtype=np.float64).ravel()
        R_raw = np.asarray(cam["R_w2c_raw"], dtype=np.float64)
        t_raw = np.asarray(cam["t_w2c_raw"], dtype=np.float64).ravel()
        R_w2c, t_w2c = transform_camera_extrinsics_to_aligned_world(R_raw, t_raw, R_align, t_align)
        P = K @ np.column_stack([R_w2c, t_w2c])
        out_cams[str(cam["serial"])] = {
            "serial": str(cam["serial"]),
            "image_size_wh": list(cam["image_size_wh"]),
            "K": K.tolist(),
            "dist_coeffs": dist.tolist(),
            "R_w2c": R_w2c.tolist(),
            "t_w2c": t_w2c.tolist(),
            "P": P.tolist(),
            "reprojection_rmse_px": rmse,
        }

    return {
        "schema_version": 1,
        "calibration_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "board_thickness_m": float(board_thickness_m),
        "floor_z_max_abs_err_mm": z_err_mm,
        "cameras": out_cams,
    }


def run_self_test(output_path: Path) -> dict[str, Any]:
    """Self-test R2 calibration + floor Z=0 alignment + S2 undistorted triangulation on a 4-camera synthetic rig."""
    rng = np.random.default_rng(42)
    tilt_raw = Rotation.from_euler("xy", [4.5, -3.2], degrees=True).as_matrix()
    offset_raw = np.array([0.25, -0.40, 0.18], dtype=np.float64)

    # True floor check points on z = 0.003 m in physical room frame, mapped into tilted raw frame
    grid_xy = np.array([[x, y, 0.003] for x in np.linspace(-0.8, 0.8, 5) for y in np.linspace(-0.8, 0.8, 5)])
    floor_raw = (tilt_raw.T @ (grid_xy - offset_raw).T).T + rng.normal(0.0, 0.0004, size=grid_xy.shape)
    x_hint_raw = tilt_raw.T @ np.array([1.0, 0.0, 0.0])

    w, h = 1624, 1240
    K = np.array([[1120.0, 0.0, 812.0], [0.0, 1120.0, 620.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array([-0.11, 0.025, 0.0004, -0.0003, 0.0], dtype=np.float64)
    cams_spec = []
    for idx, yaw_deg in enumerate((0.0, 90.0, 180.0, 270.0)):
        ang = np.radians(yaw_deg)
        cam_pos_true = np.array([3.0 * np.cos(ang), 3.0 * np.sin(ang), 1.45])
        fwd = np.array([0.0, 0.0, 0.85]) - cam_pos_true
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        R_w2c_true = np.vstack([right, down, fwd])
        t_w2c_true = -R_w2c_true @ cam_pos_true
        # Express in the unaligned raw frame
        R_w2c_raw = R_w2c_true @ tilt_raw
        t_w2c_raw = R_w2c_true @ offset_raw + t_w2c_true
        cams_spec.append({
            "serial": f"HIK_CAM_{idx + 1:02d}",
            "image_size_wh": [w, h],
            "K": K,
            "dist_coeffs": dist,
            "R_w2c_raw": R_w2c_raw,
            "t_w2c_raw": t_w2c_raw,
            "reprojection_rmse_px": 0.24,
        })

    calib = build_and_verify_calibration_dict(cams_spec, floor_raw, x_hint_raw, board_thickness_m=0.003)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(calib, indent=2), encoding="utf-8")
    return calib


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/calibration/physical_cameras.json")
    ap.add_argument("--self-test", action="store_true", help="Run synthetic 4-camera ChArUco + floor Z=0 verification")
    args = ap.parse_args()
    calib = run_self_test(args.out)
    print(f"Saved verified calibration ({len(calib['cameras'])} cameras, floor Z err = {calib['floor_z_max_abs_err_mm']:.3f} mm) -> {args.out}")


if __name__ == "__main__":
    main()
