"""Unit & Integration Tests for Module S2: Multi-View Weighted DLT Triangulation & Outlier Rejection.

Verification Criteria (`docs/方案规范.md` — S2.3):
  1. Under 6 circular cameras (`1280x720`), sigma = 2.0 px Gaussian noise + 1-2 cameras per joint injected
      with 120.0 px large errors at high fake confidence (0.8):
      3D MPJPE after outlier rejection < 8.0 mm (0.008 m).
  2. Single-frame 2.0 m 3D jump is clamped to <= 0.25 m.
  3. 6-camera 27-point single-frame Numba computation time < 0.5 ms.
"""
from __future__ import annotations

import numpy as np
import pytest

from g1_imitation.core.s2_triangulation import (
    MultiViewTriangulator,
    create_circular_camera_rig,
    project_points_3d_to_2d,
)
from g1_imitation.core.synthetic_human import synthesize_human_27_pose


def _generate_corrupted_sequence() -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(20261024)
    P_6, _ = create_circular_camera_rig(
        num_cameras=6,
        radius_m=3.2,
        camera_height_m=1.35,
        lookat_xyz_m=(0.0, 0.0, 0.90),
        image_width=1280,
        image_height=720,
        focal_px=950.0,
    )
    gt_list: list[np.ndarray] = []
    kp_2d_list: list[np.ndarray] = []
    for f in range(120):
        t = f * 0.01
        gt_3d = synthesize_human_27_pose(
            height_m=1.75,
            squat_depth_m=0.08 * (0.5 - 0.5 * np.cos(2.0 * np.pi * 0.5 * t)),
            torso_yaw_rad=0.20 * np.sin(2.0 * np.pi * 0.4 * t),
        )
        kp_2d = project_points_3d_to_2d(gt_3d, P_6, default_confidence=0.90)
        kp_2d[:, :, :2] += rng.normal(0.0, 2.0, size=(6, 27, 2))
        for j in range(27):
            n_out = int(rng.integers(1, 3))
            out_cams = rng.choice(6, size=n_out, replace=False)
            for c_idx in out_cams:
                angle = float(rng.uniform(0.0, 2.0 * np.pi))
                kp_2d[c_idx, j, 0] += 120.0 * np.cos(angle)
                kp_2d[c_idx, j, 1] += 120.0 * np.sin(angle)
                kp_2d[c_idx, j, 2] = 0.80
        gt_list.append(gt_3d)
        kp_2d_list.append(kp_2d)
    return P_6, gt_list, kp_2d_list


def test_s2_consensus_triangulation_velocity_clamp_and_speed() -> None:
    P_6, gt_list, kp_2d_list = _generate_corrupted_sequence()
    tri = MultiViewTriangulator(
        c_min=0.25,
        reproj_outlier_threshold_px=15.0,
        max_displacement_m=0.25,
        use_consensus_search=True,
    )

    mpjpe_mm_list: list[float] = []
    solve_times_ms: list[float] = []
    for gt_3d, kp_2d in zip(gt_list, kp_2d_list):
        res = tri.process_frame(kp_2d, P_6, update_history=True)
        err_mm = float(np.mean(np.linalg.norm(res.points_3d_m - gt_3d, axis=1)) * 1000.0)
        mpjpe_mm_list.append(err_mm)
        solve_times_ms.append(res.solve_time_ms)

    mean_mpjpe_mm = float(np.mean(mpjpe_mm_list))
    median_solve_ms = float(np.median(solve_times_ms))

    # Test single-frame 2.0 m 3D jump clamping on the next frame
    prev_pts = tri.prev_points_3d.copy()
    jumped_gt_3d = prev_pts + np.array([2.0, 0.0, 0.0], dtype=np.float64)
    kp_2d_jump = project_points_3d_to_2d(jumped_gt_3d, P_6, default_confidence=0.95)
    res_jump = tri.process_frame(kp_2d_jump, P_6, update_history=True)
    step_displacements_m = np.linalg.norm(res_jump.points_3d_m - prev_pts, axis=1)
    max_step_disp_m = float(np.max(step_displacements_m))

    assert mean_mpjpe_mm < 8.0, f"Expected consensus 3D MPJPE < 8.0 mm, got {mean_mpjpe_mm:.3f} mm"
    assert max_step_disp_m <= 0.25 + 1e-9, f"Expected 2.0m jump clamped to <= 0.25 m, got {max_step_disp_m:.6f} m"
    assert median_solve_ms < 0.5, f"Expected 6-cam 27-point triangulation < 0.5 ms, got {median_solve_ms:.4f} ms"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Falsified theoretical algorithm in S2.2.2: literal 2-round residual thresholding (e_k <= 15.0 px "
        "around the Round 1 all-camera least-squares DLT solution) breaks down (~64.32 mm > 8.0 mm) when "
        "2/6 cameras (33%) have simultaneous 120 px high-confidence outliers, because L2 SVD is pulled "
        "toward the two outliers and assigns large residuals to the 4 inlier cameras."
    ),
)
def test_s2_literal_2round_dlt_under_2cam_outliers() -> None:
    P_6, gt_list, kp_2d_list = _generate_corrupted_sequence()
    tri_literal = MultiViewTriangulator(
        c_min=0.25,
        reproj_outlier_threshold_px=15.0,
        max_displacement_m=0.25,
        use_consensus_search=False,
    )
    mpjpe_mm_list: list[float] = []
    for gt_3d, kp_2d in zip(gt_list, kp_2d_list):
        res = tri_literal.process_frame(kp_2d, P_6, update_history=True)
        err_mm = float(np.mean(np.linalg.norm(res.points_3d_m - gt_3d, axis=1)) * 1000.0)
        mpjpe_mm_list.append(err_mm)

    mean_mpjpe_mm = float(np.mean(mpjpe_mm_list))
    assert mean_mpjpe_mm < 8.0, f"Expected literal 2-round DLT MPJPE < 8.0 mm, got {mean_mpjpe_mm:.3f} mm"


def test_s2_continuous_residual_sweep_and_covariance() -> None:
    """Verify Step 2: sweeping a camera's residual continuously across the 15.0 px threshold (0..40 px in 0.1 px steps)
    produces a strictly continuous 3D trajectory without step jumps, and outputs valid 3x3 covariances.
    """
    P_6, _ = create_circular_camera_rig(num_cameras=6)
    gt_3d = synthesize_human_27_pose(height_m=1.75)
    kp_base = project_points_3d_to_2d(gt_3d, P_6, default_confidence=0.90)

    tri = MultiViewTriangulator(
        c_min=0.25,
        reproj_outlier_threshold_px=15.0,
        max_displacement_m=0.25,
        use_consensus_search=True,
    )

    sweep_pts: list[np.ndarray] = []
    cov_traces: list[float] = []
    for err_px in np.linspace(0.0, 40.0, 401):
        kp_s = kp_base.copy()
        kp_s[0, 20, 0] += float(err_px)
        res = tri.process_frame(kp_s, P_6, update_history=False)
        sweep_pts.append(res.points_3d_m[20].copy())
        assert res.point_covariances_m2 is not None
        cov_j = res.point_covariances_m2[20]
        assert cov_j.shape == (3, 3)
        assert np.all(np.isfinite(cov_j))
        assert np.allclose(cov_j, cov_j.T, atol=1e-10)
        eigvals = np.linalg.eigvalsh(cov_j)
        assert float(np.min(eigvals)) > 0.0
        cov_traces.append(float(np.trace(cov_j)))

    traj = np.array(sweep_pts, dtype=np.float64)
    step_jumps_mm = np.linalg.norm(np.diff(traj, axis=0), axis=1) * 1000.0
    max_step_jump_mm = float(np.max(step_jumps_mm))
    assert max_step_jump_mm < 0.25, f"Expected smooth transition (< 0.25 mm/0.1px) across threshold, got {max_step_jump_mm:.4f} mm"
