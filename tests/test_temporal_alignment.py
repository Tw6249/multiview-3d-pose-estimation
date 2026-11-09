from __future__ import annotations

import numpy as np
import pytest

from g1_imitation.core.s2_triangulation import (
    MultiViewTriangulator,
    create_circular_camera_rig,
    project_points_3d_to_2d,
)
from g1_imitation.core.s3_temporal_filter import (
    AnisotropicKalmanFilter3D,
    CausalAlphaBetaFilter3D,
    SphericalSkeletonKalmanFilter3D,
    TemporalFilter,
    VectorizedOneEuroFilter3D,
)
from g1_imitation.core.synthetic_human import synthesize_human_27_pose
from g1_imitation.core.temporal_alignment import (
    estimate_camera_offsets,
    fundamental_from_projections,
    resample_keypoints_to_common_times,
)


def _pose_at(t: float, base: np.ndarray) -> np.ndarray:
    p = base.copy()
    yaw = 0.25 * np.sin(2.0 * np.pi * 0.43 * t) + 0.08 * np.sin(2.0 * np.pi * 0.91 * t)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    p = (R @ p.T).T
    p[:, 0] += 0.30 * np.sin(2.0 * np.pi * 0.37 * t) + 0.08 * np.sin(2.0 * np.pi * 0.83 * t)
    p[:, 1] += 0.12 * np.sin(2.0 * np.pi * 0.61 * t)
    p[:, 2] += 0.05 * np.sin(2.0 * np.pi * 0.71 * t)
    return p


def _synthetic_streams():
    projections, _ = create_circular_camera_rig(num_cameras=4)
    offsets = {"c0": 0.0, "c1": 0.034, "c2": -0.022, "c3": 0.052}
    ts = np.arange(0.02, 2.98, 0.02)
    base = synthesize_human_27_pose(height_m=1.72)
    keypoints, times, Ps = {}, {}, {}
    for i, (camera, offset) in enumerate(offsets.items()):
        local_t = ts.copy()
        physical_t = local_t + offset
        projected = [project_points_3d_to_2d(_pose_at(float(t), base), projections[i:i + 1],
                                               default_confidence=0.95)[0]
                     for t in physical_t]
        keypoints[camera] = np.stack(projected)
        times[camera] = local_t
        Ps[camera] = projections[i]
    return keypoints, times, Ps, offsets


def test_fundamental_matrix_matches_projected_correspondences():
    projections, _ = create_circular_camera_rig(num_cameras=4)
    X = synthesize_human_27_pose(height_m=1.75)
    uv = project_points_3d_to_2d(X, projections[:2], default_confidence=0.9)
    F = fundamental_from_projections(projections[0], projections[1])
    xa = np.c_[uv[0, :, :2], np.ones(27)]
    xb = np.c_[uv[1, :, :2], np.ones(27)]
    residual = np.einsum("ni,ij,nj->n", xb, F, xa)
    assert np.median(np.abs(residual)) < 1e-7


def test_camera_clock_offsets_recovered_from_epipolar_motion():
    kp, ts, Ps, expected = _synthetic_streams()
    result = estimate_camera_offsets(kp, ts, Ps, reference_camera="c0", search_range_s=0.08,
                                     search_step_s=0.002, window_s=2.0, min_observations=80)
    assert result.reliable, result.failure_reason
    for camera, offset in expected.items():
        assert result.offsets_s[camera] == pytest.approx(offset, abs=0.006)
    assert result.cycle_residual_rms_s < 0.006


def test_static_views_do_not_produce_a_fabricated_clock_offset():
    kp, ts, Ps, _ = _synthetic_streams()
    for camera in kp:
        kp[camera] = np.broadcast_to(kp[camera][0], kp[camera].shape).copy()
    with pytest.raises(ValueError, match="boundary|not observable"):
        estimate_camera_offsets(kp, ts, Ps, reference_camera="c0", search_range_s=0.08,
                                search_step_s=0.002, window_s=1.0, min_observations=80)


def test_common_time_resampling_uses_offsets_and_keeps_unbracketed_points_unknown():
    kp, ts, Ps, offsets = _synthetic_streams()
    target = np.array([0.5, 1.0, 1.5])
    aligned = resample_keypoints_to_common_times(kp, ts, offsets, target, max_interp_gap_s=0.05)
    assert aligned.shape == (3, 4, 27, 3)
    assert np.all(aligned[..., 2] > 0.9)
    outside = resample_keypoints_to_common_times(kp, ts, offsets, np.array([-0.3]), max_interp_gap_s=0.05)
    assert np.all(outside[..., 2] == 0.0)


def test_alpha_beta_filter_uses_real_time_and_expires_stale_data():
    filt = CausalAlphaBetaFilter3D(alpha=0.85, beta=0.15, max_hold_s=0.05, reset_gap_s=0.15)
    pose = synthesize_human_27_pose(height_m=1.75)
    first = filt.process_frame(pose, 10.0)
    moved = pose.copy()
    moved[:, 0] += 0.04
    second = filt.process_frame(moved, 10.037)
    assert second.fresh_mask.all()
    assert np.isfinite(second.points_3d_m).all()
    assert np.all(second.points_3d_m[:, 0] > first.points_3d_m[:, 0])

    missing = np.full((27, 3), np.nan)
    held = filt.process_frame(missing, 10.074)
    expired = filt.process_frame(missing, 10.111)
    assert np.isfinite(held.points_3d_m).all()
    assert not np.isfinite(expired.points_3d_m).any()
    assert not expired.fresh_mask.any()
    with pytest.raises(ValueError):
        filt.process_frame(missing, 10.111)


def test_alpha_beta_candidate_reduces_lag_vs_literal_one_euro_on_fast_motion():
    rng = np.random.default_rng(91)
    base = synthesize_human_27_pose(height_m=1.75)
    one = TemporalFilter(freq_hz=30.0, f_c_min=1.2, beta=0.04, f_c_d=1.0, mode="one_euro")
    default_s3 = TemporalFilter(freq_hz=30.0)
    ab = CausalAlphaBetaFilter3D(alpha=0.82, beta=0.12, max_hold_s=0.08)
    dt = 1.0 / 30.0
    t = np.arange(270) * dt
    y_one, y_ab, y_s3 = [], [], []
    for ti in t:
        offset = 0.15 * np.sin(2.0 * np.pi * 3.0 * ti)
        sample = base.copy()
        sample[:, 0] += offset + rng.normal(0.0, 0.015, size=27)
        y_one.append(one.process_frame(sample, timestamp_s=float(ti)).points_3d_smooth_m[0, 0] - base[0, 0])
        y_ab.append(ab.process_frame(sample, float(ti)).points_3d_m[0, 0] - base[0, 0])
        y_s3.append(default_s3.process_frame(sample, timestamp_s=float(ti)).points_3d_smooth_m[0, 0] - base[0, 0])
    y_one, y_ab, y_s3 = np.asarray(y_one), np.asarray(y_ab), np.asarray(y_s3)
    eval_t = t[30:]
    lags = np.linspace(-0.10, 0.10, 801)

    def best_lag(y):
        errors = [np.mean((y[30:] - 0.15 * np.sin(2.0 * np.pi * 3.0 * (eval_t - lag))) ** 2)
                  for lag in lags]
        return float(lags[int(np.argmin(errors))])

    lag_one, lag_ab, lag_s3 = best_lag(y_one), best_lag(y_ab), best_lag(y_s3)
    assert abs(lag_ab) < 0.05
    assert abs(lag_s3) < 0.015
    assert lag_ab < lag_one
    assert lag_s3 < lag_one


def test_one_euro_does_not_invent_origin_for_first_missing_measurement():
    filt = VectorizedOneEuroFilter3D(freq_hz=30.0)
    missing = np.full((2, 3), np.nan)
    out, _ = filt.filter_frame(missing, dt=0.04)
    assert np.isnan(out).all()
    observed = np.array([[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan]])
    out, _ = filt.filter_frame(observed, dt=0.04)
    assert np.allclose(out[0], observed[0])
    assert np.isnan(out[1]).all()


def test_timestamped_one_euro_filter_derives_irregular_dt_and_rejects_nonmonotonic_time():
    filt = TemporalFilter(freq_hz=30.0)
    pose = synthesize_human_27_pose(height_m=1.75)
    filt.process_frame(pose, timestamp_s=1.0)
    moved = pose.copy()
    moved[:, 0] += 0.01
    out = filt.process_frame(moved, timestamp_s=1.043)
    assert np.isfinite(out.points_3d_smooth_m).all()
    with pytest.raises(ValueError, match="strictly increasing"):
        filt.process_frame(moved, timestamp_s=1.043)

    after_gap = moved.copy()
    after_gap[:, 0] += 0.1
    gap_out = filt.process_frame(after_gap, timestamp_s=1.5)
    assert np.allclose(gap_out.points_3d_smooth_m, after_gap)


def test_s2_displacement_gate_scales_with_actual_dt():
    projections, _ = create_circular_camera_rig(num_cameras=4)
    tri = MultiViewTriangulator(c_min=0.25, max_joint_speed_m_s=1.0)
    pose = synthesize_human_27_pose(height_m=1.75)
    kp0 = project_points_3d_to_2d(pose, projections, default_confidence=0.95)
    first = tri.process_frame(kp0, projections, dt_s=0.02)
    moved = pose.copy()
    moved[:, 0] += 0.20
    kp1 = project_points_3d_to_2d(moved, projections, default_confidence=0.95)
    second = tri.process_frame(kp1, projections, dt_s=0.05)
    displacement = np.linalg.norm(second.points_3d_m - first.points_3d_m, axis=1)
    assert np.nanmax(displacement) <= 0.05 + 1e-8


def test_s2_does_not_apply_stale_velocity_history_after_long_gap():
    projections, _ = create_circular_camera_rig(num_cameras=4)
    tri = MultiViewTriangulator(c_min=0.25, max_joint_speed_m_s=1.0, max_history_gap_s=0.15)
    pose = synthesize_human_27_pose(height_m=1.75)
    kp0 = project_points_3d_to_2d(pose, projections, default_confidence=0.95)
    tri.process_frame(kp0, projections, dt_s=0.02)
    moved = pose.copy()
    moved[:, 0] += 0.8
    kp1 = project_points_3d_to_2d(moved, projections, default_confidence=0.95)
    after_gap = tri.process_frame(kp1, projections, dt_s=0.5)
    displacement = np.linalg.norm(after_gap.points_3d_m - moved, axis=1)
    assert np.nanmax(displacement) < 0.01


def test_anisotropic_kalman_filter_smooths_ray_axis_more_than_lateral_axis():
    rng = np.random.default_rng(123)
    pose = synthesize_human_27_pose(height_m=1.75)
    kf = AnisotropicKalmanFilter3D()
    cov = np.zeros((27, 3, 3), dtype=np.float64)
    # X is well-constrained (3 mm), Y is poorly constrained along camera ray (65 mm), Z is 5 mm
    cov[:, 0, 0] = 0.003 ** 2
    cov[:, 1, 1] = 0.065 ** 2
    cov[:, 2, 2] = 0.005 ** 2

    dt = 1.0 / 30.0
    err_x, err_y, raw_y = [], [], []
    for k in range(150):
        t = k * dt
        gt = pose.copy()
        gt[:, 0] += 0.12 * np.sin(2.0 * np.pi * 2.0 * t)
        noisy = gt.copy()
        noisy[:, 0] += rng.normal(0.0, 0.003, size=27)
        y_noise = rng.normal(0.0, 0.050, size=27)
        noisy[:, 1] += y_noise
        res = kf.process_frame(noisy, t, cov)
        if k >= 20:
            err_x.append(np.sqrt(np.mean((res.points_3d_m[:, 0] - gt[:, 0]) ** 2)))
            err_y.append(np.sqrt(np.mean((res.points_3d_m[:, 1] - gt[:, 1]) ** 2)))
            raw_y.append(np.sqrt(np.mean(y_noise ** 2)))

    assert float(np.mean(err_x)) < 0.008
    assert float(np.mean(err_y)) < 0.65 * float(np.mean(raw_y))


def test_spherical_skeleton_kalman_filter_suppresses_radial_bone_stretching():
    rng = np.random.default_rng(456)
    base = synthesize_human_27_pose(height_m=1.75)
    sph = SphericalSkeletonKalmanFilter3D()
    elbow = base[7].copy()
    forearm_len = float(np.linalg.norm(base[9] - elbow))
    cov = np.broadcast_to((0.005 ** 2) * np.eye(3), (27, 3, 3)).copy()

    raw_lengths, sph_lengths, sph_err = [], [], []
    dt = 1.0 / 30.0
    for k in range(150):
        t = k * dt
        angle = 0.9 * np.sin(2.0 * np.pi * 1.8 * t)
        u_true = np.array([np.cos(angle), 0.0, np.sin(angle)], dtype=np.float64)
        gt = base.copy()
        gt[9] = elbow + forearm_len * u_true
        # Inject radial stretching along u_true with large directional covariance along u_true
        stretch = rng.normal(0.0, 0.045)
        tangential = rng.normal(0.0, 0.006, size=3)
        obs = gt.copy()
        obs[9] = gt[9] + stretch * u_true + tangential
        cov_frame = cov.copy()
        cov_frame[9] = (0.006 ** 2) * np.eye(3) + (0.045 ** 2) * np.outer(u_true, u_true)
        res = sph.process_frame(obs, t, cov_frame)
        if k >= 20:
            raw_lengths.append(float(np.linalg.norm(obs[9] - obs[7])))
            sph_lengths.append(float(np.linalg.norm(res.points_3d_m[9] - res.points_3d_m[7])))
            sph_err.append(float(np.linalg.norm(res.points_3d_m[9] - gt[9])))

    raw_cv = float(np.std(raw_lengths) / np.mean(raw_lengths))
    sph_cv = float(np.std(sph_lengths) / np.mean(sph_lengths))
    assert sph_cv < 0.35 * raw_cv
    assert float(np.mean(sph_err)) < 0.020
