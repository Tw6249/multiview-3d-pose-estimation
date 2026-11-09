"""Isolated evaluation of geometry-based camera clock alignment and causal filters.

Camera offsets use only 2D observations and epipolar geometry. The BVH supplies
the legacy anchor-slot hypothesis, a fixed initial world transform, and held-out
scores; it never selects camera offsets or per-method time shifts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1_imitation.core import adv_eval as ae  # noqa: E402
from g1_imitation.core.bvh import parse_bvh_and_fk  # noqa: E402
from g1_imitation.core.s2_triangulation import MultiViewTriangulator  # noqa: E402
from g1_imitation.core.s3_temporal_filter import (  # noqa: E402
    AnisotropicKalmanFilter3D,
    CausalAlphaBetaFilter3D,
    SphericalSkeletonKalmanFilter3D,
    TemporalFilter,
)
from g1_imitation.core.temporal_alignment import (  # noqa: E402
    estimate_camera_offsets,
    resample_keypoints_to_common_times,
)

CAMS = ["44310274", "44310277", "44310281", "44310284"]
S1_ROOT = ROOT / "outputs" / "advanced_take" / "s1"
RAW_ROOT = ROOT / "data" / "advanced_take_raw" / "bvh"
CAMERA_FILE = ROOT / "outputs" / "advanced_take" / "cameras_rgbonly.json"
OUT_ROOT = ROOT / "outputs" / "temporal_alignment_experiment_20261108"


def _load_projection_matrices():
    doc = json.loads(CAMERA_FILE.read_text(encoding="utf-8"))
    return {c: np.asarray(doc["cameras"][c]["P"], dtype=np.float64) for c in CAMS}


def _read_streams(take: str):
    pack = np.load(S1_ROOT / f"{take}_kp27.npz")
    keypoints, timestamps = {}, {}
    for camera in CAMS:
        keypoints[camera] = np.asarray(pack[f"{camera}_kp"], dtype=np.float64)
        timestamps[camera] = np.asarray(pack[f"{camera}_pts_s"], dtype=np.float64)
    return keypoints, timestamps


def _triangulate(keypoints, timestamps, P, *, max_speed_m_s=25.0, min_confidence=0.35):
    tri = MultiViewTriangulator(c_min=min_confidence, reproj_outlier_threshold_px=15.0,
                                max_joint_speed_m_s=max_speed_m_s)
    points, covariance, reproj, times = [], [], [], []
    for k, kp in enumerate(keypoints):
        t = float(timestamps[k])
        dt = (float(np.median(np.diff(timestamps))) if k == 0
              else float(timestamps[k] - timestamps[k - 1]))
        r = tri.process_frame(kp, np.stack([P[c] for c in CAMS]), dt_s=dt)
        points.append(r.points_3d_m)
        covariance.append(r.point_covariances_m2 if r.point_covariances_m2 is not None
                          else np.full((27, 3, 3), np.nan))
        reproj.append(r.reprojection_errors_px)
        times.append(t)
    return (np.asarray(times), np.asarray(points), np.asarray(covariance), np.asarray(reproj))


def _score(points, grid, bvh, ref_lag_frames=0):
    motion = bvh["motion_world_m"]
    idx = np.asarray(grid, dtype=np.int64) + int(ref_lag_frames)
    names = [bvh["name_to_idx"][n] for n in ae.BVH12]
    valid = (idx >= 0) & (idx < len(motion))
    ref = np.full((len(idx), 12, 3), np.nan, dtype=np.float64)
    ref[valid] = motion[idx[valid]][:, names]
    pred = points[:, 5:17]
    err = np.linalg.norm(pred - ref, axis=-1)
    good = np.isfinite(err)
    full = np.isfinite(pred).all((1, 2)) & np.isfinite(ref).all((1, 2))
    dir_err_deg = None
    bone_cv_mean = None
    if int(full.sum()) > 10:
        dp, dg = ae.segment_directions(pred[full]), ae.segment_directions(ref[full])
        Rrot = ae.rotation_only_align(
            np.concatenate([dp[n] for n in dp]),
            np.concatenate([dg[n] for n in dg]),
        )
        dir_err_deg = float(
            np.mean([
                np.mean(ae.angle_between_deg((Rrot @ dp[n].T).T, dg[n]))
                for n, _, _ in ae.SEGMENTS
            ])
        )
        bp = ae.bone_lengths(pred[full])
        bone_cv_mean = float(np.mean([np.std(bp[n]) / np.mean(bp[n]) for n in bp]))
    return dict(mpjpe_mm=float(np.mean(err[good]) * 1000.0) if good.any() else None,
                median_mm=float(np.median(err[good]) * 1000.0) if good.any() else None,
                p95_mm=float(np.percentile(err[good], 95) * 1000.0) if good.any() else None,
                bone_dir_err_deg=dir_err_deg,
                bone_length_cv_mean=bone_cv_mean,
                jitter_rms_mm_per_slot=float(ae.jitter_rms(pred - ref) * 1000.0),
                observed_fraction=float(good.mean()),
                n_joint_samples=int(good.sum()))


def _fit_world_transform(points, grid, bvh, calibration_mask, ref_lag_frames=0):
    motion = bvh["motion_world_m"]
    idx = np.asarray(grid, dtype=np.int64) + int(ref_lag_frames)
    names = [bvh["name_to_idx"][n] for n in ae.BVH12]
    valid = calibration_mask & (idx >= 0) & (idx < len(motion))
    src = points[valid, 5:17].reshape(-1, 3)
    dst = motion[idx[valid]][:, names].reshape(-1, 3)
    finite = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    if int(finite.sum()) < 30:
        raise RuntimeError("not enough calibration samples for a fixed world-frame transform")
    scale, R, offset = ae.robust_umeyama(src[finite], dst[finite], with_scale=False)
    return float(scale), R, offset


def _apply_world_transform(points, transform):
    scale, R, offset = transform
    out = np.full_like(points, np.nan)
    finite = np.isfinite(points).all(axis=2)
    if finite.any():
        out[finite] = scale * (R @ points[finite].T).T + offset
    return out


def _resampling_lookahead_ms(timestamps_by_camera, offsets_s, target_times_s):
    waits = []
    for camera, ts in timestamps_by_camera.items():
        ts = np.asarray(ts, dtype=np.float64)
        local_query = np.asarray(target_times_s, dtype=np.float64) - float(offsets_s[camera])
        right = np.searchsorted(ts, local_query, side="left")
        for i, (tq, hi) in enumerate(zip(local_query, right)):
            if hi >= len(ts):
                continue
            if abs(float(ts[hi] - tq)) <= 1e-8:
                wait_s = 0.0
            elif hi > 0:
                wait_s = float(ts[hi] + offsets_s[camera] - target_times_s[i])
            else:
                continue
            waits.append(max(0.0, wait_s))
    if not waits:
        return dict(p50_ms=None, p95_ms=None, max_ms=None)
    waits = np.asarray(waits) * 1000.0
    return dict(p50_ms=float(np.percentile(waits, 50)), p95_ms=float(np.percentile(waits, 95)),
                max_ms=float(np.max(waits)))


def _filter_tracks(t, points, covariance):
    one_euro = TemporalFilter(freq_hz=30.0, f_c_min=1.2, beta=0.04, f_c_d=1.0, z_nominal_m=0.0, mode="one_euro")
    alpha_beta = CausalAlphaBetaFilter3D(alpha=0.82, beta=0.12, max_hold_s=0.08,
                                        reset_gap_s=0.15, uncertainty_scale_m=0.03)
    kalman_3x3 = AnisotropicKalmanFilter3D(max_hold_s=0.08, reset_gap_s=0.15)
    spherical_kf = SphericalSkeletonKalmanFilter3D(max_hold_s=0.08, reset_gap_s=0.15)
    out_one, out_ab, out_kf, out_sph, out_raw = [], [], [], [], []
    age_ab = []
    timings_ms = {"one_euro": [], "alpha_beta": [], "kalman_3x3": [], "spherical_kalman": []}
    for i, (ti, p, cov) in enumerate(zip(t, points, covariance)):
        dt = (1.0 / 30.0) if i == 0 else float(ti - t[i - 1])
        t0 = time.perf_counter()
        out_one.append(one_euro.process_frame(p, dt=dt, timestamp_s=float(ti)).points_3d_smooth_m)
        timings_ms["one_euro"].append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        r = alpha_beta.process_frame(p, float(ti), cov)
        timings_ms["alpha_beta"].append((time.perf_counter() - t0) * 1000.0)
        ab_points = r.points_3d_m.copy()
        valid_ab = np.isfinite(ab_points).all(axis=1)
        ab_points[valid_ab, 2] = np.maximum(ab_points[valid_ab, 2], 0.0)
        out_ab.append(ab_points)
        age_ab.append(r.age_s)

        t0 = time.perf_counter()
        rk = kalman_3x3.process_frame(p, float(ti), cov)
        timings_ms["kalman_3x3"].append((time.perf_counter() - t0) * 1000.0)
        kf_points = rk.points_3d_m.copy()
        valid_kf = np.isfinite(kf_points).all(axis=1)
        kf_points[valid_kf, 2] = np.maximum(kf_points[valid_kf, 2], 0.0)
        out_kf.append(kf_points)

        t0 = time.perf_counter()
        rs = spherical_kf.process_frame(p, float(ti), cov)
        timings_ms["spherical_kalman"].append((time.perf_counter() - t0) * 1000.0)
        sph_points = rs.points_3d_m.copy()
        valid_sph = np.isfinite(sph_points).all(axis=1)
        sph_points[valid_sph, 2] = np.maximum(sph_points[valid_sph, 2], 0.0)
        out_sph.append(sph_points)

        out_raw.append(p)
    solve_ms = {k: float(np.median(v)) for k, v in timings_ms.items()}
    return (
        {
            "raw": np.asarray(out_raw),
            "one_euro": np.asarray(out_one),
            "alpha_beta": np.asarray(out_ab),
            "kalman_3x3": np.asarray(out_kf),
            "spherical_kalman": np.asarray(out_sph),
        },
        np.asarray(age_ab),
        solve_ms,
    )


def _diagnostic_lag_sweep(points, grid, bvh, radius_frames=18):
    motion = bvh["motion_world_m"]
    names = [bvh["name_to_idx"][n] for n in ae.BVH12]
    pred = points[:, 5:17]
    scores = {}
    for shift in range(-radius_frames, radius_frames + 1):
        idx = np.asarray(grid, dtype=np.int64) + shift
        valid_frame = (idx >= 0) & (idx < len(motion))
        if not valid_frame.any():
            scores[shift] = None
            continue
        ref = motion[idx[valid_frame]][:, names]
        err = np.linalg.norm(pred[valid_frame] - ref, axis=-1)
        valid = np.isfinite(err)
        scores[shift] = float(np.mean(err[valid]) * 1000.0) if valid.any() else None
    finite = [(k, v) for k, v in scores.items() if v is not None]
    best = min(finite, key=lambda item: item[1])[0] if finite else None
    return {"grid_frames": scores, "best_shift_frames_diagnostic_only": best,
            "best_shift_ms_diagnostic_only": None if best is None else float(best * bvh["frame_time_s"] * 1000.0)}


def run_take(take: str, calibration_seconds: float, reference_lag_frames: int):
    kp_by_cam, ts_by_cam = _read_streams(take)
    P = _load_projection_matrices()
    bvh = parse_bvh_and_fk(RAW_ROOT / f"{take}_Skeleton0.bvh")
    grid, legacy_indices = ae.build_slot_grid(ts_by_cam, bvh["n_motion_frames"])

    anchor = CAMS[0]
    anchor_indices = legacy_indices[anchor]
    keep = anchor_indices >= 0
    grid = grid[keep]
    anchor_indices = anchor_indices[keep]
    target_times = ts_by_cam[anchor][anchor_indices]
    if len(target_times) < 10:
        raise RuntimeError(f"{take}: fewer than 10 reference-camera slots")

    legacy_views = np.zeros((len(grid), len(CAMS), 27, 3), dtype=np.float64)
    for ci, camera in enumerate(CAMS):
        idx = legacy_indices[camera][keep]
        present = idx >= 0
        legacy_views[present, ci] = kp_by_cam[camera][idx[present]]

    estimate = None
    estimate_error = None
    try:
        estimate = estimate_camera_offsets(kp_by_cam, ts_by_cam, P, reference_camera=anchor,
                                            search_range_s=0.65, search_step_s=0.002,
                                            min_confidence=0.35, window_s=calibration_seconds,
                                            min_observations=80)
    except ValueError as exc:
        estimate_error = str(exc)

    legacy_t = target_times - target_times[0]
    aligned_t = legacy_t.copy()
    offset_applied = bool(estimate is not None and estimate.reliable)
    aligned_views = legacy_views
    lookahead = None
    if offset_applied:
        aligned_views = resample_keypoints_to_common_times(
            kp_by_cam, ts_by_cam, estimate.offsets_s, target_times,
            min_confidence=0.35,
        )
        lookahead = _resampling_lookahead_ms(ts_by_cam, estimate.offsets_s, target_times)
    tracks = {}
    modes = [("legacy_end_alignment", legacy_views, legacy_t)]
    if offset_applied:
        modes.append(("epipolar_clock_offsets", aligned_views, aligned_t))
    for name, views, t in modes:
        tri_t, raw, cov, reproj = _triangulate(views, t, P)
        filtered, ages, solve_ms = _filter_tracks(tri_t, raw, cov)
        tracks[name] = dict(raw=raw, covariance=cov, reprojection=reproj,
                            filters=filtered, ages=ages, times=t, solve_ms=solve_ms)

    calibration_mask = legacy_t < calibration_seconds
    evaluation_mask = ~calibration_mask
    if int(evaluation_mask.sum()) < 20:
        raise RuntimeError(f"{take}: calibration window leaves too few held-out frames")
    world_transform = _fit_world_transform(tracks["legacy_end_alignment"]["raw"], grid, bvh,
                                            calibration_mask, reference_lag_frames)
    score_grid = grid[evaluation_mask]

    result = dict(take=take, reference_camera=anchor, frames=int(len(grid)),
                  calibration_seconds=float(calibration_seconds),
                  evaluation_frames=int(evaluation_mask.sum()),
                  alignment_applied=offset_applied,
                  alignment_rejection=(estimate_error if estimate is None else estimate.failure_reason),
                  world_transform_fit="fixed rigid transform from legacy baseline calibration window; scale held at 1",
                  camera_offset_estimate=dict(
                      offsets_s=None if estimate is None else estimate.offsets_s,
                      endpoint_alignment_offsets_s={c: float(ts_by_cam[anchor][-1] - ts_by_cam[c][-1])
                                                    for c in CAMS},
                      pairwise=[] if estimate is None else [vars(e) for e in estimate.pairwise],
                      cycle_residual_rms_ms=(None if estimate is None
                                              else estimate.cycle_residual_rms_s * 1000.0),
                      reliable=False if estimate is None else estimate.reliable,
                      failure_reason=estimate_error if estimate is None else estimate.failure_reason,
                      interpolation_lookahead_ms=lookahead),
                  scores={}, diagnostics={})
    for sync_name, data in tracks.items():
        result["scores"][sync_name] = {}
        for filter_name, xyz in data["filters"].items():
            fixed_xyz = _apply_world_transform(xyz, world_transform)
            result["scores"][sync_name][filter_name] = _score(
                fixed_xyz[evaluation_mask], score_grid, bvh, reference_lag_frames)
        result["diagnostics"][sync_name] = dict(
            median_reprojection_px=float(np.nanmedian(data["reprojection"])),
            p95_reprojection_px=float(np.nanpercentile(data["reprojection"], 95)),
            fresh_point_fraction=float(np.isfinite(data["filters"]["alpha_beta"]).all(axis=2).mean()),
            stale_age_p95_ms=float(np.nanpercentile(data["ages"], 95) * 1000.0),
            filter_solve_median_ms=data["solve_ms"],
            alpha_beta_lag_sweep=_diagnostic_lag_sweep(
                _apply_world_transform(data["filters"]["alpha_beta"], world_transform)[evaluation_mask],
                score_grid, bvh),
            kalman_3x3_lag_sweep=_diagnostic_lag_sweep(
                _apply_world_transform(data["filters"]["kalman_3x3"], world_transform)[evaluation_mask],
                score_grid, bvh),
            spherical_kalman_lag_sweep=_diagnostic_lag_sweep(
                _apply_world_transform(data["filters"]["spherical_kalman"], world_transform)[evaluation_mask],
                score_grid, bvh),
            one_euro_lag_sweep=_diagnostic_lag_sweep(
                _apply_world_transform(data["filters"]["one_euro"], world_transform)[evaluation_mask],
                score_grid, bvh))
    return result


def render_summary(results, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    takes = [r["take"] for r in results]
    methods = ["raw", "one_euro", "alpha_beta", "kalman_3x3", "spherical_kalman"]
    colors = ["#3B6FB6", "#D17A22", "#3D9278", "#7A4FB3", "#C44569"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)

    ax = axes[0]
    x = np.arange(len(takes), dtype=float)
    width = 0.15
    for i, (method, color) in enumerate(zip(methods, colors)):
        values = [r["scores"]["legacy_end_alignment"][method]["mpjpe_mm"] for r in results]
        ax.bar(x + (i - 2) * width, values, width, label=method.replace("_", " "), color=color)
    ax.set_xticks(x, takes)
    ax.set_ylabel("Held-out MPJPE (mm)")
    ax.set_title("Temporal filter comparison")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1]
    for i, (method, color) in enumerate(zip(methods, colors)):
        values = [r["scores"]["legacy_end_alignment"][method]["bone_dir_err_deg"] for r in results]
        ax.bar(x + (i - 2) * width, values, width, label=method.replace("_", " "), color=color)
    ax.set_xticks(x, takes)
    ax.set_ylabel("Mean limb direction error (deg)")
    ax.set_title("Limb direction fidelity")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[2]
    for i, (method, color) in enumerate(zip(methods, colors)):
        values = [r["scores"]["legacy_end_alignment"][method]["bone_length_cv_mean"] * 100.0 for r in results]
        ax.bar(x + (i - 2) * width, values, width, label=method.replace("_", " "), color=color)
    ax.set_xticks(x, takes)
    ax.set_ylabel("Mean bone-length CV (%)")
    ax.set_title("Bone-length stability (CV %)")
    ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Timestamp alignment and causal filter evaluation", fontsize=13)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_closed_loop_eval(takes: list[str]):
    import mujoco
    from scipy.spatial.transform import Rotation
    from scripts.retarget_ablation_20261107 import configure_runtime, score_fk, EXPERIMENT
    from g1_imitation.core.constants import load_config
    from g1_imitation.core.s45c_keypoint_ik import KeypointIKRetargeter, causal_upsample
    from g1_imitation.core.s6_mujoco_closed_loop import (
        G1PhysicsPlant,
        MuJoCoClosedLoopVerifier,
        SonicPolicy,
    )

    _, asset_root = configure_runtime()
    full_cfg = load_config()
    rt_cfg = full_cfg["s45c_keypoint_ik"]
    s6_cfg = full_cfg["s6_closed_loop"]
    plant = G1PhysicsPlant(mjcf_path=asset_root / "sonic_official_g1" / "scene_43dof.xml")
    policy = SonicPolicy(s6_cfg[EXPERIMENT["policy"]])
    dt_ctrl = 0.02
    variants = ("one_euro", "raw", "alpha_beta", "kalman_3x3", "spherical_kalman")
    s2s3_dir = ROOT / "outputs" / "advanced_take" / "s2s3" / "rgbonly_lag6"
    metrics_meta = json.loads((s2s3_dir / "metrics.json").read_text(encoding="utf-8"))
    lag = int(metrics_meta["ref_lag_bvh_samples"])

    records: dict[str, dict[str, dict]] = {}
    for take in takes:
        z = np.load(s2s3_dir / f"{take}.npz")
        times = (z["grid"] + lag) / 120.0
        raw_pts = np.asarray(z["raw"], dtype=np.float64)
        cov = np.asarray(z["cov"], dtype=np.float64)
        human = np.asarray(z["bvh12"], dtype=np.float64)
        eval_start = float(times[0] + EXPERIMENT["evaluation"]["exclude_startup_s"])
        filtered_tracks, _, filter_ms = _filter_tracks(times, raw_pts, cov)

        records[take] = {}
        anchor = None
        for variant in variants:
            pts_seq = filtered_tracks[variant]
            rt = KeypointIKRetargeter(cfg=rt_cfg, model=plant.kin_model)
            rows = [rt.step(p, float(t)) for p, t in zip(pts_seq, times)]
            q_slots = np.asarray([x.q for x in rows])
            dq_slots = np.asarray([x.dq for x in rows])
            observed = np.asarray([x.observed for x in rows])
            t50, q, dq, fresh = causal_upsample(
                times, q_slots, dq_slots, observed, "hold", rt.lo, rt.hi, dt=dt_ctrl
            )
            if anchor is None:
                anchor = (
                    q[0, :2].copy(),
                    Rotation.from_quat(q[0, [4, 5, 6, 3]]).as_euler("zyx")[0],
                )
            origin_xy, yaw0 = anchor
            rot = Rotation.from_euler("z", -yaw0)
            q[:, :2] -= origin_xy
            q[:, :3] = rot.apply(q[:, :3])
            q[:, [4, 5, 6, 3]] = (rot * Rotation.from_quat(q[:, [4, 5, 6, 3]])).as_quat()
            dq[:, :3] = rot.apply(dq[:, :3])

            verifier = MuJoCoClosedLoopVerifier(
                mjcf_path=asset_root / "sonic_official_g1" / "scene_43dof.xml",
                policy=policy,
                physics_dt=s6_cfg["physics_dt_s"],
                effort_limit="actuator_range",
            )
            physical = verifier.run_stage_6b_physics(
                q, dq, ik_dt=dt_ctrl, control_dt=dt_ctrl, settle_steps=25, preview_s=EXPERIMENT["preview_s"]
            )
            run_data = {
                "t50": t50,
                "q_ref": q,
                "qpos_phys": physical.qpos_hist,
            }
            score = score_fk(plant, run_data, human, times, eval_start)
            score.update({
                "survived": bool(physical.survived),
                "joint_rmse_rad_vs_own_reference": float(physical.joint_rmse_rad),
                "support_foot_skating_m_s": float(physical.support_foot_skating_speed_ms),
                "max_tilt_deg": float(physical.max_tilt_deg),
                "path_end_error_m": float(
                    np.linalg.norm(physical.qpos_hist[-1, :2] - q[len(physical.qpos_hist) - 1, :2])
                ),
                "ik_solve_p50_ms": float(np.median([x.solve_ms for x in rows if x.observed])),
                "filter_solve_p50_ms": float(filter_ms.get(variant, 0.0)),
                "post_calib_fresh_fraction": float(np.mean(fresh)),
            })
            records[take][variant] = score
            print(f"[closed-loop] {take} {variant}:", json.dumps(score), flush=True)

    summary = {}
    held_out = [t for t in takes if t in ("Take_006", "Take_007", "Take_008")]
    for group_name, group_takes in (("all_takes", takes), ("held_out_006_007_008", held_out)):
        if not group_takes:
            continue
        summary[group_name] = {}
        for variant in variants:
            rows = [records[t][variant] for t in group_takes]
            summary[group_name][variant] = {
                "survived_count": f"{sum(int(r['survived']) for r in rows)}/{len(rows)}",
                "arms_reference_direction_deg": float(np.mean([r["arms_reference_direction_deg"] for r in rows])),
                "arms_physics_direction_deg": float(np.mean([r["arms_physics_direction_deg"] for r in rows])),
                "legs_reference_direction_deg": float(np.mean([r["legs_reference_direction_deg"] for r in rows])),
                "legs_physics_direction_deg": float(np.mean([r["legs_physics_direction_deg"] for r in rows])),
                "support_foot_skating_m_s": float(np.mean([r["support_foot_skating_m_s"] for r in rows])),
                "path_end_error_m": float(np.mean([r["path_end_error_m"] for r in rows])),
                "joint_rmse_rad_vs_own_reference": float(np.mean([r["joint_rmse_rad_vs_own_reference"] for r in rows])),
                "max_tilt_deg_mean": float(np.mean([r["max_tilt_deg"] for r in rows])),
            }

    out_doc = {
        "notes": "Identical KeypointIKRetargeter + official G1 scene_43dof.xml + SONIC v1.1 (preview=0); only S3 filter varies.",
        "summary": summary,
        "per_take": records,
    }
    out_path = OUT_ROOT / "closed_loop_report.json"
    out_path.write_text(json.dumps(out_doc, indent=2), encoding="utf-8")
    render_closed_loop_summary(records, OUT_ROOT / "closed_loop_summary.png")
    print(f"wrote {out_path}")


def run_ik_branch_eval(takes: list[str], candidate_count: int = 5,
                       switch_penalty: float | None = None, switch_margin: float | None = None,
                       pelvis_pitch_scale: float | None = None):
    """Compare warm-start-only IK against opt-in multistart branch selection on the main S3 Kalman tracks."""
    import mujoco
    from scipy.spatial.transform import Rotation
    from scripts.retarget_ablation_20261107 import configure_runtime, score_fk, EXPERIMENT
    from g1_imitation.core.constants import load_config
    from g1_imitation.core.s45c_keypoint_ik import KeypointIKRetargeter, causal_upsample
    from g1_imitation.core.s6_mujoco_closed_loop import G1PhysicsPlant, MuJoCoClosedLoopVerifier, SonicPolicy

    if candidate_count < 1:
        raise ValueError("candidate_count must be >= 1")
    _, asset_root = configure_runtime()
    full_cfg = load_config()
    base_cfg = dict(full_cfg["s45c_keypoint_ik"])
    s6_cfg = full_cfg["s6_closed_loop"]
    plant = G1PhysicsPlant(mjcf_path=asset_root / "sonic_official_g1" / "scene_43dof.xml")
    policy = SonicPolicy(s6_cfg[EXPERIMENT["policy"]])
    dt_ctrl = 0.02
    s2s3_dir = ROOT / "outputs" / "advanced_take" / "s2s3" / "selfcal_lag6"
    metrics_meta = json.loads((s2s3_dir / "metrics.json").read_text(encoding="utf-8"))
    lag = int(metrics_meta["ref_lag_bvh_samples"])
    records = {}

    for take in takes:
        z = np.load(s2s3_dir / f"{take}.npz")
        times = (z["grid"] + lag) / 120.0
        points = np.asarray(z["smooth"], dtype=np.float64)
        human = np.asarray(z["bvh12"], dtype=np.float64)
        eval_start = float(times[0] + EXPERIMENT["evaluation"]["exclude_startup_s"])
        records[take] = {}
        shared_anchor = None

        variants = [("warm_start", 1, None)]
        if candidate_count > 1:
            variants.append(("branch_search", candidate_count, None))
        if pelvis_pitch_scale is not None:
            variants.append(("pelvis_pitch_weak", 1, [1.0, float(pelvis_pitch_scale), 1.0]))
        for label, count, pelvis_axis_scale in variants:
            rt_cfg = {**base_cfg, "ik_branch_candidates": count}
            if switch_penalty is not None:
                rt_cfg["ik_branch_switch_penalty"] = float(switch_penalty)
            if switch_margin is not None:
                rt_cfg["ik_branch_switch_margin"] = float(switch_margin)
            if pelvis_axis_scale is not None:
                rt_cfg["pelvis_orientation_axis_scale"] = pelvis_axis_scale
            rt = KeypointIKRetargeter(cfg=rt_cfg, model=plant.kin_model)
            rows = [rt.step(p, float(t)) for p, t in zip(points, times)]
            q_slots = np.asarray([x.q for x in rows])
            dq_slots = np.asarray([x.dq for x in rows])
            observed = np.asarray([x.observed for x in rows])
            t50, q, dq, fresh = causal_upsample(times, q_slots, dq_slots, observed, "hold", rt.lo, rt.hi, dt=dt_ctrl)
            if shared_anchor is None:
                shared_anchor = (
                    q[0, :2].copy(),
                    Rotation.from_quat(q[0, [4, 5, 6, 3]]).as_euler("zyx")[0],
                )
            origin_xy, yaw0 = shared_anchor
            rot = Rotation.from_euler("z", -yaw0)
            q[:, :2] -= origin_xy
            q[:, :3] = rot.apply(q[:, :3])
            q[:, [4, 5, 6, 3]] = (rot * Rotation.from_quat(q[:, [4, 5, 6, 3]])).as_quat()
            dq[:, :3] = rot.apply(dq[:, :3])

            verifier = MuJoCoClosedLoopVerifier(
                mjcf_path=asset_root / "sonic_official_g1" / "scene_43dof.xml",
                policy=policy,
                physics_dt=s6_cfg["physics_dt_s"],
                effort_limit="actuator_range",
            )
            physical = verifier.run_stage_6b_physics(
                q, dq, ik_dt=dt_ctrl, control_dt=dt_ctrl, settle_steps=25, preview_s=EXPERIMENT["preview_s"]
            )
            score = score_fk(plant, {"t50": t50, "q_ref": q, "qpos_phys": physical.qpos_hist}, human, times, eval_start)
            labels = [r.ik_branch for r in rows if r.observed]
            branch_counts = {name: labels.count(name) for name in sorted(set(labels))}
            score.update({
                "survived": bool(physical.survived),
                "joint_rmse_rad_vs_own_reference": float(physical.joint_rmse_rad),
                "support_foot_skating_m_s": float(physical.support_foot_skating_speed_ms),
                "max_tilt_deg": float(physical.max_tilt_deg),
                "path_end_error_m": float(np.linalg.norm(physical.qpos_hist[-1, :2] - q[len(physical.qpos_hist)-1, :2])),
                "ik_solve_p50_ms": float(np.median([r.solve_ms for r in rows if r.observed])),
                "ik_solve_p95_ms": float(np.percentile([r.solve_ms for r in rows if r.observed], 95)),
                "branch_counts": branch_counts,
                "branch_switches": int(sum(a != b for a, b in zip(labels, labels[1:]))),
                "mean_task_cost": float(np.mean([r.ik_task_cost for r in rows if r.observed])),
                "mean_continuity_cost": float(np.mean([r.ik_continuity_cost for r in rows if r.observed])),
                "post_calib_fresh_fraction": float(np.mean(fresh)),
                "branch_candidates_configured": count,
            })
            records[take][label] = score
            print(f"[ik-branch] {take} {label}:", json.dumps(score), flush=True)

    summary = {}
    held_out = [t for t in takes if t in ("Take_006", "Take_007", "Take_008")]
    for group_name, group_takes in (("all_takes", takes), ("held_out_006_007_008", held_out)):
        if not group_takes:
            continue
        summary[group_name] = {}
        labels = ["warm_start"]
        if candidate_count > 1:
            labels.append("branch_search")
        if pelvis_pitch_scale is not None:
            labels.append("pelvis_pitch_weak")
        for label in labels:
            rows = [records[t][label] for t in group_takes]
            summary[group_name][label] = {
                "survived_count": f"{sum(int(r['survived']) for r in rows)}/{len(rows)}",
                "arms_reference_direction_deg": float(np.mean([r["arms_reference_direction_deg"] for r in rows])),
                "arms_physics_direction_deg": float(np.mean([r["arms_physics_direction_deg"] for r in rows])),
                "legs_reference_direction_deg": float(np.mean([r["legs_reference_direction_deg"] for r in rows])),
                "legs_physics_direction_deg": float(np.mean([r["legs_physics_direction_deg"] for r in rows])),
                "support_foot_skating_m_s": float(np.mean([r["support_foot_skating_m_s"] for r in rows])),
                "path_end_error_m": float(np.mean([r["path_end_error_m"] for r in rows])),
                "joint_rmse_rad_vs_own_reference": float(np.mean([r["joint_rmse_rad_vs_own_reference"] for r in rows])),
                "max_tilt_deg_mean": float(np.mean([r["max_tilt_deg"] for r in rows])),
                "ik_solve_p50_ms": float(np.mean([r["ik_solve_p50_ms"] for r in rows])),
                "ik_solve_p95_ms_max": float(np.max([r["ik_solve_p95_ms"] for r in rows])),
            }

    doc = {
        "notes": "Same S3 kalman_3x3 tracks, KeypointIKRetargeter, official G1 scene_43dof.xml and SONIC v1.1. Only IK warm-start candidate count differs.",
        "candidate_count": candidate_count,
        "switch_penalty": switch_penalty,
        "switch_margin": switch_margin,
        "pelvis_pitch_scale": pelvis_pitch_scale,
        "summary": summary,
        "per_take": records,
    }
    out_path = OUT_ROOT / "ik_branch_report.json"
    out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    render_ik_branch_summary(records, OUT_ROOT / "ik_branch_summary.png")
    print(f"wrote {out_path}")


def render_ik_branch_summary(records: dict[str, dict[str, dict]], path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    takes = list(records.keys())
    labels = ["warm_start"]
    if any("branch_search" in r for r in records.values()):
        labels.append("branch_search")
    if any("pelvis_pitch_weak" in r for r in records.values()):
        labels.append("pelvis_pitch_weak")
    colors = ["#4E79A7", "#E15759", "#59A14F"]
    fields = (
        ("arms_reference_direction_deg", "Reference arm direction (deg)"),
        ("arms_physics_direction_deg", "Physical arm direction (deg)"),
        ("legs_physics_direction_deg", "Physical leg direction (deg)"),
        ("support_foot_skating_m_s", "Support foot skating (m/s)"),
        ("path_end_error_m", "Path end error (m)"),
        ("ik_solve_p50_ms", "IK solve p50 (ms)"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    x = np.arange(len(takes), dtype=float)
    width = 0.8 / len(labels)
    for ax, (field, title) in zip(axes.flat, fields):
        for i, (label, color) in enumerate(zip(labels, colors)):
            vals = [records[t][label][field] for t in takes]
            ax.bar(x + (i - (len(labels) - 1) / 2) * width, vals, width,
                   label=label.replace("_", " "), color=color)
        ax.set_xticks(x, [t.replace("Take_", "") for t in takes])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes.flat[0].legend(frameon=False)
    title = (
        "Pelvis Pitch Observability: Full vs Weak Local-Y Constraint"
        if "pelvis_pitch_weak" in labels and "branch_search" not in labels
        else "IK Branch Selection: Warm Start vs Multi-Start Candidates"
    )
    fig.suptitle(title, fontsize=13)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def render_closed_loop_summary(records: dict[str, dict[str, dict]], path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    takes = list(records.keys())
    variants = ["one_euro", "raw", "alpha_beta", "kalman_3x3", "spherical_kalman"]
    colors = ["#D17A22", "#3B6FB6", "#3D9278", "#7A4FB3", "#C44569"]
    fields = (
        ("arms_reference_direction_deg", "Ref arm dir error (deg)"),
        ("arms_physics_direction_deg", "Phys arm dir error (deg)"),
        ("legs_reference_direction_deg", "Ref leg dir error (deg)"),
        ("legs_physics_direction_deg", "Phys leg dir error (deg)"),
        ("support_foot_skating_m_s", "Support foot skating (m/s)"),
        ("path_end_error_m", "Path end error vs ref (m)"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), constrained_layout=True)
    x = np.arange(len(takes), dtype=float)
    width = 0.15
    for ax, (field, title) in zip(axes.flat, fields):
        for i, (variant, color) in enumerate(zip(variants, colors)):
            vals = [records[t][variant][field] for t in takes]
            ax.bar(x + (i - 2) * width, vals, width, label=variant.replace("_", " "), color=color)
        ax.set_xticks(x, [t.replace("Take_", "") for t in takes])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle("G1 Closed-Loop (S45c Keypoint IK + SONIC v1.1): S3 Filter Comparison", fontsize=13)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--takes", nargs="+", default=["Take_006", "Take_007", "Take_008"])
    parser.add_argument("--calibration-seconds", type=float, default=2.0)
    parser.add_argument("--reference-lag-frames", type=int, default=6,
                        help="fixed BVH scoring lag; never fit separately for methods")
    parser.add_argument("--closed-loop", action="store_true",
                        help="also run 5-take KeypointIK + SONIC v1.1 closed-loop comparison")
    parser.add_argument("--closed-loop-takes", nargs="+",
                        default=["Take_002", "Take_006", "Take_007", "Take_008", "Take_012"])
    parser.add_argument("--ik-branch-eval", action="store_true",
                        help="compare warm-start IK against candidate-branch selection using main Kalman S3 tracks")
    parser.add_argument("--ik-branch-candidates", type=int, default=5)
    parser.add_argument("--ik-branch-switch-penalty", type=float, default=None)
    parser.add_argument("--ik-branch-switch-margin", type=float, default=None)
    parser.add_argument("--ik-pelvis-pitch-scale", type=float, default=None,
                        help="opt-in local pelvis-y orientation scale ablation")
    parser.add_argument("--ik-branch-takes", nargs="+",
                        default=["Take_002", "Take_006", "Take_007", "Take_008", "Take_012"])
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for take in args.takes:
        row = run_take(take, args.calibration_seconds, args.reference_lag_frames)
        results.append(row)
        print(take, json.dumps(row["camera_offset_estimate"]["offsets_s"], sort_keys=True),
              json.dumps(row["scores"], sort_keys=True), flush=True)
    output = dict(method=("2D epipolar offset estimation; BVH used for the common legacy frame mapping, "
                          "initial rigid world calibration, and held-out scoring"),
                  results=results)
    path = OUT_ROOT / "report.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    render_summary(results, OUT_ROOT / "comparison.png")
    print(f"wrote {path}")
    if args.closed_loop:
        run_closed_loop_eval(args.closed_loop_takes)
    if args.ik_branch_eval:
        run_ik_branch_eval(args.ik_branch_takes, candidate_count=args.ik_branch_candidates,
                           switch_penalty=args.ik_branch_switch_penalty,
                           switch_margin=args.ik_branch_switch_margin,
                           pelvis_pitch_scale=args.ik_pelvis_pitch_scale)


if __name__ == "__main__":
    main()
