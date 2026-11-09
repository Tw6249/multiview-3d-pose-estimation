"""Timestamp alignment helpers for asynchronous multi-view keypoint streams.

Clock offsets are estimated from calibrated epipolar consistency only. The
absolute time origin remains arbitrary; the reference camera is fixed at zero.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairwiseOffsetEstimate:
    camera_a: str
    camera_b: str
    offset_b_minus_a_s: float
    median_sampson_error_px: float
    contrast_px: float
    observations: int
    frames: int


@dataclass(frozen=True)
class CameraOffsetEstimate:
    offsets_s: dict[str, float]
    pairwise: tuple[PairwiseOffsetEstimate, ...]
    reference_camera: str
    cycle_residual_rms_s: float
    reliable: bool
    failure_reason: str | None


def fundamental_from_projections(P_a: np.ndarray, P_b: np.ndarray) -> np.ndarray:
    """Return F such that x_b.T @ F @ x_a = 0 for two 3x4 camera matrices."""
    P_a = np.asarray(P_a, dtype=np.float64)
    P_b = np.asarray(P_b, dtype=np.float64)
    if P_a.shape != (3, 4) or P_b.shape != (3, 4):
        raise ValueError("projection matrices must both have shape (3,4)")
    _, _, vt = np.linalg.svd(P_a)
    C_a = vt[-1]
    if abs(C_a[3]) < 1e-12:
        raise ValueError("camera center is at infinity")
    C_a /= C_a[3]
    e_b = P_b @ C_a
    ex = np.array([[0.0, -e_b[2], e_b[1]],
                   [e_b[2], 0.0, -e_b[0]],
                   [-e_b[1], e_b[0], 0.0]])
    F = ex @ P_b @ np.linalg.pinv(P_a)
    norm = np.linalg.norm(F[:2, :2])
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("degenerate camera pair")
    return F / norm


def _validate_streams(keypoints_by_camera, timestamps_by_camera, projections_by_camera):
    cameras = list(keypoints_by_camera)
    if len(cameras) < 2 or set(cameras) != set(timestamps_by_camera) or set(cameras) != set(projections_by_camera):
        raise ValueError("keypoints, timestamps and projections must have the same >=2 camera keys")
    kp_out, ts_out, P_out = {}, {}, {}
    n_joints = None
    for camera in cameras:
        kp = np.asarray(keypoints_by_camera[camera], dtype=np.float64)
        ts = np.asarray(timestamps_by_camera[camera], dtype=np.float64)
        P = np.asarray(projections_by_camera[camera], dtype=np.float64)
        if kp.ndim != 3 or kp.shape[2] != 3 or ts.shape != (len(kp),) or P.shape != (3, 4):
            raise ValueError(f"invalid stream shapes for camera {camera!r}")
        if len(ts) < 2 or not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
            raise ValueError(f"timestamps for camera {camera!r} must be finite and strictly increasing")
        if n_joints is None:
            n_joints = kp.shape[1]
        elif kp.shape[1] != n_joints:
            raise ValueError("all cameras must have the same joint count")
        kp_out[camera], ts_out[camera], P_out[camera] = kp, ts, P
    return cameras, kp_out, ts_out, P_out


def _interpolate_2d(ts: np.ndarray, kp: np.ndarray, query_s: np.ndarray, min_confidence: float,
                    max_gap_s: float) -> tuple[np.ndarray, np.ndarray]:
    n_query, n_joints = len(query_s), kp.shape[1]
    uv = np.full((n_query, n_joints, 2), np.nan, dtype=np.float64)
    conf = np.zeros((n_query, n_joints), dtype=np.float64)
    right = np.searchsorted(ts, query_s, side="left")
    exact = (right < len(ts)) & (np.abs(ts[np.minimum(right, len(ts) - 1)] - query_s) <= 1e-8)
    exact_rows = np.flatnonzero(exact)
    if len(exact_rows):
        src = right[exact_rows]
        valid = (kp[src, :, 2] >= min_confidence) & np.isfinite(kp[src, :, :2]).all(axis=2)
        uv[exact_rows] = np.where(valid[..., None], kp[src, :, :2], np.nan)
        conf[exact_rows] = np.where(valid, kp[src, :, 2], 0.0)

    rows = np.flatnonzero(~exact & (right > 0) & (right < len(ts)))
    if len(rows):
        hi = right[rows]
        lo = hi - 1
        gaps = ts[hi] - ts[lo]
        allowed = (gaps > 0.0) & (gaps <= max_gap_s)
        rows, lo, hi, gaps = rows[allowed], lo[allowed], hi[allowed], gaps[allowed]
        if len(rows):
            valid = ((kp[lo, :, 2] >= min_confidence) & (kp[hi, :, 2] >= min_confidence)
                     & np.isfinite(kp[lo, :, :2]).all(axis=2) & np.isfinite(kp[hi, :, :2]).all(axis=2))
            w = ((query_s[rows] - ts[lo]) / gaps)[:, None, None]
            interp = kp[lo, :, :2] * (1.0 - w) + kp[hi, :, :2] * w
            uv[rows] = np.where(valid[..., None], interp, np.nan)
            conf[rows] = np.where(valid, np.minimum(kp[lo, :, 2], kp[hi, :, 2]), 0.0)
    return uv, conf


def _sampson_distance(F: np.ndarray, xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
    xa_h = np.concatenate([xa, np.ones((*xa.shape[:-1], 1))], axis=-1)
    xb_h = np.concatenate([xb, np.ones((*xb.shape[:-1], 1))], axis=-1)
    Fx = np.einsum("ij,...j->...i", F, xa_h)
    Ftx = np.einsum("ji,...j->...i", F, xb_h)
    num = np.sum(xb_h * Fx, axis=-1)
    den = np.sum(Fx[..., :2] ** 2, axis=-1) + np.sum(Ftx[..., :2] ** 2, axis=-1)
    return np.abs(num) / np.sqrt(np.maximum(den, 1e-12))


def _estimate_pair(camera_a, camera_b, kp_a, kp_b, ts_a, ts_b, P_a, P_b, deltas,
                   min_confidence, max_gap_s, window_s, min_observations):
    t0 = max(float(ts_a[0]), float(ts_b[0]) + float(deltas[-1]))
    t1 = min(float(ts_a[-1]), float(ts_b[-1]) + float(deltas[0]))
    if window_s is not None:
        t1 = min(t1, t0 + float(window_s))
    ta = ts_a[(ts_a >= t0) & (ts_a <= t1)]
    if len(ta) < 4:
        raise ValueError(f"insufficient temporal overlap for {camera_a}/{camera_b}")

    F = fundamental_from_projections(P_a, P_b)
    cost = np.full(len(deltas), np.inf, dtype=np.float64)
    counts = np.zeros(len(deltas), dtype=np.int64)
    frame_counts = np.zeros(len(deltas), dtype=np.int64)
    xa = kp_a[np.searchsorted(ts_a, ta), :, :2]
    ca = kp_a[np.searchsorted(ts_a, ta), :, 2]
    va = (ca >= min_confidence) & np.isfinite(xa).all(axis=2)
    for di, delta in enumerate(deltas):
        xb, cb = _interpolate_2d(ts_b, kp_b, ta - delta, min_confidence, max_gap_s)
        valid = va & (cb >= min_confidence) & np.isfinite(xb).all(axis=2)
        if int(valid.sum()) < min_observations:
            continue
        dist = _sampson_distance(F, xa, xb)
        per_frame = []
        for row in range(len(ta)):
            e = dist[row, valid[row]]
            if len(e) >= 3:
                per_frame.append(float(np.median(np.minimum(e, 20.0))))
        if len(per_frame) < 4:
            continue
        per_frame = np.asarray(per_frame)
        cost[di] = float(np.median(per_frame) + 0.25 * np.percentile(per_frame, 75))
        counts[di] = int(valid.sum())
        frame_counts[di] = len(per_frame)
    if not np.isfinite(cost).any():
        raise ValueError(f"insufficient valid epipolar observations for {camera_a}/{camera_b}")

    best = int(np.nanargmin(cost))
    delta = float(deltas[best])
    step = float(np.median(np.diff(deltas)))
    # Contrast to alternatives at least 1.5 frames away; a flat objective is ambiguous.
    dt_nom = float(np.median(np.diff(ts_a)))
    alternatives = np.abs(deltas - delta) >= max(1.5 * dt_nom, 3.0 * step)
    alternatives &= np.isfinite(cost)
    contrast = float(np.median(cost[alternatives]) - cost[best]) if alternatives.any() else 0.0
    if best < 2 or best >= len(deltas) - 2:
        raise ValueError(f"offset optimum lies at search boundary for {camera_a}/{camera_b}")
    if contrast < 0.10:
        raise ValueError(f"relative offset is not observable for {camera_a}/{camera_b} (flat epipolar cost)")
    return PairwiseOffsetEstimate(camera_a, camera_b, delta, float(cost[best]), contrast,
                                  int(counts[best]), int(frame_counts[best]))


def estimate_camera_offsets(keypoints_by_camera: dict[str, np.ndarray], timestamps_by_camera: dict[str, np.ndarray],
                            projections_by_camera: dict[str, np.ndarray], *, reference_camera: str | None = None,
                            search_range_s: float = 0.12, search_step_s: float = 0.002,
                            min_confidence: float = 0.35, max_interp_gap_s: float | None = None,
                            window_s: float | None = 2.0, min_observations: int = 80) -> CameraOffsetEstimate:
    """Estimate relative camera clock offsets using only calibrated 2D geometry.

    `offsets_s[c]` maps a stream-local timestamp to the reference clock by
    `t_common = t_local + offsets_s[c]`. The reference offset is exactly zero.
    A flat or boundary-seeking epipolar objective is rejected as unobservable.
    """
    cameras, kp, ts, P = _validate_streams(keypoints_by_camera, timestamps_by_camera, projections_by_camera)
    if reference_camera is None:
        reference_camera = cameras[0]
    if reference_camera not in cameras:
        raise ValueError("reference_camera is not present in the input streams")
    if search_range_s <= 0.0 or search_step_s <= 0.0 or min_observations < 1:
        raise ValueError("search range, step and minimum observations must be positive")
    deltas = np.arange(-search_range_s, search_range_s + search_step_s * 0.5, search_step_s)
    if max_interp_gap_s is None:
        max_interp_gap_s = 1.6 * max(float(np.median(np.diff(ts[c]))) for c in cameras)

    pairwise = []
    for i, ca in enumerate(cameras):
        for cb in cameras[i + 1:]:
            pairwise.append(_estimate_pair(ca, cb, kp[ca], kp[cb], ts[ca], ts[cb], P[ca], P[cb], deltas,
                                           min_confidence, max_interp_gap_s, window_s, min_observations))

    free = [c for c in cameras if c != reference_camera]
    col = {c: i for i, c in enumerate(free)}
    A = np.zeros((len(pairwise), len(free)), dtype=np.float64)
    y = np.empty(len(pairwise), dtype=np.float64)
    base_w = np.empty(len(pairwise), dtype=np.float64)
    for row, edge in enumerate(pairwise):
        if edge.camera_a != reference_camera:
            A[row, col[edge.camera_a]] -= 1.0
        if edge.camera_b != reference_camera:
            A[row, col[edge.camera_b]] += 1.0
        y[row] = edge.offset_b_minus_a_s
        base_w[row] = min(25.0, max(0.2, edge.contrast_px))
    if np.linalg.matrix_rank(A) < len(free):
        raise ValueError("camera offset graph is disconnected from the reference camera")

    w = base_w.copy()
    x = np.zeros(len(free), dtype=np.float64)
    for _ in range(8):
        Aw, yw = A * np.sqrt(w[:, None]), y * np.sqrt(w)
        x, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
        residual = A @ x - y
        scale = max(0.001, 1.4826 * float(np.median(np.abs(residual - np.median(residual)))))
        huber = np.minimum(1.0, (1.5 * scale) / np.maximum(np.abs(residual), 1e-12))
        w = base_w * huber
    residual = A @ x - y
    offsets = {reference_camera: 0.0, **{c: float(x[col[c]]) for c in free}}
    residual_rms_s = float(np.sqrt(np.mean(residual ** 2)))
    frame_period = float(np.median([np.median(np.diff(ts[c])) for c in cameras]))
    cycle_limit_s = max(3.0 * search_step_s, 0.5 * frame_period)
    reliable = residual_rms_s <= cycle_limit_s
    reason = None if reliable else (
        f"pairwise clock offsets disagree: cycle residual {residual_rms_s * 1000:.1f} ms "
        f"exceeds {cycle_limit_s * 1000:.1f} ms"
    )
    return CameraOffsetEstimate(offsets, tuple(pairwise), reference_camera, residual_rms_s, reliable, reason)


def resample_keypoints_to_common_times(keypoints_by_camera: dict[str, np.ndarray],
                                       timestamps_by_camera: dict[str, np.ndarray],
                                       offsets_s: dict[str, float], target_times_s: np.ndarray, *,
                                       min_confidence: float = 0.35,
                                       max_interp_gap_s: float | None = None) -> np.ndarray:
    """Resample 2D streams onto common times, returning (T,C,J,3); invalid points stay confidence 0.

    Linear interpolation can require a future camera frame. Callers must include
    that wait in the latency budget; this helper does not claim causal operation.
    """
    cameras = list(keypoints_by_camera)
    if len(cameras) < 2 or set(cameras) != set(timestamps_by_camera) or set(cameras) != set(offsets_s):
        raise ValueError("offsets must contain exactly the input camera keys")
    if not all(np.isfinite(float(v)) for v in offsets_s.values()):
        raise ValueError("camera offsets must be finite")
    kp, ts = {}, {}
    n_joints = None
    for camera in cameras:
        kp_c = np.asarray(keypoints_by_camera[camera], dtype=np.float64)
        ts_c = np.asarray(timestamps_by_camera[camera], dtype=np.float64)
        if kp_c.ndim != 3 or kp_c.shape[2] != 3 or ts_c.shape != (len(kp_c),):
            raise ValueError(f"invalid stream shapes for camera {camera!r}")
        if len(ts_c) < 2 or not np.isfinite(ts_c).all() or np.any(np.diff(ts_c) <= 0.0):
            raise ValueError(f"timestamps for camera {camera!r} must be finite and strictly increasing")
        if n_joints is None:
            n_joints = kp_c.shape[1]
        elif kp_c.shape[1] != n_joints:
            raise ValueError("all cameras must have the same joint count")
        kp[camera], ts[camera] = kp_c, ts_c
    target = np.asarray(target_times_s, dtype=np.float64)
    if target.ndim != 1 or not np.isfinite(target).all() or (len(target) > 1 and np.any(np.diff(target) <= 0)):
        raise ValueError("target times must be a finite, strictly increasing vector")
    out = np.zeros((len(target), len(cameras), n_joints, 3), dtype=np.float64)
    for ci, camera in enumerate(cameras):
        if max_interp_gap_s is None:
            gap = 1.6 * float(np.median(np.diff(ts[camera])))
        else:
            gap = float(max_interp_gap_s)
        uv, conf = _interpolate_2d(ts[camera], kp[camera], target - float(offsets_s[camera]),
                                   min_confidence, gap)
        out[:, ci, :, :2] = np.nan_to_num(uv, nan=0.0)
        out[:, ci, :, 2] = conf
    return out
