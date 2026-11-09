"""Evaluation helpers for comparing vision skeletons with a BVH-derived reference.

Skeleton layout (12 body joints, shared by vision and BVH):
  order = [L/R shoulder, L/R elbow, L/R wrist, L/R hip, L/R knee, L/R ankle]
  == COCO indices 5..16 == BVH nodes LeftArm, RightArm, LeftForeArm, ... LeftFoot, RightFoot.
Units: metres, degrees, seconds. World = BVH template-aligned Z-up frame (room registration unverified).
Every function is pure; NaN means "not observed" and is never imputed or clamped here.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

JOINTS12 = ["l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
            "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
J = {n: i for i, n in enumerate(JOINTS12)}
BVH12 = ["LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "LeftHand", "RightHand",
         "LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg", "LeftFoot", "RightFoot"]

# (name, proximal, distal) limb segments whose direction is compared
SEGMENTS = [("l_upper_arm", "l_shoulder", "l_elbow"), ("r_upper_arm", "r_shoulder", "r_elbow"),
            ("l_forearm", "l_elbow", "l_wrist"), ("r_forearm", "r_elbow", "r_wrist"),
            ("l_thigh", "l_hip", "l_knee"), ("r_thigh", "r_hip", "r_knee"),
            ("l_shank", "l_knee", "l_ankle"), ("r_shank", "r_knee", "r_ankle")]
BONES = [(n, a, b) for n, a, b in SEGMENTS]

# interior joint angle = angle at `mid` between (mid->a) and (mid->b)
ANGLES = [("l_knee_flex", "l_hip", "l_knee", "l_ankle"), ("r_knee_flex", "r_hip", "r_knee", "r_ankle"),
          ("l_elbow_flex", "l_shoulder", "l_elbow", "l_wrist"), ("r_elbow_flex", "r_shoulder", "r_elbow", "r_wrist")]


def end_aligned_index(pts_s: float, n_motion: int, n_container: int, ratio: int = 4, fps: float = 30.0):
    """Legacy provisional clock hypothesis: last video frame <-> last BVH frame, 4 BVH samples per video frame.

    Returns None outside the BVH range (never clamped). NOT a verified hardware clock mapping.
    """
    if not np.isfinite(pts_s) or pts_s < 0 or min(n_motion, n_container) < 1:
        raise ValueError("invalid timestamp or sample count")
    sample = int(np.floor(pts_s * fps + 0.5))
    b = n_motion - 1 - ratio * (n_container - 1 - sample)
    return b if 0 <= b < n_motion else None


def build_slot_grid(pts_s_by_cam: dict, n_motion: int, shift_slots: dict | None = None):
    """30 Hz slot grid on BVH indices. Returns (bvh_idx[n_slots], {cam: frame_idx[n_slots] or -1}).

    `shift_slots[cam]` moves that camera's frames by whole 30 Hz slots (diagnostic only, default 0).
    A slot keeps every expected time even when a camera has no frame there (-1).
    """
    grid = np.arange((n_motion - 1) % 4, n_motion, 4)
    pos = {int(b): k for k, b in enumerate(grid)}
    out = {}
    for cam, pts in pts_s_by_cam.items():
        idx = np.full(len(grid), -1, dtype=np.int64)
        sh = 0 if not shift_slots else int(shift_slots.get(cam, 0))
        for i, t in enumerate(pts):
            b = end_aligned_index(float(t), n_motion, len(pts))
            if b is None:
                continue
            k = pos[b] + sh
            if 0 <= k < len(grid):
                if idx[k] != -1:
                    raise ValueError("duplicate PTS slot")
                idx[k] = i
        out[cam] = idx
    return grid, out


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True, weights: np.ndarray | None = None):
    """Least-squares similarity dst ~ s R src + t. Returns (s, R, t); R in SO(3)."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    w = np.ones(len(src)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    mu_s = (w[:, None] * src).sum(0)
    mu_d = (w[:, None] * dst).sum(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = (w[:, None, None] * (xd[:, :, None] * xs[:, None, :])).sum(0)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1.0
    R = U @ D @ Vt
    var_s = (w * (xs ** 2).sum(1)).sum()
    s = float(np.trace(np.diag(S) @ D) / var_s) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def robust_umeyama(src, dst, iters: int = 8, trim: float = 0.8, with_scale: bool = True):
    """Umeyama refit on the best `trim` fraction of residuals (robust to a few bad triangulations)."""
    keep = np.ones(len(src), bool)
    for _ in range(iters):
        s, R, t = umeyama(src[keep], dst[keep], with_scale)
        r = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
        thr = np.quantile(r, trim)
        new = r <= thr
        if (new == keep).all():
            break
        keep = new
    return s, R, t


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def segment_directions(p: np.ndarray) -> dict:
    """p: (T,12,3). Unit vectors proximal->distal for limbs, plus pelvis/shoulder axes and spine."""
    d = {n: _unit(p[:, J[b]] - p[:, J[a]]) for n, a, b in SEGMENTS}
    d["pelvis_axis"] = _unit(p[:, J["l_hip"]] - p[:, J["r_hip"]])
    d["shoulder_axis"] = _unit(p[:, J["l_shoulder"]] - p[:, J["r_shoulder"]])
    mid_h = 0.5 * (p[:, J["l_hip"]] + p[:, J["r_hip"]])
    mid_s = 0.5 * (p[:, J["l_shoulder"]] + p[:, J["r_shoulder"]])
    d["spine"] = _unit(mid_s - mid_h)
    return d


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    c = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(c))


def interior_angles(p: np.ndarray) -> dict:
    out = {}
    for n, a, m, b in ANGLES:
        out[n] = angle_between_deg(_unit(p[:, J[a]] - p[:, J[m]]), _unit(p[:, J[b]] - p[:, J[m]]))
    # trunk-relative hip / shoulder abduction-flexion magnitude
    d = segment_directions(p)
    out["l_hip_flex"] = angle_between_deg(d["spine"], -d["l_thigh"])
    out["r_hip_flex"] = angle_between_deg(d["spine"], -d["r_thigh"])
    out["l_shoulder_flex"] = angle_between_deg(-d["spine"], d["l_upper_arm"])
    out["r_shoulder_flex"] = angle_between_deg(-d["spine"], d["r_upper_arm"])
    return out


def bone_lengths(p: np.ndarray) -> dict:
    return {n: np.linalg.norm(p[:, J[b]] - p[:, J[a]], axis=1) for n, a, b in BONES}


def rotation_only_align(dir_src: np.ndarray, dir_dst: np.ndarray):
    """Single rotation R minimising sum |R src - dst|^2 over stacked unit vectors (Kabsch)."""
    s = dir_src.reshape(-1, 3)
    d = dir_dst.reshape(-1, 3)
    ok = np.isfinite(s).all(1) & np.isfinite(d).all(1)
    H = s[ok].T @ d[ok]
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.linalg.det(Vt.T @ U.T)])
    return Vt.T @ D @ U.T


def jitter_rms(e: np.ndarray) -> float:
    """RMS of frame-to-frame change of an error trajectory (T,K,3) in metres; NaNs skipped."""
    de = np.diff(e, axis=0)
    de = de[np.isfinite(de).all(-1)]
    return float(np.sqrt(np.mean(np.sum(de ** 2, axis=-1)))) if len(de) else float("nan")


def anchor_heading(q: np.ndarray, dq: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Re-express the reference root trajectory in the frame of its first sample's heading and ground position."""
    yaw0 = Rotation.from_quat(q[0, [4, 5, 6, 3]]).as_euler("zyx")[0]
    Rz = Rotation.from_euler("z", -yaw0)
    q, dq = q.copy(), dq.copy()
    q[:, :2] -= q[0, :2]
    q[:, :3] = Rz.apply(q[:, :3])
    q[:, [4, 5, 6, 3]] = (Rz * Rotation.from_quat(q[:, [4, 5, 6, 3]])).as_quat()
    dq[:, :3] = Rz.apply(dq[:, :3])
    return q, dq, float(np.degrees(yaw0))


def local_path(xy: np.ndarray, yaw0: float) -> np.ndarray:
    """Project 2D root XY path into the initial heading frame."""
    return Rotation.from_euler("z", -yaw0).apply(np.c_[xy - xy[0], np.zeros(len(xy))])[:, :2]


def yaw(qwxyz: np.ndarray) -> float:
    """Extract planar heading angle (rad) from wxyz quaternion."""
    f = Rotation.from_quat(qwxyz[[1, 2, 3, 0]]).apply([1.0, 0.0, 0.0])
    return float(np.arctan2(f[1], f[0]))


def to_50hz(t_s: np.ndarray, qpos: np.ndarray, observed: np.ndarray, dt: float = 0.02):
    """Resample 30 Hz slot qpos (T,36) to 50 Hz ticks, holding across unobserved gaps."""
    ok = observed & np.isfinite(qpos).all(1)
    t, q = t_s[ok], qpos[ok]
    t50 = np.arange(t[0], t[-1], dt)
    lo = np.clip(np.searchsorted(t, t50, side="right") - 1, 0, len(t) - 2)
    w = ((t50 - t[lo]) / (t[lo + 1] - t[lo]))[:, None]
    out = np.empty((len(t50), 36))
    out[:, :3] = q[lo, :3] + w * (q[lo + 1, :3] - q[lo, :3])
    out[:, 7:] = q[lo, 7:] + w * (q[lo + 1, 7:] - q[lo, 7:])
    rots = Rotation.from_quat(q[:, [4, 5, 6, 3]])
    out[:, [4, 5, 6, 3]] = Slerp(t, rots)(t50).as_quat()
    slot_dt = np.median(np.diff(t_s))
    fresh = (t[lo + 1] - t[lo]) <= 1.5 * slot_dt
    held = 0
    for i in range(1, len(out)):
        if not fresh[i]:
            out[i] = out[i - 1]
            held += 1
    return t50, out, fresh, held
