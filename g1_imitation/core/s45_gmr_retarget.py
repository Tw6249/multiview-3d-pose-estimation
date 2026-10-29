"""Module S4+S5 (offline reference retargeter): EasyMocap SMPL-X fit + GMR -> Unitree G1 reference trajectory.

Default until 2026-10-29; the live default is now the causal keypoint IK (core/s45c_keypoint_ik.py), because this
fit optimises the whole sequence and cannot run in real time. Kept for offline comparison (BVH -> GMR is the
fidelity upper bound). Replaced the removed in-house S4 (analytic SE(3) scaling) and S5 (Mink IK). The work is done by third-party,
pinned code (GMR, EasyMocap, smplx; see config `s45_gmr_retarget`) in its own venv; this module only marshals data
through `g1_imitation/gmr_bridge/worker.py` as a subprocess and converts the result for S6.

Input  : points (T,27,3) human golden keypoints, metres, right-handed Z-up world, uniform rate `fps` (Hz).
         NaN = unobserved (never imputed here).
Output : RetargetResult
           q   (T,36)  [root xyz m, root quat wxyz, 29 joints rad in SDK order] - the S6 reference layout
           dq  (T,35)  [root linear vel (world, m/s), root angular vel (body, rad/s), 29 joint vel (rad/s)],
                       finite differences of q (central inside, one-sided at the ends)
           observed (T,) frames with >= 1 observed body point; the fit fills the others from neighbours, callers
                       must treat those as missing (hold), not as new observations
           fit_residual_mm (T,25) per body25 joint (NaN = unobserved), human_height_m (GMR's estimate)
Time   : NON-CAUSAL. The fit optimises the whole sequence at once (temporal smoothness both ways), so q[t] depends on
         future frames. Valid for offline evaluation/replay only; a real-time path needs a causal fit.
Failure: ValueError for a bad shape/rate or < 2 frames with observations; RuntimeError if the worker fails or times
         out (its stderr is attached). There is no fallback retargeter.
Unobservable DOF (limb twist, fingers, foot roll) are set by EasyMocap's priors, not measured.
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .constants import G1_29_JOINTS, load_config

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RetargetResult:
    q: np.ndarray
    dq: np.ndarray
    observed: np.ndarray
    fit_residual_mm: np.ndarray
    human_height_m: float
    fit_s: float
    retarget_s: float


def finite_difference_velocity(q: np.ndarray, dt: float) -> np.ndarray:
    """q (T,36) -> dq (T,35) in the MuJoCo free-joint convention (linear world, angular body)."""
    T = len(q)
    dq = np.zeros((T, 35))
    if T < 2:
        return dq
    dq[:, :3] = np.gradient(q[:, :3], dt, axis=0)
    dq[:, 6:] = np.gradient(q[:, 7:], dt, axis=0)
    R = Rotation.from_quat(q[:, [4, 5, 6, 3]])
    w = (R[:-1].inv() * R[1:]).as_rotvec() / dt                     # body-frame angular velocity between samples
    dq[1:-1, 3:6] = 0.5 * (w[:-1] + w[1:])
    dq[0, 3:6], dq[-1, 3:6] = w[0], w[-1]
    return dq


def resample_points(points: np.ndarray, fps_in: float, fps_out: float) -> tuple[np.ndarray, np.ndarray]:
    """Linear resampling of a (T,27,3) track onto a uniform fps_out grid; NaN stays NaN. Returns (t_out, pts)."""
    t_in = np.arange(len(points)) / float(fps_in)
    t_out = np.arange(0.0, t_in[-1] + 1e-9, 1.0 / float(fps_out))
    lo = np.clip(np.searchsorted(t_in, t_out, side="right") - 1, 0, max(0, len(t_in) - 2))
    hi = np.minimum(lo + 1, len(t_in) - 1)
    w = np.where(hi > lo, (t_out - t_in[lo]) * fps_in, 0.0)[:, None, None]
    return t_out, points[lo] + w * (points[hi] - points[lo])


def resample_q(t_src: np.ndarray, q: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """(T,36) reference onto t_dst: root position/joints linear, root orientation slerp; t_dst clipped to t_src."""
    t = np.clip(t_dst, t_src[0], t_src[-1])
    out = np.empty((len(t), 36))
    for c in list(range(3)) + list(range(7, 36)):
        out[:, c] = np.interp(t, t_src, q[:, c])
    out[:, [4, 5, 6, 3]] = Slerp(t_src, Rotation.from_quat(q[:, [4, 5, 6, 3]]))(t).as_quat()
    return out


def retarget_trajectory_from_3d(points: np.ndarray, fps_in: float, out_dt: float, retarget_fps: float = 30.0,
                                retargeter: "GMRRetargeter | None" = None) -> tuple[np.ndarray, np.ndarray, RetargetResult]:
    """Synthetic/offline helper: resample to GMR's 30 Hz, retarget, interpolate back to `out_dt` for S6.

    Returns (q (T_out,36), dq (T_out,35), RetargetResult at retarget_fps). Requires a fully observed track
    (synthetic input); for real tracks with gaps use GMRRetargeter.retarget and handle `observed` explicitly.
    """
    if not np.isfinite(points).all():
        raise ValueError("retarget_trajectory_from_3d expects a fully observed track")
    t_r, pts_r = resample_points(np.asarray(points, np.float64), fps_in, retarget_fps)
    res = (retargeter or GMRRetargeter()).retarget(pts_r, retarget_fps)
    t_out = np.arange(0.0, t_r[-1] + 1e-9, out_dt)
    q = resample_q(t_r, res.q, t_out)
    return q, finite_difference_velocity(q, out_dt), res


class GMRRetargeter:
    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = cfg or load_config()["s45_gmr_retarget"]
        self.python = Path(self.cfg["python"])
        self.worker = ROOT / self.cfg["worker"]
        if not self.python.exists() or not self.worker.exists():
            raise RuntimeError(f"GMR runtime missing: {self.python} / {self.worker}")

    def _run(self, mode: str, src: Path, dst: Path) -> dict:
        p = subprocess.run([str(self.python), "-I", str(self.worker), mode, str(src), str(dst)], capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=float(self.cfg["timeout_s"]))
        if p.returncode != 0 or not dst.exists():
            raise RuntimeError(f"GMR worker failed ({p.returncode}):\n{p.stderr[-4000:]}")
        with np.load(dst) as d:
            return {k: d[k] for k in d.files}

    @staticmethod
    def _to_sdk(qpos: np.ndarray, names) -> np.ndarray:
        names = [str(n) for n in names]
        if sorted(names) != sorted(G1_29_JOINTS):
            raise RuntimeError("GMR joint set differs from the SDK 29-joint set")
        q = np.asarray(qpos, np.float64).copy()
        q[:, 7:] = q[:, 7:][:, [names.index(n) for n in G1_29_JOINTS]]
        return q

    def retarget(self, points: np.ndarray, fps: float) -> RetargetResult:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 3 or pts.shape[1:] != (27, 3):
            raise ValueError("points must be (T,27,3)")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive")
        if (np.isfinite(pts).all(-1).any(-1)).sum() < 2:
            raise ValueError("need >= 2 frames with an observed point")
        with tempfile.TemporaryDirectory() as td:
            src, dst = Path(td) / "in.npz", Path(td) / "out.npz"
            np.savez(src, points=pts, fps=float(fps))
            out = self._run("points", src, dst)
        q = self._to_sdk(out["qpos"], out["joint_names"])
        if not np.isfinite(q).all():
            raise RuntimeError("GMR returned non-finite joint targets")
        return RetargetResult(q=q, dq=finite_difference_velocity(q, 1.0 / fps), observed=out["observed"].astype(bool),
                              fit_residual_mm=out["fit_residual_m"] * 1000.0, human_height_m=float(out["human_height_m"]),
                              fit_s=float(out["fit_s"]), retarget_s=float(out["retarget_s"]))

    def retarget_bvh(self, bvh_path: str | Path) -> tuple[np.ndarray, float]:
        """Reference path only (motion-capture BVH, Nokov naming): returns (q (T,36) SDK order, GMR human height)."""
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / "out.npz"
            out = self._run("bvh", Path(bvh_path), dst)
        return self._to_sdk(out["qpos"], out["joint_names"]), float(out["human_height_m"])
