"""Module S1b: Multi-Person Cross-View Association & Target Lock (between S1 and S2).

Added 2026-10-29 for multi-person scenes (data/public_mamma_session); not part of `docs/方案规范.md`.
Why: S1 crops one person per camera on its own (largest YOLOX box; in TRACK_MODE the box nearest the reprojected
track). With several people in view the cameras can crop different people, S2 keeps whichever group wins its
consensus, and the source person can change without any flag (MAMMA run 2026-10-29: 21/120 frames with mixed crops,
a silent switch at frame 93, one 0.74 m chimera skeleton).

Input (one synchronized multi-camera frame):
  - `candidates`: per camera a `ViewCandidates`: every YOLOX person box `box_xyxy (M, 4)` [px], `det_score (M,)`, and
    the RTMW golden-27 keypoints of each box `kp (M, 27, 3)` [u px, v px, score] (`extract_person_candidates`).
  - `proj_matrices (V, 3, 4)`: P = K [R | t]; world right-handed, Z up, metres, ground z = 0 (the capture volume uses XY).
  - `t_s`: frame time in seconds, strictly increasing (the session's frame time base).
Output (`TargetFrame`):
  - `keypoints_2d (V, 27, 3)`: the target's keypoints in the cameras it was associated in, score 0 elsewhere (S2 input).
  - `valid` (True only when `status` is TRACKING), `status` (TRACKING | AMBIGUOUS | OCCLUDED | LOST | UNSELECTED),
    `target_id`, `target_views`, `selected_now` (a target was locked this frame: reset S2 / S3 / S4+S5 state),
    `identity_margin_m`, the frame's hypotheses and track centroids (diagnostics), `associate_ms`.
Algorithm:
  1. Grouping (per frame, no temporal prior): exhaustive search over "one candidate or none" per camera; DLT of the 12
     COCO body joints; every used camera needs a median reprojection error <= max(floor px, rel x box height); best =
     largest sum over cameras of 1 - (error / gate)^2; repeated greedily on the unused candidates. Two hypotheses
     closer than `duplicate_dist_m` (median joint distance) are one body; the later (smaller) one is dropped.
  2. 3D tracks: Hungarian assignment of hypotheses to tracks on the median body-joint distance to a constant-velocity
     prediction, gated at base + speed x (time since last seen). Only a hypothesis seen by >= `min_views_birth`
     cameras starts a track (two cameras facing each other at equal height barely constrain identity: their epipolar
     lines are near horizontal); a new track is confirmed after `confirm_hits` frames.
  3. Target: locked once, by operator designation (world XY, or a pixel in one camera), or automatically when exactly
     one track is inside the capture volume and it is confirmed. A locked target is never handed to another track.
Failure behaviour (keypoint scores all 0 and `valid` False, so S2 returns NaN and S3 holds the last pose):
  - target not matched this frame: OCCLUDED; LOST after `target_lost_s` since it was last seen;
  - another hypothesis or track within `ambiguity_margin_m` of the target's match: AMBIGUOUS; LOST after `max_ambiguous_s`;
  - LOST clears the lock. After a designation only a new designation re-locks; in auto mode the auto rule applies again;
  - several people in the capture volume and no designation: UNSELECTED.
Verification: tests/test_s1b_person_association.py (synthetic), scripts/mamma_multiperson_eval.py (real RGB, no 3D
labels), scripts/advanced_take_s1b_regression.py (single-person regression against BVH).
"""
from __future__ import annotations

from dataclasses import dataclass
import time

from numba import njit
import numpy as np
from scipy.optimize import linear_sum_assignment

from .constants import load_config

BODY12 = np.arange(5, 17, dtype=np.int64)          # golden-27 shoulders, elbows, wrists, hips, knees, ankles
TORSO_IN_BODY12 = np.array([0, 1, 6, 7])           # left/right shoulder, left/right hip inside BODY12
POSE_BATCH = 6                                     # S1 replays a static batch-6 CUDA graph

TRACKING = "TRACKING"
AMBIGUOUS = "AMBIGUOUS"
OCCLUDED = "OCCLUDED"
LOST = "LOST"
UNSELECTED = "UNSELECTED"


@dataclass(frozen=True)
class ViewCandidates:
    """All person candidates of one camera: YOLOX boxes and the RTMW keypoints of each box."""
    box_xyxy: np.ndarray    # (M, 4) px
    det_score: np.ndarray   # (M,)
    kp: np.ndarray          # (M, 27, 3) [u px, v px, score]


@dataclass(frozen=True)
class PersonHypothesis:
    """One person seen by >= 2 cameras in one frame."""
    sel: np.ndarray          # (V,) candidate index per camera, -1 = camera not used
    body_m: np.ndarray       # (12, 3) DLT of BODY12, NaN where fewer than 2 cameras
    view_err_px: np.ndarray  # (V,) median reprojection error of each used camera, NaN if unused

    @property
    def n_views(self) -> int:
        return int(np.sum(self.sel >= 0))

    def centroid(self) -> np.ndarray:
        return _centroid(self.body_m)


@dataclass(frozen=True)
class TargetFrame:
    """Output of Module S1b for one frame."""
    keypoints_2d: np.ndarray            # (V, 27, 3) S2 input; score 0 outside target_views
    crop_bboxes_xyxy: np.ndarray        # (V, 4) target box per camera, NaN where unused
    valid: bool
    status: str
    target_id: int | None
    target_views: tuple[int, ...]
    selected_now: bool
    identity_margin_m: float            # next-best match minus the target's match; inf = no competitor; NaN = no match
    hypotheses: tuple[PersonHypothesis, ...]
    hypothesis_track_ids: tuple[int, ...]
    track_centroids_m: dict[int, np.ndarray]
    designation: str                    # none | pending | resolved | rejected_ambiguous | expired
    associate_ms: float


def extract_person_candidates(tracker, images_bgr: list[np.ndarray], max_per_view: int = 6) -> list[ViewCandidates]:
    """Run S1's YOLOX on every camera and RTMW on every box (batches of `POSE_BATCH`); no box is selected.

    `tracker` is an S1 `RTMW27VisionTracker`. At most `max_per_view` boxes (highest detector score) are kept per camera.
    """
    boxes, scores = [], []
    for img in images_bgr:
        b, s = tracker.detect_person_candidates(img)
        b = np.asarray(b, np.float64).reshape(-1, 4)
        s = np.asarray(s, np.float64).reshape(-1)
        keep = np.argsort(-s, kind="stable")[:max_per_view]
        boxes.append(b[keep])
        scores.append(s[keep])
    owner = [v for v, b in enumerate(boxes) for _ in range(len(b))]
    flat = np.concatenate(boxes, axis=0) if owner else np.zeros((0, 4))
    kp = np.zeros((len(owner), 27, 3))
    for s0 in range(0, len(owner), POSE_BATCH):
        s1 = min(s0 + POSE_BATCH, len(owner))
        kp[s0:s1] = tracker.infer_batch_from_bboxes([images_bgr[v] for v in owner[s0:s1]], flat[s0:s1],
                                                    mode_label=tracker.DETECT_MODE).keypoints_2d
    out, k = [], 0
    for b, s in zip(boxes, scores):
        out.append(ViewCandidates(box_xyxy=b, det_score=s, kp=kp[k:k + len(b)]))
        k += len(b)
    return out


@njit(cache=True)
def _triangulate_sel(kp, P, cand_view, sel, joints, c_min, X, med, nerr, ebuf):
    """Score-weighted DLT of `joints` from the candidates `sel` (V,) [-1 = camera unused]; then the median
    reprojection error of every used camera over the joints it supplied. Returns the number of triangulated joints."""
    n_v = sel.shape[0]
    n_ok = 0
    for jj in range(joints.shape[0]):
        j = joints[jj]
        m00 = 0.0; m01 = 0.0; m02 = 0.0; m11 = 0.0; m12 = 0.0; m22 = 0.0
        r0 = 0.0; r1 = 0.0; r2 = 0.0
        cnt = 0
        for v in range(n_v):
            i = sel[v]
            if i < 0:
                continue
            c = kp[i, j, 2]
            if not c >= c_min:
                continue
            u = kp[i, j, 0]
            w = kp[i, j, 1]
            p = P[cand_view[i]]
            wt = c * c
            ax0 = u * p[2, 0] - p[0, 0]; ax1 = u * p[2, 1] - p[0, 1]; ax2 = u * p[2, 2] - p[0, 2]
            bx = p[0, 3] - u * p[2, 3]
            ay0 = w * p[2, 0] - p[1, 0]; ay1 = w * p[2, 1] - p[1, 1]; ay2 = w * p[2, 2] - p[1, 2]
            by = p[1, 3] - w * p[2, 3]
            m00 += wt * (ax0 * ax0 + ay0 * ay0); m01 += wt * (ax0 * ax1 + ay0 * ay1)
            m02 += wt * (ax0 * ax2 + ay0 * ay2); m11 += wt * (ax1 * ax1 + ay1 * ay1)
            m12 += wt * (ax1 * ax2 + ay1 * ay2); m22 += wt * (ax2 * ax2 + ay2 * ay2)
            r0 += wt * (ax0 * bx + ay0 * by); r1 += wt * (ax1 * bx + ay1 * by); r2 += wt * (ax2 * bx + ay2 * by)
            cnt += 1
        X[jj, 0] = np.nan; X[jj, 1] = np.nan; X[jj, 2] = np.nan
        if cnt < 2:
            continue
        c00 = m11 * m22 - m12 * m12
        c01 = -(m01 * m22 - m12 * m02)
        c02 = m01 * m12 - m11 * m02
        det = m00 * c00 + m01 * c01 + m02 * c02
        if abs(det) < 1e-12:
            continue
        c11 = m00 * m22 - m02 * m02
        c12 = -(m00 * m12 - m01 * m02)
        c22 = m00 * m11 - m01 * m01
        X[jj, 0] = (c00 * r0 + c01 * r1 + c02 * r2) / det
        X[jj, 1] = (c01 * r0 + c11 * r1 + c12 * r2) / det
        X[jj, 2] = (c02 * r0 + c12 * r1 + c22 * r2) / det
        n_ok += 1
    for v in range(n_v):
        med[v] = np.nan
        nerr[v] = 0
        i = sel[v]
        if i < 0:
            continue
        p = P[cand_view[i]]
        k = 0
        for jj in range(joints.shape[0]):
            j = joints[jj]
            if not np.isfinite(X[jj, 0]) or not kp[i, j, 2] >= c_min:
                continue
            zc = p[2, 0] * X[jj, 0] + p[2, 1] * X[jj, 1] + p[2, 2] * X[jj, 2] + p[2, 3]
            if zc <= 1e-6:
                ebuf[k] = 1e9
            else:
                du = kp[i, j, 0] - (p[0, 0] * X[jj, 0] + p[0, 1] * X[jj, 1] + p[0, 2] * X[jj, 2] + p[0, 3]) / zc
                dv = kp[i, j, 1] - (p[1, 0] * X[jj, 0] + p[1, 1] * X[jj, 1] + p[1, 2] * X[jj, 2] + p[1, 3]) / zc
                ebuf[k] = np.sqrt(du * du + dv * dv)
            k += 1
        nerr[v] = k
        if k > 0:
            med[v] = np.median(ebuf[:k])
    return n_ok


@njit(cache=True)
def _best_group(kp, P, cand_view, cand_gate, cand_free, n_views, max_opts, joints, c_min,
                min_views, min_joints_view, min_valid_joints):
    """Exhaustive search over one free candidate or none per camera. Returns (sel (V,), score); score
    = sum over used cameras of 1 - (median error / gate)^2, so a camera near its gate adds almost nothing and a group
    that pulls in a wrong-person view (which inflates every view's error) loses to the clean smaller group;
    -inf when no group of >= `min_views` cameras passes every gate."""
    opt = np.full((n_views, max_opts + 1), -1, np.int64)
    n_opt = np.ones(n_views, np.int64)
    for i in range(cand_view.shape[0]):
        v = cand_view[i]
        if cand_free[i] and n_opt[v] <= max_opts:
            opt[v, n_opt[v]] = i
            n_opt[v] += 1
    ctr = np.zeros(n_views, np.int64)
    sel = np.full(n_views, -1, np.int64)
    best = np.full(n_views, -1, np.int64)
    best_score = -np.inf
    J = joints.shape[0]
    X = np.empty((J, 3))
    med = np.empty(n_views)
    nerr = np.zeros(n_views, np.int64)
    ebuf = np.empty(J)
    while True:
        nsel = 0
        for v in range(n_views):
            sel[v] = opt[v, ctr[v]]
            if sel[v] >= 0:
                nsel += 1
        if nsel >= min_views and nsel > best_score:
            n_ok = _triangulate_sel(kp, P, cand_view, sel, joints, c_min, X, med, nerr, ebuf)
            if n_ok >= min_valid_joints:
                ok = True
                score = 0.0
                for v in range(n_views):
                    i = sel[v]
                    if i < 0:
                        continue
                    if nerr[v] < min_joints_view or not med[v] <= cand_gate[i]:
                        ok = False
                        break
                    r = med[v] / cand_gate[i]
                    score += 1.0 - r * r
                if ok:
                    if score > best_score:
                        best_score = score
                        best[:] = sel
        v = 0
        while v < n_views:
            ctr[v] += 1
            if ctr[v] < n_opt[v]:
                break
            ctr[v] = 0
            v += 1
        if v == n_views:
            break
    return best, best_score


def _centroid(body: np.ndarray) -> np.ndarray:
    t = body[TORSO_IN_BODY12]
    ok = np.isfinite(t).all(axis=1)
    if ok.any():
        return t[ok].mean(axis=0)
    ok = np.isfinite(body).all(axis=1)
    return body[ok].mean(axis=0) if ok.any() else np.full(3, np.nan)


def _median_joint_dist(a: np.ndarray, b: np.ndarray, min_common: int) -> float:
    d = np.linalg.norm(a - b, axis=1)
    d = d[np.isfinite(d)]
    return float(np.median(d)) if len(d) >= min_common else np.inf


def camera_centers_m(proj_matrices: np.ndarray) -> np.ndarray:
    """(V, 3) camera centres in the world frame from P = [M | p4]: C = -M^-1 p4."""
    P = np.asarray(proj_matrices, np.float64)
    return np.stack([-np.linalg.solve(p[:, :3], p[:, 3]) for p in P])


def group_candidates(candidates: list[ViewCandidates], proj_matrices: np.ndarray, cfg: dict) -> list[PersonHypothesis]:
    """Step 1: per-frame cross-view grouping (see module docstring). Hypotheses are ordered as found."""
    P = np.ascontiguousarray(proj_matrices, np.float64)
    n_views = P.shape[0]
    max_opts = int(cfg["max_candidates_per_view"])
    view, local, kps, gates = [], [], [], []
    for v, vc in enumerate(candidates):
        for i in range(min(len(vc.box_xyxy), max_opts)):
            h = float(vc.box_xyxy[i, 3] - vc.box_xyxy[i, 1])
            view.append(v)
            local.append(i)
            kps.append(vc.kp[i])
            gates.append(max(float(cfg["reproj_gate_floor_px"]), float(cfg["reproj_gate_rel_box_height"]) * h))
    if len(view) < 2:
        return []
    cand_view = np.asarray(view, np.int64)
    local = np.asarray(local, np.int64)
    kp = np.ascontiguousarray(np.stack(kps), np.float64)
    gate = np.asarray(gates, np.float64)
    free = np.ones(len(view), np.bool_)
    c_min = float(cfg["min_2d_confidence"])
    J = len(BODY12)
    hyps: list[PersonHypothesis] = []
    for _ in range(int(cfg["max_persons"])):
        best, score = _best_group(kp, P, cand_view, gate, free, n_views, max_opts, BODY12, c_min,
                                  int(cfg["min_views"]), int(cfg["min_joints_per_view"]), int(cfg["min_valid_joints"]))
        if not np.isfinite(score):
            break
        X = np.empty((J, 3))
        med = np.empty(n_views)
        nerr = np.zeros(n_views, np.int64)
        _triangulate_sel(kp, P, cand_view, best, BODY12, c_min, X, med, nerr, np.empty(J))
        free[best[best >= 0]] = False
        sel = np.where(best >= 0, local[np.maximum(best, 0)], -1)
        body = X.copy()
        if any(_median_joint_dist(body, h.body_m, 4) < float(cfg["duplicate_dist_m"]) for h in hyps):
            continue
        hyps.append(PersonHypothesis(sel=sel, body_m=body, view_err_px=np.where(best >= 0, med, np.nan)))
    return hyps


class _Track:
    def __init__(self, track_id: int, hyp: PersonHypothesis, t_s: float) -> None:
        self.track_id = track_id
        self.body = hyp.body_m.copy()
        self.vel = np.zeros(3)
        self.t_last = float(t_s)
        self.hits = 1
        self.confirmed = False

    def centroid(self) -> np.ndarray:
        return _centroid(self.body)

    def predict(self, t_s: float) -> np.ndarray:
        return self.body + self.vel * (float(t_s) - self.t_last)

    def update(self, hyp: PersonHypothesis, t_s: float, cfg: dict) -> None:
        dt = float(t_s) - self.t_last
        c_old, c_new = self.centroid(), hyp.centroid()
        if dt > 0.0 and np.isfinite(c_old).all() and np.isfinite(c_new).all():
            v = (c_new - c_old) / dt
            n = float(np.linalg.norm(v))
            vmax = float(cfg["max_speed_m_s"])
            if n > vmax:
                v *= vmax / n
            a = float(cfg["velocity_smoothing"])
            self.vel = a * v + (1.0 - a) * self.vel
        self.body = np.where(np.isfinite(hyp.body_m), hyp.body_m, self.body + self.vel * dt)
        self.t_last = float(t_s)
        self.hits += 1
        self.confirmed = self.confirmed or self.hits >= int(cfg["confirm_hits"])


class MultiPersonTargetTracker:
    """Module S1b: group all candidates into 3D people, track them, and output the locked target's 2D keypoints."""

    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = dict(cfg or load_config()["s1b_person_association"])
        self.reset()
        self._warmup()

    def _warmup(self) -> None:
        """Compile the numba kernels (cached on disk) before the first real frame."""
        kp = np.zeros((2, 27, 3))
        P = np.zeros((2, 3, 4))
        vc = [ViewCandidates(np.array([[0.0, 0.0, 10.0, 10.0]]), np.ones(1), kp[:1]) for _ in range(2)]
        group_candidates(vc, P, self.cfg)
        J = len(BODY12)
        _triangulate_sel(kp, P, np.array([0, 1], np.int64), np.array([0, 1], np.int64), BODY12, 0.35,
                         np.empty((J, 3)), np.empty(2), np.zeros(2, np.int64), np.empty(J))

    def reset(self) -> None:
        self.tracks: dict[int, _Track] = {}
        self._next_id = 0
        self.target_id: int | None = None
        self._mode = "auto" if bool(self.cfg["auto_select"]) else "designated"
        self._lost = False
        self._t_prev: float | None = None
        self._amb_since: float | None = None
        self._pending_xy: np.ndarray | None = None
        self._pending_px: tuple[int, float, float] | None = None
        self._pending_since: float | None = None
        self._designation = "none"

    def designate_xy(self, xy_m) -> None:
        """Operator designation: lock the confirmed person nearest to world `xy_m` (within `designate_radius_m`, and
        at least `designate_margin_m` nearer than anyone else). Replaces the current target once resolved."""
        self._pending_xy = np.asarray(xy_m, np.float64).reshape(-1)[:2].copy()
        self._pending_px = None
        self._pending_since = None
        self._designation = "pending"

    def designate_pixel(self, view: int, u_px: float, v_px: float) -> None:
        """Operator designation by a pixel in camera `view`: the person whose box there contains it (must be unique)."""
        self._pending_px = (int(view), float(u_px), float(v_px))
        self._pending_xy = None
        self._pending_since = None
        self._designation = "pending"

    def _capture_volume(self, P: np.ndarray) -> tuple[np.ndarray, float] | None:
        cv = self.cfg["capture_volume"]
        if cv["mode"] == "none":
            return None
        if cv["mode"] == "circle":
            return np.asarray(cv["center_xy_m"], np.float64), float(cv["radius_m"])
        if cv["mode"] == "camera_ring":
            c = camera_centers_m(P)[:, :2]
            ctr = c.mean(axis=0)
            return ctr, float(cv["radius_fraction"]) * float(np.linalg.norm(c - ctr, axis=1).mean())
        raise ValueError(f"unknown capture_volume mode {cv['mode']!r}")

    def _lock(self, track_id: int) -> None:
        self.target_id = track_id
        self._lost = False
        self._amb_since = None

    def _resolve_designation(self, candidates, hyps, hyp_track, t: float) -> bool:
        c = self.cfg
        if self._pending_px is None and self._pending_xy is None:
            return False
        if self._pending_since is None:
            self._pending_since = t
        if self._pending_px is not None:
            view, u, v = self._pending_px
            hits = []
            for b, h in enumerate(hyps):
                if 0 <= view < len(h.sel) and h.sel[view] >= 0:
                    x1, y1, x2, y2 = candidates[view].box_xyxy[h.sel[view]]
                    if x1 <= u <= x2 and y1 <= v <= y2:
                        hits.append(b)
            if len(hits) > 1:
                self._pending_px = None
                self._designation = "rejected_ambiguous"
                return False
            if len(hits) == 1:
                self._pending_xy = hyps[hits[0]].centroid()[:2]
                self._pending_px = None
        if self._pending_xy is not None:
            cand = sorted((float(np.linalg.norm(trk.centroid()[:2] - self._pending_xy)), k)
                          for k, trk in self.tracks.items() if trk.confirmed and trk.t_last == t)
            if cand and cand[0][0] <= float(c["designate_radius_m"]) and (
                    len(cand) == 1 or cand[1][0] - cand[0][0] >= float(c["designate_margin_m"])):
                self._lock(cand[0][1])
                self._mode = "designated"
                self._pending_xy = None
                self._designation = "resolved"
                return True
        if t - self._pending_since > float(c["designate_timeout_s"]):
            self._pending_xy = self._pending_px = None
            self._designation = "expired"
        return False

    def step(self, candidates: list[ViewCandidates], proj_matrices: np.ndarray, t_s: float) -> TargetFrame:
        P = np.ascontiguousarray(proj_matrices, np.float64)
        if P.ndim != 3 or P.shape[1:] != (3, 4) or len(candidates) != P.shape[0]:
            raise ValueError(f"need one ViewCandidates per camera and P (V,3,4); got {len(candidates)} and {P.shape}")
        t = float(t_s)
        if self._t_prev is not None and not t > self._t_prev:
            raise ValueError("t_s must be strictly increasing")
        self._t_prev = t
        c = self.cfg
        t0 = time.perf_counter()
        hyps = group_candidates(candidates, P, c)

        # step 2: hypotheses -> tracks
        ids = list(self.tracks)
        cost = np.full((len(ids), len(hyps)), np.inf)
        for a, k in enumerate(ids):
            trk = self.tracks[k]
            pred = trk.predict(t)
            gate = min(float(c["track_gate_base_m"]) + float(c["track_gate_speed_m_s"]) * (t - trk.t_last),
                       float(c["track_gate_max_m"]))
            for b, h in enumerate(hyps):
                d = _median_joint_dist(pred, h.body_m, int(c["min_common_joints"]))
                if d <= gate:
                    cost[a, b] = d
        match: dict[int, int] = {}
        margin: dict[int, float] = {}
        if cost.size:
            rows, cols = linear_sum_assignment(np.where(np.isfinite(cost), cost, 1e6))
            for a, b in zip(rows, cols):
                if np.isfinite(cost[a, b]):
                    alt = min(np.min(np.delete(cost[a], b), initial=np.inf), np.min(np.delete(cost[:, b], a), initial=np.inf))
                    match[ids[a]] = int(b)
                    margin[ids[a]] = float(alt - cost[a, b])
        hyp_track = [-1] * len(hyps)
        for k, b in match.items():
            self.tracks[k].update(hyps[b], t, c)
            hyp_track[b] = k
        for b, h in enumerate(hyps):
            if hyp_track[b] < 0 and h.n_views >= int(c["min_views_birth"]):
                self.tracks[self._next_id] = _Track(self._next_id, h, t)
                hyp_track[b] = self._next_id
                self._next_id += 1
        for k in [k for k, trk in self.tracks.items()
                  if k != self.target_id and t - trk.t_last > float(c["track_drop_s"])]:
            del self.tracks[k]

        # step 3: target lock
        selected_now = self._resolve_designation(candidates, hyps, hyp_track, t)
        volume = self._capture_volume(P)
        if self.target_id is None and self._mode == "auto" and self._pending_xy is None and self._pending_px is None:
            inside = [k for k, trk in self.tracks.items() if volume is None
                      or np.linalg.norm(trk.centroid()[:2] - volume[0]) <= volume[1]]
            if len(inside) == 1 and self.tracks[inside[0]].confirmed and self.tracks[inside[0]].t_last == t:
                self._lock(inside[0])
                selected_now = True

        n_v = P.shape[0]
        kp_out = np.zeros((n_v, 27, 3))
        box_out = np.full((n_v, 4), np.nan)
        status, valid, views, ident = (LOST if self._lost else UNSELECTED), False, (), np.nan
        if self.target_id is not None:
            trk = self.tracks[self.target_id]
            if self.target_id in match:
                b = match[self.target_id]
                ident = margin[self.target_id]
                if ident < float(c["ambiguity_margin_m"]):
                    self._amb_since = t if self._amb_since is None else self._amb_since
                    status = LOST if t - self._amb_since > float(c["max_ambiguous_s"]) else AMBIGUOUS
                else:
                    self._amb_since = None
                    status, valid = TRACKING, True
                    sel = hyps[b].sel
                    views = tuple(int(v) for v in np.flatnonzero(sel >= 0))
                    for v in views:
                        kp_out[v] = candidates[v].kp[sel[v]]
                        box_out[v] = candidates[v].box_xyxy[sel[v]]
            else:
                status = OCCLUDED if t - trk.t_last <= float(c["target_lost_s"]) else LOST
            if status == LOST:
                self.target_id = None
                self._lost = True
                self._amb_since = None
        return TargetFrame(
            keypoints_2d=kp_out, crop_bboxes_xyxy=box_out, valid=valid, status=status, target_id=self.target_id,
            target_views=views, selected_now=selected_now, identity_margin_m=float(ident), hypotheses=tuple(hyps),
            hypothesis_track_ids=tuple(hyp_track),
            track_centroids_m={k: trk.centroid() for k, trk in self.tracks.items()},
            designation=self._designation, associate_ms=(time.perf_counter() - t0) * 1000.0)
