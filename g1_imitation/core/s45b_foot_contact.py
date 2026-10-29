"""Module S4+S5b: contact-aware foot correction applied after the GMR retargeter (GMR and SONIC unchanged).

Input : q_gmr (T,36) GMR reference on a uniform grid of `dt` s ([root xyz m, root quat wxyz, 29 joints rad, SDK
        order]); points (>=T,27,3) the human golden-27 track on the same grid (same t0, dt; extra tail ignored), metres, right-handed Z-up world with the
        ground at z = 0 (the registration convention); NaN = unobserved.
Output: FootContactResult
          q            (T,36) corrected reference (same layout)
          contact      (T,2) int8 per foot [left, right]: 1 support, 0 swing, -1 unknown (foot points missing)
          lock_weight  (T,2) blend weight in [0,1] between the GMR foot pose (0) and the locked anchor (1)
          lock_err_m   (T,2) ankle position error to its target after IK (NaN where lock_weight == 0)
Method: per foot, a causal hysteresis state machine on the HUMAN foot (heel + both toes): it enters support when the
        mean sole height < h_on AND the 3-D sole speed (backward difference over `speed_window_s`) < v_on for
        `on_hold_s`; it leaves support when height > h_off OR speed > v_off; missing points -> unknown (never locked).
        On entering support the anchor is the GMR ankle pose at that frame, made flat (yaw only) and put at the sole
        height of the robot model. Each frame, Mink differential IK (DAQP) starts from the GMR configuration and pulls
        both ankles to lerp/slerp(GMR ankle pose, anchor, lock_weight) while staying close to the GMR joints and pelvis
        pose. lock_weight ramps in over `blend_s` on touchdown; on lift-off the correction is frozen as an offset on
        the moving GMR ankle and decays over `blend_s` (never pulls a lifting foot down). Swing ankles only get
        z >= sole height.
        Both feet in swing (jump flight) -> targets equal the GMR feet, the root keeps GMR's vertical motion; the
        pelvis height is never clamped.
Time  : the contact detector is causal; the IK is per frame (no temporal smoothing). The GMR input itself is
        non-causal (whole-sequence fit), so the chain as a whole stays offline.
Units : metres, seconds, radians. Thresholds live in config `s45b_foot_contact`.
Failure: ValueError on shape/grid mismatch or non-finite q; a frame whose IK leaves the joint limits raises.
Not handled (left to GMR): heel-off / toe pivot (a heel raise ends support), sliding contacts, non-flat ground.
"""
from __future__ import annotations

from dataclasses import dataclass

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .constants import LEFT_FOOT_INDICES_27, RIGHT_FOOT_INDICES_27, load_config

SIDES = ("left", "right")
_SOLE_POINTS = (LEFT_FOOT_INDICES_27[1:], RIGHT_FOOT_INDICES_27[1:])   # big toe, small toe, heel


@dataclass(frozen=True)
class FootContactResult:
    q: np.ndarray
    contact: np.ndarray
    lock_weight: np.ndarray
    lock_err_m: np.ndarray


def detect_contacts(points: np.ndarray, dt: float, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Causal per-foot contact states (T,2) int8 and touchdown flags (T,2) bool from the human track."""
    T = len(points)
    k = max(1, round(float(cfg["speed_window_s"]) / dt))
    hold = max(1, round(float(cfg["on_hold_s"]) / dt))
    state = np.zeros((T, 2), np.int8)
    touchdown = np.zeros((T, 2), bool)
    for f, idx in enumerate(_SOLE_POINTS):
        sole = points[:, idx]                                   # (T,3,3)
        ok = np.isfinite(sole).all(axis=(1, 2))
        h = sole[..., 2].mean(1)
        c = sole.mean(1)
        cur, run = 0, 0
        for t in range(T):
            if not ok[t]:
                cur, run = -1, 0
                state[t, f] = -1
                continue
            j = t - k
            v = np.inf if j < 0 or not ok[j] else float(np.linalg.norm(c[t] - c[j])) / (k * dt)
            if cur == 1:
                if h[t] > cfg["h_off_m"] or v > cfg["v_off_m_s"]:
                    cur = 0
            else:
                cur = 0
                run = run + 1 if (h[t] < cfg["h_on_m"] and v < cfg["v_on_m_s"]) else 0
                if run >= hold:
                    cur, run = 1, 0
                    touchdown[t, f] = True
            state[t, f] = cur
    return state, touchdown


def _yaw_only(R: np.ndarray) -> np.ndarray:
    return Rotation.from_euler("z", np.arctan2(R[1, 0], R[0, 0])).as_matrix()


class FootContactCorrector:
    def __init__(self, model: mujoco.MjModel, cfg: dict | None = None) -> None:
        self.cfg = cfg or load_config()["s45b_foot_contact"]
        self.model = model
        self.feet = [f"{s}_ankle_roll_link" for s in SIDES]
        self.ids = [model.body(n).id for n in self.feet]
        self.sole_z = self._sole_height()
        self.configuration = mink.Configuration(model)
        c = self.cfg
        self.foot_tasks = [mink.FrameTask(n, "body", position_cost=c["foot_position_cost"],
                                          orientation_cost=c["foot_orientation_cost"], lm_damping=1.0) for n in self.feet]
        self.pelvis_task = mink.FrameTask("pelvis", "body", position_cost=c["pelvis_position_cost"],
                                          orientation_cost=c["pelvis_orientation_cost"], lm_damping=1.0)
        self.posture_task = mink.PostureTask(model, cost=c["posture_cost"])
        self.limits = [mink.ConfigurationLimit(model)]

    def _sole_height(self) -> float:
        """Ankle-frame height above the ground when the foot's contact geometry (spheres or boxes) touches it, flat foot."""
        m, d = self.model, mujoco.MjData(self.model)
        d.qpos[:] = 0.0
        d.qpos[3] = 1.0
        mujoco.mj_kinematics(m, d)
        b = self.ids[0]
        g = range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b])
        bottoms = []
        for i in g:
            if not m.geom_contype[i]:
                continue
            if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE:
                bottoms.append(d.geom_xpos[i][2] - m.geom_size[i][0])
            elif m.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX:
                bottoms.append(d.geom_xpos[i][2] - np.abs(d.geom_xmat[i].reshape(3, 3)[2]) @ m.geom_size[i])
        if not bottoms:
            raise ValueError("no contact spheres or boxes on the ankle body")
        return float(d.xpos[b][2] - min(bottoms))

    def _feet_fk(self, q: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        self.configuration.update(q)
        d = self.configuration.data
        return [(d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()) for b in self.ids]

    def correct(self, q_gmr: np.ndarray, points: np.ndarray, dt: float) -> FootContactResult:
        q_gmr = np.asarray(q_gmr, np.float64)
        pts = np.asarray(points, np.float64)
        if q_gmr.ndim != 2 or q_gmr.shape[1] != self.model.nq or pts.shape[1:] != (27, 3):
            raise ValueError("expected q (T,36) and points (T,27,3)")
        if len(pts) < len(q_gmr):
            raise ValueError("points must cover the q grid (same start time and dt)")
        if not np.isfinite(q_gmr).all():
            raise ValueError("non-finite GMR reference")
        T = len(q_gmr)
        pts = pts[:T]
        c = self.cfg
        state, touchdown = detect_contacts(pts, dt, c)
        step = dt / float(c["blend_s"])
        w = np.zeros(2)
        anchors: list[tuple[np.ndarray, np.ndarray] | None] = [None, None]
        out = q_gmr.copy()
        weights = np.zeros((T, 2))
        err = np.full((T, 2), np.nan)
        lo, hi = self.model.jnt_range[1:, 0], self.model.jnt_range[1:, 1]
        release: list[tuple[np.ndarray, np.ndarray] | None] = [None, None]
        last_delta: list[tuple[np.ndarray, np.ndarray]] = [(np.zeros(3), np.zeros(3))] * 2
        for t in range(T):
            feet = self._feet_fk(q_gmr[t])
            for f in range(2):
                p, R = feet[f]
                if touchdown[t, f]:
                    anchors[f] = (np.array([p[0], p[1], self.sole_z]), _yaw_only(R))
                    release[f] = None
                if state[t, f] != 1 and w[f] > 0.0 and release[f] is None and anchors[f] is not None:
                    # Lift-off / loss of observation: freeze the correction as an offset on the moving GMR foot and
                    # let it decay, so the foot follows GMR upward instead of being dragged back to the anchor.
                    # The offset is the one of the last support frame (this frame's GMR foot has already moved).
                    release[f] = last_delta[f]
                w[f] = min(1.0, w[f] + step) if state[t, f] == 1 else max(0.0, w[f] - step)
                if w[f] == 0.0:
                    release[f] = None
            weights[t] = w
            if not w.any() and all(feet[f][0][2] >= self.sole_z - 1e-6 for f in range(2)):
                continue                                            # nothing to change on this frame
            pelvis = self.configuration.get_transform_frame_to_world("pelvis", "body")
            self.pelvis_task.set_target(pelvis)
            self.posture_task.set_target(q_gmr[t])
            targets = []
            for f in range(2):
                p, R = feet[f]
                if state[t, f] == 1 and anchors[f] is not None:
                    pa, Ra = anchors[f]
                    dp, dr = pa - p, (Rotation.from_matrix(R).inv() * Rotation.from_matrix(Ra)).as_rotvec()
                    last_delta[f] = (dp, dr)
                elif release[f] is not None:
                    dp, dr = release[f]
                else:
                    dp, dr = np.zeros(3), np.zeros(3)
                p = p + w[f] * dp
                R = R @ Rotation.from_rotvec(w[f] * dr).as_matrix()
                p = np.array([p[0], p[1], max(p[2], self.sole_z)])
                targets.append(p)
                self.foot_tasks[f].set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(R), p))
            tasks = [*self.foot_tasks, self.pelvis_task, self.posture_task]
            for _ in range(int(c["ik_iterations"])):
                v = mink.solve_ik(self.configuration, tasks, dt, solver="daqp", damping=1e-3, limits=self.limits)
                self.configuration.integrate_inplace(v, dt)
            q = self.configuration.q.copy()
            if np.any(q[7:] < lo - 1e-4) or np.any(q[7:] > hi + 1e-4):
                raise ValueError(f"foot-correction IK left the joint limits at frame {t}")
            out[t] = q
            d = self.configuration.data
            for f in range(2):
                if w[f] > 0.0:
                    err[t, f] = float(np.linalg.norm(d.xpos[self.ids[f]] - targets[f]))
        return FootContactResult(q=out, contact=state, lock_weight=weights, lock_err_m=err)
