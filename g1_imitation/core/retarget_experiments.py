"""Opt-in retargeting hypotheses. Never selected by the production default.

Input: golden-27 RGB-derived points (metres, right-handed Z-up, ground=0),
capture timestamp seconds, optional triangulation covariance (m^2). No BVH input.
Output: the same G1 q/dq layout as KeypointIKRetargeter, plus experimental diagnostics.
All processing is causal. Geometric contact is an estimate, not force measurement.
Pelvis pitch/limb twist/hand orientation remain unobserved; priors do not measure them.
Missing observations remain NaN at the temporal boundary; the retargeter retains its
documented hold behavior. Contact constraints disappear on missing evidence or lift-off.
Validation: prefix invariance, Jacobian finite differences, missing-data and limit tests,
then identical-physics ablations against the original retargeter. No deployment claim.
"""
from __future__ import annotations

import mink
import mujoco
import numpy as np

from .constants import KEYPOINT_INDEX_BY_NAME as K
from .s45c_keypoint_ik import KeypointIKRetargeter, _unit
from .s3_temporal_filter import smoothing_alpha


class RelativePositionTask(mink.Task):
    """Position a-b expressed in orientation frame c; b/c=None means world.

    Uses pure Euclidean translation, avoiding SE(3)-log rotation/translation coupling.
    Target in metres; cost in the same convention as Mink position tasks.
    """
    def __init__(self, model, a, b=None, c=None, cost=1.0):
        super().__init__(cost=np.full(3, cost), gain=1.0, lm_damping=1.0)
        self.ids = tuple(None if n is None else model.body(n).id for n in (a, b, c))
        self.target = np.zeros(3)

    def compute_error(self, configuration):
        d = configuration.data
        a, b, c = self.ids
        delta = d.xpos[a] - (0.0 if b is None else d.xpos[b])
        R = np.eye(3) if c is None else d.xmat[c].reshape(3, 3)
        return R.T @ delta - self.target

    def compute_jacobian(self, configuration):
        m, d = configuration.model, configuration.data
        a, b, c = self.ids
        ja, jb, jr = (np.zeros((3, m.nv)) for _ in range(3))
        mujoco.mj_jacBody(m, d, ja, None, a)
        if b is not None:
            mujoco.mj_jacBody(m, d, jb, None, b)
        if c is not None:
            mujoco.mj_jacBody(m, d, None, jr, c)
        delta = d.xpos[a] - (0.0 if b is None else d.xpos[b])
        x, y, z = delta
        skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
        R = np.eye(3) if c is None else d.xmat[c].reshape(3, 3)
        return R.T @ (ja - jb + skew @ jr)


class ExperimentalRetargeter(KeypointIKRetargeter):
    def __init__(self, experiment, *, relations=False, support=False, **kwargs):
        self.experiment = experiment
        self.use_relations, self.use_support = relations, support
        super().__init__(**kwargs)
        if self.robot != "unitree_g1":
            raise ValueError("This experiment is bound to the G1 model only")
        rc = experiment["relations"]
        self.relation_tasks = {
            s: RelativePositionTask(self.model, self._b_hand[s], self._b_torso,
                                    self._b_torso, rc["wrist_torso_cost"]) for s in ("left", "right")}
        self.pair_task = RelativePositionTask(self.model, self._b_hand["left"], self._b_hand["right"],
                                              self._b_torso, rc["hand_pair_cost"])
        self.contact_tasks = {s: RelativePositionTask(self.model, self._b_foot[s],
                                cost=experiment["support"]["contact_position_cost"]) for s in ("left", "right")}
        if self.use_support:
            # Pitch gets a weaker PRIOR; torso/limbs/support participate in choosing a feasible allocation.
            self.t_pelvis.set_orientation_cost([self.cfg["pelvis_orientation_cost"],
                                               experiment["support"]["pelvis_pitch_cost"],
                                               self.cfg["pelvis_orientation_cost"]])

    def reset(self):
        super().reset()
        self._human_calibration = []
        self._arm_scale = None
        self._height_origin = None
        self._last_target_time = None
        self._contact_previous = {}
        self._contact_run = {s: 0.0 for s in ("left", "right")}
        self._contact_state = {s: False for s in ("left", "right")}
        self._contact_age = {s: 0.0 for s in ("left", "right")}
        self._anchors = {}
        self.contact = np.full(2, -1, np.int8)

    def step(self, points, t_s, covariance=None):
        p = np.asarray(points)
        if self.scale is None and p.shape == (27, 3) and np.isfinite(p[5:17]).all():
            self._human_calibration.append(p.copy())
        if p.shape == (27, 3) and not np.isfinite(p[[5, 6, 11, 12]]).all():
            self.contact[:] = -1
            self._contact_state = {s: False for s in ("left", "right")}
            self._contact_run = {s: 0.0 for s in ("left", "right")}
            self._contact_previous.clear()
        return super().step(points, t_s)

    def _prepare_target_tasks(self, points, t_s, pelvis_rotation, torso_rotation, pelvis_position, relative_targets):
        p, P, rel, tasks = points, pelvis_position.copy(), relative_targets, []
        dt = 0.0 if self._last_target_time is None else t_s - self._last_target_time
        self._last_target_time = t_s
        if self._arm_scale is None and self._human_calibration:
            cal = np.array(self._human_calibration)
            arms = [np.linalg.norm(cal[:, K[f"{s}_elbow"]] - cal[:, K[f"{s}_shoulder"]], axis=1)
                    + np.linalg.norm(cal[:, K[f"{s}_wrist"]] - cal[:, K[f"{s}_elbow"]], axis=1)
                    for s in ("left", "right")]
            self._arm_scale = np.mean([self.L[s]["upper"] + self.L[s]["fore"] for s in ("left", "right")]) / np.median(arms)
        if self.use_relations and self._arm_scale is not None:
            mid_sh = (p[K["left_shoulder"]] + p[K["right_shoulder"]]) / 2
            sh_offset = (self.sh_off["left"] + self.sh_off["right"]) / 2
            for s in ("left", "right"):
                if np.isfinite(p[[K[f"{s}_elbow"], K[f"{s}_wrist"]]]).all():
                    task = self.relation_tasks[s]
                    task.target = sh_offset + self._arm_scale * torso_rotation.T @ (p[K[f"{s}_wrist"]] - mid_sh)
                    tasks.append(task)
            if np.isfinite(p[[K["left_wrist"], K["right_wrist"]]]).all():
                self.pair_task.target = self._arm_scale * torso_rotation.T @ (p[K["left_wrist"]] - p[K["right_wrist"]])
                tasks.append(self.pair_task)
        if self.use_support:
            c = self.experiment["support"]
            human_z = (p[K["left_hip"], 2] + p[K["right_hip"], 2]) / 2
            if self._height_origin is None:
                self._height_origin = (human_z, float(P[2]))
            hz, rz = self._height_origin
            mix = c["pelvis_height_mix"]
            P[2] = (1 - mix) * P[2] + mix * (rz + self.scale * (human_z - hz))
            for i, s in enumerate(("left", "right")):
                sole = p[[K[f"{s}_{n}"] for n in ("heel", "big_toe", "small_toe")]]
                if not np.isfinite(sole).all() or f"{s}_foot" not in rel:
                    self._contact_state[s] = False
                    self._contact_run[s] = self._contact_age[s] = 0.0
                    self._contact_previous.pop(s, None)
                    self.contact[i] = -1
                    continue
                center, h = sole.mean(0), sole[:, 2].mean()
                prev = self._contact_previous.get(s)
                speed = np.inf if prev is None or dt <= 0 or dt > 0.1 else np.linalg.norm(center - prev) / dt
                self._contact_previous[s] = center.copy()
                if self._contact_state[s]:
                    self._contact_state[s] = bool(h <= c["h_off_m"] and speed <= c["v_off_m_s"])
                    if not self._contact_state[s]:
                        self._contact_run[s] = 0.0
                else:
                    self._contact_run[s] = self._contact_run[s] + dt if h < c["h_on_m"] and speed < c["v_on_m_s"] else 0.0
                    if self._contact_run[s] >= c["on_hold_s"]:
                        self._contact_state[s] = True
                        self._anchors[s] = P + rel[f"{s}_foot"]
                        self._contact_age[s] = 0.0
                self.contact[i] = int(self._contact_state[s])
                if self._contact_state[s]:
                    self._contact_age[s] += dt
                    task = self.contact_tasks[s]
                    task.target = self._anchors[s].copy()
                    task.cost[:] = c["contact_position_cost"] * min(1.0, self._contact_age[s] / c["contact_blend_s"])
                    tasks.append(task)
                else:
                    self._contact_age[s] = 0.0
        return P, rel, tasks


class SkeletonTemporalFilter:
    """Causal fixed-length skeleton: root plus normalized direction states.

    Lengths are medians of the first calibration_s, then frozen. Covariance changes
    filter gain, never supplies missing points. Unsupported rotations stay unknown.
    Filtering uses current/past points only; no ground clamp, extrapolation or BVH.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.start = self.prev_t = None
        self.samples = []
        self.lengths = None
        self.state, self.previous, self.derivative = {}, {}, {}

    def _filter(self, key, value, dt, direction=False, confidence=1.0):
        v = np.asarray(value, float)
        if not np.isfinite(v).all():
            self.previous.pop(key, None)
            return np.full(3, np.nan)
        if key not in self.state or key not in self.previous:
            self.state[key], self.previous[key], self.derivative[key] = v.copy(), v.copy(), np.zeros(3)
        vel = (v - self.previous[key]) / dt
        ad = smoothing_alpha(self.cfg["derivative_cutoff_hz"], dt)
        self.derivative[key] += ad * (vel - self.derivative[key])
        fc = self.cfg["direction_cutoff_hz"] if direction else self.cfg["root_cutoff_hz"]
        if direction:
            fc += self.cfg["direction_beta"] * np.linalg.norm(self.derivative[key])
        alpha = float(smoothing_alpha(fc, dt)) * confidence
        out = self.state[key] + alpha * (v - self.state[key])
        if direction:
            u = _unit(out)
            out = v.copy() if u is None else u
        self.state[key], self.previous[key] = out, v.copy()
        return out.copy()

    @staticmethod
    def _vectors(p):
        hips = (p[11] + p[12]) / 2
        shoulders = (p[5] + p[6]) / 2
        out = {"hip_width": p[11] - p[12], "shoulder_width": p[5] - p[6], "trunk": shoulders - hips}
        for s in ("left", "right"):
            for name, a, b in (("upper", "shoulder", "elbow"), ("fore", "elbow", "wrist"),
                               ("thigh", "hip", "knee"), ("shank", "knee", "ankle")):
                out[f"{s}_{name}"] = p[K[f"{s}_{b}"]] - p[K[f"{s}_{a}"]]
        return out

    def step(self, points, t_s, covariance=None):
        p = np.asarray(points, float)
        if p.shape != (27, 3) or (self.prev_t is not None and t_s <= self.prev_t):
            raise ValueError("expected (27,3) and increasing capture times")
        dt = 1 / 30 if self.prev_t is None else t_s - self.prev_t
        self.prev_t = t_s
        if self.start is None:
            self.start = t_s
        vec = self._vectors(p)
        if self.lengths is None:
            self.samples.append({k: np.linalg.norm(v) for k, v in vec.items()})
            if t_s - self.start >= self.cfg["calibration_s"]:
                lengths = {k: float(np.nanmedian([s[k] for s in self.samples])) for k in vec}
                if all(np.isfinite(v) and v > 1e-5 for v in lengths.values()):
                    self.lengths = lengths
            # Retargeter independently calibrates; no artificial zeros for missing joints.
            return p.copy()
        conf = np.ones(27)
        if covariance is not None:
            sig2 = np.trace(covariance, axis1=-2, axis2=-1)
            conf = 1 / (1 + sig2 / self.cfg["uncertainty_scale_m"] ** 2)
            conf = np.clip(np.nan_to_num(conf, nan=self.cfg["min_confidence"]), self.cfg["min_confidence"], 1.0)
        filtered_vec = {}
        for key, v in vec.items():
            u = _unit(v)
            if u is None:
                self.previous.pop(key, None)
                filtered_vec[key] = np.full(3, np.nan)
            else:
                # Per-vector confidence uses the endpoint evidence, not unrelated joints.
                if key == "hip_width": ids = [11, 12]
                elif key == "shoulder_width": ids = [5, 6]
                elif key == "trunk": ids = [5, 6, 11, 12]
                else:
                    s, name = key.split("_", 1)
                    a, b = {"upper": ("shoulder", "elbow"), "fore": ("elbow", "wrist"),
                            "thigh": ("hip", "knee"), "shank": ("knee", "ankle")}[name]
                    ids = [K[f"{s}_{a}"], K[f"{s}_{b}"]]
                filtered_vec[key] = self.lengths[key] * self._filter(key, u, dt, True, float(np.min(conf[ids])))
        o = np.full_like(p, np.nan)
        root = self._filter("root", (p[11] + p[12]) / 2, dt, confidence=float(min(conf[11], conf[12])))
        sh = root + filtered_vec["trunk"]
        o[11], o[12] = root + filtered_vec["hip_width"] / 2, root - filtered_vec["hip_width"] / 2
        o[5], o[6] = sh + filtered_vec["shoulder_width"] / 2, sh - filtered_vec["shoulder_width"] / 2
        for s in ("left", "right"):
            for name, a, b in (("upper", "shoulder", "elbow"), ("fore", "elbow", "wrist"),
                               ("thigh", "hip", "knee"), ("shank", "knee", "ankle")):
                o[K[f"{s}_{b}"]] = o[K[f"{s}_{a}"]] + filtered_vec[f"{s}_{name}"]
            for n in ("heel", "big_toe", "small_toe", "index_mcp", "pinky_mcp"):
                j, anchor = K[f"{s}_{n}"], K[f"{s}_{'ankle' if n in ('heel', 'big_toe', 'small_toe') else 'wrist'}"]
                o[j] = o[anchor] + self._filter(f"offset_{j}", p[j] - p[anchor], dt, confidence=float(min(conf[j], conf[anchor])))
        for j in range(5):
            o[j] = sh + self._filter(f"face_{j}", p[j] - (p[5] + p[6]) / 2, dt, confidence=float(conf[j]))
        o[~np.isfinite(p).all(1)] = np.nan
        return o


def constrain_joint_trajectory(q, t, lo, hi, vmax, amax):
    """Causal joint-only rate governor; root reference is untouched.

    q [T,36], t seconds, joint bounds radians, limits rad/s and rad/s^2.
    The stopping-distance envelope avoids joint-limit overshoot. No physical claim.
    Returns governed q and per-frame intervention mask; raises if constraints conflict.
    """
    q, t, lo, hi = (np.asarray(x, float) for x in (q, t, lo, hi))
    if q.ndim != 2 or q.shape[1] != 7 + len(lo) or len(q) != len(t) or not len(q):
        raise ValueError("trajectory dimensions mismatch")
    if not np.isfinite(q).all() or not np.isfinite(t).all() or vmax <= 0 or amax <= 0:
        raise ValueError("non-finite trajectory or non-positive experimental limits")
    if np.any(q[0, 7:] < lo) or np.any(q[0, 7:] > hi):
        raise ValueError("initial joint state outside limits")
    out = q.copy()
    velocity = np.zeros(len(lo))
    changed = np.zeros(len(q), bool)
    for k in range(1, len(q)):
        dt = float(t[k] - t[k - 1])
        if dt <= 0:
            raise ValueError("non-increasing timestamps")
        prev = out[k - 1, 7:]
        v_stop_hi = np.sqrt((amax * dt) ** 2 + 2 * amax * np.maximum(hi - prev, 0)) - amax * dt
        v_stop_lo = np.sqrt((amax * dt) ** 2 + 2 * amax * np.maximum(prev - lo, 0)) - amax * dt
        lower = np.maximum.reduce([np.full_like(prev, -vmax), velocity - amax * dt, -v_stop_lo])
        upper = np.minimum.reduce([np.full_like(prev, vmax), velocity + amax * dt, v_stop_hi])
        if np.any(lower > upper + 1e-7):
            raise RuntimeError("joint governor infeasible; reject sequence")
        velocity = np.clip((q[k, 7:] - prev) / dt, lower, upper)
        out[k, 7:] = prev + velocity * dt
        changed[k] = bool(np.max(np.abs(out[k, 7:] - q[k, 7:])) > 1e-7)
    return out, changed
