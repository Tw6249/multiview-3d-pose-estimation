"""Module S4+S5 (real-time retargeter): golden-27 keypoints -> Unitree G1 reference, causal, one frame at a time.

Replaces the EasyMocap SMPL-X fit + GMR path for live use; that path stays as the offline reference. No body
model, no sequence optimisation: q[t] depends only on points[0..t].

Input  : `step(points (27,3), t_s)`: golden keypoints of ONE frame, metres, right-handed Z-up registered world (ground
         z = 0), NaN = unobserved; t_s = capture time in seconds (strictly increasing).
Output : `KeypointIKFrame`
           q  (36,) [root xyz m, root quat wxyz, 29 joints rad, SDK order] - the S6 reference layout
           dq (35,) backward difference against the previous output [root lin vel world m/s, root ang vel body
                    rad/s, joint vel rad/s]; zero on held / calibration frames
           observed: False while calibrating (default stance) or when the trunk points are missing (q held)
           held_groups: limbs whose keypoints were missing this frame (their joints keep the previous solution)
           solve_ms: wall time of this call
Method :
  1. Calibration: median human bone lengths over the first `calibration_s` of frames with observed legs ->
     root-displacement scale s = G1 leg length / human leg length. During calibration the output is the default
     stance turned to the hip-line heading (observed=False): an explicit start-up phase, not a guess. The output
     then ramps linearly from that stance to the retargeted pose over `startup_blend_s` (held_groups "blend").
  2. Direction transfer: every limb segment of the robot points along the human segment direction, with the robot's
     own segment lengths (hip-knee, knee-ankle, shoulder-elbow, elbow-wrist). Pelvis yaw/roll from the hip line,
     torso frame from the shoulder line and the hip->shoulder axis, foot frame from heel / big toe / small toe,
     hand frame from wrist / index MCP / pinky MCP.
  3. Mink differential IK (DAQP), warm-started from the previous frame, `ik_iterations` steps, joint limits enforced.
     The hand frame is NOT an IK task of the whole arm. Noisy index/pinky MCP points can flip it by ~180 deg; as a
     whole-arm task that pulled the shoulder/elbow into a wrong branch (Take_012, S2 raw input: left upper arm 33 deg
     off its own input). Default `wrist_iterations` = 0: hand orientation is not tracked (unmeasured; wrists stay at
     the posture default). With `wrist_iterations` > 0 only the three wrist joints of each side are turned towards
     the hand frame after step 3 (damped Gauss-Newton, clipped to the joint limits); measured on the real takes, that
     gave wrist references up to p95 1300 deg/s and no gain in limb fidelity, so it is off by default.
Assumptions (not measurements):
  - Pelvis pitch is not observable from two hip points. The hip->shoulder (trunk) inclination is split between the
    pelvis (hip joints) and the waist by `pelvis_trunk_share` (0 = pelvis upright, 1 = pelvis follows the trunk).
  - Hand frame (only if `wrist_iterations` > 0): G1 wrist zero = anatomical neutral (fingers along the forearm,
    index MCP above pinky MCP). It has never been checked against ground truth.
  - Upper-arm twist with a straight elbow and leg twist with a straight knee are unobservable; a weak posture task
    pulls them to the default stance.
  - Root xy displacement = s * human mid-hip displacement since calibration end; root height puts the lowest robot
    sole point at the human sole clearance (flight phases are kept). Root position is not a SONIC input; it only
    makes the reference self-consistent and scorable.
Failure: ValueError for a wrong shape / non-increasing time; RuntimeError if IK leaves the joint limits or returns
         non-finite values. Missing data is held, never extrapolated.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .constants import G1_29_JOINTS, KEYPOINT_INDEX_BY_NAME as K, load_config

SIDES = ("left", "right")
_Z = np.array([0.0, 0.0, 1.0])


@dataclass
class KeypointIKFrame:
    q: np.ndarray
    dq: np.ndarray
    observed: bool
    held_groups: tuple[str, ...] = field(default_factory=tuple)
    solve_ms: float = 0.0


def _unit(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    return v / n if np.isfinite(n) and n > 1e-6 else None


def _frame_zx(z: np.ndarray, y_hint: np.ndarray) -> np.ndarray | None:
    """Rotation with z axis = z and y axis = y_hint orthogonalised against z (columns x, y, z)."""
    z = _unit(z)
    if z is None:
        return None
    y = _unit(y_hint - np.dot(y_hint, z) * z)
    if y is None:
        return None
    return np.column_stack([np.cross(y, z), y, z])


def _frame_xz(x: np.ndarray, z_hint: np.ndarray) -> np.ndarray | None:
    """Rotation with x axis = x and z axis = z_hint orthogonalised against x."""
    x = _unit(x)
    if x is None:
        return None
    z = _unit(z_hint - np.dot(z_hint, x) * x)
    if z is None:
        return None
    return np.column_stack([x, np.cross(z, x), z])


def _ok(p: np.ndarray, *names: str) -> bool:
    return bool(np.isfinite(p[[K[n] for n in names]]).all())


class KeypointIKRetargeter:
    def __init__(self, model: mujoco.MjModel | None = None, cfg: dict | None = None,
                 default_q_29: np.ndarray | None = None) -> None:
        self.cfg = cfg or load_config()["s45c_keypoint_ik"]
        if model is None:
            from .s6_mujoco_closed_loop import G1PhysicsPlant
            model = G1PhysicsPlant().kin_model
        if model.nq != 36 or [model.joint(i).name for i in range(1, model.njnt)] != list(G1_29_JOINTS):
            raise ValueError("KeypointIKRetargeter needs the 29-joint G1 kinematic model (nq 36, SDK order)")
        self.model = model
        if default_q_29 is None:
            from .s6_mujoco_closed_loop import COMMON_DEFAULT_Q_29
            default_q_29 = COMMON_DEFAULT_Q_29
        self.default_q = np.asarray(default_q_29, np.float64).copy()
        self._robot_geometry()
        c = self.cfg
        self.configuration = mink.Configuration(model)
        mk = lambda n, p, o: mink.FrameTask(n, "body", position_cost=p, orientation_cost=o, lm_damping=1.0)
        self.t_pelvis = mk("pelvis", c["pelvis_position_cost"], c["pelvis_orientation_cost"])
        self.t_torso = mk("torso_link", 0.0, c["torso_orientation_cost"])
        self.t_knee = {s: mk(f"{s}_knee_link", c["knee_position_cost"], 0.0) for s in SIDES}
        self.t_foot = {s: mk(f"{s}_ankle_roll_link", c["foot_position_cost"], c["foot_orientation_cost"]) for s in SIDES}
        self.t_elbow = {s: mk(f"{s}_elbow_link", c["elbow_position_cost"], 0.0) for s in SIDES}
        self.t_hand = {s: mk(f"{s}_wrist_yaw_link", c["wrist_position_cost"], 0.0) for s in SIDES}
        self._wrist_body = {s: model.body(f"{s}_wrist_yaw_link").id for s in SIDES}
        wj = {s: [model.joint(f"{s}_wrist_{a}_joint").id for a in ("roll", "pitch", "yaw")] for s in SIDES}
        self._wrist_dof = {s: model.jnt_dofadr[wj[s]].copy() for s in SIDES}
        self._wrist_qadr = {s: model.jnt_qposadr[wj[s]].copy() for s in SIDES}
        self._wrist_rng = {s: model.jnt_range[wj[s]].copy() for s in SIDES}
        self._wd = mujoco.MjData(model)
        self._jr = np.zeros((3, model.nv))
        self.t_posture = mink.PostureTask(model, cost=c["posture_cost"])
        q_neutral = np.r_[0.0, 0.0, self.pelvis_rest_z, 1.0, 0.0, 0.0, 0.0, self.default_q]
        self.t_posture.set_target(q_neutral)
        self.limits = [mink.ConfigurationLimit(model)]
        self.lo, self.hi = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
        self.reset()

    # ---- robot geometry (FK at the zero pose, pelvis at the origin) -------------------------------------------
    def _robot_geometry(self) -> None:
        m, d = self.model, mujoco.MjData(self.model)
        d.qpos[:] = 0.0
        d.qpos[3] = 1.0
        mujoco.mj_kinematics(m, d)
        x = lambda n: d.xpos[m.body(n).id].copy()
        self.hip_off, self.sh_off, self.L = {}, {}, {}
        for s in SIDES:
            knee, ankle = x(f"{s}_knee_link"), x(f"{s}_ankle_pitch_link")
            # virtual hip centre above the knee at hip-pitch height, so the rest thigh is vertical like a human's
            self.hip_off[s] = np.array([knee[0], knee[1], x(f"{s}_hip_pitch_link")[2]])
            elbow, wrist = x(f"{s}_elbow_link"), x(f"{s}_wrist_yaw_link")
            self.sh_off[s] = np.array([elbow[0], elbow[1], x(f"{s}_shoulder_roll_link")[2]]) - x("torso_link")
            self.L[s] = dict(thigh=float(np.linalg.norm(knee - self.hip_off[s])), shank=float(np.linalg.norm(ankle - knee)),
                             upper=float(np.linalg.norm(elbow - (self.sh_off[s] + x("torso_link")))),
                             fore=float(np.linalg.norm(wrist - elbow)))
            self.ankle_roll_off = x(f"{s}_ankle_roll_link") - ankle            # in the (rest = identity) foot frame
        self.torso_off = x("torso_link")
        # sole contact points of the foot in its own frame (box corners or sphere bottoms of colliding geoms)
        b = m.body("left_ankle_roll_link").id
        pts = []
        for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]):
            if not m.geom_contype[g]:
                continue
            p, sz = m.geom_pos[g], m.geom_size[g]
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                pts += [p + np.array([sx * sz[0], sy * sz[1], -sz[2]]) for sx in (-1, 1) for sy in (-1, 1)]
            elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
                pts.append(p - np.array([0.0, 0.0, sz[0]]))
        if not pts:
            raise ValueError("no sole contact geometry on the ankle body")
        self.sole_local = np.array(pts)
        self.pelvis_rest_z = float(-min(self.ankle_roll_off[2] + x("left_ankle_pitch_link")[2] + self.sole_local[:, 2].min(), 0.0))
        self.robot_leg = self.L["left"]["thigh"] + self.L["left"]["shank"]

    # ---- state -------------------------------------------------------------------------------------------------
    def reset(self) -> None:
        self._calib: list[np.ndarray] = []
        self._t_calib_start: float | None = None
        self.scale: float | None = None
        self._origin_h: np.ndarray | None = None
        self._origin_r = np.zeros(2)
        self._t_prev: float | None = None
        self._q_prev: np.ndarray | None = None
        self._dirs: dict[str, np.ndarray] = {}
        self._foot_R = {s: np.eye(3) for s in SIDES}
        self._hand_R = {s: np.eye(3) for s in SIDES}
        self._clear = 0.0
        self._q_stance: np.ndarray | None = None
        self._t_calib_end = 0.0
        q0 = np.r_[0.0, 0.0, self.pelvis_rest_z, 1.0, 0.0, 0.0, 0.0, self.default_q]
        self.configuration.update(q0)

    def _bones(self, p: np.ndarray) -> np.ndarray:
        b = []
        for s in SIDES:
            b += [np.linalg.norm(p[K[f"{s}_knee"]] - p[K[f"{s}_hip"]]), np.linalg.norm(p[K[f"{s}_ankle"]] - p[K[f"{s}_knee"]])]
        return np.array(b)

    def _solve_wrists(self, q: np.ndarray, sides: list[str]) -> np.ndarray:
        """Turn only the wrist roll/pitch/yaw joints of `sides` towards the hand frames (damped Gauss-Newton)."""
        m, d = self.model, self._wd
        d.qpos[:] = q
        lam = float(self.cfg["wrist_damping"])
        for _ in range(int(self.cfg["wrist_iterations"])):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            for s in sides:
                b = self._wrist_body[s]
                e = Rotation.from_matrix(self._hand_R[s] @ d.xmat[b].reshape(3, 3).T).as_rotvec()   # world frame
                mujoco.mj_jacBody(m, d, None, self._jr, b)
                J = self._jr[:, self._wrist_dof[s]]
                step = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), e)
                qa, rng = self._wrist_qadr[s], self._wrist_rng[s]
                d.qpos[qa] = np.clip(d.qpos[qa] + step, rng[:, 0], rng[:, 1])
        return d.qpos.copy()

    # ---- one frame -----------------------------------------------------------------------------------------------
    def step(self, points: np.ndarray, t_s: float) -> KeypointIKFrame:
        t0 = time.perf_counter()
        p = np.asarray(points, np.float64)
        if p.shape != (27, 3):
            raise ValueError("points must be (27,3)")
        if self._t_prev is not None and not t_s > self._t_prev:
            raise ValueError("t_s must be strictly increasing")
        dt = None if self._t_prev is None else float(t_s - self._t_prev)
        self._t_prev = float(t_s)

        trunk_ok = _ok(p, "left_hip", "right_hip", "left_shoulder", "right_shoulder")
        if self.scale is None:                                          # calibration phase
            legs_ok = _ok(p, "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")
            if legs_ok:
                if self._t_calib_start is None:
                    self._t_calib_start = float(t_s)
                self._calib.append(self._bones(p))
            if self._t_calib_start is not None and t_s - self._t_calib_start >= float(self.cfg["calibration_s"]) \
                    and len(self._calib) >= 3:
                b = np.median(np.array(self._calib), axis=0)
                self.scale = float(self.robot_leg / (0.5 * (b[0] + b[1] + b[2] + b[3])))
            q = np.r_[0.0, 0.0, self.pelvis_rest_z, 1.0, 0.0, 0.0, 0.0, self.default_q]
            if trunk_ok:
                lat = p[K["left_hip"]] - p[K["right_hip"]]
                yaw = float(np.arctan2(lat[1], lat[0]) - np.pi / 2)
                q[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
            self.configuration.update(q)
            self._q_prev = self._q_stance = q
            self._t_calib_end = float(t_s)
            return KeypointIKFrame(q=q, dq=np.zeros(35), observed=False, held_groups=("calibration",),
                                   solve_ms=1e3 * (time.perf_counter() - t0))
        if not trunk_ok:
            q = self._q_prev.copy()
            return KeypointIKFrame(q=q, dq=np.zeros(35), observed=False, held_groups=("trunk",),
                                   solve_ms=1e3 * (time.perf_counter() - t0))

        held: list[str] = []
        lh, rh, ls, rs = (p[K[n]] for n in ("left_hip", "right_hip", "left_shoulder", "right_shoulder"))
        mid_hip, mid_sh = 0.5 * (lh + rh), 0.5 * (ls + rs)
        trunk_up = _unit(mid_sh - mid_hip)
        beta = float(self.cfg["pelvis_trunk_share"])
        R_t = _frame_zx(trunk_up, ls - rs)
        R_p = _frame_zx((1.0 - beta) * _Z + beta * trunk_up, lh - rh)
        if R_t is None or R_p is None:
            q = self._q_prev.copy()
            return KeypointIKFrame(q=q, dq=np.zeros(35), observed=False, held_groups=("trunk",),
                                   solve_ms=1e3 * (time.perf_counter() - t0))

        seg = {}
        for s in SIDES:
            for name, a, b in (("thigh", "hip", "knee"), ("shank", "knee", "ankle"),
                               ("upper", "shoulder", "elbow"), ("fore", "elbow", "wrist")):
                key = f"{s}_{name}"
                u = _unit(p[K[f"{s}_{b}"]] - p[K[f"{s}_{a}"]]) if _ok(p, f"{s}_{a}", f"{s}_{b}") else None
                if u is None:
                    held.append(key)
                    u = self._dirs.get(key)
                else:
                    self._dirs[key] = u
                seg[key] = u
            heel, big, small = (p[K[f"{s}_{n}"]] for n in ("heel", "big_toe", "small_toe"))
            if _ok(p, f"{s}_heel", f"{s}_big_toe", f"{s}_small_toe"):
                lat = (small - big) if s == "left" else (big - small)
                R = _frame_xz(0.5 * (big + small) - heel, np.cross(0.5 * (big + small) - heel, lat))
                if R is not None:
                    self._foot_R[s] = R
                else:
                    held.append(f"{s}_foot")
            else:
                held.append(f"{s}_foot")
            w, ii, pp = (p[K[n]] for n in (f"{s}_wrist", f"{s}_index_mcp", f"{s}_pinky_mcp"))
            R = _frame_xz(0.5 * (ii + pp) - w, ii - pp) if _ok(p, f"{s}_wrist", f"{s}_index_mcp", f"{s}_pinky_mcp") else None
            if R is not None:
                self._hand_R[s] = R
            else:
                held.append(f"{s}_hand")

        # robot targets relative to the pelvis origin P (P added after grounding)
        rel = {}
        torso_rel = R_p @ self.torso_off
        for s in SIDES:
            hip = R_p @ self.hip_off[s]
            if seg[f"{s}_thigh"] is not None and seg[f"{s}_shank"] is not None:
                knee = hip + self.L[s]["thigh"] * seg[f"{s}_thigh"]
                ankle = knee + self.L[s]["shank"] * seg[f"{s}_shank"]
                rel[f"{s}_knee"] = knee
                rel[f"{s}_foot"] = ankle + self._foot_R[s] @ self.ankle_roll_off
            sh = torso_rel + R_t @ self.sh_off[s]
            if seg[f"{s}_upper"] is not None and seg[f"{s}_fore"] is not None:
                elbow = sh + self.L[s]["upper"] * seg[f"{s}_upper"]
                rel[f"{s}_elbow"] = elbow
                rel[f"{s}_hand"] = elbow + self.L[s]["fore"] * seg[f"{s}_fore"]

        # root position: scaled mid-hip displacement; height from the sole clearance
        robot_mid_hip_rel = R_p @ (0.5 * (self.hip_off["left"] + self.hip_off["right"]))
        if self._origin_h is None:
            self._origin_h = mid_hip[:2].copy()
        xy = self._origin_r + self.scale * (mid_hip[:2] - self._origin_h) - robot_mid_hip_rel[:2]
        sole_h = [p[K[f"{s}_{n}"]][2] for s in SIDES for n in ("heel", "big_toe", "small_toe") if _ok(p, f"{s}_{n}")]
        if sole_h:
            self._clear = max(0.0, float(min(sole_h)))
        sole_r = [(rel[f"{s}_foot"] + self.sole_local @ self._foot_R[s].T)[:, 2].min() for s in SIDES if f"{s}_foot" in rel]
        z = self._clear - min(sole_r) if sole_r else float(self._q_prev[2])
        P = np.array([xy[0], xy[1], z])

        c = self.cfg
        SE3 = lambda R, t: mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(R), t)
        self.t_pelvis.set_target(SE3(R_p, P))
        self.t_torso.set_target(SE3(R_t, P + torso_rel))
        tasks = [self.t_pelvis, self.t_torso, self.t_posture]
        pos_only = []                                   # (task, body, target position) of the position-only tasks
        for s in SIDES:
            if f"{s}_knee" in rel:
                self.t_foot[s].set_target(SE3(self._foot_R[s], P + rel[f"{s}_foot"]))
                pos_only.append((self.t_knee[s], f"{s}_knee_link", P + rel[f"{s}_knee"]))
                tasks += [self.t_knee[s], self.t_foot[s]]
            if f"{s}_elbow" in rel:
                pos_only += [(self.t_elbow[s], f"{s}_elbow_link", P + rel[f"{s}_elbow"]),
                             (self.t_hand[s], f"{s}_wrist_yaw_link", P + rel[f"{s}_hand"])]
                tasks += [self.t_elbow[s], self.t_hand[s]]
        for _ in range(int(c["ik_iterations"])):
            # mink's frame error is the SE(3) log, whose translation part is coupled to the rotation error even at
            # zero orientation cost; a target rotation equal to the current one makes it a pure position error
            for task, body, pos in pos_only:
                cur = self.configuration.get_transform_frame_to_world(body, "body").rotation()
                task.set_target(mink.SE3.from_rotation_and_translation(cur, pos))
            v = mink.solve_ik(self.configuration, tasks, 1.0, solver="daqp", damping=float(c["ik_damping"]),
                              limits=self.limits)
            self.configuration.integrate_inplace(v, 1.0)
        q = self._solve_wrists(self.configuration.q, [s for s in SIDES if f"{s}_elbow" in rel])
        self.configuration.update(q)
        if not np.isfinite(q).all():
            raise RuntimeError("keypoint IK returned non-finite values")
        if np.any(q[7:] < self.lo - 1e-4) or np.any(q[7:] > self.hi + 1e-4):
            raise RuntimeError("keypoint IK left the joint limits")
        q[3:7] /= np.linalg.norm(q[3:7])
        a = (t_s - self._t_calib_end) / max(float(c["startup_blend_s"]), 1e-9)
        if a < 1.0:                                                     # start-up ramp from the calibration stance
            st = self._q_stance
            q_out = q.copy()
            q_out[:3] = st[:3] + a * (q[:3] - st[:3])
            q_out[7:] = st[7:] + a * (q[7:] - st[7:])
            r = Rotation.from_quat(np.stack([st[[4, 5, 6, 3]], q[[4, 5, 6, 3]]]))
            q_out[[4, 5, 6, 3]] = (r[0] * Rotation.from_rotvec(a * (r[0].inv() * r[1]).as_rotvec())).as_quat()
            q = q_out
            held.append("blend")
        dq = np.zeros(35)
        if dt is not None and self._q_prev is not None:
            dq[:3] = (q[:3] - self._q_prev[:3]) / dt
            r0, r1 = (Rotation.from_quat(x[[4, 5, 6, 3]]) for x in (self._q_prev, q))
            dq[3:6] = (r0.inv() * r1).as_rotvec() / dt
            dq[6:] = (q[7:] - self._q_prev[7:]) / dt
        self._q_prev = q
        return KeypointIKFrame(q=q, dq=dq, observed=True, held_groups=tuple(held),
                               solve_ms=1e3 * (time.perf_counter() - t0))


def causal_upsample(t_slot: np.ndarray, q: np.ndarray, dq: np.ndarray, observed: np.ndarray, mode: str,
                    lo: np.ndarray, hi: np.ndarray, dt: float = 0.02):
    """Retarget slots -> control ticks every `dt` s from the first observed slot; a tick uses the newest slot at or
    before it (`hold`) or extrapolates it with that slot's own backward-difference velocity (`extrapolate`, joints
    clipped to [lo, hi]). No future slot is read. Returns (t_tick, q, dq, observed of the slot used)."""
    t0 = t_slot[np.argmax(observed)]
    t_tick = np.arange(t0, t_slot[-1] + 1e-9, dt)
    k = np.searchsorted(t_slot, t_tick + 1e-9, side="right") - 1
    qo, dqo = q[k].copy(), dq[k].copy()
    if mode == "extrapolate":
        h = (t_tick - t_slot[k])[:, None]
        qo[:, :3] += h * dq[k, :3]
        qo[:, 7:] = np.clip(qo[:, 7:] + h * dq[k, 6:], lo, hi)
        R = Rotation.from_quat(q[k][:, [4, 5, 6, 3]]) * Rotation.from_rotvec(h * dq[k, 3:6])
        qo[:, [4, 5, 6, 3]] = R.as_quat()
    elif mode != "hold":
        raise ValueError(mode)
    return t_tick, qo, dqo, observed[k]


def retarget_sequence(points: np.ndarray, t_s: np.ndarray, retargeter: KeypointIKRetargeter | None = None) -> dict:
    """Offline driver that feeds frames one at a time (identical to streaming): returns q (T,36), dq (T,35),
    observed (T,), solve_ms (T,), held (list of tuples)."""
    r = retargeter or KeypointIKRetargeter()
    r.reset()
    out = [r.step(p, float(t)) for p, t in zip(np.asarray(points, np.float64), np.asarray(t_s, np.float64))]
    return dict(q=np.array([f.q for f in out]), dq=np.array([f.dq for f in out]),
                observed=np.array([f.observed for f in out]), solve_ms=np.array([f.solve_ms for f in out]),
                held=[f.held_groups for f in out])
