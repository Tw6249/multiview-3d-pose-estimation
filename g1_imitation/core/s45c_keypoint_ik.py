"""Module S4+S5 (real-time retargeter): golden-27 keypoints -> humanoid reference, causal, one frame at a time.

Supports both configured humanoid robots (`unitree_g1`: 29-DoF, nq=36, nv=35; `roban_s17`: 21-DoF, nq=28, nv=27).
Replaces the EasyMocap SMPL-X fit + GMR path for live use; that path stays as the offline reference. No body
model, no sequence optimisation: q[t] depends only on points[0..t].

Input  : `step(points (27,3), t_s)`: golden keypoints of ONE frame, metres, right-handed Z-up registered world (ground
         z = 0), NaN = unobserved; t_s = capture time in seconds (strictly increasing).
Output : `KeypointIKFrame`
           q  (7 + n_dof,) [root xyz m, root quat wxyz, n_dof joints rad, robot joint order] - the S6 reference layout
           dq (6 + n_dof,) backward difference against the previous output [root lin vel world m/s, root ang vel body
                    rad/s, joint vel rad/s]; zero on held / calibration frames
           observed: False while calibrating (default stance) or when the trunk points are missing (q held)
           held_groups: limbs whose keypoints were missing this frame (their joints keep the previous solution)
           solve_ms: wall time of this call
Method :
  1. Calibration: median human bone lengths over the first `calibration_s` of frames with observed legs ->
     root-displacement scale s = robot leg length / human leg length. During calibration the output is the default
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
     the hand frame after step 3 (damped Gauss-Newton, clipped to the joint limits) on robots with 3-DoF wrists.
Assumptions (not measurements):
  - Pelvis pitch is not observable from two hip points. The hip->shoulder (trunk) inclination is split between the
    pelvis (hip joints) and the waist by `pelvis_trunk_share` (0 = pelvis upright, 1 = pelvis follows the trunk).
  - Hand frame (only if `wrist_iterations` > 0): G1 wrist zero = anatomical neutral (fingers along the forearm,
    index MCP above pinky MCP). It has never been checked against ground truth.
  - Upper-arm twist with a straight elbow and leg twist with a straight knee are unobservable; a weak posture task
    pulls them to the default stance.
  - Root xy displacement = s * human mid-hip displacement since calibration end; root height puts the lowest robot
    sole point at the human sole clearance (flight phases are kept).
Failure: ValueError for a wrong shape / non-increasing time; RuntimeError if IK leaves the joint limits or returns
         non-finite values. Missing data is held, never extrapolated.
"""
from __future__ import annotations

from pathlib import Path
import time
from dataclasses import dataclass, field

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .constants import (
    G1_29_JOINTS,
    KEYPOINT_INDEX_BY_NAME as K,
    ROBAN_21_JOINTS,
    ROBAN_HEFT_21_JOINTS,
    ROBOT_JOINTS,
    SUPPORTED_ROBOTS,
    load_config,
)

SIDES = ("left", "right")
_Z = np.array([0.0, 0.0, 1.0])


@dataclass
class KeypointIKFrame:
    q: np.ndarray
    dq: np.ndarray
    observed: bool
    held_groups: tuple[str, ...] = field(default_factory=tuple)
    solve_ms: float = 0.0
    ik_branch: str = "warm_start"
    ik_task_cost: float = 0.0
    ik_continuity_cost: float = 0.0
    ik_candidates_evaluated: int = 1


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
    def __init__(
        self,
        model: mujoco.MjModel | None = None,
        cfg: dict | None = None,
        default_q_29: np.ndarray | None = None,
        *,
        robot: str | None = None,
        default_q: np.ndarray | None = None,
    ) -> None:
        full_cfg = load_config()
        self.cfg = cfg or full_cfg["s45c_keypoint_ik"]
        if robot is None:
            if model is not None and model.nq == 28:
                robot = "roban_s17"
            elif model is not None and model.nq == 36:
                robot = "unitree_g1"
            else:
                robot = self.cfg.get("robot") or full_cfg.get("default_robot", "unitree_g1")
        if robot not in SUPPORTED_ROBOTS:
            raise ValueError(f"Unsupported robot {robot!r}; expected one of {SUPPORTED_ROBOTS}")
        self.robot = robot
        expected_joints = list(ROBOT_JOINTS[self.robot])
        self.nj = len(expected_joints)
        self.nq = 7 + self.nj
        self.nv = 6 + self.nj

        robots_cfg = self.cfg.get("robots") or full_cfg["s45c_keypoint_ik"]["robots"]
        rcfg = robots_cfg[self.robot]
        self._b_pelvis = str(rcfg["pelvis_body"])
        self._b_torso = str(rcfg["torso_body"])
        fmt = lambda tpl, s: str(tpl).format(side=s, s=s[0])
        self._b_hip = {s: fmt(rcfg["hip_body_template"], s) for s in SIDES}
        self._b_knee = {s: fmt(rcfg["knee_body_template"], s) for s in SIDES}
        self._b_ankle = {s: fmt(rcfg["ankle_body_template"], s) for s in SIDES}
        self._b_foot = {s: fmt(rcfg["foot_body_template"], s) for s in SIDES}
        self._b_shoulder = {s: fmt(rcfg["shoulder_body_template"], s) for s in SIDES}
        self._b_elbow = {s: fmt(rcfg["elbow_body_template"], s) for s in SIDES}
        self._b_hand = {s: fmt(rcfg["hand_body_template"], s) for s in SIDES}
        wrist_tpls = list(rcfg.get("wrist_joint_templates", ()))

        if model is None:
            if self.robot == "unitree_g1":
                from .s6_mujoco_closed_loop import G1PhysicsPlant
                model = G1PhysicsPlant().kin_model
            else:
                from .s6_mujoco_closed_loop import RobanPhysicsPlant
                model = RobanPhysicsPlant().kin_model
        if model.nq != self.nq or [model.joint(i).name for i in range(1, model.njnt)] != expected_joints:
            raise ValueError(
                f"KeypointIKRetargeter({self.robot}) needs the {self.nj}-joint kinematic model (nq {self.nq})"
            )
        self.model = model
        dq_init = default_q if default_q is not None else default_q_29
        if dq_init is None:
            if self.robot == "unitree_g1":
                from .s6_mujoco_closed_loop import COMMON_DEFAULT_Q_29
                dq_init = COMMON_DEFAULT_Q_29
            else:
                from .s6_mujoco_closed_loop import COMMON_DEFAULT_Q_ROBAN_21
                dq_init = COMMON_DEFAULT_Q_ROBAN_21
        self.default_q = np.asarray(dq_init, np.float64).copy()
        if self.default_q.shape != (self.nj,):
            raise ValueError(f"default_q for {self.robot} must have shape ({self.nj},), got {self.default_q.shape}")
        self._robot_geometry()
        c = self.cfg
        self.configuration = mink.Configuration(model)
        mk = lambda n, p, o: mink.FrameTask(n, "body", position_cost=p, orientation_cost=o, lm_damping=1.0)
        self.t_pelvis = mk(self._b_pelvis, c["pelvis_position_cost"], c["pelvis_orientation_cost"])
        self.t_torso = mk(self._b_torso, 0.0, c["torso_orientation_cost"])
        pelvis_axis_scale = np.asarray(c.get("pelvis_orientation_axis_scale", [1.0, 1.0, 1.0]), dtype=np.float64)
        torso_axis_scale = np.asarray(c.get("torso_orientation_axis_scale", [1.0, 1.0, 1.0]), dtype=np.float64)
        if pelvis_axis_scale.shape != (3,) or torso_axis_scale.shape != (3,) \
                or not np.isfinite(pelvis_axis_scale).all() or not np.isfinite(torso_axis_scale).all() \
                or np.any(pelvis_axis_scale < 0.0) or np.any(torso_axis_scale < 0.0):
            raise ValueError("orientation axis scales must be finite non-negative 3-vectors [local x,y,z]")
        # FrameTask's final three residual components are local-frame rotation axes [x,y,z].
        self.t_pelvis.cost[3:] *= pelvis_axis_scale
        self.t_torso.cost[3:] *= torso_axis_scale
        self.t_knee = {s: mk(self._b_knee[s], c["knee_position_cost"], 0.0) for s in SIDES}
        self.t_foot = {s: mk(self._b_foot[s], c["foot_position_cost"], c["foot_orientation_cost"]) for s in SIDES}
        self.t_elbow = {s: mk(self._b_elbow[s], c["elbow_position_cost"], 0.0) for s in SIDES}
        self.t_hand = {s: mk(self._b_hand[s], c["wrist_position_cost"], 0.0) for s in SIDES}
        self._wrist_body = {s: model.body(self._b_hand[s]).id for s in SIDES}
        self._has_wrists = len(wrist_tpls) == 3
        if self._has_wrists:
            wj = {s: [model.joint(fmt(t, s)).id for t in wrist_tpls] for s in SIDES}
            self._wrist_dof = {s: model.jnt_dofadr[wj[s]].copy() for s in SIDES}
            self._wrist_qadr = {s: model.jnt_qposadr[wj[s]].copy() for s in SIDES}
            self._wrist_rng = {s: model.jnt_range[wj[s]].copy() for s in SIDES}
        else:
            self._wrist_dof = {}
            self._wrist_qadr = {}
            self._wrist_rng = {}
        self._wd = mujoco.MjData(model)
        self._jr = np.zeros((3, model.nv))
        self.t_posture = mink.PostureTask(model, cost=c["posture_cost"])
        q_neutral = np.r_[0.0, 0.0, self.pelvis_rest_z, 1.0, 0.0, 0.0, 0.0, self.default_q]
        self.t_posture.set_target(q_neutral)
        self.limits = [mink.ConfigurationLimit(model)]
        self.lo, self.hi = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
        self._joint_qadr = {
            str(model.joint(i).name): int(model.jnt_qposadr[i])
            for i in range(1, model.njnt)
        }
        self._last_ik_branch = "warm_start"
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
            knee, ankle = x(self._b_knee[s]), x(self._b_ankle[s])
            # virtual hip centre above the knee at hip-pitch height, so the rest thigh is vertical like a human's
            self.hip_off[s] = np.array([knee[0], knee[1], x(self._b_hip[s])[2]])
            elbow, wrist = x(self._b_elbow[s]), x(self._b_hand[s])
            self.sh_off[s] = np.array([elbow[0], elbow[1], x(self._b_shoulder[s])[2]]) - x(self._b_torso)
            self.L[s] = dict(thigh=float(np.linalg.norm(knee - self.hip_off[s])), shank=float(np.linalg.norm(ankle - knee)),
                             upper=float(np.linalg.norm(elbow - (self.sh_off[s] + x(self._b_torso)))),
                             fore=float(np.linalg.norm(wrist - elbow)))
            self.ankle_roll_off = x(self._b_foot[s]) - ankle              # in the (rest = identity) foot frame
        self.torso_off = x(self._b_torso)
        # sole contact points of the foot in its own frame (box corners or sphere bottoms of colliding geoms)
        b = m.body(self._b_foot["left"]).id
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
        self.pelvis_rest_z = float(-min(self.ankle_roll_off[2] + x(self._b_ankle["left"])[2] + self.sole_local[:, 2].min(), 0.0))
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
        self._last_ik_branch = "warm_start"
        q0 = np.r_[0.0, 0.0, self.pelvis_rest_z, 1.0, 0.0, 0.0, 0.0, self.default_q]
        self.configuration.update(q0)

    def _branch_seed_patterns(self, q_warm: np.ndarray) -> list[tuple[str, np.ndarray]]:
        """Generate bounded arm/elbow alternative initial states around the causal warm start."""
        count = int(self.cfg.get("ik_branch_candidates", 1))
        if count <= 1 or self.robot != "unitree_g1":
            return [("warm_start", q_warm.copy())]
        offset = float(self.cfg.get("ik_branch_seed_offset_rad", 0.8))
        patterns = [("warm_start", {})]
        for side in SIDES:
            for sign, label in ((1.0, "plus"), (-1.0, "minus")):
                patterns.append((f"{side}_{label}", {
                    f"{side}_shoulder_roll_joint": sign * offset,
                    f"{side}_elbow_joint": sign * offset,
                }))
        out = []
        for name, changes in patterns[:count]:
            seed = q_warm.copy()
            for joint_name, delta in changes.items():
                qadr = self._joint_qadr.get(joint_name)
                if qadr is not None:
                    jid = self.model.joint(joint_name).id
                    seed[qadr] = np.clip(seed[qadr] + delta, *self.model.jnt_range[jid])
            out.append((name, seed))
        return out

    def _ik_candidate_cost(self, tasks: list, q_warm: np.ndarray) -> tuple[float, float, float]:
        """Return (total, observation task, continuity) costs for the current configuration."""
        observation_sq = 0.0
        posture_sq = 0.0
        for task in tasks:
            error = np.asarray(task.compute_error(self.configuration), dtype=np.float64)
            weighted = np.asarray(task.cost, dtype=np.float64) * error
            value = float(np.dot(weighted, weighted))
            if task is self.t_posture:
                posture_sq += value
            else:
                observation_sq += value
        continuity_weight = float(self.cfg.get("ik_branch_continuity_weight", 0.15))
        delta = self.configuration.q[7:] - q_warm[7:]
        continuity = float(np.dot(delta, delta)) * continuity_weight
        total = observation_sq + posture_sq + continuity
        return total, observation_sq, continuity

    def _bones(self, p: np.ndarray) -> np.ndarray:
        b = []
        for s in SIDES:
            b += [np.linalg.norm(p[K[f"{s}_knee"]] - p[K[f"{s}_hip"]]), np.linalg.norm(p[K[f"{s}_ankle"]] - p[K[f"{s}_knee"]])]
        return np.array(b)

    def _solve_wrists(self, q: np.ndarray, sides: list[str]) -> np.ndarray:
        """Turn only the wrist roll/pitch/yaw joints of `sides` towards the hand frames (damped Gauss-Newton)."""
        if not self._has_wrists or int(self.cfg["wrist_iterations"]) <= 0:
            return q.copy()
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
    def _prepare_target_tasks(self, points, t_s, pelvis_rotation, torso_rotation, pelvis_position, relative_targets):
        """Extension point for opt-in experiments; the default reference and tasks are unchanged."""
        return pelvis_position, relative_targets, []

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
            return KeypointIKFrame(q=q, dq=np.zeros(self.nv), observed=False, held_groups=("calibration",),
                                   solve_ms=1e3 * (time.perf_counter() - t0))
        if not trunk_ok:
            q = self._q_prev.copy()
            return KeypointIKFrame(q=q, dq=np.zeros(self.nv), observed=False, held_groups=("trunk",),
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
            return KeypointIKFrame(q=q, dq=np.zeros(self.nv), observed=False, held_groups=("trunk",),
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
        P, rel, extra_tasks = self._prepare_target_tasks(p, t_s, R_p, R_t, P, rel)

        c = self.cfg
        SE3 = lambda R, t: mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(R), t)
        self.t_pelvis.set_target(SE3(R_p, P))
        self.t_torso.set_target(SE3(R_t, P + torso_rel))
        tasks = [self.t_pelvis, self.t_torso, self.t_posture, *extra_tasks]
        pos_only = []                                   # (task, body, target position) of the position-only tasks
        for s in SIDES:
            if f"{s}_knee" in rel:
                self.t_foot[s].set_target(SE3(self._foot_R[s], P + rel[f"{s}_foot"]))
                pos_only.append((self.t_knee[s], self._b_knee[s], P + rel[f"{s}_knee"]))
                tasks += [self.t_knee[s], self.t_foot[s]]
            if f"{s}_elbow" in rel:
                pos_only += [(self.t_elbow[s], self._b_elbow[s], P + rel[f"{s}_elbow"]),
                             (self.t_hand[s], self._b_hand[s], P + rel[f"{s}_hand"])]
                tasks += [self.t_elbow[s], self.t_hand[s]]
        q_warm = self.configuration.q.copy()
        candidates = self._branch_seed_patterns(q_warm)
        evaluated = []
        for branch_name, seed in candidates:
            self.configuration.update(seed)
            for _ in range(int(c["ik_iterations"])):
                # Mink's SE(3) log couples translation with rotation unless position-only targets use current FK rotation.
                for task, body, pos in pos_only:
                    cur = self.configuration.get_transform_frame_to_world(body, "body").rotation()
                    task.set_target(mink.SE3.from_rotation_and_translation(cur, pos))
                v = mink.solve_ik(self.configuration, tasks, 1.0, solver="daqp", damping=float(c["ik_damping"]),
                                  limits=self.limits)
                self.configuration.integrate_inplace(v, 1.0)
            q_candidate = self.configuration.q.copy()
            total_cost, observation_cost, continuity_cost = self._ik_candidate_cost(tasks, q_warm)
            evaluated.append((total_cost, branch_name, q_candidate, observation_cost, continuity_cost))

        switch_penalty = float(c.get("ik_branch_switch_penalty", 0.02))
        ranked = [
            (row[0] + (switch_penalty if row[1] != self._last_ik_branch else 0.0), row)
            for row in evaluated
        ]
        selected = min(ranked, key=lambda item: item[0])[1]
        warm_result = next(row for row in evaluated if row[1] == "warm_start")
        switch_margin = float(c.get("ik_branch_switch_margin", 0.0))
        selected_rank = next(score for score, row in ranked if row[1] == selected[1])
        warm_rank = next(score for score, row in ranked if row[1] == "warm_start")
        if selected[1] != "warm_start" and selected_rank > warm_rank - switch_margin:
            selected = warm_result
        _, branch_name, q_solution, observation_cost, continuity_cost = selected
        self.configuration.update(q_solution)
        self._last_ik_branch = branch_name
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
        dq = np.zeros(self.nv)
        if dt is not None and self._q_prev is not None:
            dq[:3] = (q[:3] - self._q_prev[:3]) / dt
            r0, r1 = (Rotation.from_quat(x[[4, 5, 6, 3]]) for x in (self._q_prev, q))
            dq[3:6] = (r0.inv() * r1).as_rotvec() / dt
            dq[6:] = (q[7:] - self._q_prev[7:]) / dt
        self._q_prev = q
        return KeypointIKFrame(q=q, dq=dq, observed=True, held_groups=tuple(held),
                               solve_ms=1e3 * (time.perf_counter() - t0),
                               ik_branch=branch_name,
                               ik_task_cost=float(np.sqrt(max(observation_cost, 0.0))),
                               ik_continuity_cost=float(np.sqrt(max(continuity_cost, 0.0))),
                               ik_candidates_evaluated=len(evaluated))


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


def retarget_sequence(
    points: np.ndarray,
    t_s: np.ndarray,
    retargeter: KeypointIKRetargeter | None = None,
    *,
    robot: str | None = None,
) -> dict:
    """Offline driver that feeds frames one at a time (identical to streaming): returns q (T,7+n_dof), dq (T,6+n_dof),
    observed (T,), solve_ms (T,), held (list of tuples)."""
    r = retargeter or KeypointIKRetargeter(robot=robot)
    r.reset()
    out = [r.step(p, float(t)) for p, t in zip(np.asarray(points, np.float64), np.asarray(t_s, np.float64))]
    return dict(q=np.array([f.q for f in out]), dq=np.array([f.dq for f in out]),
                observed=np.array([f.observed for f in out]), solve_ms=np.array([f.solve_ms for f in out]),
                held=[f.held_groups for f in out])


_ROBAN_TO_HEFT_ORDER = np.array([ROBAN_21_JOINTS.index(n) for n in ROBAN_HEFT_21_JOINTS], dtype=np.int32)
_HEFT_TO_ROBAN_ORDER = np.argsort(_ROBAN_TO_HEFT_ORDER)
_HEFT_JOINT_DIRECTION = np.array([-1.0] + [1.0] * 20, dtype=np.float64)


def export_heft_deploy_csv(path: str | Path, q_traj_28: np.ndarray, dq_traj_27: np.ndarray | None = None) -> Path:
    """Export a Roban S17 trajectory (q: (T,28) in ROBAN_21_JOINTS order, dq: (T,27)) to a 49-column LejuLab HEFT CSV.

    Applies the ROBAN_21_JOINTS -> ROBAN_HEFT_21_JOINTS permutation and the `waist_yaw_joint` sign flip (`-1`)
    expected by `HeftRLController` / `MotionTrajectory`.
    """
    q = np.asarray(q_traj_28, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 28:
        raise ValueError(f"Expected q_traj_28 of shape (T, 28), got {q.shape}")
    if dq_traj_27 is None:
        dq_joints = np.zeros((q.shape[0], 21), dtype=np.float64)
    else:
        dq = np.asarray(dq_traj_27, dtype=np.float64)
        if dq.shape != (q.shape[0], 27):
            raise ValueError(f"Expected dq_traj_27 of shape ({q.shape[0]}, 27), got {dq.shape}")
        dq_joints = dq[:, 6:]
    jpos_heft = q[:, 7:][:, _ROBAN_TO_HEFT_ORDER] * _HEFT_JOINT_DIRECTION
    jvel_heft = dq_joints[:, _ROBAN_TO_HEFT_ORDER] * _HEFT_JOINT_DIRECTION
    mat = np.hstack([q[:, :7], jpos_heft, jvel_heft])
    header_cols = (
        ["body_pos_x", "body_pos_y", "body_pos_z", "body_quat_w", "body_quat_x", "body_quat_y", "body_quat_z"]
        + [f"joint_pos{i:02d}" for i in range(21)]
        + [f"joint_vel_{i:02d}" for i in range(21)]
    )
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(dst, mat, delimiter="\t", header="\t".join(header_cols), comments="")
    return dst


def load_heft_deploy_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a 49-column LejuLab HEFT deploy CSV into S6 layout `(q (T, 28), dq (T, 27))` in ROBAN_21_JOINTS order."""
    raw = np.loadtxt(Path(path), delimiter="\t", skiprows=1, ndmin=2)
    if raw.shape[1] != 49:
        raise ValueError(f"Expected 49-column HEFT deploy CSV, got {raw.shape}")
    pos = raw[:, 0:3]
    quat = raw[:, 3:7]
    jpos_roban = (raw[:, 7:28] * _HEFT_JOINT_DIRECTION)[:, _HEFT_TO_ROBAN_ORDER]
    jvel_roban = (raw[:, 28:49] * _HEFT_JOINT_DIRECTION)[:, _HEFT_TO_ROBAN_ORDER]
    q = np.hstack([pos, quat, jpos_roban])
    dq = np.hstack([np.zeros((raw.shape[0], 6), dtype=np.float64), jvel_roban])
    return q, dq
