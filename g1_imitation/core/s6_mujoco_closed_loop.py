"""Module S6: MuJoCo Dual-Stage Closed-Loop Verification (Stage 6A Kinematic & Stage 6B Physics RL WBC with SONIC v1.1).

Specification Reference: `docs/方案规范.md` — 模块 S6
Input:
  - `q_ik_traj`: np.ndarray of shape (T, 36) at 100 Hz (dt = 0.01 s) from Module S4+S5 (default: causal keypoint IK
    `solve_ik_trajectory_from_3d`; offline reference: SMPL-X + GMR `solve_gmr_trajectory_from_3d`), `[x,y,z,qw,qx,qy,qz, 29 joints]`.
  - `dq_ik_traj`: np.ndarray of shape (T, 35) at 100 Hz (dt = 0.01 s) from Module S4+S5 (`[vx,vy,vz,wx,wy,wz, 29 joint vels]`).
Output:
  - `Stage6AKinematicReport` (Stage 6A: Pure Kinematic Replay via `mj_forward`):
    - Verifies visual consistency, joint limit compliance, ground non-penetration, and kinematic foot skating.
  - `Stage6BPhysicsReport` (Stage 6B: configured-rate physics + 50 Hz SONIC v1.1 whole-body tracker via `mj_step`):
    - `survived`: bool (pelvis height z > 0.45 m and body tilt < 45 deg throughout trajectory)
    - `min_pelvis_height_m`: float
    - `max_tilt_deg`: float
    - `joint_rmse_rad`: float (RMSE between physical joint angles q_phys(t) and IK reference q_IK(t))
    - `support_foot_skating_speed_ms`: float (average horizontal foot speed during support F_z > 20 N)
    - `mean_torque_nm`, `max_torque_nm`, `torque_delta_rms_nm`
    - Full trajectory arrays (`qpos_hist`, `q_joints_phys_hist`, `q_joints_ref_hist`, `torques_hist`,
      `left_fz_hist`, `right_fz_hist`, `left_foot_speed_xy_hist`, `right_foot_speed_xy_hist`)
Failure Behavior:
  - Raises `ValueError` on malformed trajectory shapes or non-finite inputs.
  - Detects and records exact fall timestamp/reason if `z_pelvis <= 0.45 m` or `tilt >= 45 deg`.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation, Slerp

from .constants import G1_29_JOINTS, load_config
from .s45_gmr_retarget import resample_points, retarget_trajectory_from_3d
from .s45c_keypoint_ik import KeypointIKRetargeter, causal_upsample, retarget_sequence
from .synthetic_human import synthesize_human_27_pose


# Plant constants shared by every controller (so controllers stay comparable):
#   COMMON_DEFAULT_Q_29: initial standing posture of the plant (the pinned G1 sim2sim default).
#   COMMON_TORQUE_KP_29: per-joint PD-gain table that bounds the motor torque envelope, as in the pinned plant
#                        (effort = min(kp, actuator force range)). It is NOT the controller's PD gain:
#                        the controller supplies its own trained Kp/Kd (see `SonicPolicy.kp/.kd`).
COMMON_DEFAULT_Q_29 = np.array(
    [
        -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
        -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0,
        0.0, -0.4, 0.0, 1.2, 0.0, 0.0, 0.0,
    ],
    dtype=np.float64,
)

COMMON_TORQUE_KP_29 = np.array(
    [
        100.0, 100.0, 100.0, 150.0, 40.0, 40.0,
        100.0, 100.0, 100.0, 150.0, 40.0, 40.0,
        150.0, 150.0, 150.0,
        40.0, 40.0, 40.0, 40.0, 4.0, 4.0, 4.0,
        40.0, 40.0, 40.0, 40.0, 4.0, 4.0, 4.0,
    ],
    dtype=np.float64,
)

DEFAULT_HEIGHT_M = 0.793


@dataclass(frozen=True)
class Stage6AKinematicReport:
    """Output of Stage 6A pure kinematic replay (`mj_forward`)."""
    num_frames: int
    min_pelvis_height_m: float
    max_tilt_deg: float
    min_foot_z_m: float
    support_foot_skating_speed_ms: float
    joint_limit_violation_rate: float
    qpos_hist: np.ndarray


@dataclass(frozen=True)
class Stage6BPhysicsReport:
    """Output of Stage 6B physics + 50 Hz RL closed-loop simulation (`mj_step`)."""
    num_policy_steps: int
    survived: bool
    min_pelvis_height_m: float
    max_tilt_deg: float
    joint_rmse_rad: float
    joint_rmse_deg: float
    support_foot_skating_speed_ms: float
    mean_torque_nm: float
    max_torque_nm: float
    torque_delta_rms_nm: float
    qpos_hist: np.ndarray
    q_joints_phys_hist: np.ndarray
    q_joints_ref_hist: np.ndarray
    torques_hist: np.ndarray
    left_fz_hist: np.ndarray
    right_fz_hist: np.ndarray
    left_foot_speed_xy_hist: np.ndarray
    right_foot_speed_xy_hist: np.ndarray
    controller: str = "sonic_v11"


def compute_tilt_deg_from_quat_wxyz(quat_wxyz: np.ndarray) -> float:
    """Compute body tilt angle (degrees) between body +Z axis and world +Z axis."""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    q = q / float(np.linalg.norm(q))
    R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    cos_tilt = float(np.clip(R[2, 2], -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_tilt)))


def _rotation6(matrix: np.ndarray) -> np.ndarray:
    """First two matrix columns, row-major flattening (the official SONIC layout, not column stacking)."""
    return np.asarray(matrix)[..., :, :2].reshape(-1)


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class SonicPolicy:
    """Self-contained SONIC v1.1 (GEAR-SONIC) whole-body tracker: ONNX encoder + decoder, 50 Hz, CPU.

    Input per tick (`step`): reference joint angles q_ref[29] (rad), joint velocities dq_ref[29] (rad/s) and the
    reference root orientation (scipy Rotation, world); all SDK joint order. The measured plant state comes from
    `G1PhysicsPlant.state()`.
    Output: 29 joint position targets (rad, SDK order). `kp`/`kd` are the controller's own trained PD gains.
    Reference window: the G1 encoder takes ten samples spaced `reference_step_s` apart (v1.1: 0.1 s -> 0..900 ms;
    low_latency: 0.02 s -> 0..180 ms). `step` repeats the CURRENT sample in all ten slots (no preview);
    `step_window` takes ten explicit samples, which a causal deployment can only fill by delaying the tracked
    reference (see `MuJoCoClosedLoopVerifier.run_stage_6b_physics(preview_s=...)`). No prediction is supplied.
    The root XY/height are not inputs of this policy.
    Failure: non-finite reference or action, wrong ONNX widths or a hash mismatch raise ValueError.
    Conventions ported from NVlabs/GR00T-WholeBodyControl 45c03fac (Apache-2.0).
    """

    name = "sonic_v11"
    WINDOW = 10

    def __init__(self, cfg: dict | None = None) -> None:
        full = load_config()
        self.cfg = cfg or full["s6_closed_loop"]["sonic_v11"]
        assets_root = Path(full["runtime"]["robot_assets_root"])
        enc_path = assets_root / self.cfg["encoder_onnx_relative_path"]
        dec_path = assets_root / self.cfg["decoder_onnx_relative_path"]
        for path, digest in ((enc_path, self.cfg["encoder_onnx_sha256"]), (dec_path, self.cfg["decoder_onnx_sha256"])):
            if _sha256(path) != digest:
                raise ValueError(f"SONIC asset fingerprint mismatch: {path.name}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        self.encoder = ort.InferenceSession(str(enc_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.decoder = ort.InferenceSession(str(dec_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.encoder_width = int(self.cfg["encoder_width"])
        if self.encoder.get_inputs()[0].shape != [1, self.encoder_width]:
            raise ValueError("Unexpected SONIC encoder observation layout")
        if self.decoder.get_inputs()[0].shape != [1, int(self.cfg["decoder_width"])]:
            raise ValueError("Unexpected SONIC decoder observation layout")
        self.order = np.array([G1_29_JOINTS.index(n) for n in self.cfg["policy_joint_names"]])
        if sorted(self.order.tolist()) != list(range(29)):
            raise ValueError("SONIC joint mapping is not bijective")
        self.inverse = np.argsort(self.order)
        self.kp = np.array(self.cfg["kp_sdk_order"], dtype=np.float64)
        self.kd = np.array(self.cfg["kd_sdk_order"], dtype=np.float64)
        self.default_q = np.array(self.cfg["default_q_sdk_order"], dtype=np.float64)
        self.scale = np.array(self.cfg["action_scale_sdk_order"], dtype=np.float64)
        if not (self.kp.shape == self.kd.shape == self.default_q.shape == self.scale.shape == (29,)):
            raise ValueError("SONIC configuration vectors must have 29 entries")
        self.heading_frame = self.cfg["orientation_frame"] == "heading"
        self.reference_step_s = float(self.cfg["reference_step_s"])
        self.history: deque = deque(maxlen=10)
        self.last_action = np.zeros(29, dtype=np.float64)

    def reset(self) -> None:
        self.history = deque(maxlen=10)
        self.last_action = np.zeros(29, dtype=np.float64)

    def step(
        self,
        ref_q_29: np.ndarray,
        ref_dq_29: np.ndarray,
        ref_rotation: Rotation,
        q_29: np.ndarray,
        dq_29: np.ndarray,
        quat_wxyz: np.ndarray,
        omega_body: np.ndarray,
    ) -> np.ndarray:
        ref_q = np.asarray(ref_q_29, dtype=np.float64)
        ref_dq = np.asarray(ref_dq_29, dtype=np.float64)
        if ref_q.shape != (29,) or ref_dq.shape != (29,) or not ref_rotation.single:
            raise ValueError("Require a single 29-joint SONIC reference sample")
        rots = Rotation.concatenate([ref_rotation] * self.WINDOW)
        return self.step_window(np.tile(ref_q, (self.WINDOW, 1)), np.tile(ref_dq, (self.WINDOW, 1)), rots,
                                q_29, dq_29, quat_wxyz, omega_body)

    def step_window(
        self,
        ref_q_win: np.ndarray,
        ref_dq_win: np.ndarray,
        ref_rotations: Rotation,
        q_29: np.ndarray,
        dq_29: np.ndarray,
        quat_wxyz: np.ndarray,
        omega_body: np.ndarray,
    ) -> np.ndarray:
        """Ten reference samples (10,29) q/dq and 10 root rotations at offsets k * reference_step_s, k = 0..9."""
        ref_q = np.asarray(ref_q_win, dtype=np.float64)
        ref_dq = np.asarray(ref_dq_win, dtype=np.float64)
        if (ref_q.shape != (self.WINDOW, 29) or ref_dq.shape != (self.WINDOW, 29) or ref_rotations.single
                or len(ref_rotations) != self.WINDOW
                or not np.isfinite(np.r_[ref_q.ravel(), ref_dq.ravel(), ref_rotations.as_quat().ravel()]).all()):
            raise ValueError("Require a finite ten-sample 29-joint SONIC reference window (q, dq, root orientation)")
        rotation = Rotation.from_quat(np.asarray(quat_wxyz, dtype=np.float64)[[1, 2, 3, 0]])
        current = (
            np.asarray(omega_body, dtype=np.float64).copy(),
            (np.asarray(q_29, dtype=np.float64) - self.default_q)[self.order],
            np.asarray(dq_29, dtype=np.float64)[self.order],
            self.last_action.copy(),
            rotation.inv().apply([0.0, 0.0, -1.0]),
        )
        if not self.history:
            # Official StateLogger::makeZeroEntry_ uses a zero quaternion, so its gravity term for missing
            # history evaluates to +Z (not the -Z of a valid identity rotation).
            empty = (np.zeros(3), np.zeros(29), np.zeros(29), np.zeros(29), np.array([0.0, 0.0, 1.0]))
            self.history.extend([empty] * (self.WINDOW - 1))
        self.history.append(current)

        anchor = rotation
        if self.heading_frame:
            forward = rotation.apply([1.0, 0.0, 0.0])
            anchor = Rotation.from_euler("z", np.arctan2(forward[1], forward[0]))
        enc = np.zeros(self.encoder_width, dtype=np.float32)
        # Mode 0 selects the G1 encoder; the unused teleoperation/SMPL branches stay zero.
        enc[4:294] = ref_q[:, self.order].ravel()
        enc[294:584] = ref_dq[:, self.order].ravel()
        enc[584:644] = np.concatenate([_rotation6(m) for m in (anchor.inv() * ref_rotations).as_matrix()])
        history = np.concatenate([np.concatenate([row[j] for row in self.history]) for j in range(5)])
        token = self.encoder.run(["encoded_tokens"], {"obs_dict": enc[None]})[0].reshape(64)
        obs = np.r_[token, history].astype(np.float32)[None]
        action = self.decoder.run(["action"], {"obs_dict": obs})[0].reshape(29).astype(np.float64)
        if not np.isfinite(action).all():
            raise ValueError("Nonfinite SONIC action; terminate trial")
        self.last_action = action.copy()
        return self.default_q + self.scale * action[self.inverse]


def build_body_kinematic_model(mjcf_path: str | Path) -> mujoco.MjModel:
    """Floating base + the 29 G1 body joints only (nq 36, nv 35, SDK joint order), from the same MJCF as the plant.

    Finger bodies, finger actuators and sensors are removed; link geometry and joint ranges are unchanged. Used for
    FK/IK/rendering of 36-dimensional configurations, never for dynamics.
    """
    spec = mujoco.MjSpec.from_file(str(mjcf_path))
    roots = [spec.body(f"{s}_hand_{f}_0_link") for s in ("left", "right") for f in ("thumb", "middle", "index")]
    dead = {j.name for r in roots for j in _subtree_joints(r)}
    for a in [a for a in spec.actuators if a.target in dead]:
        spec.delete(a)
    for s in list(spec.sensors):
        spec.delete(s)
    for r in roots:
        spec.delete(r)
    model = spec.compile()
    names = [model.joint(i).name for i in range(1, model.njnt)]
    if model.nq != 36 or names != list(G1_29_JOINTS):
        raise ValueError(f"Kinematic model is not the 29-joint G1: nq={model.nq}")
    return model


def _subtree_joints(body):
    for j in body.joints:
        yield j
    for c in body.bodies:
        yield from _subtree_joints(c)


class G1PhysicsPlant:
    """Self-contained Unitree G1 29-DoF MuJoCo plant (physics step and torque limits from config, 50 Hz PD decimation).

    The 29 body joints and their actuators are looked up by name, so a model with extra joints (e.g. NVIDIA's
    official 43-DoF G1 with Dex3 hands) also loads; extra actuators receive zero torque (passive).
    Defaults (model, physics_dt, effort_limit) come from config `s6_closed_loop`.
    `effort_limit`: "common_kp_capped" = min(COMMON_TORQUE_KP_29, actuator range);
    "actuator_range" = the model's actuator force range, as in the official SONIC MuJoCo sim.
    """

    def __init__(self, mjcf_path: str | Path | None = None, physics_dt: float | None = None,
                 effort_limit: str | None = None) -> None:
        cfg = load_config()
        s6 = cfg["s6_closed_loop"]
        physics_dt = s6["physics_dt_s"] if physics_dt is None else physics_dt
        effort_limit = s6["effort_limit"] if effort_limit is None else effort_limit
        if mjcf_path is None:
            assets_root = Path(cfg["runtime"]["robot_assets_root"])
            mjcf_path = assets_root / cfg["s6_closed_loop"]["mjcf_relative_path"]
        self.mjcf_path = Path(mjcf_path).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.mjcf_path))
        self._kin_model: mujoco.MjModel | None = None
        self.model.opt.timestep = float(physics_dt)
        self.data = mujoco.MjData(self.model)

        self.joint_ids = np.array([self.model.joint(n).id for n in G1_29_JOINTS], dtype=np.int32)
        if self.model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or self.model.jnt_qposadr[0] != 0:
            raise ValueError("Expected a floating base as the first joint")
        self.qadr = self.model.jnt_qposadr[self.joint_ids].copy()
        self.vadr = self.model.jnt_dofadr[self.joint_ids].copy()
        act_of_joint = {int(self.model.actuator_trnid[a, 0]): a for a in range(self.model.nu)
                        if self.model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT}
        if any(int(j) not in act_of_joint for j in self.joint_ids):
            raise ValueError("Every G1 body joint needs a joint actuator")
        self.act_ids = np.array([act_of_joint[int(j)] for j in self.joint_ids], dtype=np.int32)
        if not np.allclose(self.model.actuator_gear[self.act_ids, 0], 1.0):
            raise ValueError("Expected unit-gear torque actuators")
        self.limits = self.model.jnt_range[self.joint_ids].copy()
        act_range = self.model.jnt_actfrcrange[self.joint_ids, 1]
        if effort_limit == "common_kp_capped":
            self.effort = np.minimum(COMMON_TORQUE_KP_29, act_range)
        elif effort_limit == "actuator_range":
            self.effort = act_range.copy()
        else:
            raise ValueError(f"Unknown effort_limit {effort_limit!r}")
        if not np.all(self.effort > 0.0):
            raise ValueError("Actuator force ranges must be positive")

        self.left_ankle_body_id = self.model.body("left_ankle_roll_link").id
        self.right_ankle_body_id = self.model.body("right_ankle_roll_link").id
        self.left_foot_geoms = set(
            range(
                int(self.model.body_geomadr[self.left_ankle_body_id]),
                int(self.model.body_geomadr[self.left_ankle_body_id] + self.model.body_geomnum[self.left_ankle_body_id]),
            )
        )
        self.right_foot_geoms = set(
            range(
                int(self.model.body_geomadr[self.right_ankle_body_id]),
                int(self.model.body_geomadr[self.right_ankle_body_id] + self.model.body_geomnum[self.right_ankle_body_id]),
            )
        )
        self.reset()

    def reset(self, q_joints_init: np.ndarray | None = None) -> None:
        q_j = COMMON_DEFAULT_Q_29 if q_joints_init is None else np.asarray(q_joints_init, dtype=np.float64)
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[:7] = [0.0, 0.0, DEFAULT_HEIGHT_M, 1.0, 0.0, 0.0, 0.0]
        self.data.qpos[self.qadr] = q_j
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    @property
    def kin_model(self) -> mujoco.MjModel:
        """29-joint (nq 36) kinematic twin of the plant model; see `build_body_kinematic_model`."""
        if self._kin_model is None:
            self._kin_model = build_body_kinematic_model(self.mjcf_path)
        return self._kin_model

    def qpos36(self) -> np.ndarray:
        """[pelvis xyz, wxyz, 29 body joints in SDK order]."""
        return np.r_[self.data.qpos[:7], self.data.qpos[self.qadr]]

    def set_qpos36(self, q36: np.ndarray) -> None:
        self.data.qpos[:7] = q36[:7]
        self.data.qpos[self.qadr] = q36[7:]

    def state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.data.qpos[self.qadr].copy(),
            self.data.qvel[self.vadr].copy(),
            self.data.qpos[3:7].copy(),
            self.data.qvel[3:6].copy(),
        )

    def step_pd(self, q_des_29: np.ndarray, kp: np.ndarray, kd: np.ndarray, control_dt: float = 0.02) -> np.ndarray:
        """Execute `round(control_dt / physics_dt)` PD sub-steps (config physics_dt_s) with the CONTROLLER's Kp/Kd (SDK order)."""
        kp = np.asarray(kp, dtype=np.float64)
        kd = np.asarray(kd, dtype=np.float64)
        if kp.shape != (29,) or kd.shape != (29,):
            raise ValueError("kp and kd must have 29 entries")
        target = np.clip(np.asarray(q_des_29, dtype=np.float64), self.limits[:, 0], self.limits[:, 1])
        substeps = max(1, round(float(control_dt) / float(self.model.opt.timestep)))
        last_tau = np.zeros(29, dtype=np.float64)
        for _ in range(substeps):
            q = self.data.qpos[self.qadr]
            dq = self.data.qvel[self.vadr]
            tau = np.clip(kp * (target - q) - kd * dq, -self.effort, self.effort)
            self.data.ctrl[self.act_ids] = tau
            mujoco.mj_step(self.model, self.data)
            last_tau = tau.copy()
        return last_tau

    def measure_foot_vertical_forces_n(self) -> tuple[float, float]:
        """Compute exact vertical ground reaction forces (F_z in Newtons) on left and right feet."""
        fz_left = 0.0
        fz_right = 0.0
        cf = np.zeros(6, dtype=np.float64)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            mujoco.mj_contactForce(self.model, self.data, i, cf)
            f_world = con.frame.reshape(3, 3).T @ cf[:3]
            if con.geom1 in self.left_foot_geoms or con.geom2 in self.left_foot_geoms:
                fz_left += abs(float(f_world[2]))
            if con.geom1 in self.right_foot_geoms or con.geom2 in self.right_foot_geoms:
                fz_right += abs(float(f_world[2]))
        return fz_left, fz_right


class MuJoCoClosedLoopVerifier:
    """Module S6 implementation: runs Stage 6A (Kinematic) and Stage 6B (Physics RL Closed-Loop)."""

    def __init__(
        self,
        mjcf_path: str | Path | None = None,
        policy: SonicPolicy | None = None,
        support_fz_threshold_n: float = 20.0,
        survival_min_pelvis_z_m: float = 0.45,
        survival_max_tilt_deg: float = 45.0,
        physics_dt: float | None = None,
        effort_limit: str | None = None,
    ) -> None:
        self.plant = G1PhysicsPlant(mjcf_path=mjcf_path, physics_dt=physics_dt, effort_limit=effort_limit)
        self.policy = policy or SonicPolicy()
        self.support_fz_threshold_n = float(support_fz_threshold_n)
        self.survival_min_pelvis_z_m = float(survival_min_pelvis_z_m)
        self.survival_max_tilt_deg = float(survival_max_tilt_deg)

    @staticmethod
    def split_reference(q_ik_36: np.ndarray, dq_ik_35: np.ndarray) -> tuple[np.ndarray, np.ndarray, Rotation]:
        """Module S4+S5 (GMR) output (q in R^36, dq in R^35) -> SONIC reference (q[29], dq[29], root orientation).

        No clipping, rate limiting or smoothing is applied here; the policy sees exactly what the retargeter produced.
        """
        q_ik = np.asarray(q_ik_36, dtype=np.float64)
        dq_ik = np.asarray(dq_ik_35, dtype=np.float64)
        return q_ik[7:].copy(), dq_ik[6:].copy(), Rotation.from_quat(q_ik[[4, 5, 6, 3]])

    def run_stage_6a_kinematic(
        self,
        q_ik_traj: np.ndarray,
        dt: float = 0.01,
    ) -> Stage6AKinematicReport:
        """Stage 6A: Pure kinematic replay (`d.qpos[:] = q_t^{IK}`, `mj_forward`)."""
        q_traj = np.asarray(q_ik_traj, dtype=np.float64)
        if q_traj.ndim != 2 or q_traj.shape[1] != 36:
            raise ValueError(f"Expected q_ik_traj of shape (T, 36), got {q_traj.shape}")

        m = self.plant.model
        d = mujoco.MjData(m)
        l_bid = self.plant.left_ankle_body_id
        r_bid = self.plant.right_ankle_body_id

        q_min = self.plant.limits[:, 0]
        q_max = self.plant.limits[:, 1]
        viol_count = 0

        min_pelvis_z = float("inf")
        max_tilt = 0.0
        min_foot_z = float("inf")
        skating_speeds: list[float] = []
        prev_l_pos: np.ndarray | None = None
        prev_r_pos: np.ndarray | None = None

        for t_idx in range(q_traj.shape[0]):
            q_t = q_traj[t_idx]
            self.plant.set_qpos36(q_t)
            mujoco.mj_forward(m, d)

            q_j = q_t[7:]
            if np.any(q_j < q_min - 1e-5) or np.any(q_j > q_max + 1e-5):
                viol_count += 1

            pelvis_z = float(q_t[2])
            min_pelvis_z = min(min_pelvis_z, pelvis_z)
            tilt_deg = compute_tilt_deg_from_quat_wxyz(q_t[3:7])
            max_tilt = max(max_tilt, tilt_deg)

            l_pos = d.xpos[l_bid].copy()
            r_pos = d.xpos[r_bid].copy()
            min_foot_z = min(min_foot_z, float(l_pos[2]), float(r_pos[2]))

            if prev_l_pos is not None and prev_r_pos is not None:
                # Check kinematic support foot speed when foot is near ground (z <= 0.055 m)
                if l_pos[2] <= 0.055:
                    skating_speeds.append(float(np.linalg.norm((l_pos[:2] - prev_l_pos[:2]) / dt)))
                if r_pos[2] <= 0.055:
                    skating_speeds.append(float(np.linalg.norm((r_pos[:2] - prev_r_pos[:2]) / dt)))
            prev_l_pos = l_pos
            prev_r_pos = r_pos

        mean_skating = float(np.mean(skating_speeds)) if skating_speeds else 0.0
        viol_rate = float(viol_count) / float(max(1, q_traj.shape[0]))
        return Stage6AKinematicReport(
            num_frames=int(q_traj.shape[0]),
            min_pelvis_height_m=float(min_pelvis_z),
            max_tilt_deg=float(max_tilt),
            min_foot_z_m=float(min_foot_z),
            support_foot_skating_speed_ms=mean_skating,
            joint_limit_violation_rate=viol_rate,
            qpos_hist=q_traj.copy(),
        )

    def run_stage_6b_physics(
        self,
        q_ik_traj: np.ndarray,
        dq_ik_traj: np.ndarray,
        ik_dt: float = 0.01,
        control_dt: float = 0.02,
        settle_steps: int = 25,
        preview_s: float = 0.0,
    ) -> Stage6BPhysicsReport:
        """Stage 6B: configured-rate Physics + 50 Hz RL Whole-Body Control closed-loop verification.

        `preview_s` = how far ahead of the tracked sample the policy may see. In a causal deployment this equals an
        added human->robot delay: the robot tracks the sample received `preview_s` ago, so window slots at offsets
        <= preview_s hold real (already received) samples and later slots repeat the newest received one.
        0 (default) = current sample repeated. Slots past the clip end hold its last sample. Joint RMSE is scored
        against the tracked (delayed) sample, i.e. the delay itself is not counted as tracking error.
        """
        q_traj = np.asarray(q_ik_traj, dtype=np.float64)
        dq_traj = np.asarray(dq_ik_traj, dtype=np.float64)
        if q_traj.ndim != 2 or q_traj.shape[1] != 36:
            raise ValueError(f"Expected q_ik_traj of shape (T, 36), got {q_traj.shape}")
        if dq_traj.shape != (q_traj.shape[0], 35):
            raise ValueError(f"Expected dq_ik_traj of shape ({q_traj.shape[0]}, 35), got {dq_traj.shape}")

        stride = max(1, round(float(control_dt) / float(ik_dt)))
        if preview_s < 0.0:
            raise ValueError("preview_s must be >= 0")
        offsets = np.round(np.minimum(np.arange(self.policy.WINDOW) * self.policy.reference_step_s, preview_s)
                           / float(ik_dt)).astype(int)
        indices = list(range(0, q_traj.shape[0], stride))
        n_steps = len(indices)

        self.plant.reset()
        self.policy.reset()

        # Smoothly settle from default posture to the initial frame's reference over `settle_steps`
        q0, _, rot0 = self.split_reference(q_traj[0], np.zeros(35, dtype=np.float64))
        slerp = Slerp([0.0, 1.0], Rotation.concatenate([Rotation.identity(), rot0]))
        for s in range(settle_steps):
            alpha = float(s + 1) / float(max(1, settle_steps))
            q_s = (1.0 - alpha) * COMMON_DEFAULT_Q_29 + alpha * q0
            q_cur, dq_cur, quat_cur, omega_cur = self.plant.state()
            q_des = self.policy.step(q_s, np.zeros(29), slerp(alpha), q_cur, dq_cur, quat_cur, omega_cur)
            self.plant.step_pd(q_des, self.policy.kp, self.policy.kd, control_dt=control_dt)

        qpos_hist = np.zeros((n_steps, 36), dtype=np.float64)
        q_joints_phys_hist = np.zeros((n_steps, 29), dtype=np.float64)
        q_joints_ref_hist = np.zeros((n_steps, 29), dtype=np.float64)
        torques_hist = np.zeros((n_steps, 29), dtype=np.float64)
        left_fz_hist = np.zeros(n_steps, dtype=np.float64)
        right_fz_hist = np.zeros(n_steps, dtype=np.float64)
        left_speed_hist = np.zeros(n_steps, dtype=np.float64)
        right_speed_hist = np.zeros(n_steps, dtype=np.float64)

        l_bid = self.plant.left_ankle_body_id
        r_bid = self.plant.right_ankle_body_id
        prev_l_pos = self.plant.data.xpos[l_bid].copy()
        prev_r_pos = self.plant.data.xpos[r_bid].copy()

        survived = True
        min_pelvis_z = float("inf")
        max_tilt = 0.0
        support_skating_samples: list[float] = []

        for step_idx, frame_idx in enumerate(indices):
            q_ik = q_traj[frame_idx]
            dq_ik = dq_traj[frame_idx]
            ref_q, ref_dq, ref_rot = self.split_reference(q_ik, dq_ik)

            win = np.minimum(frame_idx + offsets, q_traj.shape[0] - 1)
            q_cur, dq_cur, quat_cur, omega_cur = self.plant.state()
            q_des = self.policy.step_window(q_traj[win, 7:], dq_traj[win, 6:],
                                            Rotation.from_quat(q_traj[win][:, [4, 5, 6, 3]]),
                                            q_cur, dq_cur, quat_cur, omega_cur)
            tau = self.plant.step_pd(q_des, self.policy.kp, self.policy.kd, control_dt=control_dt)

            qpos_now = self.plant.qpos36()
            qpos_hist[step_idx] = qpos_now
            q_joints_phys_hist[step_idx] = qpos_now[7:]
            q_joints_ref_hist[step_idx] = q_ik[7:]
            torques_hist[step_idx] = tau

            pelvis_z = float(qpos_now[2])
            tilt_deg = compute_tilt_deg_from_quat_wxyz(qpos_now[3:7])
            min_pelvis_z = min(min_pelvis_z, pelvis_z)
            max_tilt = max(max_tilt, tilt_deg)
            if pelvis_z <= self.survival_min_pelvis_z_m or tilt_deg >= self.survival_max_tilt_deg:
                survived = False

            fz_l, fz_r = self.plant.measure_foot_vertical_forces_n()
            left_fz_hist[step_idx] = fz_l
            right_fz_hist[step_idx] = fz_r

            l_pos = self.plant.data.xpos[l_bid].copy()
            r_pos = self.plant.data.xpos[r_bid].copy()
            v_l_xy = float(np.linalg.norm((l_pos[:2] - prev_l_pos[:2]) / control_dt))
            v_r_xy = float(np.linalg.norm((r_pos[:2] - prev_r_pos[:2]) / control_dt))
            prev_l_pos = l_pos
            prev_r_pos = r_pos
            left_speed_hist[step_idx] = v_l_xy
            right_speed_hist[step_idx] = v_r_xy

            if fz_l > self.support_fz_threshold_n:
                support_skating_samples.append(v_l_xy)
            if fz_r > self.support_fz_threshold_n:
                support_skating_samples.append(v_r_xy)

        joint_err = q_joints_phys_hist - q_joints_ref_hist
        joint_rmse_rad = float(np.sqrt(np.mean(joint_err**2)))
        support_skating_ms = float(np.mean(support_skating_samples)) if support_skating_samples else 0.0
        mean_torque = float(np.mean(np.abs(torques_hist)))
        max_torque = float(np.max(np.abs(torques_hist)))
        torque_delta_rms = (
            float(np.sqrt(np.mean(np.diff(torques_hist, axis=0) ** 2))) if n_steps > 1 else 0.0
        )

        return Stage6BPhysicsReport(
            num_policy_steps=n_steps,
            survived=survived,
            min_pelvis_height_m=float(min_pelvis_z),
            max_tilt_deg=float(max_tilt),
            joint_rmse_rad=joint_rmse_rad,
            joint_rmse_deg=float(np.degrees(joint_rmse_rad)),
            support_foot_skating_speed_ms=support_skating_ms,
            mean_torque_nm=mean_torque,
            max_torque_nm=max_torque,
            torque_delta_rms_nm=torque_delta_rms,
            qpos_hist=qpos_hist,
            q_joints_phys_hist=q_joints_phys_hist,
            q_joints_ref_hist=q_joints_ref_hist,
            torques_hist=torques_hist,
            left_fz_hist=left_fz_hist,
            right_fz_hist=right_fz_hist,
            left_foot_speed_xy_hist=left_speed_hist,
            right_foot_speed_xy_hist=right_speed_hist,
        )


def _walking_gait_pose(
    t: float,
    height_m: float = 1.75,
    vx_ms: float = 0.16,
    step_freq_hz: float = 0.8,
    step_length_m: float = 0.10,
    step_height_m: float = 0.055,
    yaw_amp_rad: float = 0.0,
    squat_base_m: float = 0.07,
) -> np.ndarray:
    """Synthesize a human bipedal walking/turning pose with alternating stance/swing phases."""
    # Smoothly ramp gait amplitude over first 1.0 s
    ramp = min(1.0, t / 1.0)
    phase = (t * step_freq_hz) % 1.0  # [0, 1)
    # In [0, 0.5): left foot swings, right foot in stance
    # In [0.5, 1.0): right foot swings, left foot in stance
    if phase < 0.5:
        s = phase * 2.0  # [0, 1)
        l_x = ramp * step_length_m * (s - 0.5)
        l_z = ramp * step_height_m * np.sin(np.pi * s)
        r_x = ramp * step_length_m * (0.5 - s)
        r_z = 0.0
        pelvis_lat_y = -0.018 * ramp * np.sin(np.pi * s)
    else:
        s = (phase - 0.5) * 2.0  # [0, 1)
        r_x = ramp * step_length_m * (s - 0.5)
        r_z = ramp * step_height_m * np.sin(np.pi * s)
        l_x = ramp * step_length_m * (0.5 - s)
        l_z = 0.0
        pelvis_lat_y = +0.018 * ramp * np.sin(np.pi * s)

    root_x = vx_ms * max(0.0, t - 0.5) * ramp
    yaw = yaw_amp_rad * np.sin(2.0 * np.pi * 0.2 * t) * ramp
    arm_swing = 0.22 * ramp * np.sin(2.0 * np.pi * step_freq_hz * t)

    return synthesize_human_27_pose(
        height_m=height_m,
        squat_depth_m=squat_base_m,
        torso_yaw_rad=yaw,
        root_xy_m=np.array([root_x, pelvis_lat_y]),
        l_foot_offset_m=np.array([l_x, 0.0, l_z]),
        r_foot_offset_m=np.array([r_x, 0.0, r_z]),
        l_arm_dir=np.array([-arm_swing, 0.18, -0.95]),
        r_arm_dir=np.array([+arm_swing, -0.18, -0.95]),
    )


def generate_10_diverse_human_motions(
    duration_s: float = 10.0,
    fps_hz: float = 100.0,
) -> dict[str, np.ndarray]:
    """Generate 10 distinct 10-second (1000-frame) 27-point human motion trajectories (Section S6.3.1).

    Covers:
      1. `01_forward_walking_gait`: Bipedal forward walking (0.16 m/s, 5.5 cm foot lift, root translation)
      2. `02_turning_stepping_gait`: Turning with alternating stepping gait (+/- 0.40 rad yaw + foot lifts)
      3. `03_deep_squat_cycle`: Deep rhythmic squatting (0.20 m squat depth + torso pitch compensation)
      4. `04_waist_bend_reach`: Waist forward bend (0.45 rad / ~26 deg pitch) + bimanual forward reach
      5. `05_lateral_stepping_sway`: Lateral weight transfer + torso/hip roll + alternating foot lift
      6. `06_torso_twist_arm_wave`: Large axial torso twist (+/- 0.45 rad) + asymmetric arm waving
      7. `07_boxing_guard_combo`: Alternating punch extensions + knee bounce + torso rotation
      8. `08_overhead_bimanual_lift`: Bimanual lift from waist to overhead + squat coordination
      9. `09_tall_subject_190cm_walk_squat`: 1.90 m tall subject performing walking + squat
      10. `10_short_subject_155cm_walk_turn`: 1.55 m short subject performing walking + torso turn
    """
    n_frames = int(round(duration_s * fps_hz))
    t_vec = np.arange(n_frames, dtype=np.float64) / float(fps_hz)

    def build_seq(fn: Callable[[float], np.ndarray]) -> np.ndarray:
        out = np.zeros((n_frames, 27, 3), dtype=np.float64)
        for i, t in enumerate(t_vec):
            out[i] = fn(float(t))
        return out

    motions: dict[str, np.ndarray] = {
        "01_forward_walking_gait": build_seq(
            lambda t: _walking_gait_pose(
                t,
                height_m=1.75,
                vx_ms=0.16,
                step_freq_hz=0.8,
                step_length_m=0.10,
                step_height_m=0.055,
            )
        ),
        "02_turning_stepping_gait": build_seq(
            lambda t: _walking_gait_pose(
                t,
                height_m=1.75,
                vx_ms=0.05,
                step_freq_hz=0.8,
                step_length_m=0.08,
                step_height_m=0.05,
                yaw_amp_rad=0.40,
            )
        ),
        "03_deep_squat_cycle": build_seq(
            lambda t: synthesize_human_27_pose(
                height_m=1.75,
                squat_depth_m=0.20 * (0.5 - 0.5 * np.cos(2.0 * np.pi * 0.35 * t)),
                torso_pitch_rad=0.22 * (0.5 - 0.5 * np.cos(2.0 * np.pi * 0.35 * t)),
            )
        ),
        "04_waist_bend_reach": build_seq(
            lambda t: synthesize_human_27_pose(
                height_m=1.75,
                squat_depth_m=0.10 * (0.5 - 0.5 * np.cos(2.0 * np.pi * 0.3 * t)),
                torso_pitch_rad=0.45 * (0.5 - 0.5 * np.cos(2.0 * np.pi * 0.3 * t)),
                root_xy_m=np.array([-0.08 * (0.5 - 0.5 * np.cos(2.0 * np.pi * 0.3 * t)), 0.0]),
                l_arm_dir=np.array([0.55 * (0.5 - 0.5 * np.cos(2.0 * np.pi * 0.3 * t)), 0.20, -0.80]),
                r_arm_dir=np.array([0.55 * (0.5 - 0.5 * np.cos(2.0 * np.pi * 0.3 * t)), -0.20, -0.80]),
            )
        ),
        "05_lateral_stepping_sway": build_seq(
            lambda t: synthesize_human_27_pose(
                height_m=1.75,
                squat_depth_m=0.07,
                torso_roll_rad=0.10 * np.sin(2.0 * np.pi * 0.35 * t),
                root_xy_m=np.array([0.0, 0.04 * np.sin(2.0 * np.pi * 0.35 * t)]),
                l_foot_offset_m=np.array([0.0, 0.0, max(0.0, -0.04 * np.sin(2.0 * np.pi * 0.35 * t))]),
                r_foot_offset_m=np.array([0.0, 0.0, max(0.0, +0.04 * np.sin(2.0 * np.pi * 0.35 * t))]),
            )
        ),
        "06_torso_twist_arm_wave": build_seq(
            lambda t: synthesize_human_27_pose(
                height_m=1.75,
                squat_depth_m=0.07,
                torso_yaw_rad=0.45 * np.sin(2.0 * np.pi * 0.25 * t),
                l_arm_dir=np.array([0.25, 0.45, -0.70 + 0.35 * np.sin(2.0 * np.pi * 0.5 * t)]),
                r_arm_dir=np.array([0.25, -0.45, -0.70 - 0.35 * np.sin(2.0 * np.pi * 0.5 * t)]),
            )
        ),
        "07_boxing_guard_combo": build_seq(
            lambda t: synthesize_human_27_pose(
                height_m=1.75,
                squat_depth_m=0.08 + 0.03 * np.sin(2.0 * np.pi * 0.6 * t),
                torso_yaw_rad=0.22 * np.sin(2.0 * np.pi * 0.4 * t),
                l_forearm_dir=np.array([0.92, -0.12, -0.15 + 0.35 * np.sin(2.0 * np.pi * 0.4 * t)]),
                r_forearm_dir=np.array([0.92, 0.12, -0.15 - 0.35 * np.sin(2.0 * np.pi * 0.4 * t)]),
            )
        ),
        "08_overhead_bimanual_lift": build_seq(
            lambda t: synthesize_human_27_pose(
                height_m=1.75,
                squat_depth_m=0.07 + 0.05 * np.sin(2.0 * np.pi * 0.25 * t),
                l_arm_dir=np.array([0.20, 0.25, -0.80 + 0.60 * (0.5 - 0.5 * np.cos(2.0 * np.pi * 0.25 * t))]),
                r_arm_dir=np.array([0.20, -0.25, -0.80 + 0.60 * (0.5 - 0.5 * np.cos(2.0 * np.pi * 0.25 * t))]),
            )
        ),
        "09_tall_subject_190cm_walk_squat": build_seq(
            lambda t: _walking_gait_pose(
                t,
                height_m=1.90,
                vx_ms=0.14,
                step_freq_hz=0.75,
                step_length_m=0.10,
                step_height_m=0.05,
                squat_base_m=0.09,
            )
        ),
        "10_short_subject_155cm_walk_turn": build_seq(
            lambda t: _walking_gait_pose(
                t,
                height_m=1.55,
                vx_ms=0.12,
                step_freq_hz=0.85,
                step_length_m=0.08,
                step_height_m=0.045,
                yaw_amp_rad=0.30,
                squat_base_m=0.06,
            )
        ),
    }
    return motions


def _jump_pose(
    t: float,
    period_s: float = 2.5,
    start_s: float = 1.0,
    crouch_s: float = 0.35,
    push_s: float = 0.15,
    flight_s: float = 0.30,
    absorb_s: float = 0.15,
    recover_s: float = 0.40,
    depth_m: float = 0.15,
    forward_m: float = 0.0,
    support_only_side: str | None = None,
    height_m: float = 1.75,
    squat_base_m: float = 0.04,
) -> np.ndarray:
    """Synthetic jump cycle: crouch -> push-off -> ballistic flight (z = v0 t - g t^2 / 2) -> landing absorb -> recover.

    `support_only_side='left'` hops on the left foot with the right foot held up. Forward jumps advance the root
    linearly during flight. Ground truth only for simulation tests.
    """
    g = 9.81
    n_done = int(np.floor((t - start_s) / period_s)) if t >= start_s else -1
    tau = (t - start_s) - n_done * period_s if n_done >= 0 else -1.0
    squat, lift, x = 0.0, 0.0, forward_m * max(0, n_done)
    b = np.cumsum([crouch_s, push_s, flight_s, absorb_s, recover_s])
    if 0.0 <= tau < b[0]:
        squat = depth_m * 0.5 * (1.0 - np.cos(np.pi * tau / crouch_s))
    elif b[0] <= tau < b[1]:
        squat = depth_m * 0.5 * (1.0 + np.cos(np.pi * (tau - b[0]) / push_s))
    elif b[1] <= tau < b[2]:
        s = tau - b[1]
        lift = 0.5 * g * flight_s * s - 0.5 * g * s * s
        x += forward_m * s / flight_s
    elif b[2] <= tau < b[3]:
        squat = depth_m * 0.5 * (1.0 - np.cos(np.pi * (tau - b[2]) / absorb_s))
    elif b[3] <= tau < b[4]:
        squat = depth_m * 0.5 * (1.0 + np.cos(np.pi * (tau - b[3]) / recover_s))
    if tau >= b[2]:
        x += forward_m
    arm = 0.6 * float(np.clip(squat / max(depth_m, 1e-6), 0.0, 1.0))
    raise_side = {"left": "right", "right": "left"}.get(support_only_side or "")
    return synthesize_human_27_pose(
        height_m=height_m,
        squat_depth_m=squat_base_m + squat,
        torso_pitch_rad=0.8 * squat,
        root_xy_m=np.array([x, 0.0]),
        l_arm_dir=np.array([-arm, 0.18, -0.95]),
        r_arm_dir=np.array([-arm, -0.18, -0.95]),
        single_leg_raise_side=raise_side,
        root_z_offset_m=lift,
    )


def generate_jump_motions(duration_s: float = 10.0, fps_hz: float = 100.0) -> dict[str, np.ndarray]:
    """Synthetic jumps with a flight phase (both feet off the ground), not part of the S6.3 spec set:
      11_vertical_jump_series : 4 two-foot vertical jumps, 0.30 s flight (apex ~0.11 m)
      12_forward_jump_series  : 4 two-foot forward jumps, 0.20 m per jump
      13_single_leg_hop_left  : 6 hops on the left foot, right foot held up, 0.18 s flight
    """
    t_vec = np.arange(int(round(duration_s * fps_hz)), dtype=np.float64) / float(fps_hz)
    specs = {
        "11_vertical_jump_series": dict(),
        "12_forward_jump_series": dict(forward_m=0.20),
        "13_single_leg_hop_left": dict(period_s=1.5, depth_m=0.06, flight_s=0.18, crouch_s=0.25, push_s=0.12,
                                       absorb_s=0.12, recover_s=0.30, support_only_side="left"),
    }
    return {k: np.stack([_jump_pose(float(t), **v) for t in t_vec]) for k, v in specs.items()}


def solve_ik_trajectory_from_3d(
    points_3d_seq: np.ndarray,
    dt: float = 0.01,
    slot_hz: float = 30.0,
    retargeter: KeypointIKRetargeter | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Module S4+S5 default (causal keypoint IK, core/s45c_keypoint_ik.py) on a (T, 27, 3) track sampled every `dt` s.

    The track is sampled at `slot_hz` camera slots (linear interpolation of the given track, standing in for the
    camera rate), each slot is retargeted when it arrives, and every `dt` tick holds the newest slot (no future
    slot read). The output starts at the first observed slot, i.e. the `calibration_s` start-up stance is not
    part of the returned reference; the `startup_blend_s` ramp is. Returns (q (T',36), dq (T',35), info).
    """
    t_slot, pts = resample_points(np.asarray(points_3d_seq, np.float64), 1.0 / dt, slot_hz)
    rt = retargeter or KeypointIKRetargeter()
    out = retarget_sequence(pts, t_slot, rt)
    _, q, dq, _ = causal_upsample(t_slot, out["q"], out["dq"], out["observed"], "hold", rt.lo, rt.hi, dt=dt)
    ms = out["solve_ms"][out["observed"]]
    return q, dq, dict(retargeter="s45c_keypoint_ik", slot_hz=slot_hz, t_start_s=float(t_slot[np.argmax(out["observed"])]),
                       solve_ms_p50=float(np.median(ms)), solve_ms_max=float(ms.max()))


def solve_gmr_trajectory_from_3d(
    points_3d_seq: np.ndarray,
    dt: float = 0.01,
    retarget_fps: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Offline reference retargeter (EasyMocap SMPL-X fit + GMR, NOT causal) on a fully observed (T, 27, 3) track.

    Returns (q_traj (T',36), dq_traj (T',35)) on the same `dt` grid plus timing info. The retargeter fits the whole
    sequence (non-causal) at `retarget_fps` (GMR's own default rate) and is interpolated back to `dt`.
    """
    q, dq, res = retarget_trajectory_from_3d(points_3d_seq, fps_in=1.0 / dt, out_dt=dt, retarget_fps=retarget_fps)
    return q, dq, dict(fit_s=res.fit_s, retarget_s=res.retarget_s, human_height_m=res.human_height_m,
                       fit_residual_mm_mean=float(np.nanmean(res.fit_residual_mm)))
