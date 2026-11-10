"""Module R4: Causal Trajectory Governor, Envelope Guard, Soft-Clutch & Timeout Fallback.

Specification Reference: `自建硬件部署与前期执行手册.md` — 阶段四 4.4.4 / 阶段五 4.5.1
Input:
  - `ik_frame`: `KeypointIKFrame` (or `q (36,)`, `dq (35,)`, `observed: bool`) from Module S45c.
  - `human_points_3d`: `np.ndarray` of shape `(27, 3)` in world Z-up frame (units: metres).
  - `t_host_s`: `float`, strictly increasing host monotonic timestamp (units: seconds).
Output:
  - `SafeReferenceFrame`:
    - `q_safe`: `np.ndarray` of shape `(36,)` `[root xyz (m), root quat wxyz, 29 joints (rad)]`
    - `dq_safe`: `np.ndarray` of shape `(35,)` `[root lin vel (m/s), root ang vel (rad/s), 29 joint vels (rad/s)]`
    - `safety_state`: `str` in `{"IDLE_STAND", "SOFT_CLUTCH", "ACTIVE_TRACK", "HOLD_POSE", "RETURN_TO_STAND"}`
    - `safety_state_code`: `int` (`0..4` matching `SAFETY_STATES`)
    - `fault_reason`: `str | None`
Failure Behavior:
  - Rejects non-increasing timestamps or malformed array shapes with `ValueError`.
  - Human fall/crouch out of envelope (`z_pelvis < 0.48 m` or trunk tilt `> 40 deg`) freezes pose into `HOLD_POSE`
    with `fault_reason = "human_out_of_envelope"`, preventing the robot from following a falling human.
  - Excessive single-step joint jump (`> 0.35 rad`) triggers `HOLD_POSE` (`fault_reason = "joint_jump_exceeded"`).
  - Missing observation `<= 0.15 s` holds last safe pose with zero velocity; `> 0.15 s` smoothly interpolates back
    to the symmetric standing posture (`COMMON_DEFAULT_Q_29`) over `1.5 s`.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .constants import KEYPOINT_INDEX_BY_NAME as K
from .s45c_keypoint_ik import KeypointIKFrame
from .s6_mujoco_closed_loop import COMMON_DEFAULT_Q_29, DEFAULT_HEIGHT_M

SAFETY_STATES: tuple[str, ...] = (
    "IDLE_STAND",
    "SOFT_CLUTCH",
    "ACTIVE_TRACK",
    "HOLD_POSE",
    "RETURN_TO_STAND",
)
SAFETY_STATE_TO_CODE: dict[str, int] = {name: i for i, name in enumerate(SAFETY_STATES)}


@dataclass(frozen=True)
class SafeReferenceFrame:
    q_safe: np.ndarray
    dq_safe: np.ndarray
    safety_state: str
    safety_state_code: int
    fault_reason: str | None
    t_host_s: float


def _slerp_quat_wxyz(q0_wxyz: np.ndarray, q1_wxyz: np.ndarray, alpha: float) -> np.ndarray:
    a = float(np.clip(alpha, 0.0, 1.0))
    r = Slerp([0.0, 1.0], Rotation.from_quat([q0_wxyz[[1, 2, 3, 0]], q1_wxyz[[1, 2, 3, 0]]]))([a])[0]
    xyzw = r.as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)


def _interp_q36(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    a = float(np.clip(alpha, 0.0, 1.0))
    out = (1.0 - a) * q0 + a * q1
    out[3:7] = _slerp_quat_wxyz(q0[3:7], q1[3:7], a)
    return out


class SafetyGovernor:
    """Causal trajectory governor between S45c IK and S6/R3 controller."""

    def __init__(
        self,
        *,
        clutch_duration_s: float = 2.0,
        min_pelvis_z_m: float = 0.48,
        max_trunk_tilt_deg: float = 40.0,
        max_joint_step_rad: float = 0.20,
        fault_joint_jump_rad: float = 0.35,
        max_joint_speed_rad_s: float = 10.0,
        short_hold_timeout_s: float = 0.15,
        return_to_stand_duration_s: float = 1.5,
        default_q29: np.ndarray | None = None,
        default_height_m: float = DEFAULT_HEIGHT_M,
    ) -> None:
        self.clutch_duration_s = float(clutch_duration_s)
        self.min_pelvis_z_m = float(min_pelvis_z_m)
        self.max_trunk_tilt_deg = float(max_trunk_tilt_deg)
        self.max_joint_step_rad = float(max_joint_step_rad)
        self.fault_joint_jump_rad = float(fault_joint_jump_rad)
        self.max_joint_speed_rad_s = float(max_joint_speed_rad_s)
        self.short_hold_timeout_s = float(short_hold_timeout_s)
        self.return_to_stand_duration_s = float(return_to_stand_duration_s)
        self.default_q29 = (
            COMMON_DEFAULT_Q_29.copy() if default_q29 is None else np.asarray(default_q29, dtype=np.float64).copy()
        )
        self.default_height_m = float(default_height_m)
        self.reset()

    def _stand_q36(self, xy_yaw_ref: np.ndarray | None = None) -> np.ndarray:
        q = np.zeros(36, dtype=np.float64)
        q[2] = self.default_height_m
        q[3] = 1.0
        if xy_yaw_ref is not None and np.all(np.isfinite(xy_yaw_ref[:7])):
            q[:2] = xy_yaw_ref[:2]
            rot = Rotation.from_quat(xy_yaw_ref[[4, 5, 6, 3]])
            yaw = float(rot.as_euler("ZYX", degrees=False)[0])
            xyzw = Rotation.from_euler("Z", yaw).as_quat()
            q[3:7] = [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]
        q[7:] = self.default_q29
        return q

    def reset(self) -> None:
        self._last_t_s: float | None = None
        self._last_valid_t_s: float | None = None
        self._clutch_start_t_s: float | None = None
        self._return_start_t_s: float | None = None
        self._return_start_q: np.ndarray | None = None
        self._q_safe: np.ndarray = self._stand_q36()
        self._latched_fault: str | None = None

    def _check_human_envelope(self, pts: np.ndarray, q_cand: np.ndarray) -> str | None:
        l_hip, r_hip = pts[K["left_hip"]], pts[K["right_hip"]]
        l_sh, r_sh = pts[K["left_shoulder"]], pts[K["right_shoulder"]]
        if not (np.all(np.isfinite(l_hip)) and np.all(np.isfinite(r_hip))
                and np.all(np.isfinite(l_sh)) and np.all(np.isfinite(r_sh))):
            return "missing_trunk_keypoints"
        pelvis = 0.5 * (l_hip + r_hip)
        shoulder = 0.5 * (l_sh + r_sh)
        if float(pelvis[2]) < self.min_pelvis_z_m or float(q_cand[2]) < self.min_pelvis_z_m:
            return "human_out_of_envelope"
        trunk = shoulder - pelvis
        norm = float(np.linalg.norm(trunk))
        if norm < 1e-4:
            return "human_out_of_envelope"
        cos_tilt = float(np.clip(trunk[2] / norm, -1.0, 1.0))
        tilt_deg = float(np.degrees(np.arccos(cos_tilt)))
        if tilt_deg > self.max_trunk_tilt_deg:
            return "human_out_of_envelope"
        return None

    def step(
        self,
        ik_frame: KeypointIKFrame,
        human_points_3d: np.ndarray,
        t_host_s: float,
    ) -> SafeReferenceFrame:
        t = float(t_host_s)
        if not np.isfinite(t):
            raise ValueError("t_host_s must be finite")
        if self._last_t_s is not None and t <= self._last_t_s:
            raise ValueError(f"t_host_s must be strictly increasing ({t} <= {self._last_t_s})")
        pts = np.asarray(human_points_3d, dtype=np.float64)
        if pts.shape != (27, 3):
            raise ValueError(f"Expected human_points_3d of shape (27, 3), got {pts.shape}")
        q_in = np.asarray(ik_frame.q, dtype=np.float64)
        dq_in = np.asarray(ik_frame.dq, dtype=np.float64)
        if q_in.shape != (36,) or dq_in.shape != (35,):
            raise ValueError(f"Expected q(36,) and dq(35,), got {q_in.shape} and {dq_in.shape}")

        dt = 0.02 if self._last_t_s is None else float(t - self._last_t_s)
        self._last_t_s = t
        q_prev = self._q_safe.copy()

        # 1. Missing observation handling (short hold <= 0.15 s, then smooth return to stand over 1.5 s)
        if not bool(ik_frame.observed) or not np.all(np.isfinite(q_in)):
            if self._last_valid_t_s is None:
                self._q_safe = self._stand_q36()
                return SafeReferenceFrame(
                    q_safe=self._q_safe.copy(),
                    dq_safe=np.zeros(35, dtype=np.float64),
                    safety_state="IDLE_STAND",
                    safety_state_code=SAFETY_STATE_TO_CODE["IDLE_STAND"],
                    fault_reason=None,
                    t_host_s=t,
                )
            gap = t - self._last_valid_t_s
            if gap <= self.short_hold_timeout_s:
                return SafeReferenceFrame(
                    q_safe=self._q_safe.copy(),
                    dq_safe=np.zeros(35, dtype=np.float64),
                    safety_state="HOLD_POSE",
                    safety_state_code=SAFETY_STATE_TO_CODE["HOLD_POSE"],
                    fault_reason="observation_timeout_short_hold",
                    t_host_s=t,
                )
            if self._return_start_t_s is None:
                self._return_start_t_s = t
                self._return_start_q = self._q_safe.copy()
            alpha = (t - self._return_start_t_s) / max(self.return_to_stand_duration_s, 1e-6)
            stand_target = self._stand_q36(self._return_start_q)
            self._q_safe = _interp_q36(self._return_start_q, stand_target, alpha)
            dq_ret = np.zeros(35, dtype=np.float64)
            dq_ret[6:] = (self._q_safe[7:] - q_prev[7:]) / max(dt, 1e-6)
            return SafeReferenceFrame(
                q_safe=self._q_safe.copy(),
                dq_safe=dq_ret,
                safety_state="RETURN_TO_STAND",
                safety_state_code=SAFETY_STATE_TO_CODE["RETURN_TO_STAND"],
                fault_reason="observation_timeout_return_to_stand",
                t_host_s=t,
            )

        # 2. Envelope check on human 3D skeleton & candidate pelvis height
        env_fault = self._check_human_envelope(pts, q_in)
        if env_fault is not None:
            self._latched_fault = env_fault
            return SafeReferenceFrame(
                q_safe=self._q_safe.copy(),
                dq_safe=np.zeros(35, dtype=np.float64),
                safety_state="HOLD_POSE",
                safety_state_code=SAFETY_STATE_TO_CODE["HOLD_POSE"],
                fault_reason=env_fault,
                t_host_s=t,
            )

        # Valid observation inside safety envelope
        if self._last_valid_t_s is None or self._return_start_t_s is not None:
            self._clutch_start_t_s = t
            self._return_start_t_s = None
            self._return_start_q = None
        self._last_valid_t_s = t

        # 3. Soft-clutch engagement over 2.0 s before active tracking
        assert self._clutch_start_t_s is not None
        clutch_elapsed = t - self._clutch_start_t_s
        in_clutch = clutch_elapsed < self.clutch_duration_s
        if in_clutch:
            alpha = (clutch_elapsed + dt) / max(self.clutch_duration_s, 1e-6)
            q_target = _interp_q36(self._stand_q36(q_in), q_in, alpha)
            state = "SOFT_CLUTCH"
        else:
            q_target = q_in.copy()
            state = "ACTIVE_TRACK"

        # 4. Joint jump fault check & per-step velocity clamping
        raw_jump = float(np.max(np.abs(q_target[7:] - q_prev[7:])))
        if not in_clutch and raw_jump > self.fault_joint_jump_rad:
            self._latched_fault = "joint_jump_exceeded"
            return SafeReferenceFrame(
                q_safe=self._q_safe.copy(),
                dq_safe=np.zeros(35, dtype=np.float64),
                safety_state="HOLD_POSE",
                safety_state_code=SAFETY_STATE_TO_CODE["HOLD_POSE"],
                fault_reason="joint_jump_exceeded",
                t_host_s=t,
            )

        step_bound = min(self.max_joint_step_rad, self.max_joint_speed_rad_s * dt)
        delta_j = np.clip(q_target[7:] - q_prev[7:], -step_bound, step_bound)
        q_next = q_target.copy()
        q_next[7:] = q_prev[7:] + delta_j

        dq_next = dq_in.copy()
        dq_next[:3] = (q_next[:3] - q_prev[:3]) / max(dt, 1e-6)
        rot_prev = Rotation.from_quat(q_prev[[4, 5, 6, 3]])
        rot_next = Rotation.from_quat(q_next[[4, 5, 6, 3]])
        dq_next[3:6] = (rot_prev.inv() * rot_next).as_rotvec() / max(dt, 1e-6)
        dq_next[6:] = delta_j / max(dt, 1e-6)

        self._q_safe = q_next
        self._latched_fault = None
        return SafeReferenceFrame(
            q_safe=self._q_safe.copy(),
            dq_safe=dq_next,
            safety_state=state,
            safety_state_code=SAFETY_STATE_TO_CODE[state],
            fault_reason=None,
            t_host_s=t,
        )
