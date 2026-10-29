"""Synthetic 27-point human poses (golden keypoint layout, right-handed Z-up world, metres).

Moved unchanged from the removed S4 module; used by the S2/S3 tests, the S6 motion generator and the GMR
retargeting tests. Produces ground-truth-free *simulated* inputs only - never a substitute for perception.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .constants import KEYPOINT_INDEX_BY_NAME


def safe_normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Normalize a 3D vector; return normalized fallback if norm is degenerate."""
    v = np.asarray(vec, dtype=np.float64)
    if not np.all(np.isfinite(v)):
        return np.asarray(fallback, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        fb = np.asarray(fallback, dtype=np.float64)
        return fb / float(np.linalg.norm(fb))
    return v / n


def synthesize_human_27_pose(
    height_m: float = 1.75,
    squat_depth_m: float = 0.0,
    torso_pitch_rad: float = 0.0,
    torso_yaw_rad: float = 0.0,
    torso_roll_rad: float = 0.0,
    l_arm_dir: np.ndarray | None = None,
    r_arm_dir: np.ndarray | None = None,
    l_forearm_dir: np.ndarray | None = None,
    r_forearm_dir: np.ndarray | None = None,
    l_foot_offset_m: np.ndarray | None = None,
    r_foot_offset_m: np.ndarray | None = None,
    root_xy_m: np.ndarray | None = None,
    single_leg_raise_side: str | None = None,
    t_pose: bool = False,
    root_z_offset_m: float = 0.0,
) -> np.ndarray:
    """Generate a physically consistent 27-point human 3D skeleton in right-handed Z-up coordinates.

    Used for unit testing (S2, S3, S4, S5, S6) and synthetic trajectory generation.
    `root_z_offset_m` lifts the whole skeleton rigidly (flight phase of a jump; 0 = feet on the ground).
    """
    scale = height_m / 1.75
    w_hip = 0.105 * scale
    l_thigh = 0.43 * scale
    l_calf = 0.42 * scale
    h_ankle = 0.04 * scale
    l_spine = 0.48 * scale
    w_shldr = 0.18 * scale
    l_uarm = 0.29 * scale
    l_farm = 0.26 * scale

    xy = np.zeros(2, dtype=np.float64) if root_xy_m is None else np.asarray(root_xy_m, dtype=np.float64)
    leg_total = l_thigh + l_calf
    squat_clamped = float(np.clip(squat_depth_m, 0.0, 0.55 * leg_total))
    pelvis_z = h_ankle + leg_total - squat_clamped
    p_pelvis = np.array([xy[0], xy[1], pelvis_z], dtype=np.float64)

    R_yaw = Rotation.from_euler("z", torso_yaw_rad).as_matrix()
    R_torso = Rotation.from_euler("zyx", [torso_yaw_rad, torso_pitch_rad, torso_roll_rad]).as_matrix()

    p_l_hip = p_pelvis + R_yaw @ np.array([0.0, w_hip, 0.0])
    p_r_hip = p_pelvis + R_yaw @ np.array([0.0, -w_hip, 0.0])

    l_f_off = np.zeros(3) if l_foot_offset_m is None else np.asarray(l_foot_offset_m, dtype=np.float64)
    r_f_off = np.zeros(3) if r_foot_offset_m is None else np.asarray(r_foot_offset_m, dtype=np.float64)
    p_l_ankle = np.array([xy[0], xy[1], h_ankle]) + R_yaw @ (np.array([0.0, w_hip, 0.0]) + l_f_off)
    p_r_ankle = np.array([xy[0], xy[1], h_ankle]) + R_yaw @ (np.array([0.0, -w_hip, 0.0]) + r_f_off)

    if single_leg_raise_side == "left":
        p_l_ankle = p_l_hip + R_yaw @ np.array([0.22 * scale, 0.02 * scale, -0.55 * leg_total])
    elif single_leg_raise_side == "right":
        p_r_ankle = p_r_hip + R_yaw @ np.array([0.22 * scale, -0.02 * scale, -0.55 * leg_total])

    # Analytical 2-link knee solver in sagittal forward direction (+X of R_yaw)
    def solve_knee(p_hip: np.ndarray, p_ankle: np.ndarray) -> np.ndarray:
        vec = p_ankle - p_hip
        d = float(np.linalg.norm(vec))
        max_d = l_thigh + l_calf - 1e-5
        if d >= max_d:
            u = vec / max(d, 1e-8)
            p_ankle[:] = p_hip + max_d * u
            d = max_d
        d = max(d, abs(l_thigh - l_calf) + 1e-5)
        u = (p_ankle - p_hip) / d
        cos_alpha = (l_thigh**2 + d**2 - l_calf**2) / (2.0 * l_thigh * d)
        cos_alpha = float(np.clip(cos_alpha, -1.0, 1.0))
        sin_alpha = float(np.sqrt(max(0.0, 1.0 - cos_alpha**2)))
        fwd = R_yaw[:, 0]
        bend_dir = fwd - np.dot(fwd, u) * u
        n_bend = float(np.linalg.norm(bend_dir))
        bend_dir = (R_yaw[:, 2] if n_bend < 1e-6 else bend_dir / n_bend)
        return p_hip + l_thigh * (cos_alpha * u + sin_alpha * bend_dir)

    p_l_knee = solve_knee(p_l_hip, p_l_ankle)
    p_r_knee = solve_knee(p_r_hip, p_r_ankle)

    # Torso & head
    p_neck = p_pelvis + R_torso @ np.array([0.0, 0.0, l_spine])
    p_l_shldr = p_neck + R_torso @ np.array([0.0, w_shldr, 0.0])
    p_r_shldr = p_neck + R_torso @ np.array([0.0, -w_shldr, 0.0])
    p_nose = p_neck + R_torso @ np.array([0.08 * scale, 0.0, 0.18 * scale])
    p_l_eye = p_nose + R_torso @ np.array([-0.02 * scale, 0.03 * scale, 0.03 * scale])
    p_r_eye = p_nose + R_torso @ np.array([-0.02 * scale, -0.03 * scale, 0.03 * scale])
    p_l_ear = p_nose + R_torso @ np.array([-0.07 * scale, 0.07 * scale, 0.01 * scale])
    p_r_ear = p_nose + R_torso @ np.array([-0.07 * scale, -0.07 * scale, 0.01 * scale])

    if t_pose:
        u_l_u = R_torso @ np.array([0.0, 1.0, 0.0])
        u_r_u = R_torso @ np.array([0.0, -1.0, 0.0])
        u_l_f = R_torso @ np.array([0.0, 1.0, 0.0])
        u_r_f = R_torso @ np.array([0.0, -1.0, 0.0])
    else:
        u_l_u = R_torso @ safe_normalize(
            np.array([0.05, 0.15, -0.98]) if l_arm_dir is None else l_arm_dir,
            np.array([0.0, 0.0, -1.0]),
        )
        u_r_u = R_torso @ safe_normalize(
            np.array([0.05, -0.15, -0.98]) if r_arm_dir is None else r_arm_dir,
            np.array([0.0, 0.0, -1.0]),
        )
        u_l_f = R_torso @ safe_normalize(
            np.array([0.85, 0.0, -0.52]) if l_forearm_dir is None else l_forearm_dir,
            np.array([1.0, 0.0, 0.0]),
        )
        u_r_f = R_torso @ safe_normalize(
            np.array([0.85, 0.0, -0.52]) if r_forearm_dir is None else r_forearm_dir,
            np.array([1.0, 0.0, 0.0]),
        )

    p_l_elbow = p_l_shldr + l_uarm * u_l_u
    p_r_elbow = p_r_shldr + l_uarm * u_r_u
    p_l_wrist = p_l_elbow + l_farm * u_l_f
    p_r_wrist = p_r_elbow + l_farm * u_r_f

    # Foot triangles (flat on ground plane aligned with R_yaw)
    def make_foot(p_ank: np.ndarray, is_left: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ground_z = p_ank[2] - h_ankle
        heel = np.array([p_ank[0], p_ank[1], ground_z]) + R_yaw @ np.array([-0.05 * scale, 0.0, 0.0])
        # Left foot: big toe is medial (-Y relative to foot center), small toe is lateral (+Y)
        if is_left:
            btoe = np.array([p_ank[0], p_ank[1], ground_z]) + R_yaw @ np.array([0.13 * scale, -0.03 * scale, 0.0])
            stoe = np.array([p_ank[0], p_ank[1], ground_z]) + R_yaw @ np.array([0.13 * scale, 0.03 * scale, 0.0])
        else:
            btoe = np.array([p_ank[0], p_ank[1], ground_z]) + R_yaw @ np.array([0.13 * scale, 0.03 * scale, 0.0])
            stoe = np.array([p_ank[0], p_ank[1], ground_z]) + R_yaw @ np.array([0.13 * scale, -0.03 * scale, 0.0])
        return btoe, stoe, heel

    p_l_btoe, p_l_stoe, p_l_heel = make_foot(p_l_ankle, is_left=True)
    p_r_btoe, p_r_stoe, p_r_heel = make_foot(p_r_ankle, is_left=False)

    # Hand triangles: forward along u_l_f / u_r_f, palm down in local torso frame
    def make_hand(p_wr: np.ndarray, u_fore: np.ndarray, is_left: bool) -> tuple[np.ndarray, np.ndarray]:
        up_ref = R_torso[:, 2]
        lat = np.cross(up_ref, u_fore)
        if np.linalg.norm(lat) < 1e-5:
            lat = R_torso[:, 1]
        else:
            lat = lat / float(np.linalg.norm(lat))
        # Left hand: pinky -> index is along +lat
        if is_left:
            imcp = p_wr + 0.08 * scale * u_fore + 0.025 * scale * lat
            pmcp = p_wr + 0.08 * scale * u_fore - 0.025 * scale * lat
        else:
            imcp = p_wr + 0.08 * scale * u_fore - 0.025 * scale * lat
            pmcp = p_wr + 0.08 * scale * u_fore + 0.025 * scale * lat
        return imcp, pmcp

    p_l_imcp, p_l_pmcp = make_hand(p_l_wrist, u_l_f, is_left=True)
    p_r_imcp, p_r_pmcp = make_hand(p_r_wrist, u_r_f, is_left=False)

    out = np.zeros((27, 3), dtype=np.float64)
    idx = KEYPOINT_INDEX_BY_NAME
    out[idx["nose"]] = p_nose
    out[idx["left_eye"]] = p_l_eye
    out[idx["right_eye"]] = p_r_eye
    out[idx["left_ear"]] = p_l_ear
    out[idx["right_ear"]] = p_r_ear
    out[idx["left_shoulder"]] = p_l_shldr
    out[idx["right_shoulder"]] = p_r_shldr
    out[idx["left_elbow"]] = p_l_elbow
    out[idx["right_elbow"]] = p_r_elbow
    out[idx["left_wrist"]] = p_l_wrist
    out[idx["right_wrist"]] = p_r_wrist
    out[idx["left_hip"]] = p_l_hip
    out[idx["right_hip"]] = p_r_hip
    out[idx["left_knee"]] = p_l_knee
    out[idx["right_knee"]] = p_r_knee
    out[idx["left_ankle"]] = p_l_ankle
    out[idx["right_ankle"]] = p_r_ankle
    out[idx["left_big_toe"]] = p_l_btoe
    out[idx["left_small_toe"]] = p_l_stoe
    out[idx["left_heel"]] = p_l_heel
    out[idx["right_big_toe"]] = p_r_btoe
    out[idx["right_small_toe"]] = p_r_stoe
    out[idx["right_heel"]] = p_r_heel
    out[idx["left_index_mcp"]] = p_l_imcp
    out[idx["left_pinky_mcp"]] = p_l_pmcp
    out[idx["right_index_mcp"]] = p_r_imcp
    out[idx["right_pinky_mcp"]] = p_r_pmcp
    out[:, 2] += float(root_z_offset_m)
    return out
