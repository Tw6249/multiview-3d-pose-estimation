"""Shared topology constants, coordinate conventions, and configuration loader.

Coordinate System Convention:
- Right-handed world frame: +X forward, +Y left, +Z vertically up, ground at Z = 0 m.
- Quaternions: unit quaternions ordered [w, x, y, z].
- Positions: metres (m); Velocities: m/s; Angles: radians (rad); Time: seconds (s).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
CONFIG_PATH = PACKAGE_ROOT / "config" / "pipeline_config.json"

# COCO-WholeBody (133 keypoints) -> Golden 27 keypoints slice indices:
# - 0..16: 17 COCO body keypoints
# - 17..22: 6 foot keypoints (left/right big_toe, small_toe, heel)
# - 96, 108, 117, 129: 4 hand MCP keypoints (left index/pinky MCP, right index/pinky MCP)
GOLDEN_27_INDICES: tuple[int, ...] = (
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22,
    96, 108, 117, 129,
)

GOLDEN_27_NAMES: tuple[str, ...] = (
    # 0..16: Body 17
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    # 17..22: Foot 6 (indices 17..22 in both 133-layout and 27-layout)
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    # 23..26 in 27-layout (indices 96, 108, 117, 129 in 133-layout)
    "left_index_mcp",
    "left_pinky_mcp",
    "right_index_mcp",
    "right_pinky_mcp",
)

assert len(GOLDEN_27_INDICES) == 27
assert len(GOLDEN_27_NAMES) == 27

KEYPOINT_INDEX_BY_NAME: dict[str, int] = {name: i for i, name in enumerate(GOLDEN_27_NAMES)}

# Foot 4-point groups in the 27-keypoint array (per Module S3 specification):
# Left foot: [15:左踝, 17:左大脚趾, 18:左小脚趾, 19:左脚跟]
# Right foot: [16:右踝, 20:右大脚趾, 21:右小脚趾, 22:右脚跟]
LEFT_FOOT_INDICES_27: tuple[int, int, int, int] = (15, 17, 18, 19)
RIGHT_FOOT_INDICES_27: tuple[int, int, int, int] = (16, 20, 21, 22)
LEFT_FOOT_KEYPOINT_INDICES: tuple[int, int, int, int] = LEFT_FOOT_INDICES_27
RIGHT_FOOT_KEYPOINT_INDICES: tuple[int, int, int, int] = RIGHT_FOOT_INDICES_27

# Unitree G1 29-DoF SDK joint ordering
SDK_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
G1_29_JOINTS: tuple[str, ...] = SDK_JOINT_NAMES


def load_config(path: Path | None = None) -> dict[str, Any]:
    target = path or CONFIG_PATH
    return json.loads(target.read_text(encoding="utf-8"))
