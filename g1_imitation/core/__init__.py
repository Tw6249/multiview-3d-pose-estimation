"""Pipeline modules S1 (2D keypoints) through S6 (closed-loop simulation), per `docs/方案规范.md`."""
from .constants import (
    G1_29_JOINTS,
    GOLDEN_27_INDICES,
    GOLDEN_27_NAMES,
    KEYPOINT_INDEX_BY_NAME,
    LEFT_FOOT_INDICES_27,
    RIGHT_FOOT_INDICES_27,
    ROBAN_21_JOINTS,
    ROBAN_HEFT_21_JOINTS,
    ROBOT_JOINTS,
    SDK_JOINT_NAMES,
    SUPPORTED_ROBOTS,
    load_config,
)

__all__ = [
    "G1_29_JOINTS",
    "GOLDEN_27_INDICES",
    "GOLDEN_27_NAMES",
    "KEYPOINT_INDEX_BY_NAME",
    "LEFT_FOOT_INDICES_27",
    "RIGHT_FOOT_INDICES_27",
    "ROBAN_21_JOINTS",
    "ROBAN_HEFT_21_JOINTS",
    "ROBOT_JOINTS",
    "SDK_JOINT_NAMES",
    "SUPPORTED_ROBOTS",
    "load_config",
]


