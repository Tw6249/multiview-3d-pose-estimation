"""Pipeline modules S1 (2D keypoints) through S6 (closed-loop simulation), per `docs/方案规范.md`."""
from .constants import (
    GOLDEN_27_INDICES,
    GOLDEN_27_NAMES,
    KEYPOINT_INDEX_BY_NAME,
    LEFT_FOOT_INDICES_27,
    RIGHT_FOOT_INDICES_27,
    SDK_JOINT_NAMES,
    load_config,
)

__all__ = [
    "GOLDEN_27_INDICES",
    "GOLDEN_27_NAMES",
    "KEYPOINT_INDEX_BY_NAME",
    "LEFT_FOOT_INDICES_27",
    "RIGHT_FOOT_INDICES_27",
    "SDK_JOINT_NAMES",
    "load_config",
]

