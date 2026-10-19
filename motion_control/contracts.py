"""SI units, named joints and an explicit common monotonic clock at boundaries."""
from dataclasses import asdict, dataclass
from typing import Protocol
import numpy as np


def vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f'{name}: expected {size} finite values')
    return result


def timestamp(value):
    if type(value) is not int or value < 0:
        raise ValueError('timestamp must be a nonnegative integer in nanoseconds')
    return value


@dataclass(frozen=True)
class JointFeedback:
    timestamp_ns: int
    clock_domain: str
    joint_names: tuple[str, ...]
    positions_rad: tuple[float, ...]
    velocities_rad_s: tuple[float, ...]
    enabled: bool = True


@dataclass(frozen=True)
class JointCommand:
    schema_version: int
    sequence: int
    timestamp_ns: int
    valid_until_ns: int
    source_timestamp_ns: int | None
    clock_domain: str
    robot_id: str
    model_sha256: str
    joint_names: tuple[str, ...]
    positions_rad: tuple[float, ...]
    velocities_rad_s: tuple[float, ...]
    state: str
    reason: str

    def to_dict(self):
        return asdict(self)


class RobotAdapter(Protocol):
    """Adapters must independently enforce expiry, joint order and device stop.

    Device time is mapped to clock_domain BEFORE returning feedback. A physical
    adapter owns the vendor watchdog, motor units and acknowledgement handling.
    """
    def read_feedback(self, now_ns: int) -> JointFeedback: ...
    def send(self, command: JointCommand, now_ns: int) -> None: ...
    def stop(self, reason: str) -> None: ...
