"""Hardware-neutral acquisition contracts and calibrated device-clock mapping.

SDK adapters supply exposure timestamps, never host receive-time substitutes.
The mapping is injected from a measured synchronization/calibration procedure.
"""
from dataclasses import dataclass
from typing import Protocol
import numpy as np


def nanoseconds(value):
    if type(value) is not int or value < 0:
        raise ValueError('Expected nonnegative integer nanoseconds')
    return value


@dataclass(frozen=True)
class ClockMapping:
    device_domain: str
    common_domain: str
    device_origin_ns: int
    common_origin_ns: int
    rate_numerator: int = 1
    rate_denominator: int = 1
    uncertainty_ns: int = 0
    valid_span_ns: int = 1_000_000_000

    def map_exposure(self, device_timestamp_ns, uncertainty_budget_ns):
        nanoseconds(device_timestamp_ns)
        nanoseconds(self.device_origin_ns)
        nanoseconds(self.common_origin_ns)
        nanoseconds(self.uncertainty_ns)
        nanoseconds(uncertainty_budget_ns)
        if (not self.device_domain or not self.common_domain or
                type(self.rate_numerator) is not int or type(self.rate_denominator) is not int or
                min(self.rate_numerator,self.rate_denominator) <= 0 or self.valid_span_ns <= 0):
            raise ValueError('Invalid clock mapping')
        delta = device_timestamp_ns-self.device_origin_ns
        if abs(delta) > self.valid_span_ns or self.uncertainty_ns > uncertainty_budget_ns:
            raise ValueError('Clock mapping expired or uncertainty exceeds sync budget')
        # Subtract integer epochs before scale conversion; no float epoch rounding.
        return nanoseconds(self.common_origin_ns+(delta*self.rate_numerator)//self.rate_denominator)


@dataclass(frozen=True)
class ExposureFrame:
    camera_id: str
    exposure_timestamp_ns: int
    arrival_timestamp_ns: int
    clock_domain: str
    image_bgr: np.ndarray

    def validate(self):
        nanoseconds(self.exposure_timestamp_ns)
        nanoseconds(self.arrival_timestamp_ns)
        if self.arrival_timestamp_ns < self.exposure_timestamp_ns:
            raise ValueError('Arrival precedes exposure; check clock mapping')
        if (not self.camera_id or not self.clock_domain or self.image_bgr.dtype != np.uint8 or
                self.image_bgr.ndim != 3 or self.image_bgr.shape[2] != 3 or min(self.image_bgr.shape[:2]) == 0):
            raise ValueError('Expected identified camera and nonempty BGR uint8 frame')


class FrameSource(Protocol):
    """Return latest available frame or None without an unbounded frame backlog."""
    def poll(self) -> ExposureFrame | None: ...
    def close(self) -> None: ...


def validate_calibration(calibration):
    cameras = calibration['cameras']
    if len(cameras) < 2:
        raise ValueError('At least two calibrated cameras required')
    for name, camera in cameras.items():
        k = np.asarray(camera['K'],float)
        rotation = np.asarray(camera['R_world_to_camera'],float)
        translation = np.asarray(camera['t_world_to_camera_m'],float)
        distortion = np.asarray(camera.get('distortion',[0]*5),float)
        if (k.shape != (3,3) or rotation.shape != (3,3) or translation.shape != (3,) or
                distortion.shape not in ((4,),(5,),(8,),(12,),(14,))):
            raise ValueError(f'{name}: malformed calibration arrays')
        if not all(np.isfinite(v).all() for v in (k,rotation,translation,distortion)):
            raise ValueError(f'{name}: nonfinite calibration')
        if (min(k[0,0],k[1,1]) <= 0 or not np.allclose(k[2],[0,0,1]) or
                not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6) or
                not np.isclose(np.linalg.det(rotation),1,atol=1e-6)):
            raise ValueError(f'{name}: invalid camera intrinsics or improper rotation')
        if any(type(camera[d]) is not int or camera[d] <= 0 for d in ('width','height')):
            raise ValueError(f'{name}: invalid image dimensions')
