"""Experimental image evidence for distribution-level fusion, not a pose solver.

RTMPose axis responses are neither calibrated probabilities nor a joint 2D
distribution. Preserve them before argmax; never normalize them implicitly or
invent visibility/presence. Pixel coordinates refer to the supplied image.
No history, robot commands, reference labels, or default pipeline changes.
"""
from dataclasses import dataclass
import re

import numpy as np

from .sensors import ExposureFrame, nanoseconds


@dataclass(frozen=True)
class SimCCEvidence:
    camera_id: str
    exposure_timestamp_ns: int
    clock_domain: str
    model_sha256: str
    joint_names: tuple
    image_size_wh: tuple
    input_size_wh: tuple
    center_px: np.ndarray
    scale_px: np.ndarray
    x_response: np.ndarray
    y_response: np.ndarray
    split_ratio: float = 2.0

    def __post_init__(self):
        nanoseconds(self.exposure_timestamp_ns)
        if (not self.camera_id or not self.clock_domain or
                not re.fullmatch(r'[0-9a-f]{64}', self.model_sha256)):
            raise ValueError('Missing camera, clock, or model identity')
        for name in ('image_size_wh', 'input_size_wh'):
            value = tuple(getattr(self, name))
            if len(value) != 2 or any(type(x) is not int or x <= 0 for x in value):
                raise ValueError('Expected positive integer width/height')
            object.__setattr__(self, name, value)
        names = tuple(self.joint_names)
        if not names or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
            raise ValueError('Joint names must be unique and nonempty')
        object.__setattr__(self, 'joint_names', names)
        if not np.isfinite(self.split_ratio) or self.split_ratio <= 0:
            raise ValueError('Invalid bin-to-input-pixel ratio')
        for name in ('center_px', 'scale_px', 'x_response', 'y_response'):
            # Own the buffers: subsequent inference must not overwrite evidence.
            arr = np.array(getattr(self, name), copy=True)
            if not np.issubdtype(arr.dtype, np.floating) or not np.isfinite(arr).all():
                raise ValueError('Expected finite floating-point evidence')
            arr.flags.writeable = False
            object.__setattr__(self, name, arr)
        if (self.center_px.shape != (2,) or self.scale_px.shape != (2,) or
                np.any(self.scale_px <= 0)):
            raise ValueError('Invalid unrotated crop transform')
        for axis, response in enumerate((self.x_response, self.y_response)):
            bins = self.input_size_wh[axis] * self.split_ratio
            if not float(bins).is_integer() or response.shape != (len(names), int(bins)):
                raise ValueError('Response layout disagrees with model input')

    def bins_to_pixels(self, bins_xy):
        """Unrotated RTMLib crop: x right/y down, no clipping or undistortion."""
        bins = np.asarray(bins_xy, dtype=float)
        if bins.ndim == 0 or bins.shape[-1] != 2 or not np.isfinite(bins).all():
            raise ValueError('Expected finite (..., 2) bin coordinates')
        return bins / self.split_ratio / self.input_size_wh * self.scale_px + self.center_px - self.scale_px / 2

    def pixels_to_bins(self, pixels_xy):
        pixels = np.asarray(pixels_xy, dtype=float)
        if pixels.ndim == 0 or pixels.shape[-1] != 2 or not np.isfinite(pixels).all():
            raise ValueError('Expected finite (..., 2) image pixels')
        return (pixels - self.center_px + self.scale_px / 2) / self.scale_px * self.input_size_wh * self.split_ratio

    @property
    def response_bytes(self):
        return self.x_response.nbytes + self.y_response.nbytes

    @property
    def semantics(self):
        return dict(representation='raw_simcc_axis_response', calibrated_probability=False,
                    joint_xy_distribution=False, presence_probability=None,
                    visibility_probability=None, coordinate_frame='supplied_image_xy_pixels',
                    crop_transform='unrotated_affine', units='dimensionless_model_response')


def capture_simcc(model, frame: ExposureFrame, bbox_xyxy, joint_names, model_sha256):
    """Run a selected current crop once; return usual decoding plus raw evidence.

    This adapter is for the pinned RTMLib RTMPose COCO path, split ratio 2,
    with no OpenPose permutation. Invalid/missing crop or output raises; callers
    must treat failure as unknown, never as a previous observation. Expiry and
    inter-camera synchronization remain the consumer's responsibility.
    """
    frame.validate()
    box = np.asarray(bbox_xyxy, dtype=np.float32)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        raise ValueError('A nonempty current person crop is required')
    if getattr(model, 'to_openpose', False):
        raise ValueError('OpenPose permutation unsupported')
    crop, center, scale = model.preprocess(frame.image_bgr, box)
    outputs = model.inference(crop)
    if len(outputs) != 2 or any(np.asarray(a).ndim != 3 or len(a) != 1 for a in outputs):
        raise ValueError('Expected one-person x/y SimCC outputs')
    evidence = SimCCEvidence(frame.camera_id, frame.exposure_timestamp_ns, frame.clock_domain,
        model_sha256, tuple(joint_names), (frame.image_bgr.shape[1], frame.image_bgr.shape[0]),
        tuple(model.model_input_size), center, scale, outputs[0][0], outputs[1][0])
    keypoints, scores = model.postprocess(outputs, center, scale)
    if (np.asarray(keypoints).shape != (1, len(joint_names), 2) or
            np.asarray(scores).shape != (1, len(joint_names)) or
            not np.isfinite(keypoints).all() or not np.isfinite(scores).all()):
        raise ValueError('Invalid decoded pose')
    return keypoints[0], scores[0], evidence
