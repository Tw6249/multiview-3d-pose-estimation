"""Module R1: Hikrobot Multi-Camera Synchronized Capture (PoE PTP/Action, PoE Free-Run, Software & Line0) + Sim Backend.

Specification Reference: `自建硬件部署与前期执行手册.md` — 阶段二 4.2.2 (with pure-PoE trigger modes enabled)
Input:
  - `camera_serials`: `tuple[str, ...]` (`N_cams` in `[2, 6]`).
  - Capture config (`exposure_us = 1800.0`, `gain_db = 4.0`, `packet_size_bytes = 8192`,
    `trigger_mode` in `{"ptp_action", "ptp_freerun", "software", "line0"}`,
    `max_exposure_skew_ms = 1.0`, `bundle_timeout_ms = 12.0`).
Output:
  - `SynchronizedFrameBundle`:
    - `bundle_id`: `int` (strictly increasing trigger/bundle sequence number)
    - `images_bgr`: `np.ndarray` `(N_cams, H, W, 3)` `uint8`
    - `valid_cam_mask`: `np.ndarray` `(N_cams,)` `bool`
    - `dev_timestamps_s`: `np.ndarray` `(N_cams,)` `float64` (seconds from camera hardware clock; `np.nan` if invalid)
    - `host_monotonic_s`: `float` (`time.perf_counter()` seconds)
    - `exposure_time_s`: `float` (seconds)
    - `bundle_valid`: `bool` (`True` iff `sum(valid_cam_mask) >= 2`)
Failure Behavior:
  - Never substitutes host receive time for camera hardware `dev_timestamps_s`.
  - Any camera missing the frame or exceeding `max_exposure_skew_ms` from the bundle median is marked
    `valid_cam_mask[c] = False` and its image slice is zeroed; never reuses stale frames from previous bundles.
  - If `< 2` cameras are valid, `bundle_valid` is set to `False`.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np

SUPPORTED_TRIGGER_MODES: tuple[str, ...] = (
    "ptp_action",   # Pure PoE recommended: IEEE 1588 PTP clock sync + GigE Action1 broadcast trigger
    "ptp_freerun",  # Pure PoE zero-trigger: IEEE 1588 PTP clock sync + fixed-FPS continuous + timestamp matching
    "software",     # Pure PoE fallback: host software trigger command across cameras
    "line0",        # Physical 6-pin Hirose Line0 opto-isolated hardware pulse trigger
)


@dataclass(frozen=True)
class SynchronizedFrameBundle:
    bundle_id: int
    images_bgr: np.ndarray
    valid_cam_mask: np.ndarray
    dev_timestamps_s: np.ndarray
    host_monotonic_s: float
    exposure_time_s: float
    bundle_valid: bool
    trigger_mode: str = "ptp_action"


def assemble_synchronized_bundle(
    bundle_id: int,
    raw_bayer_or_bgr_frames: list[np.ndarray | None],
    dev_timestamps_s: np.ndarray | list[float],
    *,
    image_size_wh: tuple[int, int] = (1624, 1240),
    is_bayer_rg8: bool = True,
    max_exposure_skew_ms: float = 1.0,
    exposure_time_s: float = 0.0018,
    host_monotonic_s: float | None = None,
    trigger_mode: str = "ptp_action",
) -> SynchronizedFrameBundle:
    """Assemble a single multi-camera frame bundle with strict hardware timestamp skew gating."""
    if trigger_mode not in SUPPORTED_TRIGGER_MODES:
        raise ValueError(f"Unsupported trigger_mode {trigger_mode!r}; expected one of {SUPPORTED_TRIGGER_MODES}")
    n_cams = len(raw_bayer_or_bgr_frames)
    if n_cams < 2 or n_cams > 6:
        raise ValueError(f"Expected 2..6 cameras, got {n_cams}")
    dev_ts = np.asarray(dev_timestamps_s, dtype=np.float64).copy()
    if dev_ts.shape != (n_cams,):
        raise ValueError(f"Expected dev_timestamps_s of shape ({n_cams},), got {dev_ts.shape}")

    w, h = int(image_size_wh[0]), int(image_size_wh[1])
    images_bgr = np.zeros((n_cams, h, w, 3), dtype=np.uint8)
    present_mask = np.zeros(n_cams, dtype=bool)
    for c, frame in enumerate(raw_bayer_or_bgr_frames):
        if frame is not None and np.isfinite(dev_ts[c]):
            present_mask[c] = True
        else:
            dev_ts[c] = np.nan

    valid_mask = present_mask.copy()
    if int(np.sum(present_mask)) >= 2:
        median_ts = float(np.nanmedian(dev_ts[present_mask]))
        skew_ms = np.abs(dev_ts - median_ts) * 1e3
        for c in range(n_cams):
            if present_mask[c] and skew_ms[c] > float(max_exposure_skew_ms):
                valid_mask[c] = False
                dev_ts[c] = np.nan
    else:
        valid_mask[:] = False

    for c in range(n_cams):
        if not valid_mask[c]:
            continue
        raw = raw_bayer_or_bgr_frames[c]
        assert raw is not None
        if is_bayer_rg8 and raw.ndim == 2:
            bgr = cv2.cvtColor(raw, cv2.COLOR_BAYER_RG2BGR)
        elif raw.ndim == 3 and raw.shape[2] == 3:
            bgr = raw
        else:
            raise ValueError(f"Unexpected frame shape {raw.shape} on camera {c}")
        if (bgr.shape[1], bgr.shape[0]) != (w, h):
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        images_bgr[c] = bgr

    host_now = float(time.perf_counter() if host_monotonic_s is None else host_monotonic_s)
    bundle_valid = int(np.sum(valid_mask)) >= 2
    return SynchronizedFrameBundle(
        bundle_id=int(bundle_id),
        images_bgr=images_bgr,
        valid_cam_mask=valid_mask,
        dev_timestamps_s=dev_ts,
        host_monotonic_s=host_now,
        exposure_time_s=float(exposure_time_s),
        bundle_valid=bundle_valid,
        trigger_mode=trigger_mode,
    )


class SimulatedBayerCaptureBackend:
    """Deterministic simulated BayerRG8 capture backend for local testing before physical cameras arrive."""

    def __init__(
        self,
        camera_serials: tuple[str, ...] = ("SIM_CAM_01", "SIM_CAM_02", "SIM_CAM_03", "SIM_CAM_04"),
        *,
        image_size_wh: tuple[int, int] = (1624, 1240),
        fps_hz: float = 50.0,
        exposure_us: float = 1800.0,
        max_exposure_skew_ms: float = 1.0,
        trigger_mode: str = "ptp_action",
    ) -> None:
        self.camera_serials = tuple(camera_serials)
        self.image_size_wh = (int(image_size_wh[0]), int(image_size_wh[1]))
        self.fps_hz = float(fps_hz)
        self.exposure_time_s = float(exposure_us) * 1e-6
        self.max_exposure_skew_ms = float(max_exposure_skew_ms)
        self.trigger_mode = trigger_mode
        self._bundle_id = 0

    def grab_next_bundle(
        self,
        *,
        injected_skew_ms: np.ndarray | None = None,
        dropped_cameras: set[int] | None = None,
    ) -> SynchronizedFrameBundle:
        n_cams = len(self.camera_serials)
        w, h = self.image_size_wh
        t_base = self._bundle_id / self.fps_hz
        skews_s = (
            np.zeros(n_cams, dtype=np.float64)
            if injected_skew_ms is None
            else np.asarray(injected_skew_ms, dtype=np.float64) * 1e-3
        )
        dropped = set() if dropped_cameras is None else set(dropped_cameras)
        frames: list[np.ndarray | None] = []
        dev_ts = np.full(n_cams, np.nan, dtype=np.float64)
        for c in range(n_cams):
            if c in dropped:
                frames.append(None)
                continue
            bayer = np.full((h, w), 90 + 15 * c + (self._bundle_id % 5), dtype=np.uint8)
            frames.append(bayer)
            dev_ts[c] = t_base + float(skews_s[c])
        bundle = assemble_synchronized_bundle(
            self._bundle_id,
            frames,
            dev_ts,
            image_size_wh=self.image_size_wh,
            is_bayer_rg8=True,
            max_exposure_skew_ms=self.max_exposure_skew_ms,
            exposure_time_s=self.exposure_time_s,
            trigger_mode=self.trigger_mode,
        )
        self._bundle_id += 1
        return bundle


def load_hik_mvs_sdk() -> Any:
    """Dynamically load Hikrobot MVS Python bindings from `MVCAM_COMMON_RUNENV` without copying vendor code."""
    runenv = os.environ.get("MVCAM_COMMON_RUNENV")
    candidates = []
    if runenv:
        candidates.append(Path(runenv) / "Samples" / "Python" / "MvImport")
    candidates.append(Path(r"C:\Program Files (x86)\Common Files\MVS\Development\Samples\Python\MvImport"))
    for mv_import in candidates:
        if mv_import.exists():
            mv_str = str(mv_import.resolve())
            if mv_str not in sys.path:
                sys.path.insert(0, mv_str)
            dll_dir = mv_import.parents[2] / "Bin" / "win64"
            if dll_dir.exists() and hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(dll_dir.resolve()))
            import MvCameraControl_class as mv  # type: ignore[import-not-found]

            return mv
    raise RuntimeError(
        "Hikrobot MVS SDK not found. Install MVS Windows SDK so MVCAM_COMMON_RUNENV is set, "
        "or use SimulatedBayerCaptureBackend for offline hardware preparation."
    )


def build_hik_register_plan(
    camera_index: int,
    num_cameras: int,
    *,
    trigger_mode: str = "ptp_action",
    exposure_us: float = 1800.0,
    gain_db: float = 4.0,
    fps_hz: float = 50.0,
    packet_size_bytes: int = 8192,
    inter_packet_delay_step: int = 1200,
) -> dict[str, Any]:
    """Return the exact GenICam node dictionary for pure-PoE (`ptp_action`/`ptp_freerun`/`software`) or `line0`."""
    if trigger_mode not in SUPPORTED_TRIGGER_MODES:
        raise ValueError(f"Unsupported trigger_mode {trigger_mode!r}")
    if not (0 <= camera_index < num_cameras):
        raise ValueError(f"camera_index {camera_index} out of range 0..{num_cameras - 1}")

    plan: dict[str, Any] = {
        "PixelFormat": "BayerRG8",
        "AcquisitionMode": "Continuous",
        "ExposureAuto": "Off",
        "ExposureTime": float(exposure_us),
        "GainAuto": "Off",
        "Gain": float(gain_db),
        "GevSCPSPacketSize": int(packet_size_bytes),
        "GevSCPD": int(camera_index * inter_packet_delay_step),
    }
    if trigger_mode == "ptp_action":
        plan.update({
            "GevIEEE1588": True,
            "TriggerMode": "On",
            "TriggerSource": "Action1",
            "ActionDeviceKey": 1,
            "ActionGroupKey": 1,
            "ActionGroupMask": 1,
        })
    elif trigger_mode == "ptp_freerun":
        plan.update({
            "GevIEEE1588": True,
            "TriggerMode": "Off",
            "AcquisitionFrameRateEnable": True,
            "AcquisitionFrameRate": float(fps_hz),
        })
    elif trigger_mode == "software":
        plan.update({
            "GevIEEE1588": True,
            "TriggerMode": "On",
            "TriggerSource": "Software",
        })
    elif trigger_mode == "line0":
        plan.update({
            "TriggerMode": "On",
            "TriggerSource": "Line0",
            "TriggerActivation": "RisingEdge",
        })
    return plan
