"""Pure-PoE multi-camera synchronisation and bandwidth test (Hikrobot GigE, MVS SDK).

Steps: open every camera exclusively -> set resolution (binning 1 or 2), frame rate, staggered GevSCPD -> enable
IEEE 1588 PTP and wait until exactly one camera reports Master and the rest Slave -> acquire for `--seconds`:
  `--mode action`  : TriggerSource Action1; the host broadcasts GigE action commands at `--fps`. With PTP locked the
                     commands are scheduled (nActionTime = common future device time), so all cameras expose at the
                     same PTP instant; if scheduling is refused the run falls back to immediate commands and says so.
  `--mode freerun` : TriggerMode Off with AcquisitionFrameRate = `--fps` (clocks shared, exposure phase NOT aligned).
Output: outputs/hik_sync/<tag>/report.json and one snapshot per camera. Per camera: frames received, frame-number
gaps, frames with lost packets, achieved rate. Across cameras: bundles grouped by device timestamp (PTP time, ticks
converted with GevTimestampTickFrequency), complete-bundle ratio and exposure-start spread (median / p99 / max, ms).
Units: seconds, milliseconds, Hz, bytes. Host clock = time.perf_counter. Device timestamps are frame start times.
Failure: an unopenable camera or a PTP lock timeout aborts with a non-zero exit; camera settings are left as tested
(volatile, not saved to a UserSet). This measures the camera network only; no pose estimation runs here.
"""
from __future__ import annotations

import argparse
from ctypes import POINTER, byref, c_bool, cast, memset, sizeof, string_at
import json
import os
from pathlib import Path
import socket
import struct
import sys
import threading
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
runenv = os.environ.get("MVCAM_COMMON_RUNENV", r"C:\Program Files (x86)\MVS\Development")
sys.path.append(str(Path(runenv) / "Samples" / "Python" / "MvImport"))
from MvCameraControl_class import *  # noqa: E402,F401,F403

ACTION_DEVICE_KEY, ACTION_GROUP_KEY, ACTION_GROUP_MASK = 0x1234, 1, 1


def check(ret, what):
    if ret != 0:
        raise RuntimeError(f"{what} failed 0x{ret & 0xFFFFFFFF:08x}")


def text(arr) -> str:
    return memoryview(arr).tobytes().split(b"\x00", 1)[0].decode("gbk", errors="replace")


def enum_symbol(cam, key):
    v = MVCC_ENUMVALUE()
    if cam.MV_CC_GetEnumValue(key, v):
        return None
    entry = MVCC_ENUMENTRY()
    entry.nValue = v.nCurValue
    return None if cam.MV_CC_GetEnumEntrySymbolic(key, entry) else text(entry.chSymbolic)


def get_int(cam, key):
    v = MVCC_INTVALUE_EX()
    check(cam.MV_CC_GetIntValueEx(key, v), f"get {key}")
    return int(v.nCurValue)


def get_float(cam, key):
    v = MVCC_FLOATVALUE()
    return None if cam.MV_CC_GetFloatValue(key, v) else float(v.fCurValue)


def latch_timestamp(cam):
    check(cam.MV_CC_SetCommandValue("GevTimestampControlLatch"), "timestamp latch")
    return get_int(cam, "GevTimestampValue")


def enumerate_devices():
    for _ in range(20):
        devs = MV_CC_DEVICE_INFO_LIST()
        check(MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, devs), "enumerate")
        if devs.nDeviceNum:
            return [cast(devs.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents for i in range(devs.nDeviceNum)]
        time.sleep(0.5)
    raise RuntimeError("no camera found")


def ip_text(value: int) -> str:
    return socket.inet_ntoa(struct.pack(">I", value))


class Camera:
    def __init__(self, info):
        gige = info.SpecialInfo.stGigEInfo
        self.serial, self.ip = text(gige.chSerialNumber), ip_text(gige.nCurrentIp)
        self.host_ip = ip_text(gige.nNetExport)
        self.cam = MvCamera()
        check(self.cam.MV_CC_CreateHandle(info), f"{self.serial} create handle")
        check(self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0),
              f"{self.serial} open (close MVS or any program using the camera)")
        self.frames: list[tuple[int, int, float, int]] = []
        self.snapshot = None

    def configure(self, index, binning, fps, exposure_us, mode, scpd_step):
        cam = self.cam
        cam.MV_CC_StopGrabbing()
        for key in ("BinningHorizontal", "BinningVertical"):
            check(cam.MV_CC_SetEnumValueByString(key, f"{key}{binning}"), f"{self.serial} {key}")
        check(cam.MV_CC_SetIntValueEx("Width", get_max(cam, "Width")), f"{self.serial} width")
        check(cam.MV_CC_SetIntValueEx("Height", get_max(cam, "Height")), f"{self.serial} height")
        check(cam.MV_CC_SetEnumValueByString("PixelFormat", "BayerRG8"), f"{self.serial} pixel format")
        if exposure_us:
            check(cam.MV_CC_SetEnumValueByString("ExposureAuto", "Off"), f"{self.serial} exposure auto")
            check(cam.MV_CC_SetFloatValue("ExposureTime", float(exposure_us)), f"{self.serial} exposure")
        packet = cam.MV_CC_GetOptimalPacketSize()
        if int(packet) > 0:
            cam.MV_CC_SetIntValue("GevSCPSPacketSize", int(packet))
        check(cam.MV_CC_SetIntValueEx("GevSCPD", int(index * scpd_step)), f"{self.serial} GevSCPD")
        if mode == "action":
            check(cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", False), f"{self.serial} frame rate off")
            check(cam.MV_CC_SetEnumValueByString("TriggerMode", "On"), f"{self.serial} trigger mode")
            check(cam.MV_CC_SetEnumValueByString("TriggerSource", "Action1"), f"{self.serial} trigger source")
            check(cam.MV_CC_SetIntValueEx("ActionDeviceKey", ACTION_DEVICE_KEY), f"{self.serial} device key")
            check(cam.MV_CC_SetIntValueEx("ActionGroupKey", ACTION_GROUP_KEY), f"{self.serial} group key")
            check(cam.MV_CC_SetIntValueEx("ActionGroupMask", ACTION_GROUP_MASK), f"{self.serial} group mask")
        else:
            check(cam.MV_CC_SetEnumValueByString("TriggerMode", "Off"), f"{self.serial} trigger mode")
            check(cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True), f"{self.serial} frame rate on")
            check(cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(fps)), f"{self.serial} frame rate")
        self.width, self.height = get_int(cam, "Width"), get_int(cam, "Height")
        self.exposure_us = get_float(cam, "ExposureTime")

    def grab_loop(self, stop: threading.Event):
        frame = MV_FRAME_OUT()
        while not stop.is_set():
            memset(byref(frame), 0, sizeof(frame))
            if self.cam.MV_CC_GetImageBuffer(frame, 200) != 0 or not frame.pBufAddr:
                continue
            info = frame.stFrameInfo
            ts = (int(info.nDevTimeStampHigh) << 32) | int(info.nDevTimeStampLow)
            self.frames.append((int(info.nFrameNum), ts, time.perf_counter(), int(info.nLostPacket)))
            if len(self.frames) % 15 == 1:
                w, h = int(info.nWidth), int(info.nHeight)
                raw = np.frombuffer(string_at(frame.pBufAddr, w * h), np.uint8).reshape(h, w)
                self.snapshot = cv2.cvtColor(raw, cv2.COLOR_BAYER_RG2BGR)
            self.cam.MV_CC_FreeImageBuffer(frame)

    def close(self):
        self.cam.MV_CC_StopGrabbing()
        self.cam.MV_CC_CloseDevice()
        self.cam.MV_CC_DestroyHandle()


def get_max(cam, key):
    v = MVCC_INTVALUE_EX()
    check(cam.MV_CC_GetIntValueEx(key, v), f"get {key}")
    return int(v.nMax)


def wait_ptp(cams, timeout_s):
    for c in cams:
        check(c.cam.MV_CC_SetBoolValue("GevIEEE1588", True), f"{c.serial} enable PTP")
    t0 = time.perf_counter()
    while True:
        states = {c.serial: enum_symbol(c.cam, "GevIEEE1588Status") for c in cams}
        n_master = sum(s == "Master" for s in states.values())
        n_slave = sum(s == "Slave" for s in states.values())
        if n_master == 1 and n_slave == len(cams) - 1:
            return states, time.perf_counter() - t0
        if time.perf_counter() - t0 > timeout_s:
            raise RuntimeError(f"PTP did not lock within {timeout_s:.0f} s: {states}")
        time.sleep(0.5)


def clock_agreement(cams, freq, repeats=20):
    """Latch each camera clock back-to-back; offsets relative to camera 0 after removing host read delay."""
    rows = []
    for _ in range(repeats):
        sample = []
        for c in cams:
            h0 = time.perf_counter()
            ts = latch_timestamp(c.cam)
            sample.append((ts / freq, 0.5 * (h0 + time.perf_counter())))
        ref_dev, ref_host = sample[0]
        rows.append([(dev - ref_dev) - (host - ref_host) for dev, host in sample])
    off_ms = np.array(rows) * 1e3
    return dict(median_offset_vs_cam0_ms=np.median(off_ms, axis=0).round(3).tolist(),
                spread_of_estimate_ms=(np.percentile(off_ms, 90, axis=0) - np.percentile(off_ms, 10, axis=0)).round(3).tolist())


def issue_actions(cams, fps, seconds, freq, scheduled, stop, log):
    info, results = MV_ACTION_CMD_INFO(), MV_ACTION_CMD_RESULT_LIST()
    info.nDeviceKey, info.nGroupKey, info.nGroupMask = ACTION_DEVICE_KEY, ACTION_GROUP_KEY, ACTION_GROUP_MASK
    host_ip = cams[0].host_ip
    prefix = host_ip.rsplit(".", 2)[0] if host_ip.startswith("169.254.") else host_ip.rsplit(".", 1)[0]
    info.pBroadcastAddress = (prefix + (".255.255" if host_ip.startswith("169.254.") else ".255")).encode()
    info.bSpecialNetEnable = 1
    info.nSpecialNetIP = struct.unpack(">I", socket.inet_aton(host_ip))[0]
    period = 1.0 / fps
    lead_s = 0.5 * period
    ref_dev = latch_timestamp(cams[0].cam)
    ref_host = time.perf_counter()
    start_host = ref_host + 0.3
    n = int(round(seconds * fps))
    statuses = []
    for k in range(n):
        if stop.is_set():
            break
        fire_host = start_host + k * period
        issue_at = fire_host - (lead_s if scheduled else 0.0)
        while time.perf_counter() < issue_at:
            time.sleep(min(0.002, max(0.0, issue_at - time.perf_counter())))
        info.bActionTimeEnable = 1 if scheduled else 0
        info.nActionTime = int(ref_dev + (fire_host - ref_host) * freq) if scheduled else 0
        info.nTimeOut = 20 if k < 3 else 0
        ret = MvCamera.MV_GIGE_IssueActionCommand(info, results)
        if k < 3:
            statuses.append(dict(ret=f"0x{ret & 0xFFFFFFFF:08x}", acks=[
                (bytes(results.pResults[i].strDeviceAddress).split(b"\x00")[0].decode(), f"0x{results.pResults[i].nStatus & 0xFFFF:04x}")
                for i in range(results.nNumResults)]))
    log["issued"] = n
    log["schedule_lead_ms"] = round(lead_s * 1e3, 2) if scheduled else 0.0
    log["action_queue_size"] = get_int(cams[0].cam, "ActionQueueSize")
    log["first_command_acks"] = statuses


def bundles(cams, freq, fps):
    events = sorted((ts / freq, idx) for idx, c in enumerate(cams) for _, ts, _, _ in c.frames)
    groups, current = [], []
    for t, idx in events:
        if current and t - current[0][0] > 0.4 / fps:
            groups.append(current)
            current = []
        current.append((t, idx))
    if current:
        groups.append(current)
    complete = [g for g in groups if sorted(i for _, i in g) == list(range(len(cams)))]
    spread_ms = np.array([(max(t for t, _ in g) - min(t for t, _ in g)) * 1e3 for g in complete]) if complete else np.array([np.nan])
    return dict(groups=len(groups), complete_bundles=len(complete),
                complete_ratio=len(complete) / max(len(groups), 1),
                spread_ms_median=float(np.nanmedian(spread_ms)), spread_ms_p99=float(np.nanpercentile(spread_ms, 99)),
                spread_ms_max=float(np.nanmax(spread_ms)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("action", "freerun"), default="action")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--binning", type=int, choices=(1, 2), default=2)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--exposure-us", type=float)
    ap.add_argument("--scpd-step", type=int, default=2000)
    ap.add_argument("--ptp-timeout-s", type=float, default=60.0)
    ap.add_argument("--no-schedule", action="store_true")
    ap.add_argument("--tag")
    args = ap.parse_args()
    tag = args.tag or f"{args.mode}_b{args.binning}_{args.fps:g}fps"
    out = ROOT / "outputs" / "hik_sync" / tag
    out.mkdir(parents=True, exist_ok=True)
    MvCamera.MV_CC_Initialize()
    cams = []
    try:
        cams = sorted((Camera(info) for info in enumerate_devices()), key=lambda c: c.serial)
        for i, c in enumerate(cams):
            c.configure(i, args.binning, args.fps, args.exposure_us, args.mode, args.scpd_step)
        ptp_states, lock_s = wait_ptp(cams, args.ptp_timeout_s)
        freq = float(get_int(cams[0].cam, "GevTimestampTickFrequency"))
        report = dict(mode=args.mode, fps=args.fps, binning=args.binning, seconds=args.seconds,
                      cameras=[dict(serial=c.serial, ip=c.ip, width=c.width, height=c.height, exposure_us=c.exposure_us)
                               for c in cams],
                      host_ip=cams[0].host_ip, ptp_status=ptp_states, ptp_lock_s=round(lock_s, 1), tick_hz=freq,
                      clock_after_ptp=clock_agreement(cams, freq))
        bytes_per_frame = cams[0].width * cams[0].height
        report["expected_payload_MB_s"] = round(bytes_per_frame * args.fps * len(cams) / 1e6, 1)
        stop = threading.Event()
        for c in cams:
            check(c.cam.MV_CC_StartGrabbing(), f"{c.serial} start grabbing")
        threads = [threading.Thread(target=c.grab_loop, args=(stop,), daemon=True) for c in cams]
        for t in threads:
            t.start()
        action_log = {}
        if args.mode == "action":
            issue_actions(cams, args.fps, args.seconds, freq, not args.no_schedule, stop, action_log)
            time.sleep(0.3)
        else:
            time.sleep(args.seconds)
        stop.set()
        for t in threads:
            t.join()
        report["action"] = action_log
        per_cam = []
        for c in cams:
            nums = np.array([f[0] for f in c.frames])
            ts = np.array([f[1] for f in c.frames]) / freq
            per_cam.append(dict(serial=c.serial, frames=len(c.frames),
                                frame_number_gaps=int(np.sum(np.diff(nums) - 1)) if len(nums) > 1 else None,
                                frames_with_lost_packets=int(sum(f[3] > 0 for f in c.frames)),
                                achieved_hz=float((len(ts) - 1) / (ts[-1] - ts[0])) if len(ts) > 1 else None))
            if c.snapshot is not None:
                cv2.imencode(".jpg", c.snapshot)[1].tofile(str(out / f"{c.serial}.jpg"))
        report["per_camera"] = per_cam
        report["bundles"] = bundles(cams, freq, args.fps)
        (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({k: report[k] for k in ("ptp_status", "ptp_lock_s", "tick_hz", "clock_after_ptp",
                                                 "expected_payload_MB_s", "action", "per_camera", "bundles")},
                         indent=1, ensure_ascii=False))
    finally:
        for c in cams:
            c.close()
        MvCamera.MV_CC_Finalize()


if __name__ == "__main__":
    main()
