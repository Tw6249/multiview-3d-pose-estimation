"""Check every Hikrobot GigE camera through the MVS SDK: parameters, one frame, brightness and sharpness.

Input : cameras reachable by the MVS SDK (MVCAM_COMMON_RUNENV must point at MVS/Development). The cameras must not
        be opened by another program (close the MVS client or disconnect its devices first).
Output: outputs/hik_check/<serial>.jpg, contact_sheet.jpg and report.json (per camera: model, IP, pixel format,
        exposure us, gain dB, trigger mode, frame rate, luminance percentiles 0-255, saturated / dark pixel
        fractions, Laplacian-variance sharpness on a 1280-px-wide copy, device timestamp ticks).
Optional `--frame-rate F`: enable AcquisitionFrameRate and cap it at F fps (four 1624x1240 BayerRG8 cameras need
        about 2.0 MB per frame each, so a single 1 Gbps link carries roughly 12 fps per camera).
Optional `--auto-exposure`: raise/lower ExposureTime toward a target median luminance, capped by
        `--max-exposure-us`, then raise Gain up to `--max-gain-db`. Changes stay in the camera until power-off
        (no UserSet is saved). Free-running capture (TriggerMode Off) only; this is not a synchronisation test.
Failure: a camera that cannot be opened or grabbed is reported with its SDK error code; others still run.
"""
from __future__ import annotations

import argparse
from ctypes import POINTER, byref, c_bool, cast, memset, sizeof, string_at
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "hik_check"

runenv = os.environ.get("MVCAM_COMMON_RUNENV", r"C:\Program Files (x86)\MVS\Development")
sys.path.append(str(Path(runenv) / "Samples" / "Python" / "MvImport"))
from MvCameraControl_class import *  # noqa: E402,F401,F403

BAYER_TO_BGR = {
    0x01080008: cv2.COLOR_BAYER_GR2BGR,
    0x01080009: cv2.COLOR_BAYER_RG2BGR,
    0x0108000A: cv2.COLOR_BAYER_GB2BGR,
    0x0108000B: cv2.COLOR_BAYER_BG2BGR,
}
MONO8, RGB8, BGR8 = 0x01080001, 0x02180014, 0x02180015


def text(arr) -> str:
    raw = memoryview(arr).tobytes().split(b"\x00", 1)[0]
    return raw.decode("gbk", errors="replace")


def get_float(cam, key):
    v = MVCC_FLOATVALUE()
    return None if cam.MV_CC_GetFloatValue(key, v) else dict(cur=v.fCurValue, min=v.fMin, max=v.fMax)


def get_enum(cam, key):
    v = MVCC_ENUMVALUE()
    if cam.MV_CC_GetEnumValue(key, v):
        return None
    if hasattr(cam, "MV_CC_GetEnumEntrySymbolic"):
        entry = MVCC_ENUMENTRY()
        entry.nValue = v.nCurValue
        if cam.MV_CC_GetEnumEntrySymbolic(key, entry) == 0:
            return text(entry.chSymbolic)
    return int(v.nCurValue)


def get_int(cam, key):
    v = MVCC_INTVALUE()
    return None if cam.MV_CC_GetIntValue(key, v) else int(v.nCurValue)


def get_bool(cam, key):
    v = c_bool(False)
    return None if cam.MV_CC_GetBoolValue(key, v) else bool(v.value)


def grab(cam, timeout_ms=1500):
    frame = MV_FRAME_OUT()
    memset(byref(frame), 0, sizeof(frame))
    ret = cam.MV_CC_GetImageBuffer(frame, timeout_ms)
    if ret != 0 or not frame.pBufAddr:
        raise RuntimeError(f"grab failed 0x{ret & 0xFFFFFFFF:08x}")
    info = frame.stFrameInfo
    w, h, ptype = int(info.nWidth), int(info.nHeight), int(info.enPixelType)
    data = np.frombuffer(string_at(frame.pBufAddr, info.nFrameLen), np.uint8).copy()
    meta = dict(frame_num=int(info.nFrameNum), width=w, height=h, pixel_type=hex(ptype),
                dev_timestamp_ticks=(int(info.nDevTimeStampHigh) << 32) | int(info.nDevTimeStampLow))
    cam.MV_CC_FreeImageBuffer(frame)
    if ptype in BAYER_TO_BGR:
        bgr = cv2.cvtColor(data[: w * h].reshape(h, w), BAYER_TO_BGR[ptype])
    elif ptype == MONO8:
        bgr = cv2.cvtColor(data[: w * h].reshape(h, w), cv2.COLOR_GRAY2BGR)
    elif ptype == RGB8:
        bgr = cv2.cvtColor(data[: w * h * 3].reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    elif ptype == BGR8:
        bgr = data[: w * h * 3].reshape(h, w, 3)
    else:
        raise RuntimeError(f"unsupported pixel format {hex(ptype)}; set an 8-bit Bayer/Mono/RGB format")
    return bgr, meta


def image_stats(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (1280, int(round(gray.shape[0] * 1280 / gray.shape[1]))), interpolation=cv2.INTER_AREA)
    p = np.percentile(gray, [5, 50, 95])
    return dict(luma_p5=float(p[0]), luma_median=float(p[1]), luma_p95=float(p[2]), luma_mean=float(gray.mean()),
                saturated_frac=float((gray >= 250).mean()), dark_frac=float((gray <= 10).mean()),
                sharpness_laplacian_var=float(cv2.Laplacian(small, cv2.CV_64F).var()),
                channel_mean_bgr=[float(x) for x in bgr.reshape(-1, 3).mean(axis=0)])


def auto_expose(cam, target, max_exp_us, max_gain_db, log):
    for _ in range(8):
        for _ in range(3):
            bgr, _ = grab(cam)
        med = float(np.median(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)))
        exp, gain = get_float(cam, "ExposureTime"), get_float(cam, "Gain")
        log.append(dict(median=med, exposure_us=exp and exp["cur"], gain_db=gain and gain["cur"]))
        if abs(med - target) <= 0.12 * target or exp is None:
            return
        ratio = float(np.clip(target / max(med, 1.0), 0.25, 4.0))
        want = exp["cur"] * ratio
        new_exp = float(np.clip(want, exp["min"], min(max_exp_us, exp["max"])))
        cam.MV_CC_SetFloatValue("ExposureTime", new_exp)
        if gain is not None:
            residual = want / new_exp
            if residual > 1.05:
                cam.MV_CC_SetFloatValue("Gain", float(np.clip(gain["cur"] + 20 * np.log10(residual), gain["min"],
                                                              min(max_gain_db, gain["max"]))))
            elif ratio < 0.95 and gain["cur"] > gain["min"]:
                cam.MV_CC_SetFloatValue("Gain", float(max(gain["min"], gain["cur"] + 20 * np.log10(ratio))))
        time.sleep(0.15)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--auto-exposure", action="store_true")
    ap.add_argument("--target-median", type=float, default=110.0)
    ap.add_argument("--max-exposure-us", type=float, default=15000.0)
    ap.add_argument("--max-gain-db", type=float, default=6.0)
    ap.add_argument("--frame-rate", type=float)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    MvCamera.MV_CC_Initialize()
    for _ in range(20):
        devs = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, devs)
        if ret != 0:
            raise SystemExit(f"enum failed 0x{ret & 0xFFFFFFFF:08x}")
        if devs.nDeviceNum:
            break
        time.sleep(0.5)
    report, tiles = dict(sdk_version=hex(MvCamera.MV_CC_GetSDKVersion()), cameras=[]), []
    print(f"found {devs.nDeviceNum} device(s)")
    for i in range(devs.nDeviceNum):
        info = cast(devs.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        gige = info.SpecialInfo.stGigEInfo
        ip = gige.nCurrentIp
        row = dict(index=i, model=text(gige.chModelName), serial=text(gige.chSerialNumber),
                   user_name=text(gige.chUserDefinedName), ip=f"{ip >> 24 & 255}.{ip >> 16 & 255}.{ip >> 8 & 255}.{ip & 255}")
        cam = MvCamera()
        try:
            if cam.MV_CC_CreateHandle(info):
                raise RuntimeError("create handle failed")
            ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != 0:
                raise RuntimeError(f"open failed 0x{ret & 0xFFFFFFFF:08x} (is MVS or another program holding it?)")
            packet = cam.MV_CC_GetOptimalPacketSize()
            if int(packet) > 0:
                cam.MV_CC_SetIntValue("GevSCPSPacketSize", int(packet))
            cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
            if args.frame_rate:
                cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
                cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(args.frame_rate))
            if cam.MV_CC_StartGrabbing():
                raise RuntimeError("start grabbing failed")
            try:
                if args.auto_exposure:
                    row["auto_exposure_log"] = []
                    auto_expose(cam, args.target_median, args.max_exposure_us, args.max_gain_db, row["auto_exposure_log"])
                for _ in range(3):
                    bgr, meta = grab(cam)
                row["params"] = dict(
                    pixel_format=get_enum(cam, "PixelFormat"), width=get_int(cam, "Width"), height=get_int(cam, "Height"),
                    exposure_auto=get_enum(cam, "ExposureAuto"), exposure_us=get_float(cam, "ExposureTime"),
                    gain_auto=get_enum(cam, "GainAuto"), gain_db=get_float(cam, "Gain"),
                    trigger_mode=get_enum(cam, "TriggerMode"), trigger_source=get_enum(cam, "TriggerSource"),
                    frame_rate_enable=get_bool(cam, "AcquisitionFrameRateEnable"),
                    frame_rate=get_float(cam, "AcquisitionFrameRate"), resulting_frame_rate=get_float(cam, "ResultingFrameRate"),
                    balance_white_auto=get_enum(cam, "BalanceWhiteAuto"), packet_size=get_int(cam, "GevSCPSPacketSize"),
                    ieee1588_supported=get_bool(cam, "GevIEEE1588") is not None)
                row["frame"] = meta
                row["stats"] = image_stats(bgr)
            finally:
                cam.MV_CC_StopGrabbing()
            name = row["serial"] or f"cam{i}"
            cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(OUT / f"{name}.jpg"))
            tile = cv2.resize(bgr, (800, int(round(bgr.shape[0] * 800 / bgr.shape[1]))))
            s, prm = row["stats"], row["params"]
            cv2.putText(tile, f"{name} {row['ip']} exp={prm['exposure_us']['cur']:.0f}us med={s['luma_median']:.0f} "
                        f"sharp={s['sharpness_laplacian_var']:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            tiles.append(tile)
            print(f"[{i}] {row['model']} {name} {row['ip']}: exp {prm['exposure_us']['cur']:.0f} us, "
                  f"gain {prm['gain_db']['cur']:.1f} dB, median {s['luma_median']:.0f}, sat {s['saturated_frac']:.3f}, "
                  f"sharp {s['sharpness_laplacian_var']:.0f}, {meta['width']}x{meta['height']} {prm['pixel_format']}")
        except Exception as exc:
            row["error"] = str(exc)
            print(f"[{i}] {row['serial']} {row['ip']}: {exc}")
        finally:
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
        report["cameras"].append(row)
    if tiles:
        h = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
        tiles += [np.zeros_like(tiles[0])] * (len(tiles) % 2)
        sheet = np.vstack([np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles), 2)])
        cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(str(OUT / "contact_sheet.jpg"))
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    MvCamera.MV_CC_Finalize()


if __name__ == "__main__":
    main()
