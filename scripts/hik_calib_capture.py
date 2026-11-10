"""Synchronised checkerboard capture for multi-camera calibration (Hikrobot GigE, pure PoE).

Acquisition: same as scripts/hik_sync_capture_test.py --mode action (2x2 binning by default, IEEE 1588 PTP, scheduled
GigE action commands), so every saved bundle holds one exposure instant per camera. Calibration must use the same
binning/resolution as later capture.
Detection: checkerboard inner corners (`--pattern COLSxROWS`, e.g. 9x6). Without `--pattern` the script waits until a
board is seen and prints the detected size, then exits (run again with that size).
Saving: a bundle is written when at least `--min-views` cameras see the whole board, at least `--min-interval-s` after
the previous save, and the board moved more than `--min-move-px` (median corner shift) in some view.
  data/calib_capture/<session>/<phase>/<bundle_id>/<serial>.png (BGR, lossless) + meta.json
  meta.json: serials, device timestamps (s, PTP time base), exposure us, image size, pattern, per-view detection.
Phases: `intrinsics` (one camera at a time, board tilted all over its image), `extrinsics` (board in the middle,
seen by >= 2 cameras), `floor` (board flat on the floor, still). Window keys: q = stop, s = force-save this bundle.
Failure: missing cameras, PTP lock timeout or incomplete bundles are reported; incomplete bundles are never saved.
"""
from __future__ import annotations

import argparse
from ctypes import byref, memset, sizeof, string_at
import json
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hik_sync_capture_test import (  # noqa: E402
    MV_FRAME_OUT, Camera, MvCamera, check, enumerate_devices, get_int, issue_actions, wait_ptp)

ROOT = Path(__file__).resolve().parents[1]
FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE


def grab_frames(cam: Camera, stop: threading.Event, sink: dict, lock: threading.Lock, freq: float, fps: float):
    frame = MV_FRAME_OUT()
    while not stop.is_set():
        memset(byref(frame), 0, sizeof(frame))
        if cam.cam.MV_CC_GetImageBuffer(frame, 200) != 0 or not frame.pBufAddr:
            continue
        info = frame.stFrameInfo
        w, h = int(info.nWidth), int(info.nHeight)
        ok = int(info.nLostPacket) == 0 and int(info.nFrameLen) >= w * h
        if ok:
            raw = np.frombuffer(string_at(frame.pBufAddr, w * h), np.uint8).reshape(h, w)
            bgr = cv2.cvtColor(raw, cv2.COLOR_BAYER_RG2BGR)
        cam.cam.MV_CC_FreeImageBuffer(frame)
        if not ok:
            continue
        ts = ((int(info.nDevTimeStampHigh) << 32) | int(info.nDevTimeStampLow)) / freq
        with lock:
            sink.setdefault(int(round(ts * fps)), {})[cam.serial] = (ts, bgr)


def detect(gray, pattern):
    ok, corners = cv2.findChessboardCornersSB(gray, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ok:
        return corners.reshape(-1, 2), pattern
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags=FLAGS)
    if ok:
        pts = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)).reshape(-1, 2)
        return pts, pattern
    return None, pattern


def guess_pattern(gray):
    for cols in range(12, 3, -1):
        for rows in range(cols, 2, -1):
            for size in ((cols, rows), (rows, cols)):
                ok, _ = cv2.findChessboardCorners(gray, size, flags=FAST)
                if ok:
                    return size
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=("intrinsics", "extrinsics", "floor"), default="intrinsics")
    ap.add_argument("--pattern", help="inner corners COLSxROWS, e.g. 9x6")
    ap.add_argument("--session", default=time.strftime("%Y%m%d"))
    ap.add_argument("--binning", type=int, choices=(1, 2), default=1,
                    help="default 1 (full 1624x1240 resolution) for accurate checkerboard corner detection")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--exposure-us", type=float)
    ap.add_argument("--min-views", type=int)
    ap.add_argument("--min-interval-s", type=float, default=0.5)
    ap.add_argument("--min-move-px", type=float, default=15.0)
    ap.add_argument("--max-bundles", type=int, default=400)
    ap.add_argument("--scpd-step", type=int, default=2000)
    args = ap.parse_args()
    pattern = tuple(int(x) for x in args.pattern.lower().split("x")) if args.pattern else None
    min_views = args.min_views or {"intrinsics": 1, "extrinsics": 2, "floor": 2}[args.phase]
    out = ROOT / "data" / "calib_capture" / args.session / args.phase
    MvCamera.MV_CC_Initialize()
    cams = []
    stop = threading.Event()
    try:
        cams = sorted((Camera(info) for info in enumerate_devices()), key=lambda c: c.serial)
        print("cameras:", [c.serial for c in cams], flush=True)
        for i, c in enumerate(cams):
            c.configure(i, args.binning, args.fps, args.exposure_us, "action", args.scpd_step)
        states, lock_s = wait_ptp(cams, 60.0)
        print(f"PTP locked in {lock_s:.1f} s: {states}", flush=True)
        freq = float(get_int(cams[0].cam, "GevTimestampTickFrequency"))
        sink, lock = {}, threading.Lock()
        for c in cams:
            check(c.cam.MV_CC_StartGrabbing(), f"{c.serial} start grabbing")
        threads = [threading.Thread(target=grab_frames, args=(c, stop, sink, lock, freq, args.fps), daemon=True)
                   for c in cams]
        action_log = {}
        threads.append(threading.Thread(target=issue_actions, daemon=True,
                                        args=(cams, args.fps, 36000.0, freq, True, stop, action_log)))
        for t in threads:
            t.start()
        serials = [c.serial for c in cams]
        existing = sorted(int(p.name) for p in out.glob("[0-9][0-9][0-9][0-9]") if p.is_dir()) if out.exists() else []
        saved = (existing[-1] + 1) if existing else 0
        last_save, last_corners, incomplete = 0.0, {}, 0
        title = f"calibration capture [{args.phase}] q=quit s=save"
        while saved < args.max_bundles:
            with lock:
                keys = sorted(sink)
                ready = [k for k in keys if len(sink[k]) == len(cams)]
                bundle = sink[ready[-1]] if ready else None
                stale = [k for k in keys if ready and k <= ready[-1]] or (keys[:-30] if len(keys) > 30 else [])
                for k in stale:
                    incomplete += len(sink.pop(k)) < len(cams)
            if bundle is None:
                time.sleep(0.01)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            imgs = [bundle[s][1] for s in serials]
            ts = [bundle[s][0] for s in serials]
            grays = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in imgs]
            if pattern is None:
                found = [g for g in (guess_pattern(gr) for gr in grays) if g]
                if found:
                    print(f"detected checkerboard inner corners: {found[0][0]}x{found[0][1]} "
                          f"(rerun with --pattern {found[0][0]}x{found[0][1]})", flush=True)
                    break
                corners = [None] * len(cams)
            else:
                det_out = [detect(g, pattern) for g in grays]
                corners = [d[0] for d in det_out]
                for _, pat_found in det_out:
                    if pat_found is not None and pat_found != pattern:
                        pattern = pat_found
                        print(f"auto-locked inner corner pattern to {pattern[0]}x{pattern[1]}", flush=True)
            seen = [i for i, c in enumerate(corners) if c is not None]
            moved = any(serials[i] not in last_corners
                        or np.median(np.linalg.norm(corners[i] - last_corners[serials[i]], axis=1)) > args.min_move_px
                        for i in seen)
            key = cv2.waitKey(1) & 0xFF
            force = key == ord("s") and len(seen) >= 1
            now = time.perf_counter()
            if (force or (len(seen) >= min_views and moved and now - last_save >= args.min_interval_s)) \
                    and max(ts) - min(ts) < 1e-3:
                bdir = out / f"{saved:04d}"
                bdir.mkdir(parents=True, exist_ok=True)
                for s, im in zip(serials, imgs):
                    cv2.imencode(".png", im)[1].tofile(str(bdir / f"{s}.png"))
                (bdir / "meta.json").write_text(json.dumps(dict(
                    serials=serials, dev_timestamps_s=ts, exposure_us=[c.exposure_us for c in cams],
                    image_size_wh=[cams[0].width, cams[0].height], binning=args.binning, pattern=list(pattern),
                    phase=args.phase, detected=[c is not None for c in corners]), indent=1), encoding="utf-8")
                for i in seen:
                    last_corners[serials[i]] = corners[i]
                saved, last_save = saved + 1, now
                print(f"saved {saved:3d}: board in {[serials[i] for i in seen]}", flush=True)
            tiles = []
            for s, im, c in zip(serials, imgs, corners):
                tile = im.copy()
                if c is not None:
                    cv2.drawChessboardCorners(tile, pattern, c.reshape(-1, 1, 2).astype(np.float32), True)
                cv2.putText(tile, f"{s} {'BOARD' if c is not None else '-'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0) if c is not None else (0, 0, 255), 2)
                tiles.append(cv2.resize(tile, (406, 310)))
            tiles += [np.zeros_like(tiles[0])] * (len(tiles) % 2)
            sheet = np.vstack([np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles), 2)])
            cv2.putText(sheet, f"saved {saved}  phase {args.phase}", (10, sheet.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow(title, sheet)
            if key == ord("q"):
                break
        print(f"done: {saved} bundles in {out}; incomplete bundles dropped: {incomplete}", flush=True)
    finally:
        stop.set()
        time.sleep(0.3)
        cv2.destroyAllWindows()
        for c in cams:
            c.close()
        MvCamera.MV_CC_Finalize()


if __name__ == "__main__":
    main()
