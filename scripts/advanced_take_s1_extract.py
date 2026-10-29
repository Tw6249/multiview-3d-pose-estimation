"""AdvancedTake -> Module S1: fresh RTMW-27 2D keypoints from the raw RGB videos.

Inputs : data/advanced_take_raw/<Take>_Video_<cam>.mp4 (unchanged), the decode/PTS audit
         outputs/advanced_take_pipeline/audit/source_audit.json.
Outputs: outputs/advanced_take/s1/<Take>_kp27.npz with, per camera id,
         kp[n,27,3] = [u_px, v_px, simcc_score] in full-frame pixels, pts_s[n] (stream PTS, s),
         det_count[n] (YOLOX person candidates), bbox[n,4]; plus s1_meta.json.
Units  : pixels, seconds. No camera calibration or BVH is used here.
Failure: frame count != audit count, undecodable frame or non-finite output aborts the take;
         a frame with no YOLOX person box gets all scores = 0 (no fixed-box guess is used).
Detector path: YOLOX every frame in DETECT_MODE. TRACK_MODE needs a 3D calibration and is
         deliberately not used so the 2D evidence does not depend on any BVH-derived camera.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from g1_imitation.core.s1_vision_tracker import RTMW27VisionTracker  # noqa: E402

CAMERAS = ["44310274", "44310277", "44310281", "44310284"]
RAW = ROOT / "data" / "advanced_take_raw"
AUDIT = ROOT / "outputs" / "advanced_take_pipeline" / "audit" / "source_audit.json"
OUT = ROOT / "outputs" / "advanced_take" / "s1"
BATCH = 6


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def pick_box(cands: np.ndarray, scores: np.ndarray):
    if len(cands) == 0:
        return None
    area = np.maximum(cands[:, 2] - cands[:, 0], 1.0) * np.maximum(cands[:, 3] - cands[:, 1], 1.0)
    return cands[int(np.argmax(area * scores))]


def run_video(tracker: RTMW27VisionTracker, path: Path, expected: int) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    kps, boxes, dets, gpu_ms = [], [], [], []
    frames = []

    def flush():
        if not frames:
            return
        picked, counts = [], []
        for img in frames:
            c, s = tracker._detect_person_candidates_single_image(img)
            counts.append(len(c))
            picked.append(pick_box(c, s))
        use = [b if b is not None else np.array([0, 0, 64, 64], float) for b in picked]
        res = tracker.infer_batch_from_bboxes(frames, np.array(use), mode_label=tracker.DETECT_MODE)
        k = res.keypoints_2d.copy()
        for i, b in enumerate(picked):
            if b is None:
                k[i, :, 2] = 0.0
        if not np.isfinite(k).all():
            raise RuntimeError("non-finite S1 output")
        kps.append(k)
        boxes.extend(np.array(u) for u in use)
        dets.extend(counts)
        gpu_ms.append(res.gpu_inference_time_ms)
        frames.clear()

    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
        if len(frames) == BATCH:
            flush()
    flush()
    cap.release()
    kp = np.concatenate(kps, axis=0)
    if kp.shape[0] != expected:
        raise RuntimeError(f"{path.name}: decoded {kp.shape[0]} frames, audit says {expected}")
    return dict(kp=kp, bbox=np.array(boxes), det_count=np.array(dets, dtype=np.int16),
                gpu_ms_batch_median=float(np.median(gpu_ms)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--takes", nargs="+", default=["Take_002", "Take_006", "Take_007", "Take_008",
                                                    "Take_009", "Take_010", "Take_011", "Take_012"])
    args = ap.parse_args()
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    tracker = RTMW27VisionTracker(use_detector_every_frame=True)
    meta = dict(rtmw_onnx=str(tracker.rtmw_onnx_path), rtmw_sha256=sha256(tracker.rtmw_onnx_path),
                yolox_onnx=str(tracker.yolox_onnx_path), yolox_sha256=sha256(tracker.yolox_onnx_path),
                mode="DETECT_MODE_yolox_every_frame", bbox_expansion=tracker.bbox_expansion_factor,
                input_wh=list(tracker.model_input_wh), takes={})
    for take in args.takes:
        t0 = time.time()
        pack, tmeta = {}, {}
        for cid in CAMERAS:
            name = f"{take}_Video_{cid}.mp4"
            row = audit["videos"][name]
            if row["decoded_frames"] != int(row["stream"]["nb_frames"]) or row["decoder_warning_bytes"]:
                raise RuntimeError(f"{name} failed decode audit; not processed")
            r = run_video(tracker, RAW / name, row["decoded_frames"])
            pack[f"{cid}_kp"] = r["kp"]
            pack[f"{cid}_pts_s"] = np.array(row["pts_s"], dtype=np.float64)
            pack[f"{cid}_det_count"] = r["det_count"]
            pack[f"{cid}_bbox"] = r["bbox"]
            tmeta[cid] = dict(frames=int(r["kp"].shape[0]), video_sha256=sha256(RAW / name),
                              no_detection_frames=int((r["det_count"] == 0).sum()),
                              multi_person_frames=int((r["det_count"] > 1).sum()),
                              gpu_ms_batch_median=r["gpu_ms_batch_median"])
        np.savez_compressed(OUT / f"{take}_kp27.npz", **pack)
        meta["takes"][take] = tmeta
        print(take, f"{time.time() - t0:.1f}s", {c: (m["frames"], m["no_detection_frames"], m["multi_person_frames"])
                                                  for c, m in tmeta.items()}, flush=True)
    (OUT / "s1_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
