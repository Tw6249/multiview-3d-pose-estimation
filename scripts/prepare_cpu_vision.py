"""Fetch CPU FP32 vision weights and public MAMMA multi-camera videos.

Original data remains under data/ (ignored); models under outputs/local_cpu.
No motion labels or synthetic truth are used as perception inputs. Downloads
are checked against the local baseline manifest. Its hashes record the original
acquisition, not an upstream signature or independent accuracy validation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import zipfile

from prepare_local_cpu import ROOT, OUT, fetch, sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mamma-commit", required=True, help="Pinned cuevhv/mamma commit for calibration")
    args = ap.parse_args()
    model_dir = OUT / "vision-models"
    data_dir = ROOT / "data/mamma_cpu_raw"
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "g1_imitation/config/pipeline_config.json").read_text(encoding="utf-8"))
    pose = cfg["s1_vision_tracker"]["rtmw_fp32_onnx"]
    detector = cfg["s1_vision_tracker"]["yolox_onnx"]
    base = "https://download.openmmlab.com/mmpose/v1/projects"
    jobs = [
        (model_dir / pose, f"{base}/rtmw/onnx_sdk/{Path(pose).stem}.zip", True),
        (model_dir / detector, f"{base}/rtmposev1/onnx_sdk/{Path(detector).stem}.zip", True),
        (data_dir / "calibration.yaml", f"https://raw.githubusercontent.com/cuevhv/mamma/{args.mamma_commit}/configs/examples/calib/iphones_outdoors.yaml", False),
    ]
    video_base = "https://download.is.tue.mpg.de/mamma/assets/download/dataset_examples/pushing_and_lifting_from_ground/videos_light"
    jobs += [(data_dir / f"{cam}.mp4", f"{video_base}/{cam}.mp4", False) for cam in ("A001", "B001", "C001", "D001")]
    manifest_file = OUT / "vision_data_manifest.json"
    prior = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.exists() else []
    hashes = {r["path"]: r["sha256"] for r in prior}
    pinned_file = ROOT / "g1_imitation/config/local_cpu_assets.json"
    if pinned_file.exists():
        pinned = json.loads(pinned_file.read_text(encoding="utf-8"))
        if args.mamma_commit != pinned["mamma_calibration_commit"]:
            raise ValueError("Calibration commit differs from the CPU baseline")
        hashes.update({r["path"]: r["sha256"] for r in pinned["vision_and_data"]})

    def acquire(job):
        path, url, zipped = job
        rel = path.relative_to(ROOT).as_posix()
        if path.exists():
            data = path.read_bytes()
        else:
            print(f"Downloading {path.name}", flush=True)
            data = fetch(url)
            if zipped:
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    names = [n for n in archive.namelist() if n.endswith(".onnx")]
                    if len(names) != 1:
                        raise ValueError(f"Expected one ONNX in {url}: {names}")
                    data = archive.read(names[0])
            if not data or b"<html" in data[:256].lower():
                raise ValueError(f"Invalid asset response: {url}")
            path.write_bytes(data)
        digest = sha256(data)
        if rel in hashes and digest != hashes[rel]:
            raise ValueError(f"Asset changed since first acquisition: {rel}")
        print(f"Ready {path.name} ({len(data)} bytes)", flush=True)
        return dict(path=rel, sha256=digest, source=url, first_acquisition_verified_upstream_hash=False)

    with ThreadPoolExecutor(max_workers=4) as pool:
        manifest = list(pool.map(acquire, jobs))
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
