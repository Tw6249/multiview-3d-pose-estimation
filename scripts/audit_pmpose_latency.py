"""Fixed current crops, alternating batch/serial latency audit; no labels."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from pose_sim.body_head_experiment import load_head
from pose_sim.probabilistic_pose import PMPoseSmall
from pose_sim.motion_evaluation import statistics
from scripts.benchmark_pmpose import selected_rows
from scripts.evaluate_mpi3dhp import packets
from scripts.benchmark_joint_vision import sha


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    cv2.setNumThreads(1)
    folder = ROOT / 'outputs/public_datasets/mpi_long/inputs/s1_long'
    cal = json.loads((folder / 'cameras.json').read_text()); geometry = json.loads((ROOT / 'config/demo.json').read_text())
    iterator = packets(folder)
    try: packet = next(iterator)
    finally: iterator.close()
    row = next(selected_rows('s1_long', 1, cal['cameras'], geometry))
    items = [(packet.frames[c], row['views'][c]['person_bbox_xyxy_px']) for c in cal['cameras']]
    if args.method == 'pmpose_s':
        base = Path(sys.executable).parents[2]
        model = PMPoseSmall(base / 'research-pmpose/BBoxMaskPose', base / 'research-pmpose/PMPose-s-1.0.0.pth')
        meta = model.metadata
        functions = dict(serial=lambda: [model(f, b) for f, b in items], batch=lambda: model.predict_batch(items))
    else:
        model, meta = load_head('balanced')
        functions = dict(serial=lambda: [model(f.image_bgr, bboxes=[np.asarray(b, np.float32)]) for f, b in items])
    for _ in range(3):
        for fn in functions.values(): fn()
    timing = {n: [] for n in functions}; differences = []
    for i in range(20):
        results = {}
        for n in (list(functions) if i % 2 == 0 else list(functions)[::-1]):
            start = time.perf_counter(); results[n] = functions[n](); timing[n].append((time.perf_counter() - start) * 1000)
        if args.method == 'pmpose_s':
            differences.append(max(float(np.max(np.abs(a.keypoints_px - b.keypoints_px))) for a, b in zip(results['serial'], results['batch'])))
    if max(differences, default=0) > .05: raise ValueError('Batch changed decoded pixels')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,pstate,clocks.sm,clocks.mem,power.draw', '--format=csv,noheader'], text=True).strip()
    report = dict(method=args.method, model=meta, crop_count=len(items), repeats=20,
        timings_ms={n: statistics(t) for n, t in timing.items()}, raw_timings_ms=timing,
        maximum_batch_pixel_difference=max(differences, default=None), gpu_snapshot_after=gpu,
        input=dict(frame_id=packet.frame_id, timestamp_ns=packet.timestamp_ns, boxes={f.camera_id: b for f, b in items}, source_sha256=sha(folder / 'source.json')),
        code_sha256={str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__), ROOT / 'pose_sim/probabilistic_pose.py')},
        scope='Fixed-input microbenchmark with CPU scoring in background; includes crop prep, decode, output copy; not exposure-to-state latency or throughput guarantee. PMPose includes mass extraction, RTMPose includes point extraction.')
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report['timings_ms']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--method', choices=['medium', 'pmpose_s'], required=True); p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
