"""Create the same input contract from real, timestamped, calibrated camera images.

CSV columns: camera_id,timestamp_ns,image
Image paths are relative to the CSV. Timestamps must share the same exposure clock.
No device access or collection happens in this importer.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pose_sim.pipeline import write_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', required=True, type=Path)
    p.add_argument('--calibration', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--max-skew-ms', type=float, default=5)
    p.add_argument('--fps', type=float, default=30)
    a = p.parse_args()
    cal = json.loads(a.calibration.read_text(encoding='utf-8'))
    groups = {cid: [] for cid in cal['cameras']}
    with a.csv.open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if r['camera_id'] not in groups:
                raise ValueError('Unknown camera: '+r['camera_id'])
            groups[r['camera_id']].append((int(r['timestamp_ns']), (a.csv.parent/r['image']).resolve()))
    for cid in groups:
        groups[cid].sort()
        times = [r[0] for r in groups[cid]]
        if len(times) != len(set(times)):
            raise ValueError('Duplicate timestamps: '+cid)
    anchor = next(iter(groups))
    if len(groups) < 2 or not groups[anchor]:
        raise ValueError('At least two calibrated cameras and anchor images are required')
    if a.output.exists():
        raise ValueError('Output already exists; use a new folder')
    a.output.mkdir(parents=True)
    cfg = json.loads((ROOT/'config/demo.json').read_text())
    cfg.update(max_sync_error_ms=a.max_skew_ms, fps=a.fps)
    (a.output/'cameras.json').write_text(json.dumps(cal, indent=2))
    (a.output/'config.json').write_text(json.dumps(cfg, indent=2))
    rows, used = [], {cid: set() for cid in groups}
    timestamps = {cid: np.array([r[0] for r in seq], dtype=np.int64) for cid, seq in groups.items()}
    for frame_id, (ts, _) in enumerate(groups[anchor]):
        views = {}
        for cid, seq in groups.items():
            if not seq:
                continue
            index = int(np.argmin(np.abs(timestamps[cid]-ts)))
            t, path = seq[index]
            if abs(t-ts) > a.max_skew_ms*1e6 or index in used[cid]:
                continue
            used[cid].add(index)
            target = a.output/'images'/cid/f'{frame_id:06d}{path.suffix}'
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            views[cid] = {'image': target.relative_to(a.output).as_posix(), 'timestamp_ns': t}
        rows.append({'frame_id': frame_id, 'timestamp_ns': ts, 'views': views})
    write_jsonl(a.output/'frames.jsonl', rows)
    (a.output/'source.json').write_text(json.dumps({'type': 'real_images',
        'timestamp_requirement': 'Common exposure-time clock, supplied by user/device SDK',
        'note': 'Importer matches timestamps; it does not establish hardware synchronization.'}, indent=2))
    print('IMPORTED', len(rows), 'packets;', sum(len(r['views']) >= 2 for r in rows), 'with at least two views')


if __name__ == '__main__':
    main()
