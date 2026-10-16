"""RGB inference and reconstruction. Neither stage reads ground_truth.jsonl."""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
from .geometry import JOINTS, triangulate


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    with Path(path).open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False)+'\n')


def detect(session, mode='balanced'):
    from rtmlib import Body
    body = Body(mode=mode, backend='onnxruntime', device='cpu', to_openpose=False)
    cameras = json.loads((session/'cameras.json').read_text())['cameras']
    records, timings = [], []
    for packet in read_jsonl(session/'frames.jsonl'):
        row = {'frame_id': packet['frame_id'], 'timestamp_ns': packet['timestamp_ns'], 'views': {}}
        for cid, frame in packet['views'].items():
            img = cv2.imread(str(session/frame['image']))
            if img is None:
                raise ValueError(f'Cannot read {frame["image"]}')
            cam = cameras[cid]
            if img.shape[:2] != (cam['height'], cam['width']):
                raise ValueError('Image dimensions differ from calibration; do not silently resize')
            begin = time.perf_counter()
            boxes = body.det_model(img)
            if len(boxes) == 1:
                keypoints, scores = body.pose_model(img, bboxes=boxes)
            else:
                keypoints, scores = [], []
            elapsed = time.perf_counter()-begin
            timings.append(elapsed)
            entry = {'timestamp_ns': frame['timestamp_ns'], 'num_people_detected': len(boxes),
                     'keypoints_px': None, 'scores': None, 'inference_seconds': elapsed}
            # This first version is intentionally single-person. Multiple persons are
            # rejected rather than independently selecting potentially different people.
            if len(keypoints) == 1:
                k, s = np.asarray(keypoints[0]), np.asarray(scores[0])
                if k.shape != (17, 2) or s.shape != (17,):
                    raise ValueError(f'Expected COCO17, received {k.shape}/{s.shape}')
                if np.isfinite(k).all() and np.isfinite(s).all():
                    entry.update(keypoints_px=k.tolist(), scores=s.tolist())
            row['views'][cid] = entry
        records.append(row)
        print('DETECTED', packet['frame_id'], [v['num_people_detected'] for v in row['views'].values()], flush=True)
    write_jsonl(session/'detections.jsonl', records)
    (session/'inference_timing.json').write_text(json.dumps({'backend': 'onnxruntime_cpu',
        'mode': mode, 'image_count': len(timings), 'mean_seconds_per_image': float(np.mean(timings)),
        'note': 'Serial offline inference; not an end-to-end real-time latency measurement'}, indent=2))


def reconstruct_record(record, cameras, cfg):
    joints, quality = [None]*17, [None]*17
    usable = []
    excluded = {}
    for cid, view in record['views'].items():
        if view['keypoints_px'] is None:
            excluded[cid] = 'no unique detected person'
            continue
        if abs(view['timestamp_ns']-record['timestamp_ns']) > cfg['max_sync_error_ms']*1e6:
            excluded[cid] = 'timestamp outside synchronization tolerance'
            continue
        usable.append(cid)
    for j in range(17):
        ids = [cid for cid in usable if record['views'][cid]['scores'][j] >= cfg['confidence_threshold']
               and np.isfinite(record['views'][cid]['keypoints_px'][j]).all()]
        if len(ids) < 2:
            continue
        result = triangulate([cameras[cid] for cid in ids],
            [record['views'][cid]['keypoints_px'][j] for cid in ids],
            [record['views'][cid]['scores'][j] for cid in ids],
            cfg['ransac_threshold_px'], cfg['min_triangulation_angle_deg'])
        if result:
            joints[j] = result.pop('xyz_m')
            result['inlier_cameras'] = [ids[i] for i in result.pop('inliers')]
            quality[j] = result
    return {'frame_id': record['frame_id'], 'timestamp_ns': record['timestamp_ns'],
            'joint_names': JOINTS, 'joints_world_m': joints, 'quality': quality,
            'excluded_views': excluded}


def reconstruct(session):
    cameras = json.loads((session/'cameras.json').read_text())['cameras']
    cfg = json.loads((session/'config.json').read_text())
    rows = [reconstruct_record(r, cameras, cfg) for r in read_jsonl(session/'detections.jsonl')]
    write_jsonl(session/'poses3d.jsonl', rows)
    print('RECONSTRUCTED', len(rows), 'frames;', sum(sum(p is not None for p in r['joints_world_m']) for r in rows), 'valid joints', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['detect', 'reconstruct'])
    p.add_argument('session', type=Path)
    p.add_argument('--mode', choices=['lightweight', 'balanced', 'performance'], default='balanced')
    a = p.parse_args()
    if a.stage == 'detect':
        detect(a.session, a.mode)
    else:
        reconstruct(a.session)


if __name__ == '__main__':
    main()
