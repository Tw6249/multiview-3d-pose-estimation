"""RGB inference and reconstruction. Neither stage reads ground_truth.jsonl."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from .geometry import JOINTS, triangulate, CameraGeometry
from .runtime import create_body, warmup_body
from .sensors import validate_calibration
from .landmarks import SCHEMAS
from .inference import pose_batch,select_landmarks,infer_person_candidates,complete_selected_candidates
from .person_selection import CrossViewPersonSelector


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    with Path(path).open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False)+'\n')


def detect(session, mode='balanced', device='auto', warmup=3, profile=False,
           model_kind='body17', detector='default', batch=False, output=None, detector_postprocess='onnx',cross_view_person_selection=False):
    session = Path(session).resolve()
    output = Path(output).resolve() if output else session
    output.mkdir(parents=True, exist_ok=True)
    if warmup < 0:
        raise ValueError('warmup must be non-negative')
    started = time.perf_counter()
    body, runtime = create_body(mode, device, output/'profiles' if profile else None, model_kind, detector, detector_postprocess)
    schema, names, indices = SCHEMAS[model_kind]
    perception_signature = hashlib.sha256(json.dumps({k:v['sha256'] for k,v in runtime['models'].items()},sort_keys=True).encode()).hexdigest()
    calibration = json.loads((session/'cameras.json').read_text())
    validate_calibration(calibration)
    cameras = calibration['cameras']
    selector=CrossViewPersonSelector(cameras,json.loads((session/'config.json').read_text())) if cross_view_person_selection else None
    if selector:perception_signature=hashlib.sha256((perception_signature+'cross_view_person_selection_v1').encode()).hexdigest()
    packets = list(read_jsonl(session/'frames.jsonl'))
    if not packets or not cameras:
        raise ValueError('Session must contain frames and calibrated cameras')
    first_cam = next(iter(cameras.values()))
    warmup_seconds = warmup_body(body, first_cam['width'], first_cam['height'], warmup, len(cameras) if batch else 1)
    records, timings, packet_timings, stage_timings = [], [], [], []
    cv2.setNumThreads(1)
    def read_image(frame):
        img = cv2.imdecode(np.fromfile(session/frame['image'], dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f'Cannot read {frame["image"]}')
        return img
    process_begin = time.perf_counter()
    for packet in packets:
        packet_begin = time.perf_counter()
        row = {'frame_id': packet['frame_id'], 'timestamp_ns': packet['timestamp_ns'],
               'joint_names': names, 'landmark_schema_id': schema, 'source_id': 'single_person',
               'perception_signature': perception_signature, 'views': {}}
        camera_ids = list(packet['views'])
        io_start = time.perf_counter()
        # The queue is one synchronized packet, never an unbounded video buffer.
        with ThreadPoolExecutor(max_workers=min(4, len(camera_ids))) as pool:
            images = list(pool.map(read_image, packet['views'].values()))
        io_time = time.perf_counter()-io_start
        pending, detection_seconds, pose_seconds = [], 0.0, 0.0
        for cid, img in zip(camera_ids, images):
            frame = packet['views'][cid]
            # imread on Windows cannot reliably read Unicode absolute paths.
            cam = cameras[cid]
            if img.shape[:2] != (cam['height'], cam['width']):
                raise ValueError('Image dimensions differ from calibration; do not silently resize')
            begin = time.perf_counter()
            boxes = body.det_model(img)
            det_elapsed = time.perf_counter()-begin
            detection_seconds += det_elapsed
            if len(boxes) == 1 and not batch:
                keypoints, scores = body.pose_model(img, bboxes=boxes)
                if model_kind=='hybrid':
                    body_k,body_s=body.body_pose_model(img,bboxes=boxes)
                    keypoints[:,:17]=body_k;scores[:,:17]=body_s
            else:
                keypoints, scores = [], []
            elapsed = time.perf_counter()-begin
            pose_seconds += elapsed-det_elapsed
            timings.append(elapsed)
            entry = {'timestamp_ns': frame['timestamp_ns'], 'num_people_detected': len(boxes),
                     'keypoints_px': None, 'scores': None, 'inference_seconds': elapsed}
            # Unique detections are accepted here. Ambiguous views need independent
            # cross-view identity evidence below; detector order never chooses a person.
            if len(keypoints) == 1:
                k, s = np.asarray(keypoints[0]), np.asarray(scores[0])
                expected_count = 17 if model_kind == 'body17' else 133
                if k.shape != (expected_count, 2) or s.shape != (expected_count,):
                    raise ValueError(f'Unexpected model outputs {k.shape}/{s.shape}')
                k,s,outside=select_landmarks(k,s,indices,cam['width'],cam['height'])
                entry['outside_image_landmarks']=[names[i] for i in outside]
                if np.isfinite(k).all() and np.isfinite(s).all():
                    entry.update(keypoints_px=k.tolist(), scores=s.tolist())
            row['views'][cid] = entry
            if selector and 2<=len(boxes)<=4:
                candidate_begin=time.perf_counter()
                entry['person_candidates']=infer_person_candidates(body,model_kind,indices,cam,img,boxes,body_only=True)
                candidate_seconds=time.perf_counter()-candidate_begin
                entry['inference_seconds']+=candidate_seconds
                pose_seconds+=candidate_seconds
            if len(boxes) == 1 and batch:
                pending.append((cid, img, boxes[0]))
        if pending:
            begin = time.perf_counter()
            results = pose_batch(body.pose_model, [x[1] for x in pending], [x[2] for x in pending])
            if model_kind=='hybrid':
                # Preserve the validated batch-1 body numerics: near-tied SimCC
                # peaks can switch a RANSAC pair and substantially alter G1 IK.
                body_results=[(k[0],s[0]) for k,s in
                              (body.body_pose_model(img,bboxes=[box]) for _,img,box in pending)]
                for (k,s),(bk,bs) in zip(results,body_results):
                    k[:17]=bk;s[:17]=bs
            batch_seconds = time.perf_counter()-begin
            pose_seconds += batch_seconds
            for (cid, _, _), (k, s) in zip(pending, results):
                expected_count = 17 if model_kind == 'body17' else 133
                if k.shape != (expected_count, 2) or s.shape != (expected_count,):
                    raise ValueError('Unexpected batched pose schema')
                cam=cameras[cid]
                k,s,outside=select_landmarks(k,s,indices,cam['width'],cam['height'])
                row['views'][cid]['outside_image_landmarks']=[names[i] for i in outside]
                if np.isfinite(k).all() and np.isfinite(s).all():
                    row['views'][cid].update(keypoints_px=k.tolist(), scores=s.tolist())
                row['views'][cid]['inference_seconds'] += batch_seconds/len(pending)
        row['landmark_model_sources']={'body':'RTMPose-M' if model_kind in ('body17','hybrid') else 'RTMW',
                                       'feet_and_palms':'RTMW' if model_kind!='body17' else None}
        selection_begin=time.perf_counter()
        if selector:row=selector.update(row)
        selection_seconds=time.perf_counter()-selection_begin
        if selector:
            completion_begin=time.perf_counter()
            row=complete_selected_candidates(body,model_kind,indices,cameras,dict(zip(camera_ids,images)),row)
            pose_seconds+=time.perf_counter()-completion_begin
        stage_timings.append(dict(image_io=io_time,detection=detection_seconds,pose=pose_seconds,
                                  person_selection=selection_seconds))
        timings[-len(camera_ids):] = [v['inference_seconds'] for v in row['views'].values()]
        records.append(row)
        packet_timings.append(time.perf_counter()-packet_begin)
        print('DETECTED', packet['frame_id'], [v['num_people_detected'] for v in row['views'].values()], flush=True)
    write_jsonl(output/'detections.jsonl', records)
    processing_seconds = time.perf_counter()-process_begin
    profiles = {}
    if profile:
        for name in runtime['models']:
            path = Path(getattr(body, name).session.end_profiling())
            events = json.loads(path.read_text(encoding='utf-8'))
            counts = {}
            for event in events:
                provider = event.get('args', {}).get('provider')
                if provider:
                    counts[provider] = counts.get(provider, 0)+1
            profiles[name] = {'file': path.relative_to(output).as_posix(), 'node_events_by_provider': counts}
    (output/'inference_timing.json').write_text(json.dumps({'backend': 'onnxruntime_'+runtime['device'],
        **runtime, 'mode': mode, 'warmup_iterations': warmup, 'warmup_seconds': warmup_seconds,
        'image_count': len(timings), 'mean_seconds_per_image': float(np.mean(timings)) if timings else None,
        'p95_seconds_per_image': float(np.percentile(timings, 95)) if timings else None,
        'mean_seconds_per_packet_with_image_io': float(np.mean(packet_timings)),
        'p95_seconds_per_packet_with_image_io': float(np.percentile(packet_timings, 95)),
        'mean_stage_seconds': {k: float(np.mean([x[k] for x in stage_timings])) for k in stage_timings[0]},
        'batched_pose': batch, 'body_batch_size':1 if model_kind=='hybrid' or not batch else len(cameras),
        'landmark_schema_id': schema, 'profiling_enabled': profile,
        'processing_seconds': processing_seconds, 'wall_seconds': time.perf_counter()-started,
        'profiles': profiles,
        'note': 'Offline synchronized packets after warmup. Batched pose time is amortized per image. No capture, reconstruction, or display latency included.'}, indent=2), encoding='utf-8')


def reconstruct_record(record, cameras, cfg, geometry_cache=None):
    names = record.get('joint_names', JOINTS)
    if (not isinstance(names, list) or not names or any(not isinstance(n,str) or not n for n in names)
            or len(set(names)) != len(names)):
        raise ValueError('Landmark names must be nonempty unique strings')
    count = len(names)
    joints, quality = [None]*count, [None]*count
    usable = []
    excluded = {}
    for cid, view in record['views'].items():
        if cid not in cameras:
            raise ValueError('Detection references uncalibrated camera: '+cid)
        if view['keypoints_px'] is None:
            excluded[cid] = 'no unique source detection'
            continue
        if abs(view['timestamp_ns']-record['timestamp_ns']) > cfg['max_sync_error_ms']*1e6:
            excluded[cid] = 'timestamp outside synchronization tolerance'
            continue
        if len(view['keypoints_px']) != count or len(view['scores']) != count:
            raise ValueError('Detection shape differs from landmark schema')
        usable.append(cid)
    geometry_cache = {} if geometry_cache is None else geometry_cache
    for j in range(count):
        ids = []
        for cid in usable:
            score, point = record['views'][cid]['scores'][j], record['views'][cid]['keypoints_px'][j]
            if point is None:
                continue
            point = np.asarray(point, dtype=float)
            if point.shape != (2,):
                raise ValueError('Landmark pixel must be a 2-vector or null')
            if np.isfinite(score) and score >= cfg['confidence_threshold'] and np.isfinite(point).all():
                ids.append(cid)
        if len(ids) < 2:
            continue
        key = tuple(ids)
        if key not in geometry_cache:
            geometry_cache[key] = CameraGeometry([cameras[cid] for cid in ids])
        result = triangulate([cameras[cid] for cid in ids],
            [record['views'][cid]['keypoints_px'][j] for cid in ids],
            [record['views'][cid]['scores'][j] for cid in ids],
            cfg['ransac_threshold_px'], cfg['min_triangulation_angle_deg'], geometry_cache[key],
            solver=cfg.get('triangulation_solver','scipy'))
        if result:
            joints[j] = result.pop('xyz_m')
            result['inlier_cameras'] = [ids[i] for i in result.pop('inliers')]
            quality[j] = result
    source_metadata = {key: record[key] for key in ('source_model_sha256', 'detector_model_sha256', 'perception_mode',
                                                  'perception_signature','landmark_model_sources') if key in record}
    return {'frame_id': record['frame_id'], 'timestamp_ns': record['timestamp_ns'], **source_metadata,
            'joint_names': names, 'landmark_schema_id': record.get('landmark_schema_id','coco17'),
            'source_id': record.get('source_id','single_person'), 'joints_world_m': joints, 'quality': quality,
            'excluded_views': excluded}


def reconstruct(session, output=None):
    session = Path(session)
    output = Path(output) if output else session
    calibration = json.loads((session/'cameras.json').read_text())
    validate_calibration(calibration)
    cameras = calibration['cameras']
    cfg = json.loads((session/'config.json').read_text())
    begin = time.perf_counter()
    times, rows = [], []
    for r in read_jsonl(output/'detections.jsonl'):
        start = time.perf_counter()
        rows.append(reconstruct_record(r, cameras, cfg))
        times.append(time.perf_counter()-start)
    calibration_sha = hashlib.sha256(json.dumps(calibration,sort_keys=True,allow_nan=False).encode()).hexdigest()
    for row in rows:
        row['calibration_sha256'] = calibration_sha
    write_jsonl(output/'poses3d.jsonl', rows)
    (output/'reconstruction_timing.json').write_text(json.dumps(dict(mean_seconds=float(np.mean(times)),
        p95_seconds=float(np.percentile(times,95)), wall_seconds=time.perf_counter()-begin),indent=2))
    print('RECONSTRUCTED', len(rows), 'frames;', sum(sum(p is not None for p in r['joints_world_m']) for r in rows), 'valid joints', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['detect', 'reconstruct'])
    p.add_argument('session', type=Path)
    p.add_argument('--mode', choices=['lightweight', 'balanced', 'performance'], default='balanced')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--profile', action='store_true', help='Save ONNX operator execution evidence (adds measurement overhead)')
    p.add_argument('--model-kind', choices=SCHEMAS, default='body17')
    p.add_argument('--detector', choices=['default', 'tiny'], default='default')
    p.add_argument('--batch', action='store_true')
    p.add_argument('--detector-postprocess', choices=['onnx','cpu'], default='onnx')
    p.add_argument('--output', type=Path, help='Separate generated experiment directory; source RGB remains unchanged')
    a = p.parse_args()
    if a.stage == 'detect':
        detect(a.session, a.mode, a.device, a.warmup, a.profile, a.model_kind, a.detector, a.batch, a.output, a.detector_postprocess)
    else:
        reconstruct(a.session, a.output)


if __name__ == '__main__':
    main()
