"""Frozen RGB inference followed by reference-only scoring on a real subset.

Offline sequential replay preserves every 25 Hz frame. Processing time is not
physical exposure-to-output latency. This entry point never sends robot commands.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pose_sim.streaming_vision import StreamingVision
from pose_sim.multiview_temporal import MultiViewTemporal
from pose_sim.human_motion import HumanMotionEstimator
from pose_sim.online import ExposurePacket
from pose_sim.sensors import ExposureFrame
from pose_sim.pipeline import read_jsonl, write_jsonl, reconstruct_record
from pose_sim.motion_evaluation import compare_methods, statistics, phase_lag
from scripts.import_mpi3dhp import sha256

METHODS = ('raw_geometry', 'legacy_details', 'current')
EDGES = [('left_shoulder', 'right_shoulder'), ('left_hip', 'right_hip')]
for _side in ('left', 'right'):
    EDGES += [(_side+'_'+a, _side+'_'+b) for a, b in
              [('shoulder', 'elbow'), ('elbow', 'wrist'), ('shoulder', 'hip'), ('hip', 'knee'), ('knee', 'ankle')]]


def packets(input_folder):
    """Video-only iterator: this function has no reference path or label access."""
    source = json.loads((input_folder/'source.json').read_text())
    cal = json.loads((input_folder/'cameras.json').read_text())
    # 50 Hz sources are decimated to the 25 Hz evaluation clock; stride=1 is the original path.
    stride = source.get('frame_stride', 1)
    captures = {}
    try:
        for cid, video in source['videos'].items():
            cap = cv2.VideoCapture(video['path'])
            captures[cid] = cap
            if not cap.isOpened() or not cap.set(cv2.CAP_PROP_POS_FRAMES, source['start_frame']):
                raise ValueError('Cannot seek public video')
        for i in range(source['frames']):
            frames = {}
            index = source['start_frame']+i*stride
            stamp = round(index*1e9/(source['fps']*stride))
            for cid, cap in captures.items():
                for _ in range(stride-1 if i else 0):
                    if not cap.grab():
                        raise ValueError('Truncated video; no padding or frame reuse')
                if round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != index:
                    raise ValueError('Decoder frame index mismatch')
                ok, image = cap.read()
                if not ok:
                    raise ValueError('Truncated video; no padding or frame reuse')
                camera = cal['cameras'][cid]
                image = cv2.resize(image, (camera['width'], camera['height']), interpolation=cv2.INTER_AREA)
                frames[cid] = ExposureFrame(cid, stamp, stamp, 'mpi3dhp_frame_index', image)
            yield ExposurePacket(i, stamp, frames, 'mpi3dhp_frame_index')
    finally:
        for cap in captures.values():
            cap.release()


def infer(dataset, output):
    """Only inputs/* are opened; reference files are read by score() afterwards."""
    if output.exists():
        raise FileExistsError('Use a new result directory to preserve evidence')
    output.mkdir(parents=True)
    profile = json.loads((ROOT/'config/vision.json').read_text())
    settings = json.loads((ROOT/'config/vision_motion.json').read_text())
    geometry = json.loads((ROOT/'config/demo.json').read_text())
    cv2.setNumThreads(1)
    processor = None
    timing = {}
    code_hash = hashlib.sha256()
    for path in sorted((ROOT/'pose_sim').glob('*.py')):
        code_hash.update(path.name.encode()+path.read_bytes())
    try:
        for folder in sorted((dataset/'inputs').iterdir()):
            source = json.loads((folder/'source.json').read_text())
            cal = json.loads((folder/'cameras.json').read_text())
            # Validate immutable RGB input hashes before any inference.
            for video in source['videos'].values():
                if sha256(video['path']) != video['sha256']:
                    raise ValueError('Public video changed since import')
            if processor is None:
                processor = StreamingVision(cal, geometry, profile, state_config=settings['state'], fusion_config=settings)
            processor.cameras = cal['cameras']
            pipelines = {}
            for name in ('legacy_details', 'current'):
                cfg = copy.deepcopy(settings)
                if name == 'legacy_details':
                    cfg['auxiliary_association'] = False
                    cfg['state']['auxiliary_consistency'] = False
                pipelines[name] = [MultiViewTemporal(cal, geometry, association=cfg['association'],
                    structured=cfg['structured_reprojection'], auxiliary=cfg['auxiliary_association'],
                    history=cfg['association_history'],
                    observation_consistency=cfg.get('observation_consistency',False)), HumanMotionEstimator(cfg['state']), None]
            dest = output/folder.name
            dest.mkdir()
            rows = {name: [] for name in METHODS}
            detections = []
            elapsed = []
            stage = []
            cache = {}
            iterator = iter(packets(folder))
            for i in range(source['frames']):
                begin = time.perf_counter()
                packet = next(iterator)
                decoded = time.perf_counter()
                detection = processor.detect(packet)
                detected = time.perf_counter()
                detections.append(detection)
                # Run the accepted path first, timing it separately from ablations.
                for name in ('current', 'legacy_details'):
                    fusion, state, previous = pipelines[name]
                    pose, assigned = fusion.update(detection, previous)
                    result = state.update(pose, cal['cameras'], assigned)
                    result['uncertainty_calibration'] = dict(valid_for_domain=False, domain='mpi3dhp_real_rgb', guarantee_claimed=False)
                    pipelines[name][2] = result
                    rows[name].append(result)
                    if name == 'current':
                        stage.append(dict(decode_ms=(decoded-begin)*1000, detector_pose_ms=(detected-decoded)*1000,
                                          fusion_state_ms=(time.perf_counter()-detected)*1000))
                rows['raw_geometry'].append(reconstruct_record(detection, cal['cameras'], geometry, cache))
                elapsed.append((time.perf_counter()-begin)*1000)
                if (i+1) % 50 == 0:
                    print(folder.name, i+1, '/', source['frames'], flush=True)
            iterator.close()
            write_jsonl(dest/'detections.jsonl', detections)
            for name in METHODS:
                write_jsonl(dest/(name+'.jsonl'), rows[name])
            timing[folder.name] = dict(frames=len(detections), comparison_wall_per_frame_ms=statistics(elapsed),
                current_stage_ms={k: statistics([s[k] for s in stage]) for k in stage[0]},
                current_compute_ms=statistics([s['detector_pose_ms']+s['fusion_state_ms'] for s in stage]),
                unique_person_views=sum(v['num_people_detected'] == 1 for r in detections for v in r['views'].values()),
                total_views=len(detections)*len(cal['cameras']),
                result_sha256={name: sha256(dest/(name+'.jsonl')) for name in METHODS})
        meta = dict(profile=profile, motion_settings=settings, geometry_settings=geometry, runtime=processor.runtime,
                    pose_sim_source_sha256=code_hash.hexdigest(), timings=timing,
                    measurement='offline complete-frame replay; not live camera latency or real-time delivery',
                    reference_files_opened_during_inference=False)
        (output/'inference.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    finally:
        if processor:
            processor.close()


def extra_metrics(rows, truth):
    errors = []; angles = []; velocities = []; relative = []
    for i, (row, gt) in enumerate(zip(rows, truth)):
        p = {n: np.array(v) for n, v in zip(row['joint_names'], row['joints_world_m']) if v is not None}
        q = {n: np.array(v) for n, v in zip(gt['joint_names'], gt['joints_world_m'])}
        if all(n in p for n in ('left_hip', 'right_hip')):
            delta = (p['left_hip']+p['right_hip']-q['left_hip']-q['right_hip'])/2
            relative += [float(np.linalg.norm(p[n]-q[n]-delta)*1000) for n in q if n in p]
        r = row.get('orientations', {}).get('torso')
        if r is not None:
            up = (q['left_shoulder']+q['right_shoulder']-q['left_hip']-q['right_hip'])/2
            y = q['left_hip']-q['right_hip']; y /= np.linalg.norm(y)
            z = up-y*(up@y); z /= np.linalg.norm(z)
            target = np.column_stack((np.cross(y, z), y, z))
            errors.append(float(np.linalg.norm(Rotation.from_matrix(np.array(r)@target.T).as_rotvec())*180/np.pi))
        for side in ('left', 'right'):
            for joint, names in [('elbow', ('shoulder', 'elbow', 'wrist')), ('knee', ('hip', 'knee', 'ankle'))]:
                value = row.get('joint_rotations', {}).get(side+'_'+joint, {}).get('flexion_rad')
                if value is None:
                    continue
                a, b, c = [q[side+'_'+n] for n in names]
                a, c = a-b, c-b
                target = np.pi-np.arccos(np.clip(a@c/(np.linalg.norm(a)*np.linalg.norm(c)), -1, 1))
                angles.append(float(abs(value-target)*180/np.pi))
        if 0 < i < len(truth)-1:
            before, after = truth[i-1], truth[i+1]
            dt = (after['timestamp_ns']-before['timestamp_ns'])/1e9
            reference_velocity = (np.array(after['joints_world_m'])-before['joints_world_m'])/dt
            vv = dict(zip(row['joint_names'], row.get('velocities_world_m_s', [])))
            for n, v in zip(gt['joint_names'], reference_velocity):
                if vv.get(n) is not None:
                    velocities.append(float(np.linalg.norm(np.array(vv[n])-v)))
    return dict(hip_relative_position_error_mm=statistics(relative), torso_landmark_frame_error_deg=statistics(errors),
                hinge_flexion_error_deg=statistics(angles), differentiated_reference_velocity_error_m_s=statistics(velocities),
                descriptive_phase_lag_ms=phase_lag(rows, truth),
                full_anatomical_rotation_error=None, foot_contact_accuracy=None, foot_slip_error=None)


def failure_audit(rows, raw_rows, truth):
    """Compact, scoring-only counterexamples; labels never alter observations."""
    examples = []; large = current_large = missed_recovery = 0
    for row, raw, gt in zip(rows, raw_rows, truth):
        points = dict(zip(row['joint_names'], row['joints_world_m']))
        raw_points = dict(zip(raw['joint_names'], raw['joints_world_m']))
        states = dict(zip(row['joint_names'], row['joint_states']))
        qualities = dict(zip(raw['joint_names'], raw['quality']))
        for n, q in zip(gt['joint_names'], gt['joints_world_m']):
            if points.get(n) is None: continue
            error = float(np.linalg.norm(np.array(points[n])-q)*1000)
            raw_error = float(np.linalg.norm(np.array(raw_points[n])-q)*1000) if raw_points.get(n) is not None else None
            state = states[n]
            large += error > 500
            current_large += error > 500 and state['status'] == 'filtered'
            missed_recovery += error > 500 and raw_error is not None and raw_error < 150
            examples.append(dict(source_frame_index=gt['source_frame_index'], frame_id=gt['frame_id'], joint=n,
                current_error_mm=error, raw_geometry_error_mm=raw_error, status=state['status'],
                reason=state.get('reason'), max_std_m=state.get('max_std_m'), raw_quality=qualities.get(n)))
    return dict(outputs_over_500mm=int(large), current_observations_over_500mm=int(current_large),
                current_over_500mm_while_raw_under_150mm=int(missed_recovery),
                worst_examples=sorted(examples, key=lambda v: v['current_error_mm'], reverse=True)[:12])


def score(dataset, output):
    manifest = json.loads((dataset/'manifest.json').read_text())
    inference = json.loads((output/'inference.json').read_text())
    report = dict(dataset='MPI-INF-3DHP real RGB', protocol=manifest['configuration']['protocol'],
                  clips=[], manifest_sha256=sha256(dataset/'manifest.json'), inference_sha256=sha256(output/'inference.json'),
                  limitations=[
                      '3D reference comes from markerless multiview capture, not independent Vicon measurements.',
                      'Official training partition; pretrained model overlap is unknown; not a leaderboard result.',
                      '12 approximately corresponding body joint centers, no foot/palm orientation or contact/force reference.',
                      'World-frame errors have no Procrustes, scale, root or time alignment; hip-relative metric is reported separately.',
                      'Timing is offline throughput, not physical exposure-to-output latency. No robot control validation.',
                      'Torso/hinge errors are landmark-derived quantities, not anatomical full-rotation ground truth.',
                      'Phase lag is descriptive and quantized to 40 ms; differentiated reference velocity includes label noise.'
                  ])
    combined = {name: [] for name in METHODS}; alltruth = []
    for clip in manifest['clips']:
        truth = list(read_jsonl(dataset/'references'/(clip['id']+'.jsonl')))
        methods = {name: list(read_jsonl(output/clip['id']/(name+'.jsonl'))) for name in METHODS}
        keys = [(r['frame_id'], r['timestamp_ns']) for r in truth]
        if any([(r['frame_id'], r['timestamp_ns']) for r in rows] != keys for rows in methods.values()):
            raise ValueError('Scoring requires exact complete frame correspondence')
        paired = compare_methods(methods, truth)
        metrics = {name: dict(**compare_methods({name: rows}, truth)['methods'][name], **extra_metrics(rows, truth))
                   for name, rows in methods.items()}
        report['clips'].append(dict(id=clip['id'], frames=len(truth), paired=paired, methods=metrics,
                                    failures=failure_audit(methods['current'], methods['raw_geometry'], truth),
                                    timing=inference['timings'][clip['id']], reference_checks=clip['reference_checks']))
        offset = len(alltruth)
        alltruth.extend({**r, 'frame_id': offset+i} for i, r in enumerate(truth))
        for name, rows in methods.items():
            combined[name].extend({**r, 'frame_id': offset+i} for i, r in enumerate(rows))
    report['aggregate_paired'] = compare_methods(combined, alltruth)
    report['aggregate_individual'] = {name: compare_methods({name: rows}, alltruth)['methods'][name] for name, rows in combined.items()}
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({name: {key: values[key] for key in
        ('coverage', 'all_target_success', 'common_position_error_mm')}
        for name, values in report['aggregate_individual'].items()}, indent=2))
    return report


def preview(dataset, output):
    """Local inspection only: green reference, orange accepted 3D reprojection."""
    manifest = json.loads((dataset/'manifest.json').read_text())
    writer = cv2.VideoWriter(str(output/'real_multiview_36s.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 25, (1024, 1088))
    if not writer.isOpened():
        raise ValueError('Cannot create local review video')
    contact = []
    try:
        for clip in manifest['clips']:
            folder = dataset/'inputs'/clip['id']
            cal = json.loads((folder/'cameras.json').read_text())
            truth = list(read_jsonl(dataset/'references'/(clip['id']+'.jsonl')))
            estimates = list(read_jsonl(output/clip['id']/'current.jsonl'))
            for packet, gt, estimate in zip(packets(folder), truth, estimates):
                tiles = []
                for cid, frame in packet.frames.items():
                    image = frame.image_bgr.copy(); camera = cal['cameras'][cid]
                    for row, color in [(gt, (70, 230, 70)), (estimate, (0, 150, 255))]:
                        points = {}
                        for n, p in zip(row['joint_names'], row['joints_world_m']):
                            if p is None: continue
                            xyz = np.array(camera['R_world_to_camera'])@p+camera['t_world_to_camera_m']
                            if xyz[2] <= 0: continue
                            uv = np.array(camera['K'])@xyz; uv = uv[:2]/uv[2]
                            if np.isfinite(uv).all() and np.max(abs(uv)) < 10000:
                                points[n] = tuple(np.rint(uv).astype(int))
                        for a, b in EDGES:
                            if a in points and b in points:
                                cv2.line(image, points[a], points[b], color, 3, cv2.LINE_AA)
                    tile = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
                    cv2.putText(tile, 'camera '+cid, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
                    tiles.append(tile)
                canvas = np.zeros((1088, 1024, 3), np.uint8)
                canvas[:1024] = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
                cv2.putText(canvas, f"{clip['id']}   frame {gt['source_frame_index']}   green: reference   orange: estimate", (12, 1050), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
                cv2.putText(canvas, 'Real RGB / markerless reference / local research / no robot commands', (12, 1076), cv2.FONT_HERSHEY_SIMPLEX, .55, (210, 210, 210), 1)
                writer.write(canvas)
                if packet.frame_id == 75:
                    contact.append(cv2.resize(canvas, (512, 544)))
    finally:
        writer.release()
    # Browser-compatible H.264; retain only the verified final local preview.
    video = output/'real_multiview_36s.mp4'
    converted = output/'real_multiview_36s.h264.mp4'
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(video),
                    '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', str(converted)], check=True)
    cap = cv2.VideoCapture(str(converted)); decoded = 0
    while cap.read()[0]: decoded += 1
    cap.release()
    if decoded != sum(c['frames'] for c in manifest['clips']):
        raise ValueError('Encoded preview frame count mismatch')
    converted.replace(video)
    sheet = np.vstack([np.hstack(contact[i:i+2]) for i in range(0, len(contact), 2)])
    cv2.imencode('.jpg', sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(str(output/'contact_sheet.jpg'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--score-only', action='store_true')
    p.add_argument('--preview', action='store_true')
    a = p.parse_args()
    if not a.score_only:
        infer(a.dataset, a.output)
    score(a.dataset, a.output)
    if a.preview:
        preview(a.dataset, a.output)
