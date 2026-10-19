"""Frozen same-box PMPose no-mask ablation. Inference and scoring are separate.

Use isolated PyTorch CUDA environment for infer; fixed vision env for score.
No reference files are read during inference, no person detector is replaced.
"""
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from pose_sim.pipeline import read_jsonl
from pose_sim.person_selection import CrossViewPersonSelector
from pose_sim.body_head_experiment import replace_body
from pose_sim.probabilistic_pose import PMPoseSmall, WEIGHT_SHA256
from scripts.evaluate_mpi3dhp import packets, extra_metrics
from scripts.benchmark_joint_vision import sha

DATA = ROOT / 'outputs/public_datasets/mpi_long'
CACHE = ROOT / 'outputs/public_datasets/mpi_long_detections'
PROPOSALS = ROOT / 'outputs/public_datasets/mpi_long_person_candidates'


def selected_rows(cid, limit, cameras, geometry):
    proposals = {r['frame_id']: r for r in read_jsonl(PROPOSALS / (cid + '.jsonl'))}
    selector = CrossViewPersonSelector(cameras, geometry)
    for raw in itertools.islice(read_jsonl(CACHE / cid / 'detections.jsonl'), limit):
        row = copy.deepcopy(raw)
        if row['frame_id'] in proposals:
            p = proposals[row['frame_id']]
            if p['timestamp_ns'] != row['timestamp_ns']: raise ValueError('Candidate exposure mismatch')
            for c, v in p['views'].items(): row['views'][c]['person_candidates'] = v
            row = selector.update(row)
        yield row


def code_hash():
    files = sorted((ROOT / 'pose_sim').glob('*.py')) + [Path(__file__), ROOT / 'scripts/evaluate_mpi3dhp.py', ROOT / 'config/vision_motion.json', ROOT / 'config/demo.json']
    return hashlib.sha256(b''.join(p.relative_to(ROOT).as_posix().encode() + p.read_bytes() for p in files)).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def infer(args):
    output = args.output
    if output.exists(): raise FileExistsError(output)
    model = PMPoseSmall(args.repository, args.checkpoint)
    cv2.setNumThreads(1)
    geometry = json.loads((ROOT / 'config/demo.json').read_text())
    ids = ['s1_long', 's2_long'] if args.split == 'development' else ['s4_long', 's6_long']
    limit = 500 if args.split == 'development' else 750
    if args.limit: limit = args.limit
    freeze = dict(code_sha256=code_hash(), model=model.metadata, split=args.split,
        frames_per_clip=limit, inputs={},
        score_policy='predicted OKS; unchanged backend thresholds; presence and visibility stored separately',
        protocol='Frozen M-selected same xyxy boxes, own model crop preprocessing, no segmentation. Cached M body baseline, unchanged RTMW detail anchors. S1/S2 development then previously viewed S4/S6, not an untouched test set.',
        acceptance='All-target <=100mm and coverage nondecreasing; mean/P95 and >500mm errors nonincreasing; inspect per-subject results and cost. No default switch solely for development gains.')
    if args.expected_freeze:
        prev = json.loads(args.expected_freeze.read_text())
        if prev['code_sha256'] != freeze['code_sha256'] or prev['model'] != freeze['model']:
            raise ValueError('Implementation/model changed after freeze')
    for cid in ids:
        folder = DATA / 'inputs' / cid
        src = json.loads((folder / 'source.json').read_text())
        for v in src['videos'].values():
            if sha(v['path']) != v['sha256']: raise ValueError('Video changed')
        paths = [folder / 'source.json', folder / 'cameras.json', CACHE / cid / 'detections.jsonl', PROPOSALS / (cid + '.jsonl')]
        freeze['inputs'][cid] = {str(p): sha(p) for p in paths}
    output.mkdir(parents=True); write(output / 'freeze.json', freeze)
    for cid in ids:
        folder = DATA / 'inputs' / cid; cal = json.loads((folder / 'cameras.json').read_text()); cameras = list(cal['cameras'])
        rows = selected_rows(cid, limit, cal['cameras'], geometry)
        fields = {n: [] for n in ('keypoints_px', 'scores', 'presence', 'visibility', 'center_px', 'scale_px', 'boxes')}
        frames = []; stamps = []; timings = []; maps = []; map_indices = []; row_hash = hashlib.sha256()
        iterator = packets(folder)
        try:
            for i, (packet, row) in enumerate(zip(itertools.islice(iterator, limit), rows)):
                if (packet.frame_id, packet.timestamp_ns) != (row['frame_id'], row['timestamp_ns']): raise ValueError('Exposure mismatch')
                row_hash.update(json.dumps(row, sort_keys=True).encode())
                values = {n: [] for n in fields}; hm = []
                start = time.perf_counter()
                active = [c for c in cameras if row['views'][c].get('keypoints_px') is not None]
                predicted = dict(zip(active, model.predict_batch([(packet.frames[c], row['views'][c]['person_bbox_xyxy_px']) for c in active])))
                for c in cameras:
                    v = row['views'][c]
                    if v.get('keypoints_px') is None:
                        for n, shape in [('keypoints_px', (17, 2)), ('scores', (17,)), ('presence', (17,)), ('visibility', (17,)), ('center_px', (2,)), ('scale_px', (2,)), ('boxes', (4,))]: values[n].append(np.full(shape, np.nan))
                        hm.append(np.full((17, 64, 48), np.nan)); continue
                    prediction = predicted[c]
                    for n, attr in [('keypoints_px', 'keypoints_px'), ('scores', 'predicted_oks'), ('presence', 'presence'), ('visibility', 'visibility'), ('center_px', 'center_px'), ('scale_px', 'scale_px')]: values[n].append(getattr(prediction, attr))
                    values['boxes'].append(v['person_bbox_xyxy_px']); hm.append(prediction.spatial_mass)
                timings.append((time.perf_counter() - start) * 1000)
                for n in fields: fields[n].append(values[n])
                if i % 25 == 0: maps.append(hm); map_indices.append(i)
                frames.append(packet.frame_id); stamps.append(packet.timestamp_ns)
                if (i + 1) % 100 == 0: print(cid, i + 1, 'head_group_ms', round(np.mean(timings[3:]), 2), flush=True)
        finally: iterator.close()
        if len(frames) != limit: raise ValueError('Truncated clip')
        path = output / (cid + '.npz')
        np.savez_compressed(path, **{n: np.array(v, np.float32) for n, v in fields.items()},
            frame_id=frames, timestamp_ns=stamps, camera_ids=cameras, head_ms=timings,
            sampled_spatial_mass=np.array(maps, np.float32), sampled_frame_indices=map_indices)
        write(output / (cid + '.json'), dict(selected_rows_sha256=row_hash.hexdigest(), cache_sha256=sha(path), frames=limit))
        print('SAVED', cid, path.stat().st_size, flush=True)


def score(args):
    from pose_sim.multiview_temporal import MultiViewTemporal
    from pose_sim.human_motion import HumanMotionEstimator
    from pose_sim.motion_evaluation import compare_methods, statistics
    from scripts.evaluate_observation_recovery import transitions
    output = args.output
    if (output / 'report.json').exists(): raise FileExistsError(output / 'report.json')
    freeze = json.loads((output / 'freeze.json').read_text())
    if code_hash() != freeze['code_sha256']: raise ValueError('Code differs from inference freeze')
    settings = json.loads((ROOT / 'config/vision_motion.json').read_text()); geometry = json.loads((ROOT / 'config/demo.json').read_text())
    cases = []; combined = {m: [] for m in ('medium', 'pmpose_s')}; truth_all = []
    for cid, inputs in freeze['inputs'].items():
        for path, expected in inputs.items():
            if sha(path) != expected: raise ValueError('Input changed')
        meta = json.loads((output / (cid + '.json')).read_text()); path = output / (cid + '.npz')
        if sha(path) != meta['cache_sha256']: raise ValueError('Prediction cache changed')
        z = np.load(path); cal = json.loads((DATA / 'inputs' / cid / 'cameras.json').read_text())
        if list(z['camera_ids']) != list(cal['cameras']): raise ValueError('Camera order changed')
        results = {m: [] for m in combined}; times = {m: [] for m in combined}
        pipes = {m: (MultiViewTemporal(cal, geometry, structured=False, observation_consistency=True), HumanMotionEstimator(settings['state'])) for m in combined}
        row_hash = hashlib.sha256()
        for i, row in enumerate(selected_rows(cid, freeze['frames_per_clip'], cal['cameras'], geometry)):
            row_hash.update(json.dumps(row, sort_keys=True).encode())
            if (int(z['frame_id'][i]), int(z['timestamp_ns'][i])) != (row['frame_id'], row['timestamp_ns']): raise ValueError('Cache exposure mismatch')
            predictions = {}
            for j, c in enumerate(z['camera_ids']):
                if row['views'][str(c)].get('keypoints_px') is None: continue
                if not np.array_equal(z['boxes'][i, j], np.array(row['views'][str(c)]['person_bbox_xyxy_px'], np.float32)): raise ValueError('Crop changed')
                predictions[str(c)] = (z['keypoints_px'][i, j], z['scores'][i, j])
            changed = replace_body(row, predictions, cal['cameras'], WEIGHT_SHA256)
            for m, d in [('medium', row), ('pmpose_s', changed)]:
                fusion, state = pipes[m]; start = time.perf_counter()
                pose, assigned = fusion.update(d, results[m][-1] if results[m] else None)
                results[m].append(state.update(pose, cal['cameras'], assigned)); times[m].append((time.perf_counter() - start) * 1000)
        if row_hash.hexdigest() != meta['selected_rows_sha256']: raise ValueError('Selected input changed')
        # References become available only after frozen perception and reconstruction.
        truth = list(itertools.islice(read_jsonl(DATA / 'references' / (cid + '.jsonl')), freeze['frames_per_clip']))
        tails = {}; two_d = {}
        from pose_sim.geometry import JOINTS, project
        for m, rows in results.items():
            errors = []; within = 0
            for r, g in zip(rows, truth):
                if (r['frame_id'], r['timestamp_ns']) != (g['frame_id'], g['timestamp_ns']): raise ValueError('Reference exposure mismatch')
                p = dict(zip(r['joint_names'], r['joints_world_m']))
                for n, target in zip(g['joint_names'], g['joints_world_m']):
                    if p.get(n) is not None: errors.append(float(np.linalg.norm(np.asarray(p[n]) - target) * 1000))
            tails[m] = dict(all_error_mm=statistics(errors), over_500mm=sum(e > 500 for e in errors))
            combined[m].extend({**r, 'frame_id': cid + '/' + str(r['frame_id'])} for r in rows)
        pixel_errors = {m: [] for m in results}
        presence_pairs = []; visibility_values = []; mass_count = 0
        hdr = {level: dict(covered=0, area_sum=0.) for level in (.5, .9, .95)}
        map_lookup = {int(index): k for k, index in enumerate(z['sampled_frame_indices'])}
        for i, (row, g) in enumerate(zip(selected_rows(cid, len(truth), cal['cameras'], geometry), truth)):
            indices = [JOINTS.index(n) for n in g['joint_names']]
            for j, c in enumerate(z['camera_ids']):
                camera = cal['cameras'][str(c)]; gt = project(g['joints_world_m'], camera)
                valid = (gt[:, 0] >= 0) & (gt[:, 0] < camera['width']) & (gt[:, 1] >= 0) & (gt[:, 1] < camera['height'])
                if row['views'][str(c)].get('keypoints_px') is None: continue
                for m, xy in [('medium', np.asarray(row['views'][str(c)]['keypoints_px'])[:17]), ('pmpose_s', z['keypoints_px'][i, j])]:
                    pixel_errors[m].extend(np.linalg.norm(xy[indices] - gt, axis=-1)[valid].tolist())
                xy = (gt - z['center_px'][i, j] + .5 * z['scale_px'][i, j]) / z['scale_px'][i, j] * [47, 63]
                in_crop = (xy[:, 0] >= 0) & (xy[:, 0] <= 47) & (xy[:, 1] >= 0) & (xy[:, 1] <= 63)
                presence_pairs.extend(zip(z['presence'][i, j, indices][valid].tolist(), in_crop[valid].astype(int).tolist()))
                visibility_values.extend(z['visibility'][i, j, indices][valid].tolist())
                if i in map_lookup:
                    maps = z['sampled_spatial_mass'][map_lookup[i], j, indices]
                    for k in np.flatnonzero(valid & in_crop):
                        mass = maps[k]; x, y = np.rint(xy[k]).astype(int)
                        ordered = np.sort(mass.ravel())[::-1]; cumulative = np.cumsum(ordered, dtype=float)
                        mass_count += 1
                        for level, audit in hdr.items():
                            threshold = ordered[min(np.searchsorted(cumulative, level), len(ordered) - 1)]
                            audit['covered'] += int(mass[y, x] >= threshold)
                            audit['area_sum'] += float(np.mean(mass >= threshold))
        two_d = {m: statistics(v) for m, v in pixel_errors.items()}
        pp = np.array(presence_pairs)
        distribution_audit = dict(scope='Approximate projected 3D reference; in-image joints; not independent probability calibration. HDR uses every 25th frame and conditions on crop inclusion.',
            presence_samples=len(pp), presence_brier=float(np.mean((pp[:, 0] - pp[:, 1]) ** 2)),
            predicted_presence_mean=float(pp[:, 0].mean()), observed_crop_inclusion=float(pp[:, 1].mean()),
            visibility_summary=statistics(visibility_values), visibility_calibration=None,
            spatial_samples=mass_count, hdr={str(level): dict(coverage=a['covered'] / mass_count, mean_heatmap_area_fraction=a['area_sum'] / mass_count) for level, a in hdr.items()})
        truth_all.extend({**r, 'frame_id': cid + '/' + str(r['frame_id'])} for r in truth)
        case = dict(id=cid, paired=compare_methods(results, truth), tails=tails,
            projected_reference_pixel_error=two_d, distribution_audit=distribution_audit, head_group_ms=statistics(z['head_ms'][3:].tolist()),
            backend_ms={m: statistics(v) for m, v in times.items()},
            diagnostics={m: dict(kinematics=extra_metrics(rows, truth), recovery=transitions(rows, truth)) for m, rows in results.items()})
        cases.append(case); print('SCORED', cid, tails, flush=True)
    write(output / 'report.json', dict(freeze=freeze, cases=cases, paired=compare_methods(combined, truth_all),
        limitations=['PMPose-S without segmentation, not full BBoxMaskPose.', 'Cached M person selection; image-aligned boxes, model-specific crop affine.', '12 approximate body correspondences; projected reference is not occlusion/visibility ground truth.', 'Offline processing times exclude camera and capture latency; no robot validation.']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['infer', 'score']); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=['development', 'heldout'], default='development'); p.add_argument('--repository', type=Path); p.add_argument('--checkpoint', type=Path)
    p.add_argument('--expected-freeze', type=Path); p.add_argument('--limit', type=int)
    args = p.parse_args(); (infer if args.stage == 'infer' else score)(args)
