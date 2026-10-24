"""Estimate a room plane from natural floor texture in cached calibrated RGB.

MPI development adapter: existing calibration, green-cloth floor ROI, person
exclusion, and achromatic/bright-dot exclusion. No pose XYZ or reference labels.
Rectification heights are search seeds only: output points are triangulated
freely from the original image rays, then robustly fitted. Never assume Z=0.
"""
import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from pose_sim.geometry import dlt, project, center
from pose_sim.ground import calibration_hash, fit_ground


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def scene_features(source, cache):
    source, cache = Path(source), Path(cache)
    src, cal = load(source/'source.json'), load(cache/'cameras.json')
    detections = [json.loads(s) for s in (cache/'detections.jsonl').read_text().splitlines()]
    images, masks, data = {}, {}, {}
    for cam, info in src['videos'].items():
        cap = cv2.VideoCapture(info['path'])
        frames = []
        for index in (0, 20):
            cap.set(cv2.CAP_PROP_POS_FRAMES, src['start_frame']+index)
            ok, image = cap.read()
            if not ok:
                cap.release(); raise ValueError('Unreadable RGB input')
            frames.append(cv2.resize(image, (cal['cameras'][cam]['width'], cal['cameras'][cam]['height'])))
        cap.release()
        image = frames[0]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([25, 45, 15]), np.array([100, 255, 220]))
        # This is an inspected image-space ROI, not a 3-D ground-height label.
        mask[:670] = 0
        mask = cv2.erode(mask, np.ones((21, 21), np.uint8))
        difference = cv2.absdiff(cv2.GaussianBlur(frames[0], (9, 9), 0),
                                cv2.GaussianBlur(frames[1], (9, 9), 0)).max(axis=2)
        mask[difference > 15] = 0
        for index in (0, 20):
            uv = np.array([p if p is not None else [np.nan]*2
                           for p in (detections[index]['views'][cam]['keypoints_px'] or [])], float)
            xy = uv[np.isfinite(uv).all(axis=1)] if len(uv) else np.empty((0, 2))
            if len(xy):
                low = np.maximum(0, xy.min(axis=0).astype(int)-40)
                high = np.minimum(1024, xy.max(axis=0).astype(int)+40)
                mask[low[1]:high[1], low[0]:high[0]] = 0
        images[cam], masks[cam] = image, mask
    xx, yy = np.meshgrid(np.arange(-3, 3, .008), np.arange(-3, 3, .008))
    world = np.stack([xx, yy, np.zeros_like(xx)], axis=-1)
    features = []
    for seed in (-.5, 0., .5):
        world[:, :, 2] = seed
        for cam, image in images.items():
            uv = project(world.reshape(-1, 3), cal['cameras'][cam]).reshape(*xx.shape, 2).astype('float32')
            bird = cv2.remap(image, uv[:, :, 0], uv[:, :, 1], cv2.INTER_LINEAR)
            mask = cv2.remap(masks[cam], uv[:, :, 0], uv[:, :, 1], cv2.INTER_NEAREST)
            data[cam] = cv2.SIFT_create(contrastThreshold=.015).detectAndCompute(cv2.cvtColor(bird, cv2.COLOR_BGR2GRAY), mask)
        for a, b in combinations(data, 2):
            ka, da = data[a]; kb, db = data[b]
            if da is None or db is None or len(db) < 2:
                continue
            for m, other in cv2.BFMatcher().knnMatch(da, db, k=2):
                if m.distance >= .7*other.distance:
                    continue
                pts = np.array([ka[m.queryIdx].pt, kb[m.trainIdx].pt])
                wp = np.c_[pts*.008-3, np.full(2, seed)]
                cameras = [cal['cameras'][a], cal['cameras'][b]]
                xy = np.array([project(p, c)[0] for p, c in zip(wp, cameras)])
                try:
                    p = dlt(cameras, xy, [1, 1])
                except ValueError:
                    continue
                errors = np.array([np.linalg.norm(project(p, c)[0]-x) for c, x in zip(cameras, xy)])
                if errors.max() > 2 or any((np.array(c['R_world_to_camera'])@p+np.array(c['t_world_to_camera_m']))[2] <= 0 for c in cameras):
                    continue
                rays = np.array([p-center(c) for c in cameras]); rays /= np.linalg.norm(rays, axis=1)[:, None]
                angle = np.degrees(np.arccos(np.clip(rays[0]@rays[1], -1, 1)))
                if min(angle, 180-angle) < 5:
                    continue
                features.append(dict(point=p.tolist(), views=[a, b], pixels=xy.tolist(),
                                     reprojection_px=errors.tolist(), rectification_seed_m=seed))
    return cal, features, images


def run(source, cache, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cal, features, images = scene_features(source, cache)
    save(output/'features.json', features)
    ground = fit_ground([r['point'] for r in features], calibration_hash(cal))
    ground['scene_binding'] = dict(source=str(Path(source).resolve()),
        frame_offsets=[0, 20], scene='MPI indoor room; explicit reuse only with identical calibration and scene',
        detections_sha256=hashlib.sha256((Path(cache)/'detections.jsonl').read_bytes()).hexdigest(),
        source_metadata=load(Path(source)/'source.json'), initialization_uses_future_s=.8,
        floor_observation='Green cloth folds; achromatic/bright dots and people excluded')
    save(output/'ground.json', ground)
    for cam, image in images.items():
        for feature in features:
            if cam not in feature['views']:
                continue
            pixel = feature['pixels'][feature['views'].index(cam)]
            residual = abs(np.dot(feature['point'], ground['normal_world'])+ground['offset_m'])
            cv2.circle(image, tuple(np.round(pixel).astype(int)), 6, (0, 220, 255) if residual < .025 else (0, 0, 255), 2)
        cv2.imencode('.jpg', image)[1].tofile(output/f'features_camera_{cam}.jpg')
    print(json.dumps(ground, ensure_ascii=True), flush=True)
    return ground


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.cache, args.output)
