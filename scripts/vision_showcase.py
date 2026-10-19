"""Reproducible multi-scene RGB → detections → state → review video.

Synthetic source clock (nanoseconds), metres, world Z up. Each scene resets
tracking and body-shape estimation. No truth files enter inference or display.
Missing observations stay missing; the video runs offline at the source FPS.
Generated data live under outputs/. Failed/incomplete stages fail explicitly.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def prepare(output, specification):
    spec = read_json(specification)
    base = read_json(ROOT/spec['base_config'])
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(source_kind='synthetic_rendered_rgb', ground_truth_used_for_estimation=False,
                    playback='offline at source FPS; not a real-time throughput claim',
                    motion_asset='CesiumMan single walk clip; authored trajectories, speed and uniform scale',
                    scenes=[])
    for scene in spec['scenes']:
        cfg = {**base, **spec['defaults'], **scene['config']}
        source = output/'sources'/scene['id']
        source.mkdir(parents=True, exist_ok=True)
        cfg_path = source/'requested_config.json'
        if cfg_path.exists() and read_json(cfg_path) != cfg:
            raise ValueError('Existing scene configuration differs; use a new output folder')
        save_json(cfg_path, cfg)
        manifest['scenes'].append({k:v for k,v in scene.items() if k != 'config'} |
                                 dict(source=str(source.relative_to(output)),
                                      result=f"estimates/{scene['id']}", frames=cfg['frames'], fps=cfg['fps']))
    target = output/'manifest.json'
    if target.exists() and read_json(target) != manifest:
        raise ValueError('Existing manifest differs; use a new output folder')
    save_json(target, manifest)
    return manifest


def render(output, manifest):
    for scene in manifest['scenes']:
        source = output/scene['source']
        if (source/'render_validation.json').exists():
            if not read_json(source/'render_validation.json')['passed']:
                raise RuntimeError('Existing camera validation failed')
            rows = (source/'frames.jsonl').read_text().splitlines()
            if len(rows) != scene['frames']:
                raise RuntimeError('Incomplete render; inspect it before restarting')
            print(f"RENDER_VERIFIED {scene['id']}", flush=True)
            continue
        print(f"RENDER_SCENE {scene['id']}", flush=True)
        subprocess.run([sys.executable, str(ROOT/'scripts/run_blender.py'), '--output', str(source),
                        '--config', str(source/'requested_config.json'), '--render-device', 'OPTIX'], check=True)


def infer(output, manifest):
    import cv2
    import numpy as np
    from pose_sim.streaming_vision import StreamingVision
    from pose_sim.sensors import ExposureFrame
    from pose_sim.online import ExposurePacket
    from pose_sim.pipeline import read_jsonl
    from pose_sim.multiview_temporal import MultiViewTemporal
    from pose_sim.human_motion import HumanMotionEstimator

    engine = None
    try:
        for scene in manifest['scenes']:
            source, target = output/scene['source'], output/scene['result']
            if (target/'inference.json').exists():
                if read_json(target/'inference.json')['frames'] != scene['frames']:
                    raise RuntimeError('Incomplete existing inference')
                print(f"INFERENCE_VERIFIED {scene['id']}", flush=True)
                continue
            if target.exists():
                raise RuntimeError(f'Partial inference exists: {target}')
            frames = list(read_jsonl(source/'frames.jsonl'))
            if len(frames) != scene['frames']:
                raise RuntimeError('Incomplete source frames')
            cal, cfg = read_json(source/'cameras.json'), read_json(source/'config.json')
            target.mkdir(parents=True)
            if engine is None:
                engine = StreamingVision(cal, cfg, read_json(ROOT/'config/vision.json'),
                                         read_json(ROOT/'config/vision_state.json'),
                                         detector_workers=4, pose_workers=4)
            else:
                # Reuse warm image models only; never carry state across cuts.
                engine.cameras = cal['cameras']
            settings = read_json(ROOT/'config/vision_motion.json')
            fusion = MultiViewTemporal(cal, cfg, structured=False,
                observation_consistency=settings.get('observation_consistency',False))
            estimator = HumanMotionEstimator(read_json(ROOT/'config/vision_state.json'))
            previous = None
            timings, counts = [], []
            with ExitStack() as stack:
                files = {name:stack.enter_context((target/f'{name}.jsonl').open('w', encoding='utf-8'))
                         for name in ('detections', 'assigned_detections', 'poses3d', 'states3d')}
                for f in frames:
                    views = {}
                    for cid, view in f['views'].items():
                        image = cv2.imdecode(np.fromfile(source/view['image'], np.uint8), 1)
                        if image is None:
                            raise RuntimeError('Cannot decode source RGB')
                        stamp = view['timestamp_ns']
                        views[cid] = ExposureFrame(cid, stamp, stamp, 'synthetic_exposure', image)
                    packet = ExposurePacket(f['frame_id'], f['timestamp_ns'], views, 'synthetic_exposure')
                    started = time.perf_counter()
                    detection = engine.detect(packet)
                    pose, assigned = fusion.update(detection, previous)
                    previous = estimator.update(pose, cal['cameras'], assigned)
                    previous['uncertainty_calibration'] = dict(valid_for_domain=False,
                        domain='unvalidated_rgb', guarantee_claimed=False)
                    timings.append((time.perf_counter()-started)*1000)
                    for name, row in zip(files, (detection, assigned, pose, previous)):
                        files[name].write(json.dumps(row)+'\n')
                    body = [previous['joint_states'][i]['status'] for i in range(5, 17)]
                    counts.append(dict(filtered=body.count('filtered'), predicted=body.count('predicted'),
                                       missing=sum(previous['joints_world_m'][i] is None for i in range(5, 17))))
                    if (f['frame_id']+1) % 24 == 0:
                        print(f"INFERENCE {scene['id']} {f['frame_id']+1}/{len(frames)}", flush=True)
            save_json(target/'inference.json', dict(frames=len(frames), source_clock='synthetic_exposure',
                rgb_inputs=len(frames)*len(cal['cameras']), truth_used=False, runtime=engine.runtime,
                processing_ms=dict(mean=float(np.mean(timings)), p95=float(np.percentile(timings, 95))),
                body_joint_counts=counts,
                sha256={f:hashlib.sha256((target/f'{f}.jsonl').read_bytes()).hexdigest() for f in files}))
    finally:
        if engine:
            engine.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT/'outputs/vision_showcase')
    p.add_argument('--config', type=Path, default=ROOT/'config/vision_showcase.json')
    p.add_argument('--stage', choices=('all', 'render', 'infer', 'compose'), default='all')
    args = p.parse_args()
    output = args.output.resolve()
    manifest = prepare(output, args.config)
    if args.stage in ('all', 'render'):
        render(output, manifest)
    if args.stage in ('all', 'infer'):
        infer(output, manifest)
    if args.stage in ('all', 'compose'):
        from scripts.vision_showcase_video import compose
        compose(output, manifest)


if __name__ == '__main__':
    main()
