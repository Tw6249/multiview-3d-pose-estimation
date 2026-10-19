"""Repeatable module audit: fresh RGB inference, isolated faults and simulation.

References are opened only by the separate report/scoring script. Existing inputs
are never overwritten; results belong in an ignored, new output directory.
"""
import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def regression(output, environment):
    records = []

    class Result(unittest.TextTestResult):
        def startTest(self, test):
            self.begin = time.perf_counter()
            super().startTest(test)

        def record(self, test, status, detail=None):
            records.append(dict(id=test.id(), status=status, detail=detail,
                                seconds=time.perf_counter()-getattr(self, 'begin', time.perf_counter())))

        def addSuccess(self, test):
            super().addSuccess(test)
            self.record(test, 'passed')

        def addSkip(self, test, reason):
            super().addSkip(test, reason)
            self.record(test, 'skipped', reason)

        def addFailure(self, test, err):
            super().addFailure(test, err)
            self.record(test, 'failed', self._exc_info_to_string(err, test))

        def addError(self, test, err):
            super().addError(test, err)
            self.record(test, 'error', self._exc_info_to_string(err, test))

        def addSubTest(self, test, subtest, err):
            super().addSubTest(test, subtest, err)
            if err:
                self.record(subtest, 'failed', self._exc_info_to_string(err, test))

    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    result = unittest.TextTestRunner(verbosity=1, resultclass=Result).run(suite)
    save(output/'tests'/f'{environment}.json', dict(environment=environment, python=sys.executable,
         tests_run=result.testsRun, passed=result.wasSuccessful(), counts=dict(Counter(r['status'] for r in records)),
         records=records))
    return result.wasSuccessful()


def make_processor(cal, geometry, profile, settings):
    from pose_sim.streaming_vision import StreamingVision
    return StreamingVision(cal, geometry, profile, state_config=settings['state'],
                           detector_workers=4, pose_workers=4, fusion_config=settings)


def reset_processor(processor, cal, geometry, settings):
    from pose_sim.multiview_temporal import MultiViewTemporal
    from pose_sim.human_motion import HumanMotionEstimator
    from pose_sim.person_selection import CrossViewPersonSelector
    processor.cameras = cal['cameras']
    processor.person_selector = CrossViewPersonSelector(processor.cameras, geometry)
    processor.fusion = MultiViewTemporal(cal, geometry, association=settings['association'],
        structured=settings['structured_reprojection'], auxiliary=settings['auxiliary_association'],
        history=settings['association_history'], observation_consistency=settings['observation_consistency'])
    processor.state = HumanMotionEstimator(settings['state'])
    processor.previous = None


def infer(output):
    import cv2
    import numpy as np
    from scripts.evaluate_mpi3dhp import packets
    from pose_sim.pipeline import read_jsonl, write_jsonl, reconstruct_record
    from pose_sim.online import ExposurePacket
    from pose_sim.sensors import ExposureFrame, validate_calibration
    profile = load(ROOT/'config/vision.json')
    settings = load(ROOT/'config/vision_motion.json')
    geometry = load(ROOT/'config/demo.json')
    dataset = ROOT/'outputs/public_datasets/mpi_benchmark'
    synthetic = ROOT/'outputs/human_walk_g1'
    dest = output/'inference'
    dest.mkdir(parents=True, exist_ok=False)
    code_files = [p for folder in ('pose_sim', 'motion_control', 'config') for p in sorted((ROOT/folder).glob('*'))
                  if p.suffix in ('.py', '.json')]+[Path(__file__)]
    manifest = dict(git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        code_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in code_files}, profile=profile, settings=settings,
        reference_access_during_inference=False, workers=dict(detector=4, pose=4), clips=[],
        protocol='All six development clips, 150 consecutive frames each; no parameter tuning; frozen test partition untouched.',
        video_hashes={})
    save(output/'manifest.json', manifest)
    processor = None
    cv2.setNumThreads(1)

    def synthetic_packets():
        for row in read_jsonl(synthetic/'frames.jsonl'):
            fs = {}
            for c, v in row['views'].items():
                im = cv2.imdecode(np.fromfile(synthetic/v['image'], np.uint8), cv2.IMREAD_COLOR)
                if im is None:
                    raise ValueError('Unreadable synthetic RGB')
                fs[c] = ExposureFrame(c, v['timestamp_ns'], v['timestamp_ns'], 'synthetic_frame_index', im)
            yield ExposurePacket(row['frame_id'], row['timestamp_ns'], fs, 'synthetic_frame_index')

    try:
        folders = [(p.name, p, False) for p in sorted((dataset/'inputs').iterdir())]+[('synthetic_walk', synthetic, True)]
        for cid, folder, is_synthetic in folders:
            cal = load(folder/'cameras.json')
            validate_calibration(cal)
            cfg = load(folder/'config.json') if is_synthetic else geometry
            if not is_synthetic:
                for video in load(folder/'source.json')['videos'].values():
                    path = video['path']
                    if path not in manifest['video_hashes']:
                        manifest['video_hashes'][path] = sha(path)
                    if manifest['video_hashes'][path] != video['sha256']:
                        raise ValueError('Input video checksum mismatch')
            if processor is None:
                processor = make_processor(cal, cfg, profile, settings)
                manifest['runtime'] = processor.runtime
            reset_processor(processor, cal, cfg, settings)
            clipout = dest/cid
            clipout.mkdir()
            save(clipout/'cameras.json', cal)
            save(clipout/'config.json', cfg)
            if is_synthetic:
                frames = list(read_jsonl(folder/'frames.jsonl'))
                for row in frames:
                    for view in row['views'].values():
                        view['image'] = str((folder/view['image']).resolve())
                write_jsonl(clipout/'frames.jsonl', frames)
            iterator = iter(synthetic_packets() if is_synthetic else packets(folder))
            detections, raw, fused, states, timings = [], [], [], [], []
            cache = {}
            for i in range(len(frames) if is_synthetic else load(folder/'source.json')['frames']):
                tick = time.perf_counter()
                packet = next(iterator)
                decode = (time.perf_counter()-tick)*1000
                detection = processor.detect(packet)
                tick = time.perf_counter()
                pose, assigned = processor.fusion.update(detection, processor.previous)
                fuse_ms = (time.perf_counter()-tick)*1000
                tick = time.perf_counter()
                state = processor.state.update(pose, cal['cameras'], assigned)
                state['uncertainty_calibration'] = dict(valid_for_domain=False, guarantee_claimed=False)
                state_ms = (time.perf_counter()-tick)*1000
                processor.previous = state
                tick = time.perf_counter()
                r = reconstruct_record(detection, cal['cameras'], cfg, cache)
                raw_ms = (time.perf_counter()-tick)*1000
                detections.append(detection); raw.append(r); fused.append(pose); states.append(state)
                timings.append(dict(decode_ms=decode, **processor.last_detection_timing, fusion_ms=fuse_ms,
                                    state_ms=state_ms, raw_geometry_ms=raw_ms,
                                    skew_ms=(max(f.exposure_timestamp_ns for f in packet.frames.values())-
                                             min(f.exposure_timestamp_ns for f in packet.frames.values()))/1e6))
                if (i+1) % 50 == 0:
                    print('INFER', cid, i+1, flush=True)
            if hasattr(iterator, 'close'):
                iterator.close()
            for name, rows in [('detections', detections), ('poses3d', raw), ('fused', fused), ('states3d', states), ('timings', timings)]:
                write_jsonl(clipout/(name+'.jsonl'), rows)
            manifest['clips'].append(dict(id=cid, source=str(folder), synthetic=is_synthetic, frames=len(states),
                cameras=len(cal['cameras']), input_sha256={n:sha(folder/n) for n in
                    (('cameras.json', 'frames.jsonl') if is_synthetic else ('cameras.json', 'source.json'))},
                result_sha256={n:sha(clipout/(n+'.jsonl')) for n in ('detections','poses3d','fused','states3d','timings')}))
            save(output/'manifest.json', manifest)
            print('SAVED', cid, flush=True)
    finally:
        if processor:
            processor.close()


def faults(output):
    from pose_sim.pipeline import read_jsonl, write_jsonl
    from pose_sim.fault_injection import FAULT_TYPES, inject, BODY12, limb_landmarks
    from pose_sim.multiview_temporal import MultiViewTemporal
    from pose_sim.human_motion import HumanMotionEstimator
    folder = output/'inference/s1_walk'
    rows = list(read_jsonl(folder/'detections.jsonl'))
    cal, cfg = load(folder/'cameras.json'), load(folder/'config.json')
    settings = load(ROOT/'config/vision_motion.json')
    dest = output/'faults'; dest.mkdir(exist_ok=False)
    labels = []
    for kind in FAULT_TYPES:
        cams = sorted(cal['cameras'])
        camera = max(cams, key=lambda c: sum(r['views'][c].get('keypoints_px') is not None for r in rows[60:68]))
        event = dict(id=kind, type=kind, duration_ms=320, start_frame=60, end_frame=68,
            start_ns=rows[60]['timestamp_ns'], end_ns=rows[68]['timestamp_ns'],
            cameras=cams if kind in ('body_gap','all_lr_swap','limb_dropout') else [camera],
            affected_joints=list(BODY12), landmarks=None, params={})
        if kind == 'limb_dropout':
            event.update(landmarks=limb_landmarks(rows[0]['joint_names'], 'left', 'leg'),
                         affected_joints=['left_knee','left_ankle'], params=dict(side='left',limb='leg'))
        if kind == 'view_outlier':
            event.update(landmarks=['left_wrist'],affected_joints=['left_wrist'],
                         params=dict(offset_px=100.,direction_rad=0.,score=.99))
        if kind == 'view_time_offset':
            event['params'] = dict(lag_frames=3)
        corrupt, applied = inject(rows, [event])
        fusion = MultiViewTemporal(cal, cfg, structured=settings['structured_reprojection'],
            association=settings['association'], auxiliary=settings['auxiliary_association'],
            history=settings['association_history'], observation_consistency=settings['observation_consistency'])
        estimator = HumanMotionEstimator(settings['state'])
        previous = None; result = []
        for row in corrupt:
            pose, assigned = fusion.update(row, previous)
            previous = estimator.update(pose, cal['cameras'], assigned)
            result.append(previous)
        write_jsonl(dest/(kind+'.jsonl'), result)
        labels.extend(applied)
        print('FAULT',kind,applied[0]['applied_frames'],flush=True)
    save(dest/'events.json', labels)


def robot(output):
    import os
    from motion_control.humanoid import retarget_session
    from motion_control.humanoid_replay import replay
    from motion_control.g1_command import G1SimulationSink
    from motion_control.twist import TwistPlant, DEFAULT_Q, KP, KD
    from motion_control.unitree import SDK_JOINT_NAMES
    folder = output/'inference/synthetic_walk'
    # The existing replay renderer resolves images/<camera>/<frame>.png.
    # Hard links preserve input bytes without duplicating the RGB dataset.
    for image in (ROOT/'outputs/human_walk_g1/images').glob('*/*.png'):
        target = folder/'images'/image.parent.name/image.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            os.link(image, target)
    begin = time.perf_counter()
    rows = retarget_session(folder)
    save(output/'retarget.json', dict(frames=len(rows),accepted=sum(r['accepted'] for r in rows),
        seconds=time.perf_counter()-begin,reasons=dict(Counter(r.get('reason') for r in rows if not r['accepted'])),
        input='poses3d.jsonl; enhanced states3d is not connected to the default controller'))
    results = {}
    for name, drop in [('imitation',None),('imitation_dropout', (1.5,1.9))]:
        results[name] = replay(folder, video=name=='imitation', dropout=drop, output_name=name,
                               support_diagnostics=True)
        print('ROBOT',name,'passed=',results[name]['passed'],flush=True)
    save(output/'robot.json',results)
    save(output/'robot_provenance.json',dict(python=sys.executable,
        code_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in
                     [*sorted((ROOT/'motion_control').glob('*.py')),Path(__file__)]},
        input_sha256={n:sha(folder/n) for n in ('poses3d.jsonl','humanoid_reference.jsonl')},
        runtime=load(ROOT/'.robot-runtime.json')))
    plant = TwistPlant()
    packet = dict(hardware_enabled=False,clock_domain='offline_simulation',target_model_sha256=plant.sha256,
        joint_names=list(SDK_JOINT_NAMES),sequence=0,timestamp_ns=0,valid_until_ns=60_000_000,
        position_rad=DEFAULT_Q.tolist(),kp=KP.tolist(),kd=KD.tolist(),effort_limit_nm=plant.effort.tolist(),
        velocity_rad_s=[0.]*29,feedforward_torque_nm=[0.]*29)
    cases = [('valid',{},0,False),('expired',{},60_000_000,True),('model_mismatch',dict(target_model_sha256='bad'),0,True),
             ('hardware_enabled',dict(hardware_enabled=True),0,True),('joint_order',dict(joint_names=list(reversed(SDK_JOINT_NAMES))),0,True),
             ('position_limits',dict(position_rad=[10.]*29),0,True),('future',dict(timestamp_ns=1),0,True),
             ('profile',dict(kp=[0.]*29),0,True),('replay',{},1,True)]
    checks=[]
    for name,change,now,expected in cases:
        sink=G1SimulationSink(plant)
        if name=='replay':sink.validate(packet,0)
        try:
            sink.validate(dict(packet,**change),now); rejected=False; reason=None
        except ValueError as e:
            rejected=True; reason=str(e)
        checks.append(dict(case=name,expected_rejection=expected,rejected=rejected,passed=rejected==expected,reason=reason))
    save(output/'command_boundary.json',checks)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('tests','infer','faults','robot'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--environment',choices=('vision','robot'))
    args=parser.parse_args(); args.output=args.output.resolve()
    if args.stage=='tests':
        if not args.environment:parser.error('--environment required for tests')
        raise SystemExit(0 if regression(args.output,args.environment) else 1)
    {'infer':infer,'faults':faults,'robot':robot}[args.stage](args.output)
