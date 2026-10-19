"""Verify a complete demo and save its inputs, environment, and result fingerprints."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pose_sim.pipeline import read_jsonl


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('session', type=Path)
    p.add_argument('--require-control', action='store_true')
    a = p.parse_args()
    s = a.session.resolve()
    cfg = json.loads((s/'config.json').read_text())
    cameras = json.loads((s/'cameras.json').read_text())['cameras']
    frames = list(read_jsonl(s/'frames.jsonl'))
    ids = [r['frame_id'] for r in frames]
    require(len(frames) == cfg['frames'] and len(set(ids)) == len(ids), 'Missing/duplicate rendered frames')
    hashes = {}
    for packet in frames:
        require(set(packet['views']) == set(cameras), 'Demo packet is missing a camera')
        for cid, view in packet['views'].items():
            path = s/view['image']
            with Image.open(path) as im:
                require(im.size == (cameras[cid]['width'], cameras[cid]['height']), 'Image size differs from calibration')
                im.verify()
            hashes[view['image']] = digest(path)
    for name in ('detections.jsonl', 'poses3d.jsonl', 'ground_truth.jsonl'):
        require([r['frame_id'] for r in read_jsonl(s/name)] == ids, f'{name} frame IDs differ from source')
    validation = json.loads((s/'render_validation.json').read_text())
    require(validation['passed'], 'Camera projection self-check failed')
    metrics = json.loads((s/'metrics.json').read_text())
    require(metrics['body12_proxy_coverage'] > 0, 'No body joints reconstructed')
    for name in ('scene.blend', 'preview.png', 'preview.gif', 'preview.mp4'):
        require((s/name).stat().st_size > 0, f'Missing/empty artifact: {name}')
    for name in ('cameras.json', 'frames.jsonl', 'config.json', 'detections.jsonl', 'poses3d.jsonl', 'ground_truth.jsonl'):
        hashes[name] = digest(s/name)
    source_files = [*ROOT.glob('requirements*.txt'), ROOT/'run.ps1', ROOT/'.python-version',
                    *ROOT.glob('pose_sim/*.py'), *ROOT.glob('scripts/*.py'), *ROOT.glob('tests/*.py'),
                    *ROOT.glob('motion_control/*.py'), *ROOT.glob('config/*.json'),
                    ROOT/'assets/model_checksums.json']
    control = None
    if a.require_control:
        c = s/'control'
        control = json.loads((c/'control_report.json').read_text(encoding='utf-8'))
        require(control['passed'] and all(control['checks'].values()), 'Control checks failed')
        require(control['input_pose_sha256'] == digest(s/'poses3d.jsonl'), 'Control used different poses')
        for name, expected in control['source_sha256'].items():
            require(digest(ROOT/name) == expected, f'Control source changed since replay: {name}')
        commands = list(read_jsonl(c/'commands.jsonl'))
        require(len(commands) == control['command_count'], 'Command count differs from report')
        for name in ('commands.jsonl','observations.jsonl','feedback.jsonl','robot_model.json',
                     'ros2_trajectory.json','control_report.json','control_preview.png','control_preview.gif'):
            require((c/name).stat().st_size > 0, f'Missing control artifact: {name}')
            hashes['control/'+name] = digest(c/name)
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                          '--format=csv,noheader'], capture_output=True, text=True)
    revision = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    report = {'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'checks_passed': True,
        'frame_count': len(frames), 'image_count': len(frames)*len(cameras),
        'python': platform.python_version(), 'platform': platform.platform(), 'gpu': gpu.stdout.strip(),
        'base_git_commit': revision.stdout.strip(),
        'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
        'source_sha256': {f.relative_to(ROOT).as_posix(): digest(f) for f in sorted(source_files)},
        'input_and_result_sha256': hashes, 'asset_sha256': digest(ROOT/'assets/CesiumMan.glb'),
        'metrics': metrics, 'render_validation': validation,
        'control': control,
        'render_runtime': json.loads((s/'render_runtime.json').read_text()),
        'inference_timing': json.loads((s/'inference_timing.json').read_text())}
    (s/'reproduction.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('VERIFIED', str(s/'reproduction.json'))


if __name__ == '__main__':
    main()
