"""Compare CPU/CUDA on exactly the same session images; save independent outputs."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pose_sim.pipeline import read_jsonl, write_jsonl


def prepare(source, target, frame_limit=None):
    target.mkdir(parents=True)
    for name in ('cameras.json', 'config.json', 'ground_truth.jsonl'):
        if (source/name).exists():
            shutil.copy2(source/name, target/name)
    packets = list(read_jsonl(source/'frames.jsonl'))
    if frame_limit is not None:
        packets = packets[:frame_limit]
    for packet in packets:
        for view in packet['views'].values():
            view['image'] = str((source/view['image']).resolve())
    write_jsonl(target/'frames.jsonl', packets)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['balanced', 'lightweight', 'performance'], default='balanced')
    a = p.parse_args()
    source, output = a.session.resolve(), a.output.resolve()
    if output.exists():
        p.error('Output exists; choose a new folder')
    output.mkdir(parents=True)
    timings, metrics = {}, {}
    for device in ('cpu', 'cuda'):
        target = output/device
        prepare(source, target)
        # Independent processes avoid retained model sessions/thread pools and
        # allocator state from the preceding backend contaminating measurements.
        subprocess.run([sys.executable, '-m', 'pose_sim.pipeline', 'detect', str(target),
                        '--mode', a.mode, '--device', device], cwd=ROOT, check=True)
        subprocess.run([sys.executable, '-m', 'pose_sim.pipeline', 'reconstruct', str(target)], cwd=ROOT, check=True)
        timings[device] = json.loads((target/'inference_timing.json').read_text())
        if (target/'ground_truth.jsonl').exists():
            subprocess.run([sys.executable, '-m', 'pose_sim.evaluate', str(target)], cwd=ROOT, check=True)
            metrics[device] = json.loads((target/'metrics.json').read_text())
    profile_target = output/'cuda_profile'
    prepare(source, profile_target, frame_limit=1)
    subprocess.run([sys.executable, '-m', 'pose_sim.pipeline', 'detect', str(profile_target),
                    '--mode', a.mode, '--device', 'cuda', '--profile'], cwd=ROOT, check=True)
    profile = json.loads((profile_target/'inference_timing.json').read_text())
    for name, model in timings['cpu']['models'].items():
        if model['sha256'] != timings['cuda']['models'][name]['sha256']:
            raise RuntimeError('CPU and CUDA benchmark model hashes differ')
    if not all(q['node_events_by_provider'].get('CUDAExecutionProvider', 0) > 0
               for q in profile['profiles'].values()):
        raise RuntimeError('No CUDA operator execution recorded for one or more models')
    cpu_poses = list(read_jsonl(output/'cpu/poses3d.jsonl'))
    gpu_poses = list(read_jsonl(output/'cuda/poses3d.jsonl'))
    deltas = [float(np.linalg.norm(np.array(x)-y)*1000)
              for c, g in zip(cpu_poses, gpu_poses)
              for x, y in zip(c['joints_world_m'], g['joints_world_m']) if x is not None and y is not None]
    body_deltas = [float(np.linalg.norm(np.array(x)-y)*1000)
                   for c, g in zip(cpu_poses, gpu_poses)
                   for x, y in zip(c['joints_world_m'][5:], g['joints_world_m'][5:]) if x is not None and y is not None]
    comparison = {'mode': a.mode, 'timings': timings, 'metrics': metrics,
        'inference_speedup': timings['cpu']['mean_seconds_per_image']/timings['cuda']['mean_seconds_per_image'],
        'comparable_3d_points': len(deltas), 'mean_cpu_cuda_3d_difference_mm': float(np.mean(deltas)) if deltas else None,
        'max_cpu_cuda_3d_difference_mm': max(deltas) if deltas else None,
        'mean_cpu_cuda_body12_difference_mm': float(np.mean(body_deltas)) if body_deltas else None,
        'max_cpu_cuda_body12_difference_mm': max(body_deltas) if body_deltas else None,
        'cuda_operator_evidence': profile['profiles'],
        'note': 'Same images and model hashes, independent processes, fixed thread pools, TF32 off; three warmup iterations. Profile run separate from measured runs. Offline inference only.'}
    (output/'comparison.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')
    print('BENCHMARK_READY', str(output/'comparison.json'), flush=True)


if __name__ == '__main__':
    main()
