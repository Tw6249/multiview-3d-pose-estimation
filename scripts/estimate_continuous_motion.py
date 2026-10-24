"""Run the opt-in continuous human-state candidate on named 3-D observations.

Example:
  python -B scripts/estimate_continuous_motion.py --input SESSION/fused.jsonl
      --output outputs/continuous_trial --simulate

No camera/identity front end, ground-truth file, live transport or motor device
is opened. --simulate uses the pinned TWIST2 and the experimental C2 reference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pose_sim.continuous_state import ContinuousHumanState, load_config
from motion_control.continuous_reference import retarget_continuous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--single-frame', action='store_true')
    parser.add_argument('--simulate', action='store_true')
    args = parser.parse_args()
    cfg = load_config()
    if args.config:
        cfg.update(json.loads(args.config.read_text(encoding='utf-8')))
    payload = args.input.read_bytes()
    observations = [json.loads(line) for line in payload.decode('utf-8').splitlines()]
    if not observations:
        raise ValueError('Empty observation stream')
    args.output.mkdir(parents=True, exist_ok=False)
    estimator = ContinuousHumanState(cfg, window=not args.single_frame)
    states = [estimator.update(row) for row in observations]
    references = retarget_continuous(states, cfg)
    for name, rows in (('states.jsonl', states), ('humanoid_reference.jsonl', references)):
        (args.output/name).write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows), encoding='utf-8')
    manifest = dict(source=str(args.input.resolve()), source_sha256=hashlib.sha256(payload).hexdigest(),
                    configuration=cfg, future_frames_used=False, hardware_enabled=False,
                    accepted_references=sum(r['accepted'] for r in references),
                    frames=len(states), candidate_not_default=True)
    (args.output/'retargeting.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    if args.simulate:
        from motion_control.humanoid_replay import replay
        profile = json.loads((ROOT/'config/g1_continuous_recovery.json').read_text())
        manifest['simulation'] = replay(args.output, video=False, reference_mode='recovery_jerk_limited',
                                        trajectory_config=profile)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
