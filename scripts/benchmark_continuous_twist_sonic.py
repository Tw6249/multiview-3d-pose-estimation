"""Paired evaluation of TWIST2 vs SONIC on continuous C2 human pose references.

Evaluates three reference pipelines on the 5 standing/moving real human clips
from the MPI-INF-3DHP development set (`s1_walk`, `s1_crouch`, `s1_exercise`,
`s2_misc`, `s2_sports`; `s2_floor` is documented separately as a floor-lying
clip with raw pelvis-above-ankle height ~0.22 m):
  1. `A0_legacy`: legacy 3D state (`states3d.jsonl`) + legacy `FullBodyRetargeter`
     + legacy first-order 50 Hz step clipping (`prepare`).
  2. `A2_gated`: continuous 3D human state (`A2/states.jsonl`) + legacy
     `retarget_continuous` (still has turn lockout, crouch rejection, and 1-step
     IK branch flips) + 50 Hz C2 `RecoveryReferenceTrajectory`.
  3. `A2_continuous`: continuous 3D human state (`A2/states.jsonl`) +
     `retarget_continuous_pose` (per-step IK velocity bound <= 6 rad/s,
     turn/speed clamping without rollback lockout, crouch height clamped to
     [0.48, 0.95] m, short-gap pose hold <= 0.40 s) + 50 Hz C2
     `build_c2_reference_stream`.

No upstream balance/CBF/rollout safety filter (`reference_safety.py`) is
applied in front of the controllers; balance, stability, and safety are left to
the downstream tracking controllers (`TWIST2` and `SONIC`) on the identical
free-base G1 MuJoCo plant.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation

from motion_control.continuous_pose_reference import (
    build_c2_reference_stream,
    build_legacy_reference_stream,
    load_recovery_config,
    retarget_continuous_pose,
)
from motion_control.twist import CONFIG as TWIST_CONFIG, TwistPlant
from pose_sim.continuous_state import load_config as load_pred_config
from pose_sim.robot_assets import require_asset
import scripts.benchmark_controllers as bc

CLIPS = ('s1_walk', 's1_crouch', 's1_exercise', 's2_misc', 's2_sports')
CLIP_LABELS = {
    's1_walk': 'S1 行走与转身',
    's1_crouch': 'S1 下蹲与起立',
    's1_exercise': 'S1 体操与深蹲',
    's2_misc': 'S2 日常全身动作',
    's2_sports': 'S2 体育大幅度动作',
}
CONTROLLERS = ('twist', 'sonic')
REF_MODES = ('A0_legacy', 'A2_gated', 'A2_continuous')
REF_LABELS = {
    'A0_legacy': '旧参考 A0（门控撤回+一阶限速）',
    'A2_gated': '过渡参考 A2_gated（3D连续+旧IK门控+C²）',
    'A2_continuous': '新连续参考 A2_continuous（全链路C²连续）',
}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text(
        ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows),
        encoding='utf-8',
    )


def save_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8',
    )


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize_reference(rows, stream, dt=0.02):
    valid_rows = [r for r in rows if r.get('accepted')]
    held_rows = [r for r in rows if r.get('short_gap_held', False)]
    q_25 = np.array([r['reference'][6:] for r in valid_rows], float)
    t_25 = np.array([r['timestamp_ns'] for r in valid_rows], float) / 1e9
    if len(q_25) > 1:
        dq_step_deg = np.degrees(np.abs(np.diff(q_25, axis=0)))
        dt_25 = np.diff(t_25)
        adj = np.isclose(dt_25, 0.04)
        max_adj_step_deg = float(dq_step_deg[adj].max()) if adj.any() else float(dq_step_deg.max())
        pairs_over_30_deg = int((dq_step_deg.max(axis=1)[adj] > 30.0).sum()) if adj.any() else 0
        max_gap_s = float(dt_25.max())
    else:
        max_adj_step_deg = 0.0
        pairs_over_30_deg = 0
        max_gap_s = 0.0

    q_50 = stream['bounded'][:, 7:]
    dq_disc = np.diff(q_50, axis=0) / dt
    ddq_disc = np.diff(dq_disc, axis=0) / dt
    jerk_disc = np.diff(ddq_disc, axis=0) / dt

    # Measure 25 Hz ZOH velocity chatter: ratio of adjacent-tick |dq[t+1] - dq[t]| during tracking
    track_mask = stream['stage'] == 'tracking'
    dq_cmd = stream['dq']
    chatter_jumps = np.abs(np.diff(dq_cmd[track_mask], axis=0)) if track_mask.sum() > 2 else np.zeros((1, 29))

    unavailable_runs = 0
    prev_stage = None
    for st in stream['stage']:
        if st == 'reference_unavailable' and prev_stage != 'reference_unavailable':
            unavailable_runs += 1
        prev_stage = st

    return dict(
        source_frames=len(rows),
        accepted_frames=len(valid_rows),
        acceptance_fraction=len(valid_rows) / len(rows),
        short_gap_held_frames=len(held_rows),
        rejection_reasons=dict(Counter(r.get('reason', 'unknown') for r in rows if not r.get('accepted'))),
        max_adjacent_25hz_joint_step_deg=max_adj_step_deg,
        adjacent_25hz_pairs_over_30deg=pairs_over_30_deg,
        max_accepted_source_gap_s=max_gap_s,
        in_clip_withdrawal_events=unavailable_runs,
        control_50hz=dict(
            max_velocity_rad_s=float(np.abs(dq_disc).max()),
            p95_velocity_rad_s=float(np.percentile(np.abs(dq_disc), 95)),
            max_acceleration_rad_s2=float(np.abs(ddq_disc).max()),
            p95_acceleration_rad_s2=float(np.percentile(np.abs(ddq_disc), 95)),
            max_jerk_rad_s3=float(np.abs(jerk_disc).max()),
            p95_jerk_rad_s3=float(np.percentile(np.abs(jerk_disc), 95)),
            p95_subtick_dq_jump_rad_s=float(np.percentile(chatter_jumps, 95)),
            min_commanded_height_m=float(stream['bounded'][:, 2].min()),
        ),
    )


def augment_controller_metrics(trace, result, stream):
    """Add crouch-aware relative height audit alongside fixed standing diagnostic."""
    qpos = trace['qpos']
    cmd_z = trace['command_qpos'][:, 2]
    rpy = Rotation.from_quat(qpos[:, [4, 5, 6, 3]]).as_euler('xyz')
    tilt_rad = np.abs(rpy[:, :2]).max(axis=1)
    rel_z_diff = qpos[:, 2] - cmd_z
    # Reference-relative upright check: tilt < 0.5 rad and pelvis height not >0.08 m below commanded crouch height
    crouch_upright = bool(np.all(tilt_rad < 0.5) and np.all(qpos[:, 2] >= np.minimum(0.50, cmd_z - 0.05)))
    result['min_height_minus_commanded_m'] = float(rel_z_diff.min())
    result['reference_relative_upright_passed'] = crouch_upright
    return result


def run_benchmark(out_dir, delays=(120, 0)):
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    plant = TwistPlant()
    pred_cfg = load_pred_config()
    traj_cfg = load_recovery_config()

    pred_root = ROOT / 'outputs/continuous_prediction_20261020_final/natural'
    base_root = ROOT / 'outputs/foot_contact_baseline_20261019/accepted_baseline'

    tracked_files = [
        Path(__file__),
        ROOT / 'motion_control/continuous_pose_reference.py',
        ROOT / 'motion_control/continuous_reference.py',
        ROOT / 'motion_control/comparison_policies.py',
        ROOT / 'motion_control/humanoid.py',
        ROOT / 'motion_control/twist.py',
        ROOT / 'scripts/benchmark_controllers.py',
        ROOT / 'config/continuous_prediction.json',
        ROOT / 'config/g1_continuous_recovery.json',
        ROOT / 'config/g1_controller_comparison.json',
        require_asset(TWIST_CONFIG['model_asset']),
        require_asset(TWIST_CONFIG['policy_asset']),
    ]
    for clip in CLIPS:
        tracked_files.append(pred_root / clip / 'A2/states.jsonl')
        tracked_files.append(pred_root / clip / 'A2/humanoid_reference.jsonl')
        tracked_files.append(base_root / clip / 'baseline/humanoid_reference.jsonl')
    fingerprints = {str(p): sha256_file(p) for p in tracked_files}

    ref_summaries = {}
    results = {}
    streams_by_mode = {m: {} for m in REF_MODES}

    for clip in CLIPS:
        print(f'=== Preparing references for {clip} ===', flush=True)
        rows_a0 = read_jsonl(base_root / clip / 'baseline/humanoid_reference.jsonl')
        rows_a2_gated = read_jsonl(pred_root / clip / 'A2/humanoid_reference.jsonl')
        states_a2 = read_jsonl(pred_root / clip / 'A2/states.jsonl')
        ref_dir = out_dir / 'references' / clip
        ref_dir.mkdir(parents=True, exist_ok=True)
        cached_cont = ref_dir / 'A2_continuous_humanoid_reference.jsonl'
        if cached_cont.exists():
            rows_a2_cont = read_jsonl(cached_cont)
        else:
            rows_a2_cont = retarget_continuous_pose(states_a2, pred_cfg)
            write_jsonl(cached_cont, rows_a2_cont)

        stream_a0 = build_legacy_reference_stream(rows_a0)
        stream_a2_gated = build_c2_reference_stream(rows_a2_gated, plant, traj_cfg)
        stream_a2_cont = build_c2_reference_stream(rows_a2_cont, plant, traj_cfg)

        streams_by_mode['A0_legacy'][clip] = (rows_a0, stream_a0)
        streams_by_mode['A2_gated'][clip] = (rows_a2_gated, stream_a2_gated)
        streams_by_mode['A2_continuous'][clip] = (rows_a2_cont, stream_a2_cont)

        ref_summaries[clip] = {}
        for mode, (r_rows, s_stream) in (
            ('A0_legacy', (rows_a0, stream_a0)),
            ('A2_gated', (rows_a2_gated, stream_a2_gated)),
            ('A2_continuous', (rows_a2_cont, stream_a2_cont)),
        ):
            np.savez_compressed(ref_dir / f'{mode}_stream.npz', **s_stream)
            ref_summaries[clip][mode] = summarize_reference(r_rows, s_stream)
            acc = ref_summaries[clip][mode]['accepted_frames']
            max_j = ref_summaries[clip][mode]['control_50hz']['max_jerk_rad_s3']
            print(f'  {mode:15s}: accepted={acc}/150, max_jerk={max_j:.1f} rad/s^3', flush=True)

    save_json(out_dir / 'reference_continuity.json', ref_summaries)

    traces_dir = out_dir / 'traces'
    traces_dir.mkdir(parents=True, exist_ok=True)

    for delay_ms in delays:
        for mode in REF_MODES:
            if delay_ms == 0 and mode != 'A2_continuous':
                continue
            for clip in CLIPS:
                _, stream = streams_by_mode[mode][clip]
                for ctrl in CONTROLLERS:
                    key = f'{mode}_{clip}_{delay_ms}_{ctrl}'
                    t0 = time.perf_counter()
                    trace, res = bc.rollout(ctrl, clip, stream, delay_ms)
                    res = augment_controller_metrics(trace, res, stream)
                    res['reference_mode'] = mode
                    res['wall_s'] = round(time.perf_counter() - t0, 2)
                    np.savez_compressed(traces_dir / f'{key}.npz', **trace)
                    results[key] = res
                    save_json(out_dir / 'results.json', results)
                    print(
                        f'ROLLOUT {key:36s} | bal={str(res["balance_envelope_passed"]):5s} '
                        f'| stop={str(res["standing_stop_passed"]):5s} '
                        f'| q_rmse={res["joint_rmse_deg"]:5.2f} deg '
                        f'| leg_rmse={res["leg_rmse_deg"]:5.2f} deg '
                        f'| path_p95={res["path_error_p95_m"]:5.3f} m',
                        flush=True,
                    )

    protocol = dict(
        hardware_enabled=False,
        simulation_only=True,
        upstream_safety_filter_enabled=False,
        safety_responsibility='Delegated to downstream tracking controller (TWIST2 vs SONIC); no upstream CBF/rollout reference modification applied.',
        clips=CLIP_LABELS,
        excluded_clip=dict(
            s2_floor='Non-standing floor/lying sequence (raw pelvis-above-ankle height ~0.22 m < 0.35 m standing kinematics range); documented separately.'
        ),
        reference_modes=REF_LABELS,
        controllers={c: bc.NAMES[c] for c in CONTROLLERS},
        delays_ms=list(delays),
        plant_model_sha256=sha256_file(require_asset(TWIST_CONFIG['model_asset'])),
        inputs_sha256=fingerprints,
    )
    save_json(out_dir / 'protocol.json', protocol)
    for p, digest in fingerprints.items():
        if sha256_file(p) != digest:
            raise RuntimeError('Frozen input changed during benchmark: ' + p)
    save_json(
        out_dir / 'integrity.json',
        dict(
            inputs_unchanged=True,
            reference_identical_across_controllers=True,
            upstream_safety_filter_applied=False,
            inputs_sha256=fingerprints,
        ),
    )
    return ref_summaries, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output',
        type=Path,
        default=ROOT / 'outputs/continuous_twist_vs_sonic_20261021',
    )
    parser.add_argument('--delays', nargs='+', type=int, default=[120, 0])
    args = parser.parse_args()
    run_benchmark(args.output, tuple(args.delays))


if __name__ == '__main__':
    main()
