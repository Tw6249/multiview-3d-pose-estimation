"""Paired ablation of the full-chain continuous reference (IK branch + 50 Hz stream).

Modes (same 3D human states for all but A0; identical plant/controllers/delay):
  A0_legacy      legacy reference + first-order rate limit (2026-10-19 baseline)
  A2_continuous  2026-10-21 chain: velocity-bounded IK, group Ruckig, v_target = 0
  A3_ik          A2_continuous + shoulder branch barrier in IK only
  A3_balanced    gimbal_barrier=True + per-DOF softened Ruckig (1500 rad/s^3) with monotonicity-gated 0.45x terminal velocity
  A3_stream      A2_continuous IK + per-joint Ruckig with causal terminal velocity
  A3_full        both changes (gimbal_barrier=True + per_dof_rate)
No upstream safety filter. Metrics come straight from `benchmark_controllers.rollout`;
nothing is clipped, rescaled or dropped after the rollout. Simulation only.
"""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from motion_control.continuous_pose_reference import (
    build_c2_reference_stream,
    build_legacy_reference_stream,
    load_recovery_config,
    retarget_continuous_pose,
)
from motion_control.twist import TwistPlant
from pose_sim.continuous_state import load_config as load_pred_config
import scripts.benchmark_controllers as bc
from scripts.benchmark_continuous_twist_sonic import (
    CLIPS,
    augment_controller_metrics,
    read_jsonl,
    save_json,
    sha256_file,
    summarize_reference,
    write_jsonl,
)

MODES = ('A0_legacy', 'A2_continuous', 'A3_ik', 'A3_balanced', 'A3_stream', 'A3_full')
CONTROLLERS = ('twist', 'sonic')


def ik_arc_metrics(rows):
    errs = np.array(
        [[np.nan if e is None else e for e in r['limb_direction_error_deg']] for r in rows if r.get('accepted') and not r.get('short_gap_held')],
        float,
    )
    worst = np.nanmax(np.where(np.isfinite(errs), errs, -1.0), axis=1)
    return dict(
        mean_limb_error_deg=float(np.nanmean(errs)),
        p95_limb_error_deg=float(np.nanpercentile(errs, 95)),
        max_limb_error_deg=float(np.nanmax(errs)),
        frames_worst_limb_over_20deg=int((worst > 20).sum()),
    )


def common_a0_metrics(trace, stream, rows_a0, delay_ms=120):
    """Evaluate closed-loop robot trajectory against unchanged A0 reference on common observable ticks."""
    qpos = trace['qpos']
    t_ctrl = stream['time_s']
    first = next(r for r in rows_a0 if r.get('accepted'))
    origin = np.asarray(first['root_position_m'], float)
    hdg0 = float(first['root_heading_rad'])
    c, s = np.cos(-hdg0), np.sin(-hdg0)
    rot2 = np.array([[c, -s], [s, c]])
    q_errs, leg_errs, arm_errs, xy_errs = [], [], [], []
    delay_s = delay_ms / 1000.0
    for k, t in enumerate(t_ctrl):
        idx = int(np.floor((t - 3.0 - delay_s) / 0.04 + 1e-7))
        if not (0 <= idx < len(rows_a0)) or not rows_a0[idx].get('accepted'):
            continue
        ref = rows_a0[idx]
        q_ref = np.asarray(ref['reference'][6:], float)
        q_act = qpos[k, 7:]
        diff = q_act - q_ref
        q_errs.append(diff)
        leg_errs.append(diff[:12])
        arm_errs.append(diff[15:])
        des_xy = rot2 @ (np.asarray(ref['root_position_m'][:2], float) - origin[:2])
        xy_errs.append(np.linalg.norm(qpos[k, :2] - des_xy))
    q_errs = np.asarray(q_errs, float)
    leg_errs = np.asarray(leg_errs, float)
    arm_errs = np.asarray(arm_errs, float)
    return dict(
        common_a0_ticks=int(len(q_errs)),
        common_a0_joint_rmse_deg=float(np.degrees(np.sqrt(np.mean(q_errs ** 2)))),
        common_a0_leg_rmse_deg=float(np.degrees(np.sqrt(np.mean(leg_errs ** 2)))),
        common_a0_arm_rmse_deg=float(np.degrees(np.sqrt(np.mean(arm_errs ** 2)))),
        common_a0_path_p95_m=float(np.percentile(xy_errs, 95)),
    )


def run_combined_fault(out, plant, cfg, traj_cfg, delay_ms=120):
    """Run closed-loop evaluation on s1_walk with combined dropouts (120ms arm + 320ms legs + 600ms body)."""
    comb_root = ROOT / 'outputs/continuous_prediction_20261020_final/combined/s1_walk'
    nat_a0_rows = read_jsonl(ROOT / 'outputs/foot_contact_baseline_20261019/accepted_baseline/s1_walk/baseline/humanoid_reference.jsonl')
    rows_a0 = read_jsonl(comb_root / 'A0/humanoid_reference.jsonl')
    rows_a2_gated = read_jsonl(comb_root / 'A2/humanoid_reference.jsonl')
    states_a2 = read_jsonl(comb_root / 'A2/states.jsonl')
    rows_v1 = retarget_continuous_pose(states_a2, cfg)
    rows_v3 = retarget_continuous_pose(states_a2, cfg, gimbal_barrier=True)
    write_jsonl(out / 'references' / 's1_walk_combined_A2_continuous.jsonl', rows_v1)
    write_jsonl(out / 'references' / 's1_walk_combined_A3_ik.jsonl', rows_v3)

    spec = {
        'A0_legacy': (rows_a0, build_legacy_reference_stream(rows_a0)),
        'A2_gated': (rows_a2_gated, build_c2_reference_stream(rows_a2_gated, plant, traj_cfg)),
        'A2_continuous': (rows_v1, build_c2_reference_stream(rows_v1, plant, traj_cfg)),
        'A3_ik': (rows_v3, build_c2_reference_stream(rows_v3, plant, traj_cfg)),
        'A3_balanced': (rows_v3, build_c2_reference_stream(rows_v3, plant, traj_cfg, joint_mode='per_dof_balanced')),
        'A3_stream': (rows_v1, build_c2_reference_stream(rows_v1, plant, traj_cfg, joint_mode='per_dof_rate')),
        'A3_full': (rows_v3, build_c2_reference_stream(rows_v3, plant, traj_cfg, joint_mode='per_dof_rate')),
    }
    comb_ref_sum = {}
    comb_results = {}
    for mode, (r_rows, stream) in spec.items():
        np.savez_compressed(out / 'references' / f's1_walk_combined_{mode}_stream.npz', **stream)
        comb_ref_sum[mode] = summarize_reference(r_rows, stream)
        for ctrl in CONTROLLERS:
            key = f'{mode}|s1_walk_combined|{ctrl}'
            trace_path = out / 'traces' / (key.replace('|', '_') + '.npz')
            t0 = time.perf_counter()
            trace, res = bc.rollout(ctrl, 's1_walk_combined', stream, delay_ms)
            res = augment_controller_metrics(trace, res, stream)
            res.update(common_a0_metrics(trace, stream, nat_a0_rows, delay_ms))
            res['reference_mode'] = mode
            res['wall_s'] = round(time.perf_counter() - t0, 2)
            np.savez_compressed(trace_path, **trace)
            comb_results[key] = res
            print(
                f'COMBINED {key:36s} rel_up={res["reference_relative_upright_passed"]!s:5} '
                f'stop={res["standing_stop_passed"]!s:5} q={res["joint_rmse_deg"]:5.2f} '
                f'com_q={res["common_a0_joint_rmse_deg"]:5.2f} com_leg={res["common_a0_leg_rmse_deg"]:5.2f} '
                f'com_arm={res["common_a0_arm_rmse_deg"]:5.2f} path95={res["path_error_p95_m"]:5.3f}',
                flush=True,
            )
    save_json(out / 'combined_fault_reference_continuity.json', comb_ref_sum)
    save_json(out / 'combined_fault_results.json', comb_results)
    return comb_ref_sum, comb_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/continuous_v3_20261023')
    parser.add_argument('--delay', type=int, default=120)
    parser.add_argument('--clips', nargs='+', default=list(CLIPS))
    parser.add_argument('--skip-rollout', action='store_true')
    parser.add_argument('--combined-only', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    (out / 'references').mkdir(parents=True, exist_ok=True)
    (out / 'traces').mkdir(exist_ok=True)
    plant, cfg, traj_cfg = TwistPlant(), load_pred_config(), load_recovery_config()
    if args.combined_only:
        run_combined_fault(out, plant, cfg, traj_cfg, args.delay)
        return
    pred_root = ROOT / 'outputs/continuous_prediction_20261020_final/natural'
    base_root = ROOT / 'outputs/foot_contact_baseline_20261019/accepted_baseline'
    streams, ref_sum, ik_sum, results = {}, {}, {}, {}
    for clip in args.clips:
        states = read_jsonl(pred_root / clip / 'A2/states.jsonl')
        rows_a0 = read_jsonl(base_root / clip / 'baseline/humanoid_reference.jsonl')
        rows_v1 = retarget_continuous_pose(states, cfg)
        rows_v3 = retarget_continuous_pose(states, cfg, gimbal_barrier=True)
        write_jsonl(out / 'references' / f'{clip}_A2_continuous.jsonl', rows_v1)
        write_jsonl(out / 'references' / f'{clip}_A3_ik.jsonl', rows_v3)
        spec = {
            'A0_legacy': (rows_a0, build_legacy_reference_stream(rows_a0)),
            'A2_continuous': (rows_v1, build_c2_reference_stream(rows_v1, plant, traj_cfg)),
            'A3_ik': (rows_v3, build_c2_reference_stream(rows_v3, plant, traj_cfg)),
            'A3_balanced': (rows_v3, build_c2_reference_stream(rows_v3, plant, traj_cfg, joint_mode='per_dof_balanced')),
            'A3_stream': (rows_v1, build_c2_reference_stream(rows_v1, plant, traj_cfg, joint_mode='per_dof_rate')),
            'A3_full': (rows_v3, build_c2_reference_stream(rows_v3, plant, traj_cfg, joint_mode='per_dof_rate')),
        }
        for mode, (rows, stream) in spec.items():
            streams[(mode, clip)] = stream
            np.savez_compressed(out / 'references' / f'{clip}_{mode}_stream.npz', **stream)
            ref_sum[f'{mode}|{clip}'] = summarize_reference(rows, stream)
            if mode != 'A0_legacy':
                ik_sum[f'{mode}|{clip}'] = ik_arc_metrics(rows)
            c = ref_sum[f'{mode}|{clip}']['control_50hz']
            print(f'{clip:12s} {mode:14s} max_jerk={c["max_jerk_rad_s3"]:8.1f} p95_jerk={c["p95_jerk_rad_s3"]:7.1f} '
                  f'maxv={c["max_velocity_rad_s"]:5.2f}', flush=True)
        save_json(out / 'reference_continuity.json', ref_sum)
        save_json(out / 'ik_branch.json', ik_sum)
    if args.skip_rollout:
        return
    for clip in args.clips:
        for mode in MODES:
            for ctrl in CONTROLLERS:
                key = f'{mode}|{clip}|{ctrl}'
                t0 = time.perf_counter()
                trace, res = bc.rollout(ctrl, clip, streams[(mode, clip)], args.delay)
                res = augment_controller_metrics(trace, res, streams[(mode, clip)])
                res['reference_mode'] = mode
                res['wall_s'] = round(time.perf_counter() - t0, 2)
                np.savez_compressed(out / 'traces' / key.replace('|', '_') , **trace)
                results[key] = res
                save_json(out / 'results.json', results)
                print(f'ROLLOUT {key:32s} bal={res["balance_envelope_passed"]!s:5} stop={res["standing_stop_passed"]!s:5} '
                      f'q={res["joint_rmse_deg"]:5.2f} leg={res["leg_rmse_deg"]:5.2f} path95={res["path_error_p95_m"]:5.3f}', flush=True)
    save_json(out / 'protocol.json', dict(
        modes=list(MODES), controllers=list(CONTROLLERS), delay_ms=args.delay, hardware_enabled=False,
        upstream_safety_filter_applied=False,
        code_sha256={str(p): sha256_file(p) for p in (
            Path(__file__), ROOT / 'motion_control/continuous_pose_reference.py', ROOT / 'motion_control/twist.py',
            ROOT / 'scripts/benchmark_controllers.py')}))


if __name__ == '__main__':
    main()

