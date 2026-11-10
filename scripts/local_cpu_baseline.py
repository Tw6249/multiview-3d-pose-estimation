"""Reproduce the G1 CPU back-half baseline, with unchanged physics and thresholds.

Input: repository synthetic 27-point motions (m, Z-up, 100 Hz); optional existing
50 Hz deployment CSV (SDK joint order, rad, wxyz). Synthetic poses test IK and
physics only; the CSV tests control only, never fresh visual reconstruction.
Output: report.json and per-motion reference/physics NPZ in an ignored directory.
Failure: exceptions propagate; failed acceptance is reported and exits nonzero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from g1_imitation.core.constants import load_config
from g1_imitation.core.process_scheduling import disable_power_throttling
from g1_imitation.core.s6_mujoco_closed_loop import (
    MuJoCoClosedLoopVerifier, generate_10_diverse_human_motions, solve_ik_trajectory_from_3d,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/local_cpu/baseline")
    ap.add_argument("--reference-csv", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    s6 = cfg["s6_closed_loop"]
    report = dict(simulation_only=True, perception_executed=False,
                  python=sys.version, power_throttling_disabled=disable_power_throttling(),
                  config_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                  physics_dt_s=s6["physics_dt_s"], control_dt_s=s6["control_dt_s"],
                  policy=s6["default_policy"], motions={})
    verifier = MuJoCoClosedLoopVerifier()
    for name, pts in generate_10_diverse_human_motions(duration_s=10.0, fps_hz=100.0).items():
        start = time.perf_counter()
        q, dq, info = solve_ik_trajectory_from_3d(pts, dt=0.01)
        ik_wall = time.perf_counter() - start
        kin = verifier.run_stage_6a_kinematic(q, dt=0.01)
        start = time.perf_counter()
        phys = verifier.run_stage_6b_physics(q, dq, ik_dt=0.01, control_dt=0.02, settle_steps=25)
        row = dict(survived=bool(phys.survived), joint_rmse_rad=float(phys.joint_rmse_rad),
                   support_foot_skating_m_s=float(phys.support_foot_skating_speed_ms),
                   min_pelvis_height_m=float(phys.min_pelvis_height_m), max_tilt_deg=float(phys.max_tilt_deg),
                   joint_limit_violation_rate=float(kin.joint_limit_violation_rate),
                   torques_finite=bool(np.isfinite(phys.torques_hist).all()),
                   physics_wall_s=time.perf_counter()-start, ik_wall_s=ik_wall,
                   physics_duration_s=phys.num_policy_steps * 0.02,
                   physical_root_displacement_m=float(np.linalg.norm(phys.qpos_hist[-1,:2]-phys.qpos_hist[0,:2])),
                   info=info)
        row["acceptance_pass"] = bool(row["survived"] and row["torques_finite"]
            and row["joint_limit_violation_rate"] == 0
            and row["joint_rmse_rad"] < s6["max_joint_rmse_rad"]
            and row["support_foot_skating_m_s"] < s6["max_foot_skating_m_s"])
        np.savez_compressed(args.out / f"{name}.npz", q_ref=q, dq_ref=dq,
                            qpos_phys=phys.qpos_hist, torques=phys.torques_hist)
        report["motions"][name] = row
        (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"{name}: pass={row['acceptance_pass']} rmse={row['joint_rmse_rad']:.4f} skate={row['support_foot_skating_m_s']:.4f}", flush=True)
    if args.reference_csv:
        table = np.genfromtxt(args.reference_csv, delimiter=",", names=True, encoding="utf-8-sig")
        from g1_imitation.core.constants import G1_29_JOINTS
        qnames = [f"root_pos_{a}_m" for a in "xyz"] + [f"root_quat_{a}" for a in "wxyz"] + [f"pos_{j}" for j in G1_29_JOINTS]
        dnames = [f"root_lin_vel_{a}_ms" for a in "xyz"] + [f"root_ang_vel_{a}_rads" for a in "xyz"] + [f"vel_{j}" for j in G1_29_JOINTS]
        q, dq = (np.column_stack([table[n] for n in ns]) for ns in (qnames, dnames))
        if not np.allclose(np.diff(table["t_rel_s"]), 0.02):
            raise ValueError("Reference CSV must be 50 Hz")
        phys = verifier.run_stage_6b_physics(q, dq, ik_dt=0.02, control_dt=0.02, settle_steps=25)
        report["existing_reference_control_only"] = dict(source=str(args.reference_csv),
            sha256=hashlib.sha256(args.reference_csv.read_bytes()).hexdigest(),
            survived=bool(phys.survived), joint_rmse_rad=float(phys.joint_rmse_rad),
            max_tilt_deg=float(phys.max_tilt_deg), support_foot_skating_m_s=float(phys.support_foot_skating_speed_ms))
        np.savez_compressed(args.out / "existing_reference_control_only.npz", q_ref=q, dq_ref=dq,
                            qpos_phys=phys.qpos_hist, torques=phys.torques_hist)
    report["all_synthetic_acceptance_pass"] = all(r["acceptance_pass"] for r in report["motions"].values())
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["all_synthetic_acceptance_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
