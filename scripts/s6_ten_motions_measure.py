"""Measure the S6 acceptance quantities of tests/test_s6_closed_loop.py (same motions, same verifier settings).

Input : generate_10_diverse_human_motions (10 s, 100 Hz); retargeter `kpik` (default, causal keypoint IK) or `gmr`
        (offline SMPL-X + GMR reference, needs the GMR venv).
Output: outputs/advanced_take/s6_ten_motions_sonic_<retargeter>.json - per motion: survival, min pelvis z,
        max tilt, Stage 6A joint-limit violation rate, Stage 6B joint RMSE (rad), support-foot skating (m/s), torque
        finiteness, reference duration. These values are what the strict-xfail tables in the test record.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# single-threaded BLAS (see s6_keypoint_ik_eval.py); set before numpy loads, inherited by the workers
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "advanced_take"


def run(job):
    name, retargeter = job
    from g1_imitation.core.s6_mujoco_closed_loop import (
        MuJoCoClosedLoopVerifier, generate_10_diverse_human_motions, solve_gmr_trajectory_from_3d,
        solve_ik_trajectory_from_3d)
    pts = generate_10_diverse_human_motions(duration_s=10.0, fps_hz=100.0)[name]
    solve = solve_ik_trajectory_from_3d if retargeter == "kpik" else solve_gmr_trajectory_from_3d
    q, dq, info = solve(pts, dt=0.01)
    v = MuJoCoClosedLoopVerifier()
    r6a = v.run_stage_6a_kinematic(q, dt=0.01)
    r6b = v.run_stage_6b_physics(q, dq, ik_dt=0.01, control_dt=0.02, settle_steps=25)
    return name, dict(survived=bool(r6b.survived), min_pelvis_height_m=float(r6b.min_pelvis_height_m),
                      max_tilt_deg=float(r6b.max_tilt_deg), joint_limit_violation_rate_6a=float(r6a.joint_limit_violation_rate),
                      joint_rmse_rad=float(r6b.joint_rmse_rad), skating_ms=float(r6b.support_foot_skating_speed_ms),
                      torques_finite=bool(np.all(np.isfinite(r6b.torques_hist))), reference_s=float(len(q) * 0.01),
                      info=info)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retargeter", choices=("kpik", "gmr"), default="kpik")
    a = ap.parse_args()
    from g1_imitation.core.s6_mujoco_closed_loop import generate_10_diverse_human_motions
    names = sorted(generate_10_diverse_human_motions(duration_s=0.1, fps_hz=100.0))
    with ProcessPoolExecutor(max_workers=5) as ex:
        res = dict(ex.map(run, [(n, a.retargeter) for n in names]))
    f = OUT / f"s6_ten_motions_sonic_{a.retargeter}.json"
    f.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for n, r in res.items():
        print(f"{n:36s} surv {r['survived']!s:5s} rmse {r['joint_rmse_rad']:.3f} skate {r['skating_ms']:.3f} "
              f"tilt {r['max_tilt_deg']:4.1f} lim6a {r['joint_limit_violation_rate_6a']:.3f} ref {r['reference_s']:.2f}s")
    print("mean skate", np.mean([r["skating_ms"] for r in res.values()]), "->", f)


if __name__ == "__main__":
    main()
