"""Preview/delay study on real-human GMR references (AdvancedTake, official G1 plant from config).

Input : outputs/advanced_take/control/gmr_{bvh,smplx}_rgbonly_lag6/<Take>.npz (50 Hz q_ref (T,36), dq_ref (T,35);
        built by advanced_take_gmr_control.py). bvh = GMR on the mocap skeleton (scoring reference only, not a deployable
        input); smplx = GMR on the pure-RGB pipeline (S1-S3 + SMPL-X fit), the deployable path.
Output: outputs/advanced_take/preview_delay/real_takes.json - Stage 6B per source x take x controller@preview.
preview_s semantics as in s6_preview_delay_eval.py (= added human->robot delay; no future samples in deployment).
Path deviation is measured in each trajectory's own initial heading frame, phys vs the tracked reference sample.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

SRC = ROOT / "outputs" / "advanced_take" / "control"
OUT = ROOT / "outputs" / "advanced_take" / "preview_delay"
TAKES = ("Take_002", "Take_006", "Take_007", "Take_008", "Take_012")
GRID = {"sonic_v11": (0.0, 0.1, 0.2, 0.9), "sonic_low_latency": (0.0, 0.1, 0.18)}
DT = 0.02


def run(job: tuple[str, str]) -> tuple[str, dict]:
    from g1_imitation.core.constants import load_config
    from g1_imitation.core.s6_mujoco_closed_loop import MuJoCoClosedLoopVerifier, SonicPolicy
    from s6_official_clip_check import local_path, yaw
    src, take = job
    with np.load(SRC / f"gmr_{src}_rgbonly_lag6" / f"{take}.npz") as d:
        q, dq = d["q_ref"], d["dq_ref"]
    cfg = load_config()["s6_closed_loop"]
    ref_path = local_path(q[:, :2], yaw(q[0, 3:7]))
    res = {}
    for ctrl, previews in GRID.items():
        ver = MuJoCoClosedLoopVerifier(policy=SonicPolicy(cfg[ctrl]))
        for p in previews:
            r = ver.run_stage_6b_physics(q, dq, ik_dt=DT, control_dt=DT, settle_steps=25, preview_s=p)
            ph = local_path(r.qpos_hist[:, :2], yaw(r.qpos_hist[0, 3:7]))
            dev = np.linalg.norm(ph - ref_path[:len(ph)], axis=1)
            res[f"{ctrl}@{p:.2f}"] = dict(survived=r.survived, joint_rmse_rad=r.joint_rmse_rad,
                                          skate=r.support_foot_skating_speed_ms, max_tilt_deg=r.max_tilt_deg,
                                          path_dev_final_m=float(dev[-1]), path_dev_mean_m=float(dev.mean()))
    return f"{src}|{take}", res


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(s, t) for s in ("bvh", "smplx") for t in TAKES]
    with ProcessPoolExecutor(max_workers=7) as ex:
        report = dict(ex.map(run, jobs))
    (OUT / "real_takes.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    keys = list(next(iter(report.values())))
    for s in ("bvh", "smplx"):
        for k in keys:
            e = [report[f"{s}|{t}"][k] for t in TAKES]
            print(f"{s:5s} {k:24s} surv {sum(x['survived'] for x in e)}/{len(e)} rmse {np.mean([x['joint_rmse_rad'] for x in e]):.3f} "
                  f"skate {np.mean([x['skate'] for x in e]):.3f} dev_final {np.mean([x['path_dev_final_m'] for x in e]):.2f} "
                  f"dev_mean {np.mean([x['path_dev_mean_m'] for x in e]):.2f}")


if __name__ == "__main__":
    main()
