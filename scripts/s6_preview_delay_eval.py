"""Synthetic S6 study: how much does an added human->robot delay (= SONIC reference preview) buy?

For each cached GMR reference (outputs/advanced_take/foot_contact/gmr_<motion>.npz, built by
s6_foot_contact_eval.py) and each controller/preview pair, run Stage 6B. preview_s = D means the robot tracks the
sample received D seconds ago and window slots at offsets <= D are real samples (no future leakage in a causal
deployment); later slots repeat the newest one. Joint RMSE is scored against the tracked sample (delay excluded).
Output: outputs/advanced_take/preview_delay/report.json
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SRC = ROOT / "outputs" / "advanced_take" / "foot_contact"
OUT = ROOT / "outputs" / "advanced_take" / "preview_delay"
GRID = {"sonic_v11": (0.0, 0.1, 0.2, 0.3, 0.5, 0.9), "sonic_low_latency": (0.0, 0.06, 0.1, 0.18)}
LOWER = list(range(12))


def run_motion(path: str) -> tuple[str, dict]:
    from g1_imitation.core.constants import load_config
    from g1_imitation.core.s6_mujoco_closed_loop import MuJoCoClosedLoopVerifier, SonicPolicy
    name = Path(path).stem[4:]
    with np.load(path) as d:
        q, dq = d["q"], d["dq"]
    cfg = load_config()["s6_closed_loop"]
    res = {}
    for ctrl, previews in GRID.items():
        ver = MuJoCoClosedLoopVerifier(policy=SonicPolicy(cfg[ctrl]))
        for p in previews:
            r = ver.run_stage_6b_physics(q, dq, ik_dt=0.01, control_dt=0.02, settle_steps=25, preview_s=p)
            ref = r.q_joints_ref_hist
            on_l, on_r = r.left_fz_hist > 20.0, r.right_fz_hist > 20.0
            res[f"{ctrl}@{p:.2f}"] = dict(
                survived=r.survived, joint_rmse_rad=r.joint_rmse_rad,
                lower_rmse_rad=float(np.sqrt(np.mean((r.q_joints_phys_hist[:, LOWER] - ref[:, LOWER]) ** 2))),
                skate=r.support_foot_skating_speed_ms, max_tilt_deg=r.max_tilt_deg,
                travel_phys_m=(r.qpos_hist[-1, :2] - r.qpos_hist[0, :2]).tolist(),
                travel_ref_m=(q[-1, :2] - q[0, :2]).tolist(),
                double_support=float((on_l & on_r).mean()), airborne=float(((r.left_fz_hist < 1) & (r.right_fz_hist < 1)).mean()),
                pelvis_z_range_phys=float(np.ptp(r.qpos_hist[:, 2])), pelvis_z_range_ref=float(np.ptp(q[:, 2])))
    return name, res


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(str(f) for f in SRC.glob("gmr_*.npz"))
    if len(files) != 13:
        raise SystemExit(f"expected 13 cached GMR references in {SRC}, found {len(files)}")
    with ProcessPoolExecutor(max_workers=7) as ex:
        report = dict(sorted(ex.map(run_motion, files)))
    (OUT / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote", OUT / "report.json")


if __name__ == "__main__":
    main()
