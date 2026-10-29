"""Closed-loop evaluation of the causal keypoint-IK retargeter (core/s45c_keypoint_ik.py) with SONIC (official G1 plant).

Inputs : synthetic - generate_10_diverse_human_motions + generate_jump_motions (100 Hz), sampled at 30 Hz camera slots;
         real      - outputs/advanced_take/s2s3/rgbonly_lag6/<Take>.npz `smooth` (S3 One-Euro, causal) or
                     `raw` (S2 only), 30 Hz slots at t = (grid + lag) / 120 s.
Causal path: each 30 Hz slot is retargeted when it arrives; a 50 Hz control tick uses the newest slot at or before it
         (`hold`) or extrapolates it with its own backward-difference velocity (`extrapolate`). No future slot is read.
         Perception compute latency (S1/S2) is not simulated here; retarget wall time is reported per slot.
Output : outputs/advanced_take/keypoint_ik/report.json, and for real takes
         outputs/advanced_take/control/kpik_<input>_rgbonly_lag6/<Take>.npz (same layout as the GMR control
         runs: t50, q_ref, dq_ref, fresh, qpos_phys, q_joints_phys, left_fz, right_fz; physics of `--save-ctrl`).
`--gmr-baseline`: re-runs the stored GMR (BVH / RGB SMPL-X) 50 Hz references on the current plant with `--save-ctrl`
         and writes control/gmr_<src>_rgbonly_lag6_<ctrl>/ so advanced_take_limb_direction_eval.py compares physics
         under identical conditions. (The GMR references themselves are offline / non-causal.)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

# Single-threaded BLAS before numpy loads (inherited by the worker processes): the IK matrices are tiny, and
# multi-threaded BLAS measured p50 12.6 ms vs 1.7 ms per frame (2026-10-29, 4170 frames), so solve_ms would be wrong.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = ROOT / "outputs" / "advanced_take"
TAKES = ("Take_002", "Take_006", "Take_007", "Take_008", "Take_012")
GRID = (("sonic_v11", 0.0), ("sonic_v11", 0.1), ("sonic_v11", 0.2), ("sonic_low_latency", 0.0), ("sonic_low_latency", 0.18))
DT = 0.02
SLOT_HZ = 30.0
BVH_DT = 1.0 / 120.0


def load_input(src: str, name: str):
    if src == "syn":
        from g1_imitation.core.s6_mujoco_closed_loop import generate_10_diverse_human_motions, generate_jump_motions
        from g1_imitation.core.s45_gmr_retarget import resample_points
        m = generate_10_diverse_human_motions(duration_s=10.0, fps_hz=100.0)
        m.update(generate_jump_motions(duration_s=10.0, fps_hz=100.0))
        t, pts = resample_points(m[name], 100.0, SLOT_HZ)
        return t, pts
    lag = int(json.loads((OUT / "s2s3" / "rgbonly_lag6" / "metrics.json").read_text(encoding="utf-8"))["ref_lag_bvh_samples"])
    with np.load(OUT / "s2s3" / "rgbonly_lag6" / f"{name}.npz") as z:
        return (z["grid"] + lag) * BVH_DT, z[src]


def run(job):
    src, name, mode, save_ctrl, override, tag = job
    from g1_imitation.core.constants import load_config
    from g1_imitation.core.s45c_keypoint_ik import KeypointIKRetargeter, causal_upsample, retarget_sequence
    from g1_imitation.core.s6_mujoco_closed_loop import MuJoCoClosedLoopVerifier, SonicPolicy
    from s6_official_clip_check import local_path, yaw
    from advanced_take_gmr_control import anchor_heading
    cfg = load_config()["s6_closed_loop"]
    t_slot, pts = load_input(src, name)
    from g1_imitation.core.constants import load_config as _lc
    rt = KeypointIKRetargeter(cfg={**_lc()["s45c_keypoint_ik"], **override})
    out = retarget_sequence(pts, t_slot, rt)
    t50, q, dq, fresh = causal_upsample(t_slot, out["q"], out["dq"], out["observed"], mode, rt.lo, rt.hi, dt=DT)
    q, dq, _ = anchor_heading(q, dq)
    ref_path = local_path(q[:, :2], yaw(q[0, 3:7]))
    res = dict(slots=int(len(pts)), observed_slots=int(out["observed"].sum()), ticks=int(len(q)),
               solve_ms_p50=float(np.percentile(out["solve_ms"][out["observed"]], 50)),
               solve_ms_p95=float(np.percentile(out["solve_ms"][out["observed"]], 95)),
               solve_ms_max=float(out["solve_ms"][out["observed"]].max()),
               held_limb_slots=int(sum(any(g not in ("blend",) for g in h) for h, o in zip(out["held"], out["observed"]) if o)),
               ref_travel_m=float(np.linalg.norm(ref_path[-1])), runs={})
    for ctrl, p in GRID:
        r = MuJoCoClosedLoopVerifier(policy=SonicPolicy(cfg[ctrl])).run_stage_6b_physics(
            q, dq, ik_dt=DT, control_dt=DT, settle_steps=25, preview_s=p)
        ph = local_path(r.qpos_hist[:, :2], yaw(r.qpos_hist[0, 3:7]))
        dev = np.linalg.norm(ph - ref_path[:len(ph)], axis=1)
        res["runs"][f"{ctrl}@{p:.2f}"] = dict(survived=r.survived, joint_rmse_rad=r.joint_rmse_rad,
                                              skate=r.support_foot_skating_speed_ms, max_tilt_deg=r.max_tilt_deg,
                                              path_dev_final_m=float(dev[-1]), path_dev_mean_m=float(dev.mean()),
                                              phys_end_m=ph[-1].tolist(), ref_end_m=ref_path[len(ph) - 1].tolist())
        if src != "syn" and save_ctrl == f"{ctrl}@{p:.2f}":
            odir = OUT / "control" / f"kpik{tag}_{src}_rgbonly_lag6"
            odir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(odir / f"{name}.npz", t50=t50, q_ref=q, dq_ref=dq, fresh=fresh, qpos_phys=r.qpos_hist,
                                q_joints_phys=r.q_joints_phys_hist, left_fz=r.left_fz_hist, right_fz=r.right_fz_hist)
    return f"{src}|{name}|{mode}", res


def gmr_baseline(job):
    src, take, ctrl_p = job
    from g1_imitation.core.constants import load_config
    from g1_imitation.core.s6_mujoco_closed_loop import MuJoCoClosedLoopVerifier, SonicPolicy
    ctrl, p = ctrl_p.split("@")
    with np.load(OUT / "control" / f"gmr_{src}_rgbonly_lag6" / f"{take}.npz") as z:
        d = {k: z[k] for k in z.files}
    r = MuJoCoClosedLoopVerifier(policy=SonicPolicy(load_config()["s6_closed_loop"][ctrl])).run_stage_6b_physics(
        d["q_ref"], d["dq_ref"], ik_dt=DT, control_dt=DT, settle_steps=25, preview_s=float(p))
    odir = OUT / "control" / f"gmr_{src}_rgbonly_lag6_{ctrl}_{p.replace('.', 'p')}"
    odir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(odir / f"{take}.npz", t50=d["t50"], q_ref=d["q_ref"], dq_ref=d["dq_ref"], fresh=d["fresh"],
                        qpos_phys=r.qpos_hist, q_joints_phys=r.q_joints_phys_hist, left_fz=r.left_fz_hist, right_fz=r.right_fz_hist)
    return f"{src}|{take}", r.survived


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["hold", "extrapolate"])
    ap.add_argument("--inputs", nargs="+", default=["syn", "smooth", "raw"])
    ap.add_argument("--save-ctrl", default="sonic_v11@0.20")
    ap.add_argument("--gmr-baseline", action="store_true")
    ap.add_argument("--override", default="{}", help="JSON overrides of the s45c_keypoint_ik config (ablations)")
    ap.add_argument("--tag", default="", help="suffix for report_<tag>.json and control/kpik_<tag>_* (ablations)")
    a = ap.parse_args()
    if a.gmr_baseline:
        with ProcessPoolExecutor(max_workers=7) as ex:
            print(dict(ex.map(gmr_baseline, [(s, t, a.save_ctrl) for s in ("bvh", "smplx") for t in TAKES])))
        return
    from g1_imitation.core.s6_mujoco_closed_loop import generate_10_diverse_human_motions, generate_jump_motions
    syn = sorted(generate_10_diverse_human_motions(duration_s=1.0, fps_hz=100.0)) + sorted(generate_jump_motions(duration_s=1.0, fps_hz=100.0))
    jobs = [(s, n, m, a.save_ctrl if m == a.modes[0] else "", json.loads(a.override), f"_{a.tag}" if a.tag else "") for s in a.inputs for n in (syn if s == "syn" else TAKES)
            for m in a.modes]
    (OUT / "keypoint_ik").mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=7) as ex:
        report = dict(ex.map(run, jobs))
    (OUT / "keypoint_ik" / (f"report_{a.tag}.json" if a.tag else "report.json")).write_text(json.dumps(report, indent=1), encoding="utf-8")
    for s in a.inputs:
        for m in a.modes:
            rows = [v for k, v in report.items() if k.startswith(f"{s}|") and k.endswith(f"|{m}")]
            print(f"== {s} {m}: solve ms p50 {np.median([r['solve_ms_p50'] for r in rows]):.2f} "
                  f"p95 {np.max([r['solve_ms_p95'] for r in rows]):.2f} max {np.max([r['solve_ms_max'] for r in rows]):.1f}")
            for g in rows[0]["runs"]:
                e = [r["runs"][g] for r in rows]
                print(f"   {g:24s} surv {sum(x['survived'] for x in e)}/{len(e)} rmse {np.mean([x['joint_rmse_rad'] for x in e]):.3f} "
                      f"skate {np.mean([x['skate'] for x in e]):.3f} dev_final {np.mean([x['path_dev_final_m'] for x in e]):.2f} "
                      f"dev_mean {np.mean([x['path_dev_mean_m'] for x in e]):.2f}")


if __name__ == "__main__":
    main()
