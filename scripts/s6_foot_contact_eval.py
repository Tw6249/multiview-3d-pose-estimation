"""Synthetic S6 foot-contact study: GMR reference vs GMR + contact-aware foot correction, both closed-loop with SONIC.

Steps (each cached under outputs/advanced_take/foot_contact/):
  gmr      : retarget every synthetic motion with GMR once (30 Hz result + 100 Hz q/dq) -> gmr_<motion>.npz
  diagnose : reference-side skating - horizontal speed of the reference ankle while the synthetic human foot is planted
  closed   : SONIC closed loop on the GMR reference and on the corrected reference -> closed_loop.json
Human contact labels used for *diagnosis/scoring* come from the synthetic generator geometry; the correction layer
itself only sees the 27-point track (it never reads these labels).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1_imitation.core.s45_gmr_retarget import (  # noqa: E402
    finite_difference_velocity, resample_points, resample_q, GMRRetargeter)
from g1_imitation.core.s6_mujoco_closed_loop import (  # noqa: E402
    MuJoCoClosedLoopVerifier, generate_10_diverse_human_motions)

OUT = ROOT / "outputs" / "advanced_take" / "foot_contact"
DT = 0.01
FEET = {"left": (15, 17, 18, 19), "right": (16, 20, 21, 22)}


def load_motions(names: list[str] | None) -> dict[str, np.ndarray]:
    from g1_imitation.core.s6_mujoco_closed_loop import generate_jump_motions
    m = generate_10_diverse_human_motions(duration_s=10.0, fps_hz=100.0)
    m.update(generate_jump_motions(duration_s=10.0, fps_hz=100.0))
    return {k: v for k, v in m.items() if not names or k in names}


def gmr_cached(name: str, pts: np.ndarray) -> dict:
    f = OUT / f"gmr_{name}.npz"
    if f.exists():
        with np.load(f) as d:
            return {k: d[k] for k in d.files}
    t_r, pts_r = resample_points(pts, 1.0 / DT, 30.0)
    t0 = time.perf_counter()
    res = GMRRetargeter().retarget(pts_r, 30.0)
    t_out = np.arange(0.0, t_r[-1] + 1e-9, DT)
    q = resample_q(t_r, res.q, t_out)
    out = dict(q30=res.q, t30=t_r, q=q, dq=finite_difference_velocity(q, DT), human_height_m=res.human_height_m,
               fit_residual_mm_mean=float(np.nanmean(res.fit_residual_mm)), wall_s=time.perf_counter() - t0)
    np.savez(f, **out)
    return out


def human_planted(pts: np.ndarray, side: str, h=0.01, v=0.05) -> np.ndarray:
    """Generator-geometry label (scoring only): all foot ground points within h of z=0 and slower than v."""
    g = pts[:, FEET[side][1:]]
    low = (g[..., 2] < h).all(-1)
    sp = np.linalg.norm(np.gradient(g[..., :2].mean(1), DT, axis=0), axis=-1)
    return low & (sp < v)


def ref_feet(q: np.ndarray, model: mujoco.MjModel) -> dict[str, np.ndarray]:
    d = mujoco.MjData(model)
    ids = {s: model.body(f"{s}_ankle_roll_link").id for s in FEET}
    out = {s: np.zeros((len(q), 3)) for s in FEET}
    for i, qi in enumerate(q):
        d.qpos[:] = qi
        mujoco.mj_kinematics(model, d)
        for s in FEET:
            out[s][i] = d.xpos[ids[s]]
    return out


def ref_skating(q: np.ndarray, pts: np.ndarray, model) -> dict:
    n = min(len(q), len(pts))
    feet = ref_feet(q[:n], model)
    res = {}
    for s in FEET:
        planted = human_planted(pts[:n], s)
        sp = np.linalg.norm(np.gradient(feet[s][:, :2], DT, axis=0), axis=-1)
        res[s] = dict(planted_frac=float(planted.mean()),
                      speed_planted_mean=float(sp[planted].mean()) if planted.any() else float("nan"),
                      drift_planted_p95=float(np.percentile(sp[planted], 95)) if planted.any() else float("nan"),
                      min_z=float(feet[s][:, 2].min()), z_planted_std=float(feet[s][planted, 2].std()) if planted.any() else float("nan"))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", choices=["gmr", "diagnose", "closed"])
    ap.add_argument("--motions", nargs="*")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    motions = load_motions(a.motions)
    ver = MuJoCoClosedLoopVerifier()
    model = ver.plant.kin_model
    report = {}
    for name, pts in motions.items():
        g = gmr_cached(name, pts)
        r = report.setdefault(name, {})
        if "diagnose" in a.steps or "closed" in a.steps:
            from g1_imitation.core.s45b_foot_contact import FootContactCorrector
            t0 = time.perf_counter()
            corr = FootContactCorrector(model).correct(g["q"], pts, DT)
            r["correct_s"] = time.perf_counter() - t0
            r["contact_frac"] = {s: float((corr.contact[:, i] == 1).mean()) for i, s in enumerate(FEET)}
            r["unknown_frac"] = {s: float((corr.contact[:, i] == -1).mean()) for i, s in enumerate(FEET)}
            r["label_agreement"] = {s: float(((corr.contact[:, i] == 1) == human_planted(pts, s)[:len(corr.contact)]).mean())
                                    for i, s in enumerate(FEET)}
            r["joint_change_rms_rad"] = float(np.sqrt(np.mean((corr.q[:, 7:] - g["q"][:, 7:]) ** 2)))
            r["joint_change_max_rad"] = float(np.abs(corr.q[:, 7:] - g["q"][:, 7:]).max())
            r["root_change_max_m"] = float(np.abs(corr.q[:, :3] - g["q"][:, :3]).max())
            r["ref_gmr"] = ref_skating(g["q"], pts, model)
            r["ref_corrected"] = ref_skating(corr.q, pts, model)
        if "closed" in a.steps:
            for key, (q, dq) in {"gmr": (g["q"], g["dq"]),
                                 "corrected": (corr.q, finite_difference_velocity(corr.q, DT))}.items():
                rep = ver.run_stage_6b_physics(q, dq, ik_dt=DT, control_dt=0.02, settle_steps=25)
                r6a = ver.run_stage_6a_kinematic(q, dt=DT)
                r[f"closed_{key}"] = dict(survived=rep.survived, joint_rmse_rad=rep.joint_rmse_rad,
                                          joint_rmse_vs_gmr_rad=float(np.sqrt(np.mean(
                                              (rep.q_joints_phys_hist - g["q"][::2][:len(rep.q_joints_phys_hist), 7:]) ** 2))),
                                          skate=rep.support_foot_skating_speed_ms, max_tilt_deg=rep.max_tilt_deg,
                                          min_pelvis_z=rep.min_pelvis_height_m, limit_viol=r6a.joint_limit_violation_rate,
                                          kin_min_foot_z=r6a.min_foot_z_m)
        print(name, json.dumps(r, default=float)[:2000], flush=True)
    f = OUT / f"report{a.tag}.json"
    f.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print("wrote", f)


if __name__ == "__main__":
    main()
