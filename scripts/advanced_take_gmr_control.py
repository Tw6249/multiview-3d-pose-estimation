"""GMR-retargeted G1 references (offline reference S4+S5) -> SONIC closed loop (S6) on AdvancedTake.

Inputs : outputs/advanced_take/gmr/<run>/<Take>.npz (qpos (T,36) at the 30 Hz video slots, t_s, observed,
         joint_names) from advanced_take_gmr_retarget.py; `--ref-run` is the BVH->GMR reference run.
Resample: 30 Hz slots -> 50 Hz control ticks. Joints/root position: linear; root orientation: slerp. A tick whose
         bracketing slots are not both observed holds the last observed target (counted, never invented).
         Joint velocities: central finite differences of the 50 Hz joint reference (rad/s).
Frame  : the reference root is re-expressed in its first heading/ground position (anchor_heading); joints untouched.
Output : outputs/advanced_take/control/<run>/{<Take>.npz, metrics.json}. Physics parameters, controller and
         survival rule are the same for every run; nothing here depends on the source.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from g1_imitation.core.constants import G1_29_JOINTS  # noqa: E402
from g1_imitation.core.s6_mujoco_closed_loop import MuJoCoClosedLoopVerifier  # noqa: E402

OUT = ROOT / "outputs" / "advanced_take"
DT = 0.02
GROUPS = dict(left_leg=range(0, 6), right_leg=range(6, 12), waist=range(12, 15),
              left_arm=range(15, 22), right_arm=range(22, 29))


def anchor_heading(q, dq):
    """Re-express the reference root trajectory in the frame of its first sample's heading and ground position.

    The robot is spawned at yaw 0 and the origin; the perception world yaw/origin is a convention, so without this
    the settle phase would have to turn the robot by that arbitrary angle. Joint angles are untouched.
    MuJoCo free-joint qvel: linear part in world axes (rotated here), angular part in body axes (unchanged).
    """
    yaw0 = Rotation.from_quat(q[0, [4, 5, 6, 3]]).as_euler("zyx")[0]
    Rz = Rotation.from_euler("z", -yaw0)
    q, dq = q.copy(), dq.copy()
    q[:, :2] -= q[0, :2]
    q[:, :3] = Rz.apply(q[:, :3])
    q[:, [4, 5, 6, 3]] = (Rz * Rotation.from_quat(q[:, [4, 5, 6, 3]])).as_quat()
    dq[:, :3] = Rz.apply(dq[:, :3])
    return q, dq, float(np.degrees(yaw0))


def group_rmse_deg(a, b):
    n = min(len(a), len(b))
    d = a[:n] - b[:n]
    return {g: float(np.degrees(np.sqrt(np.mean(d[:, list(r)] ** 2)))) for g, r in GROUPS.items()} | \
           {"all": float(np.degrees(np.sqrt(np.mean(d ** 2))))}


def yaw_of(q):
    return np.unwrap(Rotation.from_quat(q[:, [4, 5, 6, 3]]).as_euler("zyx")[:, 0])


def to_50hz(t_s, qpos, observed):
    """Returns q (T50,36), fresh (T50,), held (count)."""
    ok = observed & np.isfinite(qpos).all(1)
    t, q = t_s[ok], qpos[ok]
    t50 = np.arange(t[0], t[-1], DT)
    lo = np.clip(np.searchsorted(t, t50, side="right") - 1, 0, len(t) - 2)
    w = ((t50 - t[lo]) / (t[lo + 1] - t[lo]))[:, None]
    out = np.empty((len(t50), 36))
    out[:, :3] = q[lo, :3] + w * (q[lo + 1, :3] - q[lo, :3])
    out[:, 7:] = q[lo, 7:] + w * (q[lo + 1, 7:] - q[lo, 7:])
    rots = Rotation.from_quat(q[:, [4, 5, 6, 3]])
    out[:, [4, 5, 6, 3]] = Slerp(t, rots)(t50).as_quat()
    # a tick is fresh only when both bracketing samples are adjacent observed slots (<= 1 slot apart)
    slot_dt = np.median(np.diff(t_s))
    fresh = (t[lo + 1] - t[lo]) <= 1.5 * slot_dt
    held = 0
    for i in range(1, len(out)):
        if not fresh[i]:
            out[i] = out[i - 1]
            held += 1
    return t50, out, fresh, held


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["bvh_rgbonly_lag6", "smplx_rgbonly_lag6"])
    ap.add_argument("--ref-run", default="bvh_rgbonly_lag6")
    ap.add_argument("--takes", nargs="+", default=["Take_002", "Take_006", "Take_007", "Take_008", "Take_012"])
    args = ap.parse_args()
    verifier = MuJoCoClosedLoopVerifier()
    for run in args.runs:
        odir = OUT / "control" / f"gmr_{run}"
        odir.mkdir(parents=True, exist_ok=True)
        metrics = dict(source_run=run, retargeter="GMR (pinned, shipped defaults)", controller="SONIC v1.1 (S6)",
                       dt_s=DT, survival_rule="pelvis z > 0.45 m and tilt < 45 deg (S6 config)", takes={})
        for take in args.takes:
            g = np.load(OUT / "gmr" / run / f"{take}.npz")
            names = [str(n) for n in g["joint_names"]]
            perm = np.array([names.index(n) for n in G1_29_JOINTS])          # GMR order -> SDK order (by name)
            qpos = g["qpos"].copy()
            qpos[:, 7:] = qpos[:, 7:][:, perm]
            t50, q, fresh, held = to_50hz(g["t_s"], qpos, np.asarray(g["observed"], bool))
            dq = np.zeros((len(q), 35))
            dq[:, 6:] = np.gradient(q[:, 7:], DT, axis=0)
            q, dq, yaw0 = anchor_heading(q, dq)
            t0 = time.time()
            a6 = verifier.run_stage_6a_kinematic(q, dt=DT)
            b6 = verifier.run_stage_6b_physics(q, dq, ik_dt=DT, control_dt=DT)
            n = min(len(q), len(b6.qpos_hist))
            ref_xy, ph_xy = q[:n, :2] - q[0, :2], b6.qpos_hist[:n, :2] - b6.qpos_hist[0, :2]
            yr, yp = yaw_of(q[:n]), yaw_of(b6.qpos_hist[:n])
            res = dict(slots_observed=int(np.asarray(g["observed"]).sum()), slots=int(len(g["observed"])),
                       ticks=int(len(q)), held_ticks=int(held), world_yaw0_deg_removed=yaw0,
                       stage6a=dict(min_pelvis_m=a6.min_pelvis_height_m, max_tilt_deg=a6.max_tilt_deg,
                                    min_foot_z_m=a6.min_foot_z_m, support_foot_skating_ms=a6.support_foot_skating_speed_ms,
                                    joint_limit_violation_rate=a6.joint_limit_violation_rate),
                       stage6b=dict(survived=bool(b6.survived), min_pelvis_m=b6.min_pelvis_height_m,
                                    max_tilt_deg=b6.max_tilt_deg, joint_rmse_deg_vs_own_ref=b6.joint_rmse_deg,
                                    support_foot_skating_ms=b6.support_foot_skating_speed_ms,
                                    max_torque_nm=b6.max_torque_nm),
                       path_err_mean_m=float(np.linalg.norm(ref_xy - ph_xy, axis=1).mean()),
                       ref_path_len_m=float(np.sum(np.linalg.norm(np.diff(ref_xy, axis=0), axis=1))),
                       phys_path_len_m=float(np.sum(np.linalg.norm(np.diff(ph_xy, axis=0), axis=1))),
                       yaw_change_ref_deg=float(np.degrees(yr[-1] - yr[0])),
                       yaw_change_phys_deg=float(np.degrees(yp[-1] - yp[0])), wall_s=float(time.time() - t0))
            np.savez_compressed(odir / f"{take}.npz", t50=t50, q_ref=q, dq_ref=dq, fresh=fresh,
                                qpos_phys=b6.qpos_hist, q_joints_phys=b6.q_joints_phys_hist,
                                left_fz=b6.left_fz_hist, right_fz=b6.right_fz_hist)
            metrics["takes"][take] = res
            print(run, take, "survived", b6.survived, "tilt %.1f own-rmse %.1f deg held %d" % (
                b6.max_tilt_deg, b6.joint_rmse_deg, held), flush=True)
        (odir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    # cross-run comparison against the BVH->GMR reference (joint angles on the common time axis)
    for run in args.runs:
        if run == args.ref_run:
            continue
        mpath = OUT / "control" / f"gmr_{run}" / "metrics.json"
        m = json.loads(mpath.read_text(encoding="utf-8"))
        for take in args.takes:
            a = np.load(OUT / "control" / f"gmr_{run}" / f"{take}.npz")
            r = np.load(OUT / "control" / f"gmr_{args.ref_run}" / f"{take}.npz")
            lo, hi = max(a["t50"][0], r["t50"][0]), min(a["t50"][-1], r["t50"][-1])
            ta = (a["t50"] >= lo) & (a["t50"] <= hi)
            ir = np.clip(np.searchsorted(r["t50"], a["t50"][ta]), 0, len(r["t50"]) - 1)
            ref_q = r["q_ref"][ir, 7:]
            m["takes"][take]["ref_divergence_deg_vs_bvh_gmr"] = group_rmse_deg(a["q_ref"][ta, 7:], ref_q)
            nph = min(int(ta.sum()), len(a["q_joints_phys"]))
            m["takes"][take]["phys_vs_bvh_gmr_reference_deg"] = group_rmse_deg(
                a["q_joints_phys"][np.flatnonzero(ta)[:nph]], ref_q[:nph])
        mpath.write_text(json.dumps(m, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
