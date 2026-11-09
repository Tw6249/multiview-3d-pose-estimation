"""Reproducible 2026-11-07 ablation: baseline, relationship, support, temporal, combined.

All runs use saved four-view RGB 3-D tracks, associated covariance, BVH only for
offline scoring, and the identical official G1 + SONIC v1.1 physics settings.
Outputs are local under ignored outputs/retarget_ablation_20261107/.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "retarget_ablation_20261107"
EXPERIMENT = json.loads((ROOT / "g1_imitation" / "config" / "retarget_ablation_20261107.json").read_text(encoding="utf-8"))
TAKES = EXPERIMENT["takes"]
VARIANTS = EXPERIMENT["variants"]
DT = 0.02


def configure_runtime():
    from g1_imitation.core import constants

    cfg = constants.load_config(constants.PACKAGE_ROOT / "config" / "pipeline_config.json")
    # MuJoCo 3.3.7 Windows rejects this workspace's non-ASCII absolute path.
    # The cache contains byte-identical copied assets, checked in the run manifest.
    asset_root = Path.home() / ".cache" / "g1-retarget-20261107" / "assets-copy"
    source_root = ROOT / "data/deployment_bundle/assets/robot-assets"
    for folder in ("sonic_official_g1", "controller-comparison/sonic_v11", "controller-comparison/sonic_low_latency", "leju_roban_s17"):
        if not (asset_root / folder).exists():
            shutil.copytree(source_root / folder, asset_root / folder)
    cfg["runtime"]["robot_assets_root"] = str(asset_root)
    cfg["runtime"]["python"] = str(OUT / "runtime" / "Scripts" / "python.exe")
    runtime_cfg = OUT / "runtime_config.json"
    OUT.mkdir(parents=True, exist_ok=True)
    runtime_cfg.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    constants.CONFIG_PATH = runtime_cfg
    return cfg, asset_root


def evaluate_variant(variant, points, cov, timestamps, model, asset_root, anchor=None):
    from g1_imitation.core.constants import load_config
    from g1_imitation.core.retarget_experiments import (
        ExperimentalRetargeter, SkeletonTemporalFilter, constrain_joint_trajectory,
    )
    from g1_imitation.core.s45c_keypoint_ik import KeypointIKRetargeter, causal_upsample
    from g1_imitation.core.s6_mujoco_closed_loop import (
        MuJoCoClosedLoopVerifier, SonicPolicy,
    )

    temporal_on = variant in ("temporal", "combined")
    relational_on = variant in ("relations", "combined")
    support_on = variant in ("support", "combined")
    rt_cfg = load_config()["s45c_keypoint_ik"]
    rt = (ExperimentalRetargeter(EXPERIMENT, relations=relational_on, support=support_on,
                                 cfg=rt_cfg, model=model)
          if relational_on or support_on else KeypointIKRetargeter(cfg=rt_cfg, model=model))
    temporal = SkeletonTemporalFilter(EXPERIMENT["temporal"]) if temporal_on else None
    rows, contacts = [], []
    for t, p, c in zip(timestamps, points, cov):
        if temporal is not None:
            p = temporal.step(p, float(t), c)
        result = rt.step(p, float(t))
        rows.append(result)
        contacts.append(getattr(rt, "contact", np.full(2, -1, np.int8)).copy())
    q_slots = np.asarray([x.q for x in rows])
    dq_slots = np.asarray([x.dq for x in rows])
    observed = np.asarray([x.observed for x in rows])
    t50, q, dq, fresh = causal_upsample(timestamps, q_slots, dq_slots, observed, "hold", rt.lo, rt.hi, dt=DT)
    if not fresh.all():
        raise RuntimeError("Post-calibration whole-body observation missing; reject this experiment trajectory")
    limited_fraction = 0.0
    if temporal_on:
        q, changed = constrain_joint_trajectory(
            q, t50, rt.lo, rt.hi,
            EXPERIMENT["temporal"]["joint_velocity_rad_s"],
            EXPERIMENT["temporal"]["joint_acceleration_rad_s2"],
        )
        limited_fraction = float(np.mean(changed))
        dq[1:, 6:] = np.diff(q[:, 7:], axis=0) / np.diff(t50)[:, None]
    from scipy.spatial.transform import Rotation
    if anchor is None:
        anchor = (q[0, :2].copy(), Rotation.from_quat(q[0, [4, 5, 6, 3]]).as_euler("zyx")[0])
    origin_xy, yaw0 = anchor
    rot = Rotation.from_euler("z", -yaw0)
    q[:, :2] -= origin_xy
    q[:, :3] = rot.apply(q[:, :3])
    q[:, [4, 5, 6, 3]] = (rot * Rotation.from_quat(q[:, [4, 5, 6, 3]])).as_quat()
    dq[:, :3] = rot.apply(dq[:, :3])
    policy = SonicPolicy(load_config()["s6_closed_loop"][EXPERIMENT["policy"]])
    verifier = MuJoCoClosedLoopVerifier(
        mjcf_path=asset_root / "sonic_official_g1" / "scene_43dof.xml",
        policy=policy, physics_dt=load_config()["s6_closed_loop"]["physics_dt_s"],
        effort_limit="actuator_range",
    )
    physical = verifier.run_stage_6b_physics(q, dq, ik_dt=DT, control_dt=DT,
                                              settle_steps=25, preview_s=EXPERIMENT["preview_s"])
    return {"t50": t50, "q_ref": q, "qpos_phys": physical.qpos_hist,
            "contacts": np.asarray(contacts), "fresh": fresh, "q_slots": q_slots,
            "observed": observed,
            "physical": physical, "limited_fraction": limited_fraction, "anchor": anchor,
            "solve_p50_ms": float(np.median([x.solve_ms for x in rows if x.observed]))}


def score_fk(plant, run, human, slot_times, eval_start):
    q = run["q_ref"]
    q_phys = run["qpos_phys"]
    tq = run["t50"]
    idx = np.clip(np.searchsorted(tq, slot_times), 0, len(tq) - 1)
    human = human[:len(slot_times)]
    keep = np.isfinite(human).all(axis=(1, 2)) & (slot_times >= eval_start) & (slot_times >= tq[0]) & (slot_times <= tq[-1])
    idx, g = idx[keep], human[keep]
    if not len(idx):
        return {"scored_frames": 0}
    d = mujoco.MjData(plant.kin_model)
    rb, rd = [], []
    for x in q[idx]:
        d.qpos[:] = x
        mujoco.mj_kinematics(plant.kin_model, d)
        rb.append({n: d.xpos[plant.kin_model.body(n).id].copy() for pair in (
            ("left_shoulder_pitch_link", "left_elbow_link"), ("right_shoulder_pitch_link", "right_elbow_link"),
            ("left_elbow_link", "left_wrist_roll_link"), ("right_elbow_link", "right_wrist_roll_link"),
            ("left_hip_pitch_link", "left_knee_link"), ("right_hip_pitch_link", "right_knee_link"),
            ("left_knee_link", "left_ankle_pitch_link"), ("right_knee_link", "right_ankle_pitch_link")) for n in pair})
    for x in q_phys[idx]:
        d.qpos[:] = x
        mujoco.mj_kinematics(plant.kin_model, d)
        rd.append({n: d.xpos[plant.kin_model.body(n).id].copy() for n in (
            "left_shoulder_pitch_link", "right_shoulder_pitch_link", "left_elbow_link", "right_elbow_link",
            "left_wrist_roll_link", "right_wrist_roll_link", "left_hip_pitch_link", "right_hip_pitch_link",
            "left_knee_link", "right_knee_link", "left_ankle_pitch_link", "right_ankle_pitch_link")})
    segments = (("arms", (("left_shoulder_pitch_link", "left_elbow_link", 0, 2),
                            ("right_shoulder_pitch_link", "right_elbow_link", 1, 3),
                            ("left_elbow_link", "left_wrist_roll_link", 2, 4),
                            ("right_elbow_link", "right_wrist_roll_link", 3, 5))),
                ("legs", (("left_hip_pitch_link", "left_knee_link", 6, 8),
                            ("right_hip_pitch_link", "right_knee_link", 7, 9),
                            ("left_knee_link", "left_ankle_pitch_link", 8, 10),
                            ("right_knee_link", "right_ankle_pitch_link", 9, 11))))
    human_lat = g[:, 6] - g[:, 7]
    yaw = np.arctan2(human_lat[:, 1], human_lat[:, 0]) - np.pi / 2
    c, s = np.cos(-yaw), np.sin(-yaw)
    R = np.zeros((len(g), 3, 3))
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1], R[:, 2, 2] = c, -s, s, c, 1
    scores = {"scored_frames": int(keep.sum())}
    for name, pairs in segments:
        for label, arr in (("reference", rb), ("physics", rd)):
            errors = []
            for a, b, ja, jb in pairs:
                hv = g[:, jb] - g[:, ja]
                rv = np.array([x[b] - x[a] for x in arr])
                hv = np.einsum("nij,nj->ni", R, hv)
                # Compare like limb directions in the human pelvis-heading frame.
                base = np.array([x["left_hip_pitch_link"] - x["right_hip_pitch_link"] for x in arr])
                ang = np.arctan2(base[:, 1], base[:, 0]) - np.pi / 2
                cr, sr = np.cos(-ang), np.sin(-ang)
                rot = np.zeros((len(ang), 3, 3))
                rot[:, 0, 0], rot[:, 0, 1], rot[:, 1, 0], rot[:, 1, 1], rot[:, 2, 2] = cr, -sr, sr, cr, 1
                rv = np.einsum("nij,nj->ni", rot, rv)
                dot = np.sum(hv * rv, axis=1) / (np.linalg.norm(hv, axis=1) * np.linalg.norm(rv, axis=1))
                errors.extend(np.degrees(np.arccos(np.clip(dot, -1, 1))).tolist())
            scores[f"{name}_{label}_direction_deg"] = float(np.mean(errors))
    # Source timestamps are matched directly; no per-variant temporal alignment or trimming.
    return scores


def run(args):
    from g1_imitation.core.s6_mujoco_closed_loop import G1PhysicsPlant

    cfg, asset_root = configure_runtime()
    plant = G1PhysicsPlant(mjcf_path=asset_root / "sonic_official_g1" / "scene_43dof.xml")
    records, saved = {}, {}
    variants = ["baseline"] + [v for v in args.variants if v != "baseline"]
    for take in TAKES:
        z = np.load(ROOT / "outputs" / "advanced_take" / "s2s3" / "rgbonly_lag6" / f"{take}.npz")
        metrics = json.loads((ROOT / "outputs" / "advanced_take" / "s2s3" / "rgbonly_lag6" / "metrics.json").read_text(encoding="utf-8"))
        lag = int(metrics["ref_lag_bvh_samples"])
        times = (z["grid"] + lag) / 120.0
        human = z["bvh12"]
        eval_start = times[0] + EXPERIMENT["evaluation"]["exclude_startup_s"]
        records[take], saved[take] = {}, {}
        anchor = None
        for variant in variants:
            dest = OUT / take / f"{variant}.npz"
            run_data = evaluate_variant(variant, z["raw" if variant in ("temporal", "combined") else "smooth"], z["cov"], times, plant.kin_model,
                                        asset_root, anchor=anchor)
            if variant == "baseline":
                anchor = run_data["anchor"]
            score = score_fk(plant, run_data, human, times, eval_start)
            physical = run_data["physical"]
            score.update({
                "survived": bool(physical.survived),
                "joint_rmse_rad_vs_own_reference": float(physical.joint_rmse_rad),
                "support_foot_skating_m_s": float(physical.support_foot_skating_speed_ms),
                "max_tilt_deg": float(physical.max_tilt_deg),
                "path_end_error_m": float(np.linalg.norm(physical.qpos_hist[-1, :2] - run_data["q_ref"][len(physical.qpos_hist)-1, :2])),
                "trajectory_limiter_fraction": run_data["limited_fraction"],
                "ik_solve_p50_ms": run_data["solve_p50_ms"],
                "contact_observed_fraction": float(np.mean(run_data["contacts"] >= 0)),
                "source_support_fraction": float(np.mean(run_data["contacts"] == 1)),
            })
            records[take][variant] = score
            saved[take][variant] = {"t50": run_data["t50"], "q_ref": run_data["q_ref"],
                                    "qpos_phys": physical.qpos_hist, "contacts": run_data["contacts"],
                                    "q_slots": run_data["q_slots"], "observed": run_data["observed"],
                                    "fresh": run_data["fresh"], "left_fz": physical.left_fz_hist,
                                    "right_fz": physical.right_fz_hist,
                                    "anchor_xy": run_data["anchor"][0], "anchor_yaw": run_data["anchor"][1]}
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, **saved[take][variant])
            OUT.joinpath("partial_metrics.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
            print(take, variant, json.dumps(score, ensure_ascii=False), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"experiment": EXPERIMENT, "results": records,
              "provenance": {"asset_root": str(asset_root), "robot_model": "official G1 scene_43dof.xml",
                             "controller": "SONIC v1.1", "preview_s": EXPERIMENT["preview_s"],
                             "human_input": "same saved RGB triangulation; baseline/relations/support use saved One-Euro; temporal/combined replace it with skeleton filtering and covariance",
                             "ground_truth": "BVH for scoring only; never used to create q_ref",
                             "clock": "pre-recorded alignment; no variant-specific time alignment"}}
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    make_plot(records)
    print(f"Saved {OUT / 'report.json'} and summary.png", flush=True)


def make_plot(records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    labels = {"baseline": "基线", "relations": "① 身体关系", "support": "② 骨盆 / 支撑", "temporal": "③ 时序 / 限速", "combined": "①②③ 组合"}
    colors = {"baseline": "#555555", "relations": "#2d73ce", "support": "#469650", "temporal": "#a55596", "combined": "#d2871e"}

    variants = [v for v in VARIANTS if any(v in d for d in records.values())]
    takes = list(records)
    fields = (("arms_reference_direction_deg", "运动学参考：手臂方向误差（°）"),
              ("arms_physics_direction_deg", "物理跟踪：手臂方向误差（°）"),
              ("legs_reference_direction_deg", "运动学参考：腿部方向误差（°）"),
              ("legs_physics_direction_deg", "物理跟踪：腿部方向误差（°）"),
              ("support_foot_skating_m_s", "物理支撑脚滑移速度（m/s）"),
              ("path_end_error_m", "物理根部相对自身参考的终点偏差（m）"))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for ax, (field, title) in zip(axes.flat, fields):
        vals = [[records[t].get(v, {}).get(field, np.nan) for t in takes] for v in variants]
        x = np.arange(len(takes))
        width = .8 / len(variants)
        for i, (variant, y) in enumerate(zip(variants, vals)):
            ax.bar(x + (i - (len(variants) - 1) / 2) * width, y, width, label=labels[variant], color=colors[variant])
        ax.set_title(title)
        ax.set_xticks(x, [x.replace("Take_", "") for x in takes])
        ax.grid(axis="y", alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("G1 重定向实验：相同 RGB 来源、时钟与 SONIC 仿真设置 · 所有指标越低越好")
    fig.savefig(OUT / "summary.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    run(parser.parse_args())
