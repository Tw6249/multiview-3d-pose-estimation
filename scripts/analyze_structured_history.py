"""Audit saved geometry/history runs without altering trajectories or scoring.

Report FK vs visual input at source slots, and actual delivered 50 Hz reference
finite differences separately: zero-order hold need not preserve source-rate
velocity/acceleration bounds. No time-shift fitting or per-method alignment.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.retarget_ablation_20261107 import configure_runtime
from scripts.ik_observation_gate_eval import save_json
from g1_imitation.core.s6_mujoco_closed_loop import G1PhysicsPlant
from g1_imitation.core.constants import KEYPOINT_INDEX_BY_NAME as K

OUT = ROOT / "outputs/structured_history_20261109"


def run():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    config = report["configuration"]
    _, assets = configure_runtime()
    model = G1PhysicsPlant(mjcf_path=assets / "sonic_official_g1/scene_43dof.xml").kin_model
    data = mujoco.MjData(model)
    colors = ("#3b6fb6", "#d88828", "#27845c", "#ad579e")
    diagnostics = dict(notes="No retiming. Source-rate bounds and delivered tick finite differences are different contracts.",
                       per_take={})
    for take, variants in report["results"].items():
        with np.load(ROOT / report["source"] / f"{take}.npz") as source:
            points = source["smooth"]
        diagnostics["per_take"][take] = {}
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), constrained_layout=True, sharex=True)
        for variant_index, variant in enumerate(variants):
            with np.load(OUT / take / f"{variant}.npz") as pack:
                records = dict(pack)
            times = records["slot_times"]
            valid = times >= times[0] + config["evaluation_start_s"]
            errors = {name: [] for name in config["geometry"]["chains"]}
            for frame_index in np.flatnonzero(valid & records["observed"]):
                data.qpos[:] = records["q_slots"][frame_index]
                mujoco.mj_kinematics(model, data)
                for name, chain in config["geometry"]["chains"].items():
                    side, limb = name.split("_")
                    human_names = ("shoulder", "elbow", "wrist") if limb == "arm" else ("hip", "knee", "ankle")
                    for body_start, body_end, human_start, human_end in zip(chain[:-1], chain[1:], human_names[:-1], human_names[1:]):
                        human_vector = points[frame_index, K[f"{side}_{human_end}"]] - points[frame_index, K[f"{side}_{human_start}"]]
                        robot_vector = data.xpos[model.body(body_end).id] - data.xpos[model.body(body_start).id]
                        if np.isfinite(human_vector).all() and np.linalg.norm(human_vector) > 1e-6:
                            angle = np.degrees(np.arctan2(np.linalg.norm(np.cross(human_vector, robot_vector)),
                                                         np.dot(human_vector, robot_vector)))
                            errors[name].append(float(angle))
            control_times = records["t50"]
            interval = np.diff(control_times)
            velocity = np.diff(records["q_ref"][:, 7:], axis=0) / interval[:, None]
            acceleration = np.diff(velocity, axis=0) / ((interval[1:] + interval[:-1]) / 2)[:, None]
            velocity = velocity[control_times[1:] >= times[0] + config["evaluation_start_s"]]
            acceleration = acceleration[control_times[2:] >= times[0] + config["evaluation_start_s"]]
            tick_indices = np.searchsorted(times, control_times, side="right") - 1
            reference_velocity_mismatch = np.diff(records["q_ref"][:, 7:], axis=0) / interval[:, None] - records["dq_slots"][tick_indices[1:], 6:]
            values = dict(source_fk_direction_vs_input_deg={name: float(np.mean(entries)) for name, entries in errors.items()},
                          delivered_tick_speed_max_rad_s=float(np.abs(velocity).max()),
                          delivered_tick_acceleration_max_rad_s2=float(np.abs(acceleration).max()),
                          delivered_ticks_over_source_speed_limit=int(np.sum(np.any(np.abs(velocity) >
                                                                      config["history"]["velocity_limit_rad_s"] + 1e-6, axis=1))),
                          delivered_ticks_over_source_acceleration_limit=int(np.sum(np.any(np.abs(acceleration) >
                                                                             config["history"]["acceleration_limit_rad_s2"] + 1e-6, axis=1))),
                          held_reference_dq_finite_difference_mismatch_max_rad_s=float(np.abs(reference_velocity_mismatch).max()))
            diagnostics["per_take"][take][variant] = values
            for axis, offset, label in zip(axes, (29, 31, 32), ("Right shoulder pitch", "Right shoulder yaw", "Right elbow")):
                source_time = times - times[0]
                axis.plot(source_time, np.degrees(records["q_slots"][:, offset]), color=colors[variant_index], label=variant)
                axis.plot(control_times[:len(records["qpos_phys"])] - times[0],
                          np.degrees(records["qpos_phys"][:, offset]), color=colors[variant_index], linestyle="--", alpha=.65)
                axis.set_ylabel(label + " / deg")
                axis.grid(alpha=.2)
        axes[0].legend(ncol=4)
        axes[-1].set_xlabel("Source time / s; solid: reference, dashed: actual physics; no lag correction")
        fig.suptitle(take + ": reference and controller tracking")
        fig.savefig(OUT / f"{take}_tracking.png", dpi=150)
        plt.close(fig)
    save_json(OUT / "delivery_audit.json", diagnostics)
    sections = "".join(f'<section><h2>{take}</h2><video controls preload="metadata" src="{take}_comparison.mp4"></video>'
                       f'<img src="{take}_tracking.png" alt="Solid reference and dashed physics"></section>'
                       for take in report["results"] if (OUT / f"{take}_comparison.mp4").exists())
    (OUT / "audit.html").write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>结构化与历史 IK：结果和失效审计</title><style>body{font:16px/1.7 system-ui;max-width:1600px;margin:28px auto;padding:20px;background:#f4f6f9;color:#243247}section{padding:20px;background:white;margin:24px 0}video,img{width:100%}</style>
<h1>结构化几何＋历史状态在线约束：实际结果</h1>
<p>2026-11-09 · 四组 × 五段，共 20 次同条件闭环仿真。所有片段已被看过；不属于新盲测。</p>
<p><strong>本轮没有胜出方案。</strong>结构化目标让参考对视觉输入的手臂方向误差从 8.50° 降至 3.05°，但平均物理手臂误差从 17.59° 增至 22.12°。Take_012 出现明显分支/限位与跟踪退化，不能替换主线。</p>
<p>两个历史版本在源 30 Hz 下满足 8 rad/s、60 rad/s²；但原有 50 Hz 零阶保持使实际交付位置参考差分最高达到约 13.33 rad/s、666.67 rad/s²。约束未贯穿控制交付。没有修改物理参数、控制器输出、时间对齐或视频中的姿态。</p>
<p><a href="index.html">指标表与原始对照页</a> · <a href="report.json">完整试验数据</a> · <a href="delivery_audit.json">输入几何与控制交付审计</a></p>
<section><img src="summary.png" alt="全部逐段结果"></section>
<p>视频上排为运动学参考，下排为实际物理输出；同一相机和源时刻。关节图实线为参考、虚线为物理输出，不做滞后修正。</p>'''
        + sections + '</html>', encoding="utf-8")
    for variant in config["variants"]:
        rows = [take[variant] for take in diagnostics["per_take"].values()]
        print(variant, "input arm direction", np.mean([row["source_fk_direction_vs_input_deg"][name]
                                                       for row in rows for name in ("left_arm", "right_arm")]),
              "max delivered speed", max(row["delivered_tick_speed_max_rad_s"] for row in rows),
              "max delivered acceleration", max(row["delivered_tick_acceleration_max_rad_s2"] for row in rows))


if __name__ == "__main__":
    run()
