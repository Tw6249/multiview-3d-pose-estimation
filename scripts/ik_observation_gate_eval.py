"""Saved Kalman tracks -> three causal IK variants -> identical SONIC physics.

BVH is used only by the unchanged legacy scorer. All previously inspected clips
are exploratory data. Save every frame, gate reason, trajectory and source hash.
Render reference and physical outputs directly with one shared robot camera.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.retarget_ablation_20261107 import configure_runtime, score_fk
from g1_imitation.core.constants import KEYPOINT_INDEX_BY_NAME as K
from g1_imitation.core.ik_observation_gate import ArmObservationGate, GatedKeypointIKRetargeter
from g1_imitation.core.s45c_keypoint_ik import KeypointIKRetargeter, causal_upsample
from g1_imitation.core.s6_mujoco_closed_loop import G1PhysicsPlant, MuJoCoClosedLoopVerifier, SonicPolicy

CONFIG_PATH = ROOT / "g1_imitation/config/ik_observation_gate_20261109.json"
OUT = ROOT / "outputs/ik_observation_gate_20261109"
SOURCE = ROOT / "outputs/advanced_take/s2s3/selfcal_lag6"
COLORS = ("#3b6fb6", "#d88828", "#27845c")


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    path.write_text(json.dumps(clean_json(value), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def evaluate(takes, retargeter_factory=None, extra_source_files=()):
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    full_cfg, assets = configure_runtime()
    model_path = assets / "sonic_official_g1/scene_43dof.xml"
    plant = G1PhysicsPlant(mjcf_path=model_path)
    original_ranges = plant.kin_model.jnt_range.copy()
    physics_cfg = full_cfg["s6_closed_loop"]
    policy = SonicPolicy(physics_cfg["sonic_v11"])
    lag = json.loads((SOURCE / "metrics.json").read_text())["ref_lag_bvh_samples"]
    OUT.mkdir(parents=True, exist_ok=True)
    report = dict(configuration=config, source=str(SOURCE), input_hashes={}, results={}, summary={},
                  scope="exploratory saved-vision simulation, no blind test or live latency claim",
                  physics=dict(model_sha256=digest(model_path), policy="sonic_v11",
                               physics_dt_s=physics_cfg["physics_dt_s"], control_dt_s=config["control_dt_s"],
                               preview_s=config["preview_s"], settle_steps=25, effort_limit="actuator_range"),
                  source_hashes={str(path.relative_to(ROOT)): digest(path) for path in
                                 (CONFIG_PATH, Path(__file__), ROOT / "g1_imitation/core/ik_observation_gate.py",
                                  ROOT / "g1_imitation/core/s45c_keypoint_ik.py",
                                  ROOT / "g1_imitation/core/s6_mujoco_closed_loop.py", *extra_source_files)})
    for take in takes:
        report["input_hashes"][take] = digest(SOURCE / f"{take}.npz")
        with np.load(SOURCE / f"{take}.npz") as pack:
            times = (pack["grid"] + lag) / 120.
            points = np.asarray(pack["smooth"], dtype=float)
            human = np.asarray(pack["bvh12"], dtype=float)
        anchor = None
        report["results"][take] = {}
        for variant in config["variants"]:
            retargeter = (retargeter_factory(variant, config, plant.kin_model) if retargeter_factory is not None else
                          KeypointIKRetargeter(model=plant.kin_model) if variant == "baseline" else
                          GatedKeypointIKRetargeter(config, mode=variant, model=plant.kin_model))
            monitor_cfg = config.get("observation_monitor", config)
            monitor = ArmObservationGate(monitor_cfg)
            frames, diagnostics, wall_ms, experiment_diagnostics = [], [], [], []
            for timestamp, frame_points in zip(times, points):
                diagnostics.append(monitor.inspect(frame_points, float(timestamp), retargeter.scale is None))
                start = time.perf_counter()
                frames.append(retargeter.step(frame_points, float(timestamp)))
                wall_ms.append(1000 * (time.perf_counter() - start))
                experiment_diagnostics.append(getattr(retargeter, "experiment_diagnostics", {}).copy())
            q_slots = np.array([frame.q for frame in frames])
            dq_slots = np.array([frame.dq for frame in frames])
            observed = np.array([frame.observed for frame in frames])
            control_times, reference, velocity, fresh = causal_upsample(
                times, q_slots, dq_slots, observed, "hold", retargeter.lo, retargeter.hi,
                dt=config["control_dt_s"])
            if anchor is None:
                anchor = (reference[0, :2].copy(),
                          Rotation.from_quat(reference[0, [4, 5, 6, 3]]).as_euler("zyx")[0])
            rotation = Rotation.from_euler("z", -anchor[1])
            reference[:, :2] -= anchor[0]
            reference[:, :3] = rotation.apply(reference[:, :3])
            reference[:, [4, 5, 6, 3]] = (rotation * Rotation.from_quat(reference[:, [4, 5, 6, 3]])).as_quat()
            velocity[:, :3] = rotation.apply(velocity[:, :3])
            np.testing.assert_array_equal(original_ranges, plant.kin_model.jnt_range)
            verifier = MuJoCoClosedLoopVerifier(mjcf_path=model_path, policy=policy,
                                                physics_dt=physics_cfg["physics_dt_s"],
                                                effort_limit="actuator_range")
            physical = verifier.run_stage_6b_physics(reference, velocity, ik_dt=config["control_dt_s"],
                                                    control_dt=config["control_dt_s"], settle_steps=25,
                                                    preview_s=config["preview_s"])
            scores = score_fk(plant, dict(t50=control_times, q_ref=reference, qpos_phys=physical.qpos_hist),
                              human, times, times[0] + config["evaluation_start_s"])
            after_start = times >= times[0] + config["evaluation_start_s"]
            wall = np.array(wall_ms)[after_start & observed]
            speeds = np.abs(np.diff(q_slots[:, 7:], axis=0) / np.diff(times)[:, None])
            speeds = speeds[after_start[1:] & observed[1:] & observed[:-1]]
            rejected = np.array([[bool(item[side]["reasons"]) for side in ("left", "right")]
                                 for item in diagnostics])
            recovery = (~rejected[1:] & rejected[:-1]).any(axis=1) & after_start[1:]
            scores.update(survived=bool(physical.survived),
                          support_foot_skating_m_s=float(physical.support_foot_skating_speed_ms),
                          path_end_error_m=float(np.linalg.norm(physical.qpos_hist[-1, :2] -
                                                               reference[len(physical.qpos_hist) - 1, :2])),
                          joint_rmse_rad_vs_own_reference=float(physical.joint_rmse_rad),
                          max_tilt_deg=float(physical.max_tilt_deg),
                          ik_wall_p50_ms=float(np.percentile(wall, 50)),
                          ik_wall_p95_ms=float(np.percentile(wall, 95)),
                          ik_wall_p99_ms=float(np.percentile(wall, 99)),
                          joint_speed_max_rad_s=float(speeds.max()),
                          joint_speed_p99_rad_s=float(np.percentile(speeds, 99)),
                          recovery_max_joint_step_rad=float(np.max(np.abs(np.diff(q_slots[:, 7:], axis=0)[recovery])))
                          if recovery.any() else None,
                          rejected_arm_frames=int(rejected[after_start].sum()),
                          scored_arm_frames=int(after_start.sum() * 2),
                          swivel_unknown_arm_frames=int(sum(not item[side]["swivel_observable"]
                                                           for index, item in enumerate(diagnostics)
                                                           if after_start[index] for side in ("left", "right"))))
            active_history = [item for index, item in enumerate(experiment_diagnostics)
                              if after_start[index] and item.get("history_active")]
            scores["history_constrained_frames"] = len(active_history)
            scores["history_acceleration_max_rad_s2"] = max(
                [item["history_acceleration_max_rad_s2"] for item in active_history], default=0.)
            scores["history_box_violation_max_rad"] = max(
                [item["history_box_violation_rad"] for item in active_history], default=0.)
            acceleration = np.diff(dq_slots[:, 6:], axis=0) / np.diff(times)[:, None]
            scores["joint_acceleration_max_rad_s2"] = float(np.max(np.abs(acceleration[after_start[1:]])))
            dest = OUT / take
            dest.mkdir(exist_ok=True)
            np.savez_compressed(dest / f"{variant}.npz", t50=control_times, q_ref=reference,
                                qpos_phys=physical.qpos_hist, q_slots=q_slots, dq_slots=dq_slots,
                                slot_times=times, observed=observed, fresh=fresh, rejected=rejected,
                                left_fz=physical.left_fz_hist, right_fz=physical.right_fz_hist,
                                torques=physical.torques_hist, anchor_xy=anchor[0], anchor_yaw=anchor[1])
            save_json(dest / f"{variant}_diagnostics.json", dict(frames=diagnostics,
                       held=[frame.held_groups for frame in frames], ik_wall_ms=wall_ms,
                       experiment=experiment_diagnostics))
            report["results"][take][variant] = scores
            save_json(OUT / "report.json", report)
            print(take, variant, json.dumps(scores), flush=True)
    for group, selected_takes in (("all", takes),
                                  ("previously_inspected_006_007_008", [take for take in takes if take in
                                                                    ("Take_006", "Take_007", "Take_008")])):
        if not selected_takes:
            continue
        report["summary"][group] = {}
        for variant in config["variants"]:
            rows = [report["results"][take][variant] for take in selected_takes]
            values = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]
                      if key not in ("survived", "recovery_max_joint_step_rad")}
            values["survived_count"] = f"{sum(row['survived'] for row in rows)}/{len(rows)}"
            report["summary"][group][variant] = values
    save_json(OUT / "report.json", report)
    plot(report)


def plot(report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fields = (("arms_physics_direction_deg", "Physical arms / deg"),
              ("legs_physics_direction_deg", "Physical legs / deg"),
              ("support_foot_skating_m_s", "Support foot skating / m/s"),
              ("path_end_error_m", "End error vs own reference / m"),
              ("joint_speed_max_rad_s", "Maximum reference joint speed / rad/s"),
              ("ik_wall_p95_ms", "IK wrapper p95 / ms (saved-track replay)"))
    takes = list(report["results"])
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for axis, (field, title) in zip(axes.flat, fields):
        for index, variant in enumerate(report["configuration"]["variants"]):
            width = .8 / len(report["configuration"]["variants"])
            axis.bar(np.arange(len(takes)) + width * (index - (len(report["configuration"]["variants"]) - 1) / 2),
                     [report["results"][take][variant][field] for take in takes], width,
                     color=COLORS[index], label=variant)
        axis.set_xticks(np.arange(len(takes)), [take[-3:] for take in takes])
        axis.set_title(title)
        axis.grid(axis="y", alpha=.2)
    axes.flat[0].legend()
    fig.suptitle("Causal IK comparison: exploratory G1 + SONIC v1.1, preview=0")
    fig.savefig(OUT / "summary.png", dpi=160)
    plt.close(fig)
    for take in takes:
        records = {variant: dict(np.load(OUT / take / f"{variant}.npz"))
                   for variant in report["configuration"]["variants"]}
        fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
        for index, (variant, record) in enumerate(records.items()):
            axis_times = record["slot_times"] - record["slot_times"][0]
            for axis, offset, title in zip(axes, (29, 31, 32),
                                          ("Right shoulder pitch", "Right shoulder yaw", "Right elbow")):
                axis.plot(axis_times, np.degrees(record["q_slots"][:, offset]), color=COLORS[index],
                          label=f"{variant}: reference")
                axis.set_ylabel(title + " / deg")
                axis.grid(alpha=.2)
        baseline = records["baseline"]
        for axis in axes:
            for timestamp in (baseline["slot_times"] - baseline["slot_times"][0])[baseline["rejected"][:, 1]]:
                axis.axvline(timestamp, color="red", alpha=.18, linewidth=.7)
        axes[0].legend(loc="best")
        axes[-1].set_xlabel("Source time / s; red lines: rejected right arm observation")
        fig.savefig(OUT / f"{take}_joint_traces.png", dpi=140)
        plt.close(fig)


def render(take):
    import cv2
    from g1_imitation.core.s6_mujoco_closed_loop import G1PhysicsPlant
    full_cfg, assets = configure_runtime()
    config = json.loads(CONFIG_PATH.read_text())
    variants = config["variants"]
    records = {variant: dict(np.load(OUT / take / f"{variant}.npz")) for variant in variants}
    with np.load(SOURCE / f"{take}.npz") as source:
        points = source["smooth"]
    times = records["baseline"]["slot_times"]
    model = G1PhysicsPlant(mjcf_path=assets / "sonic_official_g1/scene_43dof.xml").kin_model
    renderer = mujoco.Renderer(model, height=280, width=320)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.distance, camera.elevation, camera.azimuth = 3.5, -20, 48
    destination = OUT / f"{take}_comparison.mp4"
    canvas_width = 320 * (len(variants) + 1)
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), 30., (canvas_width, 680))
    if not writer.isOpened():
        raise RuntimeError("video writer failed")
    edges = [(5, 6), (11, 12), (5, 11), (6, 12), (5, 7), (7, 9), (6, 8), (8, 10),
             (11, 13), (13, 15), (12, 14), (14, 16)]
    paths = np.concatenate([record["qpos_phys"][:, :2] for record in records.values()])
    path_min, path_range = paths.min(0) - .15, np.ptp(paths, axis=0) + .3
    def label(image, string, position, scale=.46, color=(55, 55, 55)):
        cv2.putText(image, string, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    sample_indices = {len(times) // 3, len(times) // 2, 2 * len(times) // 3}
    try:
        for frame_index, timestamp in enumerate(times):
            canvas = np.full((680, canvas_width, 3), 248, dtype=np.uint8)
            label(canvas, f"{take} | source t={timestamp-times[0]:.2f}s | actual simulation, no pose smoothing", (12, 24), .65)
            label(canvas, "S3 human (root centered)", (8, 65))
            pose = points[frame_index].copy()
            pose[:, :2] -= np.nanmean(pose[[11, 12], :2], axis=0)
            horizontal = .75 * pose[:, 0] - .66 * pose[:, 1]
            vertical = pose[:, 2] + .2 * (.66 * pose[:, 0] + .75 * pose[:, 1])
            pixels = np.c_[150 + 110 * horizontal, 325 - 125 * vertical]
            for start, end in edges:
                if np.isfinite(pixels[[start, end]]).all():
                    cv2.line(canvas, tuple(pixels[start].astype(int)), tuple(pixels[end].astype(int)), (110, 90, 40), 2, cv2.LINE_AA)
            label(canvas, "Actual root XY paths (m)", (8, 395))
            for column, (variant, record) in enumerate(records.items(), 1):
                tick = int(np.searchsorted(record["t50"], timestamp, side="right") - 1)
                label(canvas, variant + " / reference", (320 * column + 6, 65))
                label(canvas, variant + " / physics", (320 * column + 6, 395))
                if tick < 0:
                    continue
                baseline_tick = max(0, int(np.searchsorted(records["baseline"]["t50"], timestamp, side="right") - 1))
                camera.lookat[:] = [*records["baseline"]["q_ref"][baseline_tick, :2], .65]
                for row, key in ((75, "q_ref"), (405, "qpos_phys")):
                    if tick >= len(record[key]):
                        label(canvas, "Simulation ended", (320 * column + 20, row + 100))
                        continue
                    data.qpos[:] = record[key][tick]
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera)
                    image = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                    canvas[row:row+275, 320*column:320*(column+1)] = image[:275]
                rejected = record["rejected"][frame_index]
                if rejected.any():
                    label(canvas, "QUALITY FLAG: " + "/".join(side for side, bad in zip(("L", "R"), rejected) if bad),
                          (320*column+6, 372), color=(40, 40, 210))
                path = record["qpos_phys"][:min(tick+1, len(record["qpos_phys"])), :2]
                pixels_path = np.c_[20 + (path[:, 0] - path_min[0]) / path_range[0] * 270,
                                    650 - (path[:, 1] - path_min[1]) / path_range[1] * 225].astype(np.int32)
                if len(path) > 1:
                    color = tuple(int(COLORS[column-1][offset:offset+2], 16) for offset in (5, 3, 1))
                    cv2.polylines(canvas, [pixels_path], False, color, 1, cv2.LINE_AA)
            writer.write(canvas)
            if frame_index in sample_indices:
                cv2.imencode(".png", canvas)[1].tofile(str(OUT / f"{take}_frame_{frame_index:04d}.png"))
            if frame_index % 150 == 0:
                print(take, "render", frame_index, "/", len(times), flush=True)
    finally:
        writer.release()
        renderer.close()
    print("rendered", destination, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--takes", nargs="+")
    parser.add_argument("--render", nargs="*", default=[])
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text())
    if not args.render_only:
        evaluate(args.takes or config["takes"])
    for take in args.render:
        render(take)


if __name__ == "__main__":
    main()
