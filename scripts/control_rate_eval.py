"""Step 1/2 of the retargeting plan: 50 Hz reference generation and controller separation.

Same saved Kalman tracks, G1 model and physics for every run. BVH is used only by the legacy scorer.
Reports reference fidelity, physics fidelity with the controller delay excluded and included, and an
audit of the reference actually delivered to the controller (finite differences of q, q/dq consistency).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.retarget_ablation_20261107 import configure_runtime, score_fk
from scripts import ik_observation_gate_eval as shared
from g1_imitation.core.control_rate_reference import retarget_at_control_rate
from g1_imitation.core.process_scheduling import disable_power_throttling
from g1_imitation.core.s45c_keypoint_ik import KeypointIKRetargeter, causal_upsample
from g1_imitation.core.s3_temporal_filter import TemporalFilter
from g1_imitation.core.structured_history_ik import StructuredHistoryRetargeter
from g1_imitation.core.s6_mujoco_closed_loop import G1PhysicsPlant, MuJoCoClosedLoopVerifier, SonicPolicy

CONFIG_PATH = ROOT / "g1_imitation/config/control_rate_20261109.json"
OUT = ROOT / "outputs/control_rate_20261109"
SOURCE = ROOT / "outputs/advanced_take/s2s3/selfcal_lag6"
THROTTLING_DISABLED = False


def filter_velocities(times, raw, covariance, saved_points):
    """Re-run the main S3 filter to obtain its velocity state; it must reproduce the saved positions."""
    temporal = TemporalFilter()
    results = [temporal.process_frame(p, timestamp_s=float(t), covariance_m2=c) for p, t, c in zip(raw, times, covariance)]
    points = np.array([r.points_3d_smooth_m for r in results])
    finite = np.isfinite(points) & np.isfinite(saved_points)
    if not np.array_equal(np.isfinite(points), np.isfinite(saved_points)) or np.abs(points - saved_points)[finite].max() > 1e-6:
        raise RuntimeError("re-run S3 filter does not reproduce the saved Kalman track")
    return np.array([r.velocities_ms for r in results]), float(temporal.kalman_3x3.velocity_decay_tau_s)


def build_reference(kind, times, points, model, history_cfg, dt, prediction=None):
    if kind == "slot30_hold":
        retargeter = KeypointIKRetargeter(model=model)
        frames = [retargeter.step(p, float(t)) for p, t in zip(points, times)]
        observed = np.array([f.observed for f in frames])
        t50, q, dq, fresh = causal_upsample(times, np.array([f.q for f in frames]), np.array([f.dq for f in frames]),
                                            observed, "hold", retargeter.lo, retargeter.hi, dt=dt)
        source = np.searchsorted(times, t50 + 1e-9, side="right") - 1
        solve = np.array([f.solve_ms for f in frames])[observed]
        return t50, q, dq, t50 - times[source], solve, len(times) / (times[-1] - times[0])
    retargeter = (StructuredHistoryRetargeter(history_cfg, structured=False, history=True, model=model)
                  if kind == "tick50_history" else KeypointIKRetargeter(model=model))
    if kind == "tick50_predict":
        out = retarget_at_control_rate(retargeter, times, points, dt, velocities=prediction[0],
                                       velocity_decay_tau_s=prediction[1])
    else:
        out = retarget_at_control_rate(retargeter, times, points, dt)
    first = int(np.argmax(out["observed"]))
    keep = slice(first, None)
    if not out["observed"][keep].all():
        raise RuntimeError("post-calibration observation missing; reject this trajectory")
    return (out["t"][keep], out["q"][keep].copy(), out["dq"][keep].copy(), out["observation_age_s"][keep],
            out["solve_ms"][keep], 1.0 / dt)


def delivery_audit(t50, q, dq, start):
    interval = np.diff(t50)
    velocity = np.diff(q[:, 7:], axis=0) / interval[:, None]
    acceleration = np.diff(velocity, axis=0) / interval[1:, None]
    v_mask, a_mask = t50[1:] >= start, t50[2:] >= start
    mismatch = np.abs(velocity - dq[1:, 6:])[v_mask]
    return dict(delivered_speed_max_rad_s=float(np.abs(velocity[v_mask]).max()),
                delivered_speed_p99_rad_s=float(np.percentile(np.abs(velocity[v_mask]).max(1), 99)),
                delivered_accel_max_rad_s2=float(np.abs(acceleration[a_mask]).max()),
                delivered_accel_p99_rad_s2=float(np.percentile(np.abs(acceleration[a_mask]).max(1), 99)),
                dq_vs_q_difference_max_rad_s=float(mismatch.max()))


def evaluate(takes, variants=None):
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if variants:
        config["variants"] = list(variants)
    if config["variants"][0] != "baseline":
        raise ValueError("baseline must run first: it fixes the shared initial anchor")
    history_cfg = json.loads((ROOT / config["history_config"]).read_text(encoding="utf-8"))
    full_cfg, assets = configure_runtime()
    model_path = assets / "sonic_official_g1/scene_43dof.xml"
    plant = G1PhysicsPlant(mjcf_path=model_path)
    ranges = plant.kin_model.jnt_range.copy()
    physics_cfg = full_cfg["s6_closed_loop"]
    policies = {name: SonicPolicy(physics_cfg[name]) for name in ("sonic_v11", "sonic_low_latency")}
    lag = json.loads((SOURCE / "metrics.json").read_text())["ref_lag_bvh_samples"]
    dt = config["control_dt_s"]
    OUT.mkdir(parents=True, exist_ok=True)
    report = dict(configuration=config, results={}, summary={}, power_throttling_disabled=THROTTLING_DISABLED,
                  input_hashes={take: shared.digest(SOURCE / f"{take}.npz") for take in takes},
                  source_hashes={str(p.relative_to(ROOT)): shared.digest(p) for p in (
                      CONFIG_PATH, Path(__file__), ROOT / "g1_imitation/core/control_rate_reference.py",
                      ROOT / "g1_imitation/core/structured_history_ik.py",
                      ROOT / "g1_imitation/core/s45c_keypoint_ik.py",
                      ROOT / "g1_imitation/core/s6_mujoco_closed_loop.py")},
                  physics=dict(model_sha256=shared.digest(model_path), physics_dt_s=physics_cfg["physics_dt_s"],
                               control_dt_s=dt, settle_steps=25, effort_limit="actuator_range"))
    for take in takes:
        with np.load(SOURCE / f"{take}.npz") as pack:
            times = (pack["grid"] + lag) / 120.
            points = np.asarray(pack["smooth"], dtype=float)
            human = np.asarray(pack["bvh12"], dtype=float)
            raw, covariance = np.asarray(pack["raw"], dtype=float), np.asarray(pack["cov"], dtype=float)
        prediction = None
        if any(config["variant_specs"][v]["reference"] == "tick50_predict" for v in config["variants"]):
            prediction = filter_velocities(times, raw, covariance, points)
        start = times[0] + config["evaluation_start_s"]
        references, anchor = {}, None
        report["results"][take] = {}
        for variant in config["variants"]:
            spec = config["variant_specs"][variant]
            if spec["reference"] not in references:
                t50, q, dq, age, solve, rate = build_reference(spec["reference"], times, points, plant.kin_model,
                                                               history_cfg, dt, prediction)
                if anchor is None:
                    anchor = (q[0, :2].copy(), Rotation.from_quat(q[0, [4, 5, 6, 3]]).as_euler("zyx")[0])
                rotation = Rotation.from_euler("z", -anchor[1])
                q[:, :2] -= anchor[0]
                q[:, :3] = rotation.apply(q[:, :3])
                q[:, [4, 5, 6, 3]] = (rotation * Rotation.from_quat(q[:, [4, 5, 6, 3]])).as_quat()
                dq[:, :3] = rotation.apply(dq[:, :3])
                references[spec["reference"]] = (t50, q, dq, age, solve, rate)
            t50, q, dq, age, solve, rate = references[spec["reference"]]
            np.testing.assert_array_equal(ranges, plant.kin_model.jnt_range)
            verifier = MuJoCoClosedLoopVerifier(mjcf_path=model_path, policy=policies[spec["policy"]],
                                                physics_dt=physics_cfg["physics_dt_s"], effort_limit="actuator_range")
            physical = verifier.run_stage_6b_physics(q, dq, ik_dt=dt, control_dt=dt, settle_steps=25,
                                                    preview_s=spec["preview_s"])
            aligned = score_fk(plant, dict(t50=t50, q_ref=q, qpos_phys=physical.qpos_hist), human, times, start)
            delayed = score_fk(plant, dict(t50=t50 + spec["preview_s"], q_ref=q, qpos_phys=physical.qpos_hist),
                               human, times, start)
            scores = dict(
                arms_reference_direction_deg=aligned["arms_reference_direction_deg"],
                legs_reference_direction_deg=aligned["legs_reference_direction_deg"],
                arms_physics_delay_excluded_deg=aligned["arms_physics_direction_deg"],
                legs_physics_delay_excluded_deg=aligned["legs_physics_direction_deg"],
                arms_physics_end_to_end_deg=delayed["arms_physics_direction_deg"],
                legs_physics_end_to_end_deg=delayed["legs_physics_direction_deg"],
                added_controller_delay_s=spec["preview_s"],
                observation_age_p50_ms=float(1000 * np.median(age)),
                observation_age_p95_ms=float(1000 * np.percentile(age, 95)),
                survived=bool(physical.survived),
                support_foot_skating_m_s=float(physical.support_foot_skating_speed_ms),
                path_end_error_m=float(np.linalg.norm(physical.qpos_hist[-1, :2] - q[len(physical.qpos_hist) - 1, :2])),
                joint_rmse_rad_vs_own_reference=float(physical.joint_rmse_rad),
                max_tilt_deg=float(physical.max_tilt_deg),
                ik_call_p50_ms=float(np.median(solve)), ik_call_p95_ms=float(np.percentile(solve, 95)),
                ik_calls_per_s=float(rate), **delivery_audit(t50, q, dq, start))
            dest = OUT / take
            dest.mkdir(exist_ok=True)
            np.savez_compressed(dest / f"{variant}.npz", t50=t50, q_ref=q, dq_ref=dq, qpos_phys=physical.qpos_hist,
                                slot_times=times, observation_age_s=age, rejected=np.zeros((len(times), 2), bool),
                                left_fz=physical.left_fz_hist, right_fz=physical.right_fz_hist,
                                torques=physical.torques_hist, anchor_xy=anchor[0], anchor_yaw=anchor[1])
            report["results"][take][variant] = scores
            shared.save_json(OUT / "report.json", report)
            print(take, variant, json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                             for k, v in scores.items()}), flush=True)
    for group, selected in (("all", takes), ("previously_inspected_006_007_008",
                                             [t for t in takes if t in ("Take_006", "Take_007", "Take_008")])):
        if not selected:
            continue
        report["summary"][group] = {}
        for variant in config["variants"]:
            rows = [report["results"][t][variant] for t in selected]
            values = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "survived"}
            values["survived_count"] = f"{sum(r['survived'] for r in rows)}/{len(rows)}"
            report["summary"][group][variant] = values
    shared.save_json(OUT / "report.json", report)
    plot(report)


def plot(report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fields = (("arms_reference_direction_deg", "Reference arms / deg"),
              ("arms_physics_end_to_end_deg", "Physical arms, delay included / deg"),
              ("legs_physics_end_to_end_deg", "Physical legs, delay included / deg"),
              ("support_foot_skating_m_s", "Support foot skating / m/s"),
              ("delivered_speed_max_rad_s", "Delivered reference speed max / rad/s"),
              ("delivered_accel_max_rad_s2", "Delivered reference accel max / rad/s^2"))
    variants = report["configuration"]["variants"]
    takes = list(report["results"])
    colors = ("#3b6fb6", "#d88828", "#27845c", "#7b8794", "#ad579e", "#c0392b")
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    width = .8 / len(variants)
    for axis, (field, title) in zip(axes.flat, fields):
        for index, variant in enumerate(variants):
            axis.bar(np.arange(len(takes)) + width * (index - (len(variants) - 1) / 2),
                     [report["results"][t][variant][field] for t in takes], width, color=colors[index], label=variant)
        axis.set_xticks(np.arange(len(takes)), [t[-3:] for t in takes])
        axis.set_title(title)
        axis.grid(axis="y", alpha=.2)
        if "accel" in field or "speed" in field:
            axis.set_yscale("log")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("50 Hz reference generation and controller separation (G1 simulation, same inputs)")
    fig.savefig(OUT / "summary.png", dpi=150)
    plt.close(fig)


def main():
    global OUT, THROTTLING_DISABLED
    THROTTLING_DISABLED = disable_power_throttling()
    parser = argparse.ArgumentParser()
    parser.add_argument("--takes", nargs="+")
    parser.add_argument("--render", nargs="*", default=[])
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.out:
        OUT = ROOT / args.out
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not args.render_only:
        evaluate(args.takes or config["takes"], args.variants)
    run_config = OUT / "run_config.json"
    if args.variants:
        config["variants"] = list(args.variants)
        OUT.mkdir(parents=True, exist_ok=True)
        run_config.write_text(json.dumps(config, indent=2), encoding="utf-8")
    shared.CONFIG_PATH, shared.OUT = (run_config if run_config.exists() else CONFIG_PATH), OUT
    shared.COLORS = ("#3b6fb6", "#d88828", "#27845c", "#7b8794", "#ad579e", "#c0392b")
    for take in args.render:
        shared.render(take)


if __name__ == "__main__":
    main()
