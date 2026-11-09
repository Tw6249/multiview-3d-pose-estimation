"""Independent diagnostics on saved ablation trajectories; never modifies q or physics.

BVH is used only here for scoring. Source-scale normalization is computed once per
take and shared across branches. Timing diagnostic scans lags but never changes
primary same-timestamp scores, videos or trajectories. Contact labels are visual
heuristics, not force ground truth. Min-clearance numbers are raw, never clamped.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retarget_ablation_20261107 import OUT, EXPERIMENT, configure_runtime
from g1_imitation.core.s45c_keypoint_ik import KeypointIKRetargeter
from g1_imitation.core.s6_mujoco_closed_loop import G1PhysicsPlant

NAMES = ["left_shoulder_pitch_link", "right_shoulder_pitch_link", "left_elbow_link", "right_elbow_link",
         "left_wrist_roll_link", "right_wrist_roll_link", "left_hip_pitch_link", "right_hip_pitch_link",
         "left_knee_link", "right_knee_link", "left_ankle_pitch_link", "right_ankle_pitch_link"]


def angles(p):
    out = []
    for a, b, c in ((0, 2, 4), (1, 3, 5), (6, 8, 10), (7, 9, 11)):
        u, v = p[:, a] - p[:, b], p[:, c] - p[:, b]
        cosine = np.sum(u * v, axis=1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
        out.append(180 - np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    return np.array(out).T


def heading(p):
    lat = p[:, 6] - p[:, 7]
    return np.arctan2(lat[:, 1], lat[:, 0]) - np.pi / 2


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def torso_local_hands(p, arm_length):
    hips, shoulders = p[:, 6:8].mean(1), p[:, 0:2].mean(1)
    z = shoulders - hips
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    y = p[:, 0] - p[:, 1]
    y -= np.sum(y * z, axis=1, keepdims=True) * z
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    R = np.stack([np.cross(y, z), y, z], axis=2)
    return np.einsum("nki,njk->nji", R, p[:, 4:6] - shoulders[:, None]) / arm_length


def fk(model, q, sole_local):
    d = mujoco.MjData(model)
    ids = [model.body(n).id for n in NAMES]
    feet = [model.body(f"{s}_ankle_roll_link").id for s in ("left", "right")]
    poses, clearances = [], []
    for x in q:
        d.qpos[:] = x
        mujoco.mj_kinematics(model, d)
        poses.append(d.xpos[ids].copy())
        clearances.append([float((d.xpos[b] + sole_local @ d.xmat[b].reshape(3, 3).T)[:, 2].min()) for b in feet])
    return np.asarray(poses), np.asarray(clearances)


def main():
    cfg, assets = configure_runtime()
    plant = G1PhysicsPlant(mjcf_path=assets / "sonic_official_g1/scene_43dof.xml")
    model = plant.kin_model
    rt = KeypointIKRetargeter(model=model)
    robot_arm = np.mean([rt.L[s]["upper"] + rt.L[s]["fore"] for s in ("left", "right")])
    result = {}
    for take in EXPERIMENT["takes"]:
        z = np.load(ROOT / "outputs/advanced_take/s2s3/rgbonly_lag6" / f"{take}.npz")
        slot_times = (z["grid"] + 6) / 120
        human = z["bvh12"]
        cal = human[:16]
        human_arm = np.nanmedian([np.linalg.norm(cal[:, b] - cal[:, a], axis=1) + np.linalg.norm(cal[:, c] - cal[:, b], axis=1)
                                 for a, b, c in ((0, 2, 4), (1, 3, 5))])
        human_leg = np.nanmedian([np.linalg.norm(cal[:, b] - cal[:, a], axis=1) + np.linalg.norm(cal[:, c] - cal[:, b], axis=1)
                                 for a, b, c in ((6, 8, 10), (7, 9, 11))])
        baseline = dict(np.load(OUT / take / "baseline.npz"))
        times = baseline["t50"]
        k = np.searchsorted(slot_times, times, side="right") - 1
        keep = (times >= slot_times[0] + 1.2) & np.isfinite(human[k]).all((1, 2))
        g = human[k][keep]
        hg = angles(g)
        human_rel = torso_local_hands(g, human_arm)
        hand_distance = np.linalg.norm(g[:, 4] - g[:, 5], axis=1) / human_arm
        human_yaw = heading(g)
        root = g[:, 6:8].mean(1)
        angle = -human_yaw[0]
        rot2 = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        expected_path = (root[:, :2] - root[0, :2]) @ rot2.T * (rt.robot_leg / human_leg)
        expected_z = (root[:, 2] - root[0, 2]) * rt.robot_leg / human_leg
        result[take] = {}
        base_initial = baseline["q_ref"][keep][0]
        for variant in EXPERIMENT["variants"]:
            d = dict(np.load(OUT / take / f"{variant}.npz"))
            assert np.array_equal(d["t50"], times), "same control clock required"
            row = {}
            for label, key in (("reference", "q_ref"), ("physics", "qpos_phys")):
                q = d[key][keep]
                pos, clearance = fk(model, q, rt.sole_local)
                a = angles(pos)
                rel = torso_local_hands(pos, robot_arm)
                dist = np.linalg.norm(pos[:, 4] - pos[:, 5], axis=1) / robot_arm
                source_range = np.percentile(hg, 95, axis=0) - np.percentile(hg, 5, axis=0)
                robot_range = np.percentile(a, 95, axis=0) - np.percentile(a, 5, axis=0)
                amplitude = np.divide(robot_range, source_range, out=np.full(4, np.nan), where=source_range > 5)
                root_yaw = heading(pos)
                initial_heading = fk(model, baseline["q_ref"][keep][:1], rt.sole_local)[0]
                yaw0 = heading(initial_heading)[0]
                angle = -yaw0
                r2 = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                robot_path = (q[:, :2] - base_initial[:2]) @ r2.T
                error = robot_path - expected_path
                lag_values, lag_errors = [], []
                # Diagnostic only. Positive lag: robot's joints resemble an earlier human state.
                for lag in range(-10, 11):
                    ia = np.arange(len(a))
                    ig = ia - lag
                    valid = (ig >= 0) & (ig < len(hg))
                    lag_values.append(lag * .02)
                    lag_errors.append(np.mean((a[ia[valid]] - hg[ig[valid]]) ** 2))
                row[label] = {
                    "hand_distance_mae_arm_lengths": float(np.mean(np.abs(dist - hand_distance))),
                    "wrist_torso_rmse_arm_lengths": float(np.sqrt(np.mean(np.sum((rel - human_rel) ** 2, axis=2)))),
                    "elbow_flex_amplitude_ratio": np.round(amplitude[:2], 4).tolist(),
                    "knee_flex_amplitude_ratio": np.round(amplitude[2:], 4).tolist(),
                    "yaw_change_mae_deg": float(np.mean(np.abs(np.degrees(wrap((root_yaw - yaw0) - (human_yaw - human_yaw[0])))))),
                    "path_vs_scaled_bvh_mean_m": float(np.mean(np.linalg.norm(error, axis=1))),
                    "pelvis_vertical_motion_mae_m": float(np.mean(np.abs((q[:, 2] - base_initial[2]) - expected_z))),
                    "sole_clearance_min_m": float(clearance.min()),
                    "sole_clearance_p95_m": float(np.percentile(clearance, 95)),
                    "best_flexion_lag_s_diagnostic_only": float(lag_values[int(np.argmin(lag_errors))]),
                    "same_time_flexion_rmse_deg": float(np.sqrt(np.mean((a - hg) ** 2))),
                    "joint_velocity_max_rad_s": float(np.max(np.abs(np.diff(q[:, 7:], axis=0) / .02))),
                    "joint_acceleration_max_rad_s2": float(np.max(np.abs(np.diff(q[:, 7:], n=2, axis=0) / .02 ** 2))),
                }
            row["input_missing_body_frames"] = int(np.sum(~np.isfinite(z["raw"][:, 5:17]).all((1, 2))))
            result[take][variant] = row
    output = {"notes": {"normalization": "source arm/leg length from first 0.5 s BVH, evaluation only, shared by every method",
                         "lag": "diagnostic search +/- 0.2 s, never applied to primary scores or videos",
                         "path": "source path in initial heading frame, scaled by leg ratio; one fixed origin and baseline heading for all robot variants",
                         "clearance": "raw geometry, negative penetrations retained", "amplitude": "p95-p5 joint flexion; ratio undefined below 5 deg source range"},
              "results": result}
    (OUT / "diagnostics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    # Hash all copied model assets, controller checkpoints and evaluated source files.
    source = ROOT / "data/deployment_bundle/assets/robot-assets"
    manifest = {"python": sys.version, "packages": {}, "assets": {}, "source": {}}
    for pkg in ("numpy", "scipy", "mujoco", "mink", "qpsolvers", "daqp", "onnxruntime", "opencv-python", "matplotlib"):
        manifest["packages"][pkg] = importlib.metadata.version(pkg)
    for folder in ("sonic_official_g1", "controller-comparison/sonic_v11"):
        for f in sorted((assets / folder).rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(assets)
            a, b = hashlib.sha256(f.read_bytes()).hexdigest(), hashlib.sha256((source / rel).read_bytes()).hexdigest()
            if a != b:
                raise RuntimeError(f"asset copy changed: {rel}")
            manifest["assets"][str(rel)] = a
    for f in [ROOT / "g1_imitation/core/retarget_experiments.py", ROOT / "g1_imitation/core/s45c_keypoint_ik.py",
              ROOT / "g1_imitation/core/s6_mujoco_closed_loop.py", ROOT / "g1_imitation/config/retarget_ablation_20261107.json",
              ROOT / "scripts/retarget_ablation_20261107.py", ROOT / "scripts/render_retarget_ablation_20261107.py",
              ROOT / "scripts/analyze_retarget_ablation_20261107.py", ROOT / "tests/test_retarget_experiments.py"]:
        manifest["source"][str(f.relative_to(ROOT))] = hashlib.sha256(f.read_bytes()).hexdigest()
    for take in EXPERIMENT["takes"]:
        f = ROOT / "outputs/advanced_take/s2s3/rgbonly_lag6" / f"{take}.npz"
        manifest["source"][str(f.relative_to(ROOT))] = hashlib.sha256(f.read_bytes()).hexdigest()
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Saved diagnostics and asset/source manifest", flush=True)


if __name__ == "__main__":
    main()
