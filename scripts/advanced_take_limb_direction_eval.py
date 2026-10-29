"""Retargeter-independent scoring: G1 limb directions vs the human (BVH) limb directions, OFFLINE only.

Joint angles of different retargeters are not comparable, so every run is scored the same way:
  human  : BVH12 joints (s2s3/<tag>/<Take>.npz `bvh12`, BVH read at the slot times incl. lag) -> unit vectors
           shoulder->elbow, elbow->wrist, hip->knee, knee->ankle (left/right).
  robot  : MuJoCo FK of the run's G1 configuration (kinematic reference `q_ref` and SONIC physics `qpos_phys`) ->
           the same segments between the *_shoulder_pitch, *_elbow, *_wrist_roll, *_hip_pitch, *_knee,
           *_ankle_pitch link origins.
  frame  : each side is rotated into its own pelvis heading frame (yaw of the left-right hip axis), so the world
           yaw/origin conventions and the root drift drop out. Pitch/roll of the body stay in.
  time   : each video slot is paired with the nearest 50 Hz control tick (<= 10 ms).
Metric : mean angle (deg) between robot and human segment directions per segment, plus the mean over arms/legs.
Output : outputs/advanced_take/limb_direction_eval.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from g1_imitation.core import adv_eval as ae  # noqa: E402
from g1_imitation.core.s6_mujoco_closed_loop import G1PhysicsPlant  # noqa: E402

OUT = ROOT / "outputs" / "advanced_take"
BVH_DT = 1.0 / 120.0
TAG = "rgbonly_lag6"
TAKES = ["Take_002", "Take_006", "Take_007", "Take_008", "Take_012"]
SEGS = {"l_upper_arm": ("left_shoulder_pitch_link", "left_elbow_link", "l_shoulder", "l_elbow"),
        "r_upper_arm": ("right_shoulder_pitch_link", "right_elbow_link", "r_shoulder", "r_elbow"),
        "l_forearm": ("left_elbow_link", "left_wrist_roll_link", "l_elbow", "l_wrist"),
        "r_forearm": ("right_elbow_link", "right_wrist_roll_link", "r_elbow", "r_wrist"),
        "l_thigh": ("left_hip_pitch_link", "left_knee_link", "l_hip", "l_knee"),
        "r_thigh": ("right_hip_pitch_link", "right_knee_link", "r_hip", "r_knee"),
        "l_shank": ("left_knee_link", "left_ankle_pitch_link", "l_knee", "l_ankle"),
        "r_shank": ("right_knee_link", "right_ankle_pitch_link", "r_knee", "r_ankle")}
RUNS = {  # label -> (control folder, file pattern)
    "GMR_bvh": ("gmr_bvh_rgbonly_lag6", "{take}.npz"),
    "GMR_rgb_smplx": ("gmr_smplx_rgbonly_lag6", "{take}.npz"),
    # same references re-simulated on the current S6 plant with SONIC v1.1 @ preview 0.2 s, and the causal keypoint IK
    "GMR_bvh_v11p02": ("gmr_bvh_rgbonly_lag6_sonic_v11_0p20", "{take}.npz"),
    "GMR_rgb_smplx_v11p02": ("gmr_smplx_rgbonly_lag6_sonic_v11_0p20", "{take}.npz"),
    "KPIK_rgb_smooth_v11p02": ("kpik_smooth_rgbonly_lag6", "{take}.npz"),
    "KPIK_rgb_raw_v11p02": ("kpik_raw_rgbonly_lag6", "{take}.npz"),
}


def heading_rot(left, right):
    """Rotation (N,3,3) that removes the yaw of the left-right axis (heading frame: +x forward, +y left)."""
    lat = left - right
    yaw = np.arctan2(lat[:, 1], lat[:, 0]) - np.pi / 2
    c, s = np.cos(-yaw), np.sin(-yaw)
    R = np.zeros((len(yaw), 3, 3))
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1], R[:, 2, 2] = c, -s, s, c, 1
    return R


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def robot_dirs(model, data, qs):
    ids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
           for seg in SEGS.values() for n in seg[:2]}
    pos = {n: np.zeros((len(qs), 3)) for n in ids}
    for i, q in enumerate(qs):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        for n, b in ids.items():
            pos[n][i] = data.xpos[b]
    R = heading_rot(pos["left_hip_pitch_link"], pos["right_hip_pitch_link"])
    return {k: np.einsum("nij,nj->ni", R, unit(pos[b] - pos[a])) for k, (a, b, _, _) in SEGS.items()}


def human_dirs(g):
    J = ae.J
    R = heading_rot(g[:, J["l_hip"]], g[:, J["r_hip"]])
    return {k: np.einsum("nij,nj->ni", R, unit(g[:, J[d]] - g[:, J[c]])) for k, (_, _, c, d) in SEGS.items()}


def main():
    plant = G1PhysicsPlant()
    model, data = plant.kin_model, mujoco.MjData(plant.kin_model)
    lag = int(json.loads((OUT / "s2s3" / TAG / "metrics.json").read_text(encoding="utf-8"))["ref_lag_bvh_samples"])
    res = {}
    for take in TAKES:
        z = np.load(OUT / "s2s3" / TAG / f"{take}.npz")
        t_slot = (z["grid"] + lag) * BVH_DT
        g = z["bvh12"]
        res[take] = {}
        for label, (folder, pat) in RUNS.items():
            f = OUT / "control" / folder / pat.format(take=take)
            if not f.exists():
                continue
            c = np.load(f)
            t50 = c["t50"]
            m = np.isfinite(g).all((1, 2)) & (t_slot >= t50[0]) & (t_slot <= t50[-1])
            k = np.clip(np.searchsorted(t50, t_slot[m]), 0, len(t50) - 1)
            hd = human_dirs(g[m])
            out = {}
            for kind, arr in (("reference", c["q_ref"]), ("physics", c["qpos_phys"])):
                kk = np.clip(k, 0, len(arr) - 1)
                rd = robot_dirs(model, data, arr[kk])
                seg = {s: float(np.mean(ae.angle_between_deg(rd[s], hd[s]))) for s in SEGS}
                seg["arms_mean"] = float(np.mean([seg[s] for s in SEGS if "arm" in s]))
                seg["legs_mean"] = float(np.mean([seg[s] for s in SEGS if s.endswith(("thigh", "shank"))]))
                out[kind] = seg
            res[take][label] = out
            print(take, f"{label:14s}", "ref arms %.1f legs %.1f | phys arms %.1f legs %.1f" % (
                out["reference"]["arms_mean"], out["reference"]["legs_mean"],
                out["physics"]["arms_mean"], out["physics"]["legs_mean"]), flush=True)
    (OUT / "limb_direction_eval.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
