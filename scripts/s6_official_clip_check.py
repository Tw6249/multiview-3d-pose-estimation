"""Check the SONIC integration on NVIDIA's own example reference clips (no GMR, no perception).

Input : a directory of official clips, NVlabs/GR00T-WholeBodyControl@45c03fac gear_sonic_deploy/reference/example/<clip>/
        {joint_pos, joint_vel, body_pos, body_quat, body_lin_vel, body_ang_vel}.csv: 50 Hz, joints in IsaacLab (=SONIC
        policy) order, body 0 = pelvis, quaternions wxyz, world frame. Downloaded from GitHub LFS, treated as data.
Output: outputs/advanced_take/official_clips/report[_<tag>].json - Stage 6B per clip x controller x preview.
Plant: default = configured S6 plant; --mjcf/--physics-dt/--effort-limit swap it (e.g. the official SONIC sim2sim
scene gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml, 200 Hz, actuator-range torque limits).
Purpose: if SONIC drifts / fails on its own training-style clips with its native window, the fault is in our
integration (plant, observation, PD), not in GMR or perception.
Travel is compared in each trajectory's own initial heading frame (the robot starts at yaw 0, the clips do not).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1_imitation.core.constants import G1_29_JOINTS, load_config  # noqa: E402
from g1_imitation.core.s6_mujoco_closed_loop import (  # noqa: E402
    MuJoCoClosedLoopVerifier, SonicPolicy)

OUT = ROOT / "outputs" / "advanced_take" / "official_clips"
DT = 0.02
GRID = {"sonic_v11": (0.0, 0.9), "sonic_low_latency": (0.0, 0.18)}


def load_clip(d: Path, policy_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """-> q (T,36) [pelvis xyz, wxyz, 29 SDK-order joints], dq (T,35) [lin world, ang body, joints]."""
    rd = lambda n: np.loadtxt(d / f"{n}.csv", delimiter=",", skiprows=1, ndmin=2)
    jp, jv = rd("joint_pos"), rd("joint_vel")
    pos, quat = rd("body_pos")[:, :3], rd("body_quat")[:, :4]
    lin, ang = rd("body_lin_vel")[:, :3], rd("body_ang_vel")[:, :3]
    idx = [G1_29_JOINTS.index(n) for n in policy_names]            # policy column j -> SDK index
    T = len(jp)
    q, dq = np.zeros((T, 36)), np.zeros((T, 35))
    q[:, :3], q[:, 3:7] = pos, quat / np.linalg.norm(quat, axis=1, keepdims=True)
    q[:, 7:][:, idx] = jp
    dq[:, :3] = lin
    dq[:, 3:6] = Rotation.from_quat(q[:, [4, 5, 6, 3]]).inv().apply(ang)
    dq[:, 6:][:, idx] = jv
    if not np.isfinite(q).all() or not np.isfinite(dq).all():
        raise ValueError(f"non-finite clip {d.name}")
    return q, dq


def local_path(xy: np.ndarray, yaw0: float) -> np.ndarray:
    return Rotation.from_euler("z", -yaw0).apply(np.c_[xy - xy[0], np.zeros(len(xy))])[:, :2]


def yaw(qwxyz: np.ndarray) -> float:
    f = Rotation.from_quat(qwxyz[[1, 2, 3, 0]]).apply([1.0, 0.0, 0.0])
    return float(np.arctan2(f[1], f[0]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip_root")
    ap.add_argument("--mjcf", default=None)
    ap.add_argument("--physics-dt", type=float, default=0.001)
    ap.add_argument("--effort-limit", default="common_kp_capped", choices=("common_kp_capped", "actuator_range"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cfg = load_config()["s6_closed_loop"]
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for d in sorted(Path(a.clip_root).iterdir()):
        q, dq = load_clip(d, cfg["sonic_v11"]["policy_joint_names"])
        ref_path = local_path(q[:, :2], yaw(q[0, 3:7]))
        r_clip = report.setdefault(d.name, {"frames": len(q), "ref_travel_m": float(np.linalg.norm(ref_path[-1]))})
        for ctrl, previews in GRID.items():
            ver = MuJoCoClosedLoopVerifier(mjcf_path=a.mjcf, policy=SonicPolicy(cfg[ctrl]),
                                           physics_dt=a.physics_dt, effort_limit=a.effort_limit)
            for p in previews:
                r = ver.run_stage_6b_physics(q, dq, ik_dt=DT, control_dt=DT, settle_steps=25, preview_s=p)
                ph = local_path(r.qpos_hist[:, :2], yaw(r.qpos_hist[0, 3:7]))
                dev = np.linalg.norm(ph - ref_path[:len(ph)], axis=1)
                r_clip[f"{ctrl}@{p:.2f}"] = dict(
                    survived=r.survived, joint_rmse_rad=r.joint_rmse_rad, skate=r.support_foot_skating_speed_ms,
                    max_tilt_deg=r.max_tilt_deg, path_dev_final_m=float(dev[-1]), path_dev_p95_m=float(np.percentile(dev, 95)),
                    phys_end_local_m=ph[-1].tolist(), ref_end_local_m=ref_path[len(ph) - 1].tolist(),
                    airborne=float(((r.left_fz_hist < 1) & (r.right_fz_hist < 1)).mean()))
                print(d.name, ctrl, p, {k: (round(v, 3) if isinstance(v, float) else v)
                                        for k, v in r_clip[f"{ctrl}@{p:.2f}"].items()}, flush=True)
    report["_plant"] = dict(mjcf=str(ver.plant.mjcf_path), physics_dt=a.physics_dt, effort_limit=a.effort_limit)
    (OUT / f"report{'_' + a.tag if a.tag else ''}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
