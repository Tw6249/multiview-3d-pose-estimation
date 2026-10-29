"""OFFLINE scoring of the RGB-only world against the BVH (the BVH never feeds back into perception or control).

Inputs : s2s3/<tag>/<Take>.npz produced with cameras_rgbonly.json (raw 27-pt tracks + BVH12 read with the lag).
Output : s2s3/<tag>/registration_vs_bvh.json
Metrics (12 body joints, held-out takes):
  - conv_aligned_mpjpe_mm : error after aligning ONLY the conventions RGB cannot know (world yaw + horizontal origin,
                            3 DoF, one per take). Gravity, ground height and metric scale errors stay in.
  - similarity_*          : the full 7-DoF take similarity, decomposed into scale ratio, up-axis tilt and
                            vertical offset (what the RGB-only registration got wrong).
  - procrustes_mpjpe_mm   : shape error after the 7-DoF similarity (same as the s2s3 metrics).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from g1_imitation.core import adv_eval as ae  # noqa: E402

OUT = ROOT / "outputs" / "advanced_take"


def yaw_xy_align(p, g):
    """Least-squares rotation about z + xy translation mapping p onto g (both (N,3))."""
    pc, gc = p[:, :2] - p[:, :2].mean(0), g[:, :2] - g[:, :2].mean(0)
    H = pc.T @ gc
    a = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    t = np.r_[g[:, :2].mean(0) - (R[:2, :2] @ p[:, :2].mean(0)), 0.0]
    return R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rgbonly_lag6")
    ap.add_argument("--takes", nargs="+", default=["Take_006", "Take_007", "Take_008"])
    args = ap.parse_args()
    res = {}
    for take in args.takes:
        z = np.load(OUT / "s2s3" / args.tag / f"{take}.npz")
        p, g = z["raw"][:, 5:17], z["bvh12"]
        full = np.isfinite(p).all((1, 2)) & np.isfinite(g).all((1, 2))
        P, G = p[full].reshape(-1, 3), g[full].reshape(-1, 3)
        R, t = yaw_xy_align(P, G)
        e_conv = np.linalg.norm((R @ P.T).T + t - G, axis=1)
        s, Rs, ts = ae.umeyama(P, G)
        e_sim = np.linalg.norm((s * (Rs @ P.T)).T + ts - G, axis=1)
        tilt = float(np.degrees(np.arccos(np.clip((Rs @ [0, 0, 1])[2], -1, 1))))
        ank_v = np.nanpercentile(p[full][:, 10:12, 2], 5)
        ank_b = np.nanpercentile(g[full][:, 10:12, 2], 5)
        per_joint = np.linalg.norm(((R @ P.T).T + t - G).reshape(-1, 12, 3), axis=2).mean(0) * 1000
        res[take] = dict(frames=int(full.sum()), conv_aligned_mpjpe_mm=float(e_conv.mean() * 1000),
                         conv_aligned_per_joint_mm=dict(zip(ae.JOINTS12, map(float, per_joint))),
                         conv_aligned_mean_z_error_mm=float(np.mean(((R @ P.T).T + t - G)[:, 2]) * 1000),
                         procrustes_mpjpe_mm=float(e_sim.mean() * 1000),
                         similarity_scale_bvh_over_rgb=float(s), similarity_up_tilt_deg=tilt,
                         ankle_z_p5_rgb_m=float(ank_v), ankle_z_p5_bvh_m=float(ank_b))
        print(take, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[take].items() if k != "conv_aligned_per_joint_mm"})
    (OUT / "s2s3" / args.tag / "registration_vs_bvh.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
