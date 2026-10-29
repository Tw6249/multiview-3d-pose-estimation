"""Synthetic clutter scene, step 1 of 3: per-frame SMPL-X body meshes for Blender (run in the GMR venv).

Motion source: the EasyMocap SMPL-X fit of one AdvancedTake take (outputs/advanced_take/smplx/rgbonly_lag6/<Take>.npz,
30 Hz, AMASS convention, world right-handed Z-up, metres). The fit itself came from our RGB 27-point track, so the
motion is a real human's but smoothed by the fit; it is only a motion source for rendering, not a reference for the
real take. Hands use the SMPL-X mean (relaxed) pose; expression and jaw are zero.
Output: <out>.npz with verts (T, V, 3) float32 [m, horizontally re-centred so the clip's mean pelvis XY is the origin,
        ground = lowest foot vertex over the clip at z = 0], faces (F, 3), part (V,) = argmax LBS joint of each vertex,
        joints (T, 55, 3) float32 (SMPL-X joints in the same frame), t_s (T,), source info.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import smplx
import torch

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "g1_imitation" / "config" / "pipeline_config.json").read_text(encoding="utf-8"))["s45_gmr_retarget"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", default="Take_002")
    ap.add_argument("--start", type=int, default=0, help="first 30 Hz slot")
    ap.add_argument("--frames", type=int, default=180)
    ap.add_argument("--out", required=True)
    ap.add_argument("--describe", action="store_true", help="print pelvis path / heading per second and exit")
    a = ap.parse_args()
    z = np.load(ROOT / "outputs" / "advanced_take" / "smplx" / "rgbonly_lag6" / f"{a.take}.npz")
    fps = float(z["mocap_frame_rate"])
    if a.describe:
        from scipy.spatial.transform import Rotation
        yaw = Rotation.from_rotvec(z["root_orient"]).apply([0.0, 0.0, 1.0])     # body forward (SMPL +Z) in world
        for k in range(0, len(z["trans"]), int(fps)):
            print(f"t={k / fps:5.1f}s slot {k:4d} pelvis xy {np.round(z['trans'][k, :2], 2)} "
                  f"heading {np.degrees(np.arctan2(yaw[k, 1], yaw[k, 0])):7.1f} deg obs {bool(z['observed'][k])}")
        return
    sl = slice(a.start, a.start + a.frames)
    if not z["observed"][sl].all():
        raise RuntimeError("segment contains unobserved slots; choose another range")
    bm = smplx.create(str(Path(CFG["smplx_model_dir"]).parent), model_type="smplx", gender="neutral", num_betas=10, use_pca=False,
                      ext="npz", batch_size=1)
    T = len(z["trans"][sl])
    t = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32)
    with torch.no_grad():
        o = bm(betas=t(z["betas"][None, :10]).expand(T, -1), global_orient=t(z["root_orient"][sl]),
               body_pose=t(z["pose_body"][sl]), transl=t(z["trans"][sl]),
               left_hand_pose=torch.zeros(T, 45), right_hand_pose=torch.zeros(T, 45), jaw_pose=torch.zeros(T, 3),
               leye_pose=torch.zeros(T, 3), reye_pose=torch.zeros(T, 3), expression=torch.zeros(T, 10),
               return_verts=True)
    verts, joints = o.vertices.numpy(), o.joints.numpy()[:, :55]
    shift = np.r_[-joints[:, 0, :2].mean(axis=0), -verts[:, :, 2].min()]
    verts += shift
    joints += shift
    part = bm.lbs_weights.numpy().argmax(axis=1)
    np.savez_compressed(a.out, verts=verts.astype(np.float32), faces=bm.faces.astype(np.int32), part=part.astype(np.int16),
                        joints=joints.astype(np.float32), t_s=np.arange(T) / fps, fps=fps, take=a.take, start_slot=a.start,
                        shift_m=shift)
    print(f"{a.take} slots {a.start}..{a.start + T - 1}: {T} frames at {fps} Hz, pelvis z {joints[:, 0, 2].min():.2f}-"
          f"{joints[:, 0, 2].max():.2f} m, height {verts[0, :, 2].max():.2f} m -> {a.out}")


if __name__ == "__main__":
    main()
