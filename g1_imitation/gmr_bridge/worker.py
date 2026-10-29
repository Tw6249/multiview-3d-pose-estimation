"""Module S4/S5 replacement worker: EasyMocap SMPL-X fit + GMR retargeting to Unitree G1 (third-party, pinned).

Runs ONLY in the GMR venv (pipeline_config.json -> s45_gmr_retarget.python); the pipeline calls it as a
subprocess through `g1_imitation.core.s45_gmr_retarget`. Nothing here changes either library's objective,
weights or IK parameters; the code only converts layouts and conventions.

Fit    : EasyMocap `pipeline.smpl_from_keypoints3d` (shape from limb lengths, global R/T, body pose, hands, expression;
         default weights, L-BFGS over the whole sequence -> NOT causal). 27 golden points -> body25 (+21+21 hand, +51
         face slots): Neck / MidHip are midpoints of the shoulders / hips; hands observe wrist, index MCP (#5) and
         pinky MCP (#17) only; every other slot and every NaN point has confidence 0. EasyMocap fills fully unobserved
         frames from neighbours; those frames are returned as observed=False.
Retarget: GMR `GeneralMotionRetargeting("smplx" -> robot)` per frame with GMR's own human-height estimate.
Units  : metres, radians, seconds; world right-handed Z-up (ground z = 0 not required by the fit).
Formats: EasyMocap reads SMPL-X v1.0 shapedirs (10 shape + 10 expression); the v1.1 file (300 + 100) is re-packed.
         EasyMocap (Rh, Th about the model origin) -> SMPL-X standard (global_orient about the pelvis, transl);
         verified mesh-identical. Betas are zero-padded 10 -> 16 (AMASS/GMR layout, identical shape).
Unobservable DOF (limb twist, fingers, foot roll) come from EasyMocap's regularisers: a prior, not a measurement.

CLI:   python worker.py points  <in.npz: points (T,27,3), fps>  <out.npz>
       python worker.py bvh     <in.bvh (Nokov naming, Y-up, cm)> <out.npz>
Output npz: qpos (T,36) [root xyz, root quat wxyz, 29 joints in GMR order], joint_names, observed (T,),
       human_height_m, and for `points` also betas (16,), fit_residual_m (T,25) (NaN where unobserved).
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CFG = json.loads((ROOT / "g1_imitation" / "config" / "pipeline_config.json").read_text(encoding="utf-8"))["s45_gmr_retarget"]
sys.path.insert(0, CFG["easymocap_root"])
sys.path.insert(0, CFG["gmr_root"])
import mujoco  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import easymocap.smplmodel.body_model as em_body_model  # noqa: E402
from easymocap.dataset.config import CONFIG  # noqa: E402
from easymocap.pipeline.basic import smpl_from_keypoints3d  # noqa: E402
from easymocap.pipeline.weight import load_weight_pose, load_weight_shape  # noqa: E402
from easymocap.smplmodel.body_model import SMPLlayer  # noqa: E402
from general_motion_retargeting import GeneralMotionRetargeting  # noqa: E402
from general_motion_retargeting.utils.lafan1 import load_bvh_file  # noqa: E402
from general_motion_retargeting.utils.smpl import get_smplx_data_offline_fast, load_smplx_file  # noqa: E402

_load_bodydata = em_body_model.load_bodydata


def _load_bodydata_v11(model_type, model_path, gender):
    data = _load_bodydata(model_type, model_path, gender)
    sd = np.asarray(data["shapedirs"])
    if model_type == "smplx" and sd.shape[-1] == 400:
        data["shapedirs"] = np.concatenate([sd[..., :10], sd[..., 300:310]], axis=-1)
    return data


em_body_model.load_bodydata = _load_bodydata_v11

G27_TO_B25 = {0: 0, 6: 2, 8: 3, 10: 4, 5: 5, 7: 6, 9: 7, 12: 9, 14: 10, 16: 11, 11: 12, 13: 13, 15: 14,
              2: 15, 1: 16, 4: 17, 3: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24}
N_JOINTS = 25 + 21 + 21 + 51
BODY25_NAMES = CONFIG["body25"]["joint_names"]


def to_easymocap(p27: np.ndarray) -> np.ndarray:
    T = len(p27)
    k = np.zeros((T, N_JOINTS, 4))
    ok = np.isfinite(p27).all(-1)
    for g, b in G27_TO_B25.items():
        k[ok[:, g], b, :3] = p27[ok[:, g], g]
        k[ok[:, g], b, 3] = 1.0
    for dst, (a, b) in ((1, (5, 6)), (8, (11, 12))):                      # Neck, MidHip
        m = ok[:, a] & ok[:, b]
        k[m, dst, :3] = 0.5 * (p27[m, a] + p27[m, b])
        k[m, dst, 3] = 1.0
    for base, wrist, idx_mcp, pinky_mcp in ((25, 9, 23, 24), (46, 10, 25, 26)):  # left hand, right hand
        for off, g in ((0, wrist), (5, idx_mcp), (17, pinky_mcp)):
            k[ok[:, g], base + off, :3] = p27[ok[:, g], g]
            k[ok[:, g], base + off, 3] = 1.0
    return k


def body_model():
    md = Path(CFG["smplx_model_dir"])
    return SMPLlayer(str(md / "SMPLX_NEUTRAL.pkl"), model_type="smplx", gender="neutral", device=torch.device("cpu"),
                     regressor_path=str(Path(CFG["easymocap_root"]) / "data" / "smplx" / "J_regressor_body25_smplx.txt"))


def to_amass(bm, params):
    """EasyMocap (Rh, Th about the model origin) -> SMPL-X standard (global_orient about the pelvis, transl)."""
    poses, shapes, Rh, Th = (np.asarray(params[k], np.float64) for k in ("poses", "shapes", "Rh", "Th"))
    if np.abs(poses[:, :3]).max() > 1e-8:
        raise RuntimeError("EasyMocap root pose expected to be zero (global rotation lives in Rh)")
    with torch.no_grad():
        j = bm(poses=np.zeros_like(poses[:1]), shapes=shapes[:1], expression=np.zeros((1, 10)),
               return_verts=False, return_smpl_joints=True, return_tensor=False)[0]
    j0 = j[0]
    trans = Th + Rotation.from_rotvec(Rh).as_matrix() @ j0 - j0
    return dict(root_orient=Rh, pose_body=poses[:, 3:66], trans=trans, betas=np.pad(shapes[0], (0, 6)))


def fit_smplx(points27: np.ndarray, bm=None):
    """(T,27,3) -> (AMASS dict, observed (T,), fit residual (T,25) m)."""
    bm = bm or body_model()
    k = to_easymocap(np.asarray(points27, np.float64))
    observed = k[:, :25, 3].sum(1) > 0.01
    if observed.sum() < 2:
        raise ValueError("fewer than 2 frames with an observed body point")
    args = SimpleNamespace(model="smplx", gender="neutral", opts={}, verbose=False, robust3d=False)
    params = smpl_from_keypoints3d(bm, k, CONFIG["body25"], args, weight_shape=load_weight_shape("smplx", {}),
                                   weight_pose=load_weight_pose("smplx", {}))
    est = bm(return_verts=False, return_tensor=False, **params)
    res = np.linalg.norm(est[:, :25] - k[:, :25, :3], axis=-1)
    res[k[:, :25, 3] == 0] = np.nan
    return to_amass(bm, params), observed, res


def _new_retargeter(src, height):
    return GeneralMotionRetargeting(src_human=src, tgt_robot=CFG["robot"], actual_human_height=height, verbose=False)


def _joint_names(retarget):
    m = retarget.model
    return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]


def retarget_smplx(amass: dict, fps: float):
    """AMASS-convention SMPL-X sequence -> (qpos (T,36), joint names, GMR human height)."""
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "motion.npz"
        np.savez(f, **amass, gender="neutral", mocap_frame_rate=float(fps))
        smplx_data, bm, out, height = load_smplx_file(str(f), str(Path(CFG["smplx_model_dir"]).parent))
        frames, _ = get_smplx_data_offline_fast(smplx_data, bm, out, tgt_fps=fps)
        smplx_data.close()  # GMR leaves the NpzFile open; Windows cannot delete an open file
    if len(frames) != len(amass["trans"]):
        raise RuntimeError(f"GMR resampled {len(frames)} frames for {len(amass['trans'])} inputs")
    r = _new_retargeter("smplx", height)
    q = np.array([np.asarray(r.retarget(fr), np.float64).copy() for fr in frames])
    return q, _joint_names(r), float(height)


def retarget_bvh(path: str):
    """Nokov-convention BVH (Y-up, cm) -> (qpos at the native BVH rate, joint names, GMR human height)."""
    frames, height = load_bvh_file(str(path), format="nokov")
    r = _new_retargeter("bvh_nokov", height)
    q = np.array([np.asarray(r.retarget(fr), np.float64).copy() for fr in frames])
    return q, _joint_names(r), float(height)


def main(argv):
    mode, src, dst = argv[1], argv[2], argv[3]
    t0 = time.perf_counter()
    if mode == "points":
        d = np.load(src)
        pts, fps = np.asarray(d["points"], np.float64), float(d["fps"])
        if pts.ndim != 3 or pts.shape[1:] != (27, 3):
            raise ValueError("points must be (T,27,3)")
        amass, observed, res = fit_smplx(pts)
        t_fit = time.perf_counter() - t0
        q, names, height = retarget_smplx(amass, fps)
        np.savez(dst, qpos=q, joint_names=np.array(names), observed=observed, human_height_m=height,
                 betas=amass["betas"], fit_residual_m=res, body25_names=np.array(BODY25_NAMES), **{
                     f"smplx_{k}": v for k, v in amass.items()},
                 fit_s=t_fit, retarget_s=time.perf_counter() - t0 - t_fit)
    elif mode == "bvh":
        q, names, height = retarget_bvh(src)
        np.savez(dst, qpos=q, joint_names=np.array(names), observed=np.isfinite(q).all(1), human_height_m=height,
                 retarget_s=time.perf_counter() - t0)
    else:
        raise ValueError(f"unknown mode {mode}")


if __name__ == "__main__":
    main(sys.argv)
