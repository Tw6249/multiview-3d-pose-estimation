"""AdvancedTake: GMR retargeting to Unitree G1 from the RGB SMPL-X fit (default S4+S5) or from the BVH (reference).

Runs in the GMR venv (pipeline_config.json -> s45_gmr_retarget.python). Uses gmr_bridge/worker.py unchanged.
Sources:
  smplx : outputs/advanced_take/smplx/<tag>/<Take>.npz (advanced_take_smplx_fit.py), 30 Hz slots.
  bvh   : data/advanced_take_raw/bvh/<Take>_Skeleton0.bvh (Chingmu, Nokov naming/units) retargeted at the native
          120 Hz and sampled at the same 30 Hz slots (grid + lag BVH samples). Reference arm only.
Output : outputs/advanced_take/gmr/<source>_<tag>/<Take>.npz
         qpos (T,36) [root xyz m, root quat wxyz, 29 joints rad in GMR order], t_s (T,), observed, joint_names.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "g1_imitation" / "gmr_bridge"))
import worker  # noqa: E402

OUT = ROOT / "outputs" / "advanced_take"
RAW = ROOT / "data" / "advanced_take_raw"
BVH_DT = 1.0 / 120.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["smplx", "bvh"], required=True)
    ap.add_argument("--tag", default="rgbonly_lag6", help="vision tag (smplx fit folder / slot grid + lag)")
    ap.add_argument("--takes", nargs="+", default=["Take_002", "Take_006", "Take_007", "Take_008", "Take_012"])
    args = ap.parse_args()
    odir = OUT / "gmr" / f"{args.source}_{args.tag}"
    odir.mkdir(parents=True, exist_ok=True)
    lag = int(json.loads((OUT / "s2s3" / args.tag / "metrics.json").read_text(encoding="utf-8"))["ref_lag_bvh_samples"])
    summary = {}
    for take in args.takes:
        slots = np.load(OUT / "s2s3" / args.tag / f"{take}.npz")["grid"]
        t0 = time.perf_counter()
        if args.source == "smplx":
            f = np.load(OUT / "smplx" / args.tag / f"{take}.npz", allow_pickle=True)
            amass = {k: f[k] for k in ("root_orient", "pose_body", "trans", "betas")}
            qpos, names, height = worker.retarget_smplx(amass, 30.0)
            if len(qpos) != len(slots):
                raise RuntimeError(f"{take}: {len(qpos)} frames for {len(slots)} slots")
            observed = np.asarray(f["observed"], bool)
            t_s = (slots + lag) * BVH_DT
        else:
            q_all, names, height = worker.retarget_bvh(RAW / "bvh" / f"{take}_Skeleton0.bvh")
            idx = slots + lag
            observed = (idx >= 0) & (idx < len(q_all))
            qpos = np.full((len(slots), 36), np.nan)
            qpos[observed] = q_all[idx[observed]]
            t_s = idx * BVH_DT
        np.savez(odir / f"{take}.npz", qpos=qpos, t_s=t_s, observed=observed, joint_names=np.array(names),
                 human_height_m=height)
        summary[take] = dict(frames=int(len(qpos)), observed=int(observed.sum()), human_height_m=float(height),
                             wall_s=float(time.perf_counter() - t0))
        print(take, summary[take], flush=True)
    (odir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
