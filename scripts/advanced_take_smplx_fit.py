"""AdvancedTake: EasyMocap SMPL-X fit of the RGB-only 27-point tracks (inspection copy of the S4+S5 fit stage).

Runs in the GMR venv (pipeline_config.json -> s45_gmr_retarget.python). Uses gmr_bridge/worker.py unchanged.
Input  : outputs/advanced_take/s2s3/<tag>/<Take>.npz  raw (T,27,3) metres, Z-up RGB world, 30 Hz slots.
Output : outputs/advanced_take/smplx/<tag>/<Take>.npz (AMASS convention: root_orient, pose_body, trans,
         betas (16,), gender, mocap_frame_rate, observed, grid, fit_residual_m) and fit_summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "g1_imitation" / "gmr_bridge"))
import worker  # noqa: E402

OUT = ROOT / "outputs" / "advanced_take"
FPS = 30.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rgbonly_lag6")
    ap.add_argument("--takes", nargs="+", default=["Take_002", "Take_006", "Take_007", "Take_008", "Take_012"])
    args = ap.parse_args()
    bm = worker.body_model()
    odir = OUT / "smplx" / args.tag
    odir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for take in args.takes:
        z = np.load(OUT / "s2s3" / args.tag / f"{take}.npz")
        amass, observed, res = worker.fit_smplx(z["raw"], bm)
        np.savez(odir / f"{take}.npz", **amass, gender="neutral", mocap_frame_rate=FPS, observed=observed,
                 grid=z["grid"], fit_residual_m=res)
        summary[take] = dict(slots=int(len(res)), observed_slots=int(observed.sum()),
                             residual_mean_mm=float(np.nanmean(res) * 1000),
                             residual_per_joint_mm={n: float(np.nanmean(res[:, i]) * 1000)
                                                    for i, n in enumerate(worker.BODY25_NAMES)},
                             betas=amass["betas"].tolist(), gmr_height_estimate_m=float(1.66 + 0.1 * amass["betas"][0]))
        print(take, "fit residual %.1f mm" % summary[take]["residual_mean_mm"], flush=True)
    (odir / "fit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
