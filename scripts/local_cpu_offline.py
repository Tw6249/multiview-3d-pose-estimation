"""Real four-view RGB -> S1/S1b/S2/S3 -> causal G1 IK -> SONIC physics, on CPU.

Input: public MAMMA outdoor MP4s plus calibrated camera poses (metres, Z-up,
Hamilton wxyz camera-to-world). Intrinsics are scaled to decoded resolution.
Frames share nominal 30 Hz sequence time; per-view decoder timestamps are
checked and saved, but physical exposure synchronization is a dataset assumption.
Output: raw perception, covariance, filtered points, references, physical states,
overlays and report. No motion labels enter the pipeline. Missing observations
remain unknown/expire through the existing filters; no fallback reference is used.
This is offline simulation, with consistency metrics only (no 3D ground truth).
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from g1_imitation.core.constants import load_config
from g1_imitation.core.process_scheduling import disable_power_throttling
from g1_imitation.core.s1_vision_tracker import RTMW27VisionTracker
from g1_imitation.core.s1b_person_association import MultiPersonTargetTracker, extract_person_candidates
from g1_imitation.core.s2_triangulation import MultiViewTriangulator
from g1_imitation.core.s3_temporal_filter import TemporalFilter
from g1_imitation.core.s45c_keypoint_ik import KeypointIKRetargeter, causal_upsample, retarget_sequence
from g1_imitation.core.s6_mujoco_closed_loop import MuJoCoClosedLoopVerifier
from g1_imitation.core.adv_eval import anchor_heading

EDGES = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]


def camera_projection(camera, width, height):
    """Dataset camera-to-world pose -> projection in resized image pixels."""
    fx, fy, cx, cy = camera["intrinsics"]
    source_w, source_h = camera["resolution"]
    K = np.array([[fx*width/source_w, 0, cx*width/source_w],
                  [0, fy*height/source_h, cy*height/source_h], [0,0,1.]])
    quat = np.asarray(camera["rotation_quaternion"])
    if not np.isclose(np.linalg.norm(quat), 1., atol=1e-5):
        raise ValueError("Non-unit calibration quaternion")
    R = Rotation.from_quat(quat[[1,2,3,0]]).as_matrix().T
    t = -R @ np.array(camera["translation"])
    if np.any(np.asarray(camera["distortion_coeffs"]) != 0):
        raise ValueError("This runner requires undistorted input; nonzero distortion is unsupported")
    return K @ np.column_stack([R, t])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data/mamma_cpu_raw")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/local_cpu/offline_mamma")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--target", nargs=3, default=["A001", "440", "440"], metavar=("CAM", "U", "V"),
                    help="Designated person in 1280x720 image coordinates (historical standing target by default)")
    args = ap.parse_args()
    if args.frames < 60 or args.start_frame < 0:
        raise ValueError("Need at least 60 frames and a nonnegative start")
    args.out.mkdir(parents=True, exist_ok=True)
    eco = disable_power_throttling()
    cfg = load_config()
    cameras = yaml.safe_load((args.data / "calibration.yaml").read_text(encoding="utf-8"))["cameras"]
    names = sorted(cameras)
    P = np.stack([camera_projection(cameras[n], 1280, 720) for n in names])
    caps = [cv2.VideoCapture(str(args.data / f"{n}.mp4")) for n in names]
    metadata = {}
    for name, cap in zip(names, caps):
        if not cap.isOpened() or not np.isclose(cap.get(cv2.CAP_PROP_FPS), 30., atol=0.01):
            raise ValueError(f"Invalid/non-30Hz video: {name}")
        metadata[name] = dict(width=cap.get(cv2.CAP_PROP_FRAME_WIDTH), height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                              fps=cap.get(cv2.CAP_PROP_FPS), frames=cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isclose(metadata[name]["width"] / metadata[name]["height"], 16/9):
            raise ValueError("Calibration scaling requires matching image aspect ratio")
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    tracker = RTMW27VisionTracker(device="cpu",
        rtmw_onnx_path=Path(cfg["runtime"]["rtmlib_cache_root"]) / cfg["s1_vision_tracker"]["rtmw_fp32_onnx"])
    if tracker.session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("Expected CPU-only inference")
    association = MultiPersonTargetTracker()
    cam, u, v = args.target
    association.designate_pixel(names.index(cam), float(u), float(v))
    tri = flt = None
    raw, smooth, cov, reproj, keypoints, statuses, boxes, pts_times, timings = [], [], [], [], [], [], [], [], []
    start = time.perf_counter()
    try:
        for k in range(args.frames):
            step_start = time.perf_counter()
            imgs, times = [], []
            for name, cap in zip(names, caps):
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError(f"Decode failed at {name}, frame {args.start_frame+k}")
                times.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.)
                imgs.append(cv2.resize(image, (1280,720)))
            expected = (args.start_frame + k) / 30.
            if not np.allclose(times, expected, atol=0.002):
                raise ValueError(f"Unexpected video timestamps at {k}: {times} versus {expected}")
            candidates = extract_person_candidates(tracker, imgs)
            result = association.step(candidates, P, k/30.)
            if result.selected_now:
                if tri is not None:
                    raise RuntimeError("Unexpected target reselection; reset requires a new output segment")
                s2 = cfg["s2_triangulation"]
                tri = MultiViewTriangulator(c_min=s2["min_2d_confidence"],
                    reproj_outlier_threshold_px=s2["reprojection_error_threshold_px"],
                    max_joint_speed_m_s=s2["max_interframe_displacement_m"] * s2["nominal_fps_hz"])
                flt = TemporalFilter(freq_hz=30., mode=cfg["s3_temporal_filter"]["mode"])
            x = np.full((27,3), np.nan)
            c = np.full((27,3,3), np.nan)
            e = np.full(27, np.nan)
            y = x.copy()
            if tri is not None:
                rec = tri.process_frame(result.keypoints_2d, P, dt_s=1/30.)
                x, c, e = rec.points_3d_m, rec.point_covariances_m2, rec.reprojection_errors_px
                y = flt.process_frame(x, timestamp_s=k/30., covariance_m2=c).points_3d_smooth_m
            raw.append(x.copy()); smooth.append(y.copy()); cov.append(c.copy()); reproj.append(e.copy())
            keypoints.append(result.keypoints_2d.copy()); statuses.append(result.status)
            boxes.append(result.crop_bboxes_xyxy.copy()); pts_times.append(times)
            timings.append(time.perf_counter()-step_start)
            if k % 20 == 0 or k == args.frames-1:
                tiles = []
                for vi, image in enumerate(imgs):
                    for a,b in EDGES:
                        pa,pb = result.keypoints_2d[vi,[a,b]]
                        if pa[2] > .35 and pb[2] > .35:
                            cv2.line(image, tuple(pa[:2].astype(int)), tuple(pb[:2].astype(int)), (0,230,255), 3)
                    if k == 0 and vi == names.index(cam):
                        cv2.circle(image, (int(float(u)),int(float(v))), 12, (0,0,255), 3)
                    cv2.putText(image, f"{names[vi]} f{k} {result.status}", (12,30), cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,255),2)
                    tiles.append(cv2.resize(image,(640,360)))
                sheet = np.vstack([np.hstack(tiles[:2]),np.hstack(tiles[2:])])
                cv2.imencode(".jpg", sheet)[1].tofile(str(args.out / f"overlay_{k:04d}.jpg"))
                print(f"RGB frame {k+1}/{args.frames}: {result.status}, {np.isfinite(y).all(axis=1).sum()}/27 points", flush=True)
    finally:
        for cap in caps:
            cap.release()
    perception_wall = time.perf_counter()-start
    raw, smooth, reproj = map(np.asarray,(raw,smooth,reproj))
    np.savez_compressed(args.out / "perception.npz", raw=raw, smooth=smooth, covariance=cov,
        reprojection_px=reproj, keypoints_2d=keypoints, boxes=boxes, decoder_pts_s=pts_times,
        timestamps_s=np.arange(args.frames)/30., statuses=statuses, projection_matrices=P)
    # Ignore only initial pre-lock unknown frames, never remove failures in a selected segment.
    valid_trunk = np.isfinite(smooth[:,[5,6,11,12]]).all(axis=(1,2))
    if not valid_trunk.any():
        raise RuntimeError("No observable trunk: perception saved; no synthetic/fallback motion generated")
    first = int(np.flatnonzero(valid_trunk)[0])
    rt = KeypointIKRetargeter()
    ts = np.arange(first,args.frames)/30.
    # A short filter hold is useful for its internal state, but it is not a new
    # measurement. Preserve per-point missingness at the command boundary.
    ik_points = smooth[first:].copy()
    ik_points[~np.isfinite(raw[first:]).all(axis=2)] = np.nan
    ret = retarget_sequence(ik_points, ts, rt)
    if not ret["observed"].any():
        raise RuntimeError("No IK observations after calibration; perception saved")
    t50,q,dq,fresh = causal_upsample(ts,ret["q"],ret["dq"],ret["observed"],"hold",rt.lo,rt.hi,dt=.02)
    q,dq,_ = anchor_heading(q,dq)
    phys = MuJoCoClosedLoopVerifier().run_stage_6b_physics(q,dq,ik_dt=.02,control_dt=.02,settle_steps=25)
    np.savez_compressed(args.out / "control.npz", t50=t50,q_ref=q,dq_ref=dq,fresh=fresh,
        qpos_phys=phys.qpos_hist,torques=phys.torques_hist,ik_observed=ret["observed"],ik_points=ik_points)
    finite_e = reproj[:,5:17][np.isfinite(reproj[:,5:17])]
    report = dict(simulation_only=True, source="MAMMA public outdoor example", input_metadata=metadata,
        config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
        target=args.target, frames=args.frames, start_frame=args.start_frame,
        input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      [args.data / "calibration.yaml"] + [args.data / f"{n}.mp4" for n in names]},
        vision_model_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                             [tracker.rtmw_onnx_path,tracker.yolox_onnx_path]},
        time_basis="nominal 30 Hz aligned sequence PTS; hardware exposure synchronization not measured",
        world="dataset calibration, metres, right-handed Z-up; ground z=0 assumption",
        precision="FP32 vision CPU; not bit-equivalent to historical FP16 GPU",
        provider=tracker.session.get_providers(), power_throttling_disabled=eco,
        status_counts=dict(Counter(statuses)), first_observed_trunk_slot=first,
        body_reprojection_median_px=float(np.median(finite_e)) if len(finite_e) else None,
        body_reprojection_p95_px=float(np.percentile(finite_e,95)) if len(finite_e) else None,
        perception_wall_s=perception_wall, perception_slots_per_wall_s=args.frames/perception_wall,
        frame_wall_ms_p50=float(np.median(timings)*1000), frame_wall_ms_p95=float(np.percentile(timings,95)*1000),
        ik_observed_slots=int(ret["observed"].sum()), policy=cfg["s6_closed_loop"]["default_policy"],
        survived=bool(phys.survived), joint_rmse_rad=float(phys.joint_rmse_rad),
        support_foot_skating_m_s=float(phys.support_foot_skating_speed_ms),
        max_tilt_deg=float(phys.max_tilt_deg), min_pelvis_height_m=float(phys.min_pelvis_height_m),
        torques_finite=bool(np.isfinite(phys.torques_hist).all()),
        physical_root_displacement_m=float(np.linalg.norm(phys.qpos_hist[-1,:2]-phys.qpos_hist[0,:2])),
        accuracy_ground_truth_available=False,
        missing_observation_policy="S2 missing points masked to NaN at IK input; IK holds missing limbs/trunk")
    report["quality_acceptance"] = dict(
        survived=report["survived"], torques_finite=report["torques_finite"],
        joint_rmse_below_0_12_rad=report["joint_rmse_rad"] < cfg["s6_closed_loop"]["max_joint_rmse_rad"],
        foot_skating_below_0_04_m_s=report["support_foot_skating_m_s"] < cfg["s6_closed_loop"]["max_foot_skating_m_s"])
    report["all_quality_criteria_pass"] = all(report["quality_acceptance"].values())
    (args.out / "report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
