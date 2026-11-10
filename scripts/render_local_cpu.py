"""Render saved CPU offline evidence, without rerunning or altering physics.

Top: four RGB views and measured 2D points. Bottom: IK reference (kinematic)
and saved SONIC physical states. Samples use recorded relative times; robot
camera framing is identical for both panels. No physical parameter is changed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import mujoco
import numpy as np
from g1_imitation.core.s6_mujoco_closed_loop import G1PhysicsPlant

EDGES = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=ROOT / "outputs/local_cpu/offline_mamma_full")
    ap.add_argument("--data", type=Path, default=ROOT / "data/mamma_cpu_raw")
    args = ap.parse_args()
    report = json.loads((args.run / "report.json").read_text(encoding="utf-8"))
    with np.load(args.run / "perception.npz") as z:
        kp, ts, status = z["keypoints_2d"], z["timestamps_s"], z["statuses"]
    with np.load(args.run / "control.npz") as z:
        tc, reference, physical = z["t50"], z["q_ref"], z["qpos_phys"]
    names = sorted(report["input_metadata"])
    caps = [cv2.VideoCapture(str(args.data / f"{name}.mp4")) for name in names]
    for cap in caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, report["start_frame"])
    plant = G1PhysicsPlant()
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 135., -15., 3.5
    renderer = mujoco.Renderer(plant.model, width=640, height=360)
    writer = cv2.VideoWriter(str(args.run / "preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),30.,(1280,720))
    if not writer.isOpened():
        raise RuntimeError("Video writer failed")
    try:
        for frame,t in enumerate(ts):
            top = []
            for v, cap in enumerate(caps):
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError("Unexpected decode failure while rendering")
                image = cv2.resize(image,(320,180))
                for a,b in EDGES:
                    if min(kp[frame,v,a,2],kp[frame,v,b,2]) > .35:
                        cv2.line(image,tuple((kp[frame,v,a,:2]/4).astype(int)),
                                 tuple((kp[frame,v,b,:2]/4).astype(int)),(0,230,255),2)
                cv2.putText(image,names[v],(8,20),0,.5,(0,0,255),1)
                top.append(image)
            canvas = np.full((720,1280,3),245,np.uint8)
            canvas[40:220] = np.hstack(top)
            cv2.putText(canvas,f"CPU offline RGB -> G1 | t={t:.2f}s | {status[frame]}",(20,28),0,.7,(30,30,30),2)
            idx = int(np.searchsorted(tc,t+1e-9,side="right")-1)
            if idx >= 0:
                camera.lookat[:] = [*reference[idx,:2],.8]
                for col, q in enumerate((reference[idx],physical[idx])):
                    plant.set_qpos36(q)
                    mujoco.mj_forward(plant.model,plant.data)
                    renderer.update_scene(plant.data,camera=camera)
                    canvas[280:640,col*640:(col+1)*640] = cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
            else:
                cv2.putText(canvas,"Waiting for identity / scale calibration",(200,470),0,1,(50,50,50),2)
            cv2.putText(canvas,"IK reference (kinematic)",(20,260),0,.7,(30,30,30),2)
            cv2.putText(canvas,"SONIC v1.1 saved physics",(660,260),0,.7,(30,30,30),2)
            cv2.putText(canvas,"Simulation only | no 3D ground truth | 30 fps playback, not real-time compute",(20,685),0,.7,(30,30,30),2)
            writer.write(canvas)
            if frame in (100,len(ts)-1):
                cv2.imencode(".jpg",canvas)[1].tofile(str(args.run / f"preview_{frame:04d}.jpg"))
    finally:
        writer.release()
        renderer.close()
        for cap in caps:
            cap.release()
    check = cv2.VideoCapture(str(args.run / "preview.mp4"))
    count = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
    check.release()
    if count != len(ts):
        raise RuntimeError(f"Output frame count {count} != {len(ts)}")
    print(f"Rendered {count} frames: {args.run / 'preview.mp4'}")


if __name__ == "__main__":
    main()
