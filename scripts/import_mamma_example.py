"""Convert the public MAMMA four-camera example for local research evaluation.

No ground-truth poses are bundled. Multiple interacting actors make this a
negative/robustness case for the current single-person frontend. Frame-index
alignment follows the published demo, not measured hardware exposure sync.
Source videos and calibration remain in ignored outputs; never redistribute.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.request
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pose_sim.pipeline import write_jsonl
from pose_sim.sensors import validate_calibration

COMMIT='588492f18876e2ed6888b2d26929047cb6b575e7'
RAW=f'https://raw.githubusercontent.com/cuevhv/mamma/{COMMIT}/'
VIDEO_BASE='https://download.is.tue.mpg.de/mamma/assets/download/dataset_examples/pushing_and_lifting_from_ground/videos_light/'


def parse_pinned_calibration(text,width=1280,height=720):
    # Deliberately restricted to the reviewed upstream demo's numeric-array
    # schema, not a general YAML loader and never an eval of downloaded code.
    parts=re.split(r'(?m)^  ([A-D]001):\s*$',text)
    if len(parts)!=9:raise ValueError('Pinned example calibration layout changed')
    cameras={}
    for name,block in zip(parts[1::2],parts[2::2]):
        if 'camera_model: pinhole' not in block or 'distortion_model: radtan' not in block:
            raise ValueError('Unsupported public calibration model')
        def array(key):
            match=re.search(r'\b'+key+r':\s*(\[[^\]]+\])',block)
            if not match:raise ValueError('Missing '+key)
            return np.asarray(json.loads(match.group(1)),float)
        intr=array('intrinsics');resolution=array('resolution');trans=array('translation');quat=array('rotation_quaternion')
        if abs(np.linalg.norm(quat)-1)>.001:raise ValueError('Nonunit calibration quaternion')
        # Published loader specifies T_world_camera, Hamilton [w,x,y,z].
        r=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix().T
        sx,sy=width/resolution[0],height/resolution[1]
        cameras[name]=dict(K=[[intr[0]*sx,0,intr[2]*sx],[0,intr[1]*sy,intr[3]*sy],[0,0,1]],
            R_world_to_camera=r.tolist(),t_world_to_camera_m=(-r@trans).tolist(),
            distortion=array('distortion_coeffs').tolist(),width=width,height=height)
    calibration=dict(schema_version=1,units='metre',world_axes='provider world, Z-up',cameras=cameras)
    validate_calibration(calibration);return calibration


def run(cache,output,start=60,frames=30):
    cache=Path(cache);output=Path(output)
    if output.exists():raise FileExistsError('Preserve public evidence')
    cache.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True)
    for name,path in [('calibration.yaml','configs/examples/calib/iphones_outdoors.yaml'),('LICENSE.txt','LICENSE')]:
        if not (cache/name).exists():
            with urllib.request.urlopen(RAW+path,timeout=30) as response:(cache/name).write_bytes(response.read())
    calibration=parse_pinned_calibration((cache/'calibration.yaml').read_text())
    captures={};source_hashes={}
    for c in calibration['cameras']:
        path=cache/(c+'.mp4')
        if not path.exists():
            with urllib.request.urlopen(VIDEO_BASE+c+'.mp4',timeout=30) as response,path.open('wb') as file:
                while chunk:=response.read(1024*1024):file.write(chunk)
        source_hashes[c]=hashlib.sha256(path.read_bytes()).hexdigest()
        cap=cv2.VideoCapture(str(path))
        if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS)-30)>.01:raise ValueError('Unexpected source video')
        if (cap.get(3),cap.get(4))!=(3840,2160):raise ValueError('Source geometry differs from pinned calibration')
        cap.set(cv2.CAP_PROP_POS_FRAMES,start);captures[c]=cap
        (output/'images'/c).mkdir(parents=True)
    packets=[]
    try:
        for i in range(frames):
            stamp=round((i+start)/30*1e9);packet=dict(frame_id=i,timestamp_ns=stamp,views={})
            for c,cap in captures.items():
                ok,img=cap.read()
                if not ok:raise ValueError('Truncated public video')
                img=cv2.resize(img,(1280,720),interpolation=cv2.INTER_AREA);rel=f'images/{c}/{i:06d}.jpg'
                cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,95])[1].tofile(str(output/rel))
                packet['views'][c]=dict(timestamp_ns=stamp,image=rel)
            packets.append(packet)
    finally:
        for cap in captures.values():cap.release()
    write_jsonl(output/'frames.jsonl',packets)
    (output/'cameras.json').write_text(json.dumps(calibration,indent=2))
    cfg=json.loads((ROOT/'config/demo.json').read_text());cfg.update(fps=30,frames=frames,width=1280,height=720)
    (output/'config.json').write_text(json.dumps(cfg,indent=2))
    manifest=dict(source='MAMMA official public demo',repository_commit=COMMIT,source_video_sha256=source_hashes,
        source_calibration_sha256=hashlib.sha256((cache/'calibration.yaml').read_bytes()).hexdigest(),
        license_url=RAW+'LICENSE',license_scope='non-commercial local research; no redistribution',
        start_frame=start,frames=frames,frame_alignment='published demo frame indices; hardware exposure skew unverified',
        reference_labels_available=False,purpose='real RGB multi-person rejection and occlusion challenge; not quantitative 3D accuracy')
    (output/'source.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start',type=int,default=60);p.add_argument('--frames',type=int,default=30);a=p.parse_args()
    run(a.cache,a.output,a.start,a.frames)
