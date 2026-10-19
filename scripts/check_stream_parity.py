"""Check concurrent detector execution on identical RGB, separately from live timing."""
import argparse
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pose_sim.streaming_vision import StreamingVision
from pose_sim.online import ExposurePacket
from pose_sim.sensors import ExposureFrame
from pose_sim.pipeline import read_jsonl


def run(source,output,pose_workers=1):
    source=Path(source);output=Path(output)
    if output.exists():raise FileExistsError(output)
    cal=json.loads((source/'cameras.json').read_text());cfg=json.loads((source/'config.json').read_text())
    profile=json.loads((ROOT/'config/vision.json').read_text());engine=StreamingVision(cal,cfg,profile,detector_workers=4,pose_workers=pose_workers)
    pool=engine.pool;pose_pool=engine.pose_pool;delta=score_delta=0.;count=0;counts_match=True;coverage_match=True
    try:
        for row in read_jsonl(source/'frames.jsonl'):
            t=time.perf_counter_ns();frames={}
            for c,v in row['views'].items():
                im=cv2.imdecode(np.fromfile(source/v['image'],np.uint8),cv2.IMREAD_COLOR)
                frames[c]=ExposureFrame(c,t,time.perf_counter_ns(),'host_monotonic',im)
            packet=ExposurePacket(row['frame_id'],t,frames)
            engine.pool=None;engine.pose_pool=None;serial=engine.detect(packet)
            engine.pool=pool;engine.pose_pool=pose_pool;parallel=engine.detect(packet)
            for c in frames:
                a,b=serial['views'][c],parallel['views'][c]
                counts_match &= a['num_people_detected']==b['num_people_detected']
                coverage_match &= (a['keypoints_px'] is None)==(b['keypoints_px'] is None)
                if a['keypoints_px'] is not None and b['keypoints_px'] is not None:
                    delta=max(delta,float(np.max(np.abs(np.array(a['keypoints_px'])-b['keypoints_px']))))
                    score_delta=max(score_delta,float(np.max(np.abs(np.array(a['scores'])-b['scores']))))
            count+=1
    finally:
        engine.pool=pool;engine.pose_pool=pose_pool;engine.close()
    result=dict(frames=count,max_keypoint_difference_px=delta,max_score_difference=score_delta,
        person_counts_identical=bool(counts_match),coverage_identical=bool(coverage_match),
        passed=bool(counts_match and coverage_match and delta<.01 and score_delta<1e-5),
        scope='same hybrid model; body remains batch 1; four detector workers',pose_workers=pose_workers)
    output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pose-workers',type=int,default=1,choices=(1,2,4))
    a=p.parse_args();run(a.source,a.output,a.pose_workers)
