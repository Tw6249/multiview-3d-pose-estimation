"""Run causal joint reconstruction/state on recorded detections; no RGB inference or truth input."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pose_sim.pipeline import read_jsonl,write_jsonl
from pose_sim.multiview_temporal import MultiViewTemporal
from pose_sim.human_motion import HumanMotionEstimator
from pose_sim.motion_evaluation import statistics


def run(source,detections,output,structured=False,ground_height=None,history=False,observation_consistency=None):
    source=Path(source);output=Path(output);detections=Path(detections)
    if output.exists():raise FileExistsError('Use a new output directory')
    output.mkdir(parents=True)
    cal=json.loads((source/'cameras.json').read_text());cfg=json.loads((source/'config.json').read_text())
    settings=json.loads((ROOT/'config/vision_motion.json').read_text())
    consistency=settings.get('observation_consistency',False) if observation_consistency is None else observation_consistency
    fusion=MultiViewTemporal(cal,cfg,structured=structured,history=history,observation_consistency=consistency)
    state=HumanMotionEstimator({**settings['state'],'ground_height_m':ground_height})
    poses=[];states=[];assigned_rows=[];previous=None;times=[]
    for detection in read_jsonl(detections):
        start=time.perf_counter();pose,assigned=fusion.update(detection,previous)
        previous=state.update(pose,cal['cameras'],assigned)
        previous['uncertainty_calibration']=dict(valid_for_domain=False,domain='unvalidated_rgb',guarantee_claimed=False)
        poses.append(pose);states.append(previous);assigned_rows.append(assigned);times.append((time.perf_counter()-start)*1000)
    write_jsonl(output/'poses3d.jsonl',poses)
    sha=hashlib.sha256((output/'poses3d.jsonl').read_bytes()).hexdigest()
    for row in states:row['source_poses_sha256']=sha
    write_jsonl(output/'states3d.jsonl',states);write_jsonl(output/'assigned_detections.jsonl',assigned_rows)
    report=dict(input_detections_sha256=hashlib.sha256(detections.read_bytes()).hexdigest(),frames=len(states),
        processing_ms=statistics(times),structured=structured,association_history=history,ground_height_m=ground_height,
        observation_consistency=consistency,state_reacquisition=settings['state'].get('state_reacquisition',False),
        correction_count=sum(len(r['association'].get('corrections',[])) for r in poses),
        unresolved_association_count=sum(len(r['association'].get('ambiguities',[])) for r in poses))
    (output/'refinement.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--detections',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--structured',action='store_true');p.add_argument('--ground-height',type=float)
    p.add_argument('--history',action='store_true',help='Experimental, rejected as default after RGB regression')
    p.add_argument('--no-observation-consistency',action='store_true',help='Replay the prior observation baseline')
    a=p.parse_args();run(a.source,a.detections,a.output,a.structured,a.ground_height,a.history,False if a.no_observation_consistency else None)
