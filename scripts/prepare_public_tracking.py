"""Public RGB detections -> source-derived G1 references, without reference labels."""
import argparse
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pose_sim.pipeline import read_jsonl,write_jsonl
from pose_sim.person_selection import CrossViewPersonSelector
from pose_sim.multiview_temporal import MultiViewTemporal
from pose_sim.human_motion import HumanMotionEstimator
from motion_control.humanoid import FullBodyRetargeter,read_body
from scripts.import_mpi3dhp import sha256


def run(dataset,detections,candidates,clip,output):
    if output.exists():raise FileExistsError('Preserve tracking input evidence')
    output.mkdir(parents=True);cal=json.loads((dataset/'inputs'/clip/'cameras.json').read_text())
    cfg=json.loads((ROOT/'config/demo.json').read_text());settings=json.loads((ROOT/'config/vision_motion.json').read_text())
    selector=CrossViewPersonSelector(cal['cameras'],cfg);fusion=MultiViewTemporal(cal,cfg,structured=False,observation_consistency=True)
    state=HumanMotionEstimator(settings['state']);retargeter=FullBodyRetargeter(input_filtered=True);previous=None;refs=[]
    sparse={r['frame_id']:r for r in read_jsonl(candidates/(clip+'.jsonl'))}
    for detection in read_jsonl(detections/clip/'detections.jsonl'):
        d=copy.deepcopy(detection)
        if d['frame_id'] in sparse:
            proposal=sparse[d['frame_id']]
            if proposal['timestamp_ns']!=d['timestamp_ns']:raise ValueError('Candidate exposure mismatch')
            for c,ps in proposal['views'].items():d['views'][c]['person_candidates']=ps
        d=selector.update(d);pose,assigned=fusion.update(d,previous);previous=state.update(pose,cal['cameras'],assigned)
        row=dict(frame_id=d['frame_id'],timestamp_ns=d['timestamp_ns'],accepted=False)
        try:row.update(retargeter.solve(read_body(previous),d['timestamp_ns']));row['accepted']=True
        except (ValueError,KeyError,IndexError,TypeError) as e:row['reason']=str(e)
        refs.append(row)
    write_jsonl(output/'humanoid_reference.jsonl',refs)
    meta=dict(source_kind='MPI-INF-3DHP real RGB detections',ground_truth_used=False,frames=len(refs),accepted=sum(r['accepted'] for r in refs),
        input_detections_sha256=sha256(detections/clip/'detections.jsonl'),candidate_sha256=sha256(candidates/(clip+'.jsonl')),
        calibration_sha256=sha256(dataset/'inputs'/clip/'cameras.json'),
        limitation='Offline uncalibrated perception-to-kinematic simulation reference; no hardware permission, complete foot rotations or measured support contacts.')
    (output/'retargeting.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print(clip,meta['accepted'],'/',len(refs),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--detections',type=Path,required=True)
    p.add_argument('--candidates',type=Path,required=True);p.add_argument('--clip',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.dataset,a.detections,a.candidates,a.clip,a.output)
