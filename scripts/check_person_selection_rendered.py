"""Rendered RGB regression and source-only tracking references, without copying RGB."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pose_sim.pipeline import read_jsonl,write_jsonl
from pose_sim.streaming_vision import StreamingVision
from pose_sim.person_selection import CrossViewPersonSelector
from pose_sim.multiview_temporal import MultiViewTemporal
from pose_sim.human_motion import HumanMotionEstimator
from pose_sim.motion_evaluation import compare_methods
from motion_control.humanoid import FullBodyRetargeter,read_body
from scripts.import_mpi3dhp import sha256


def run(manifest_dir,output):
    if output.exists():raise FileExistsError('Preserve rendered evidence')
    output.mkdir(parents=True);manifest=json.loads((manifest_dir/'manifest.json').read_text())
    settings=json.loads((ROOT/'config/vision_motion.json').read_text());profile=json.loads((ROOT/'config/vision.json').read_text())
    processor=None;clips=[];combined={m:[] for m in ('baseline','selected')};truth_all=[];sparse=[]
    try:
        for clip in manifest['scenes']:
            source=Path(clip['source']);cal=json.loads((source/'cameras.json').read_text());cfg=json.loads((source/'config.json').read_text())
            detections=list(read_jsonl(clip['detections']));frames={r['frame_id']:r for r in read_jsonl(source/'frames.jsonl')};selected=[]
            selector=CrossViewPersonSelector(cal['cameras'],cfg)
            import copy
            for original in detections:
                d=copy.deepcopy(original)
                for c,v in d['views'].items():
                    if not 2<=v['num_people_detected']<=4:continue
                    if processor is None:processor=StreamingVision(cal,cfg,profile)
                    processor.cameras=cal['cameras'];im=cv2.imdecode(np.fromfile(source/frames[d['frame_id']]['views'][c]['image'],np.uint8),cv2.IMREAD_COLOR)
                    if im is None:raise ValueError('Unreadable RGB')
                    boxes=processor.body.det_model(im)
                    if len(boxes)!=v['num_people_detected']:raise ValueError('Detector changed')
                    v['person_candidates']=processor.infer_candidates(c,im,boxes)
                    sparse.append(dict(scene=clip['id'],frame_id=d['frame_id'],camera=c,candidates=v['person_candidates']))
                selected.append(selector.update(d))
            rows={};refs=[]
            for method,ds in (('baseline',detections),('selected',selected)):
                fusion=MultiViewTemporal(cal,cfg,structured=False,observation_consistency=True)
                estimator=HumanMotionEstimator({**settings['state'],'state_reacquisition':False});previous=None;states=[]
                retargeter=FullBodyRetargeter(input_filtered=True) if method=='selected' else None
                for d in ds:
                    pose,assigned=fusion.update(d,previous);previous=estimator.update(pose,cal['cameras'],assigned);states.append(previous)
                    if retargeter:
                        ref=dict(frame_id=d['frame_id'],timestamp_ns=d['timestamp_ns'],accepted=False)
                        try:ref.update(retargeter.solve(read_body(previous),d['timestamp_ns']));ref['accepted']=True
                        except (ValueError,KeyError,IndexError,TypeError) as e:ref['reason']=str(e)
                        refs.append(ref)
                rows[method]=states
            dest=output/clip['id'];dest.mkdir();write_jsonl(dest/'humanoid_reference.jsonl',refs)
            (dest/'retargeting.json').write_text(json.dumps(dict(source_kind='rendered_rgb_estimates',ground_truth_used=False,
                source_detections=clip['detections'],source_detections_sha256=sha256(clip['detections']),
                calibration_sha256=sha256(source/'cameras.json'),selected_reference_frames=sum(r['accepted'] for r in refs),
                frames=len(refs),input_state='human_state_v2; observation consistency and current person selection'),indent=2),encoding='utf-8')
            truth=list(read_jsonl(source/'ground_truth.jsonl'))
            if not all(len(v)==len(truth)==clip['frames'] for v in rows.values()):raise ValueError('Rendered coverage mismatch')
            clips.append(dict(id=clip['id'],paired=compare_methods(rows,truth),accepted_references=sum(r['accepted'] for r in refs)))
            for m,rs in rows.items():combined[m].extend({**r,'frame_id':clip['id']+'/'+str(r['frame_id'])} for r in rs)
            truth_all.extend({**r,'frame_id':clip['id']+'/'+str(r['frame_id'])} for r in truth)
            print(clip['id'],'references',sum(r['accepted'] for r in refs),'/',len(refs),flush=True)
        write_jsonl(output/'candidates.jsonl',sparse)
        report=dict(clips=clips,paired=compare_methods(combined,truth_all),limitations='Stylized rendered walking proxy; reference optimization and dynamics remain separate from real-human accuracy.')
        (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    finally:
        if processor:processor.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.manifest,a.output)
