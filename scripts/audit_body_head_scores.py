"""Additional all-output tail audit from frozen body-head pixel caches."""
import argparse
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from pose_sim.body_head_experiment import replace_body
from pose_sim.pipeline import read_jsonl
from pose_sim.multiview_temporal import MultiViewTemporal
from pose_sim.human_motion import HumanMotionEstimator
from pose_sim.person_selection import CrossViewPersonSelector
from pose_sim.motion_evaluation import statistics
from scripts.benchmark_joint_vision import sha


def run(root):
    root=Path(root);dest=root/'tail_audit.json'
    if dest.exists():raise FileExistsError(dest)
    freeze=json.loads((root/'freeze.json').read_text());limit=freeze['frames_per_clip']
    settings=json.loads(Path('config/vision_motion.json').read_text());cfg=json.loads(Path('config/demo.json').read_text());cases=[]
    for cid,inputs in freeze['inputs'].items():
        for path,expected in inputs.items():
            if sha(path)!=expected:raise ValueError('Input changed')
        folder=Path('outputs/public_datasets/mpi_long/inputs')/cid;cal=json.loads((folder/'cameras.json').read_text())
        ds=list(read_jsonl(Path('outputs/public_datasets/mpi_long_detections')/cid/'detections.jsonl'))[:limit]
        proposals={r['frame_id']:r for r in read_jsonl(Path('outputs/public_datasets/mpi_long_person_candidates')/(cid+'.jsonl'))};selector=CrossViewPersonSelector(cal['cameras'],cfg);selected=[]
        for row in ds:
            d=copy.deepcopy(row)
            if d['frame_id'] in proposals:
                p=proposals[d['frame_id']]
                if p['timestamp_ns']!=d['timestamp_ns']:raise ValueError('Candidate clock mismatch')
                for c,v in p['views'].items():d['views'][c]['person_candidates']=v
                d=selector.update(d)
            selected.append(d)
        results={}
        for method in freeze['models']:
            z=np.load(root/(cid+'_'+method+'_body.npz'));fusion=MultiViewTemporal(cal,cfg,structured=False,observation_consistency=True);state=HumanMotionEstimator(settings['state']);rows=[];previous=None
            if len(z['timestamp_ns'])!=len(ds):raise ValueError('Cache count mismatch')
            for i,d in enumerate(selected):
                if int(z['timestamp_ns'][i])!=d['timestamp_ns']:raise ValueError('Cache exposure mismatch')
                predictions={str(c):(z['keypoints_px'][i,j],z['scores'][i,j]) for j,c in enumerate(z['camera_ids']) if np.isfinite(z['keypoints_px'][i,j]).all()}
                changed=replace_body(d,predictions,cal['cameras'],freeze['models'][method]['sha256']);pose,assigned=fusion.update(changed,previous);previous=state.update(pose,cal['cameras'],assigned);rows.append(previous)
            results[method]=rows
        truth=list(read_jsonl(Path('outputs/public_datasets/mpi_long/references')/(cid+'.jsonl')))[:limit];metrics={}
        for method,rows in results.items():
            error=[];current=[];examples=[]
            for row,g in zip(rows,truth):
                if row['timestamp_ns']!=g['timestamp_ns']:raise ValueError('Reference clock mismatch')
                p=dict(zip(row['joint_names'],row['joints_world_m']));states=dict(zip(row['joint_names'],row['joint_states']))
                for n,target in zip(g['joint_names'],g['joints_world_m']):
                    e=float(np.linalg.norm(np.array(p[n])-target)) if p.get(n) is not None else np.nan;error.append(e);current.append(states.get(n,{}).get('status')=='filtered')
                    if e>.5:examples.append(dict(frame_id=row['frame_id'],joint=n,error_m=e,current=current[-1]))
            error=np.array(error);current=np.array(current);metrics[method]=dict(targets=len(error),available=int(np.isfinite(error).sum()),over_500mm=int((error>.5).sum()),current_over_500mm=int(((error>.5)&current).sum()),all_error_m=statistics(error[np.isfinite(error)].tolist()),worst_examples=sorted(examples,key=lambda x:x['error_m'],reverse=True)[:8])
            np.savez_compressed(root/(cid+'_'+method+'_errors.npz'),error_m=error,current=current)
        cases.append(dict(id=cid,methods=metrics));print(cid,{m:v['over_500mm'] for m,v in metrics.items()},flush=True)
    result=dict(cases=cases,scope='Additional scoring only; no estimator parameter changes; frozen RGB head pixel cache.')
    dest.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)
