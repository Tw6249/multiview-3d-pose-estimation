"""Same RGB/crops/association/body backend, paired M versus X body head.

Reference annotations are opened only after both methods' inference per clip.
No duplicate video, full trajectory, altered person selection or detail head.
"""
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from pose_sim.body_head_experiment import load_head,replace_body
from pose_sim.pipeline import read_jsonl
from pose_sim.person_selection import CrossViewPersonSelector
from pose_sim.multiview_temporal import MultiViewTemporal
from pose_sim.human_motion import HumanMotionEstimator
from pose_sim.motion_evaluation import compare_methods,statistics
from scripts.evaluate_mpi3dhp import packets,extra_metrics
from scripts.evaluate_observation_recovery import transitions
from scripts.benchmark_joint_vision import sha


def run(output,split,expected_freeze=None):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);cv2.setNumThreads(1)
    dataset=Path('outputs/public_datasets/mpi_long');cached=Path('outputs/public_datasets/mpi_long_detections');proposals=Path('outputs/public_datasets/mpi_long_person_candidates')
    ids=['s1_long','s2_long'] if split=='development' else ['s4_long','s6_long'];limit=500 if split=='development' else 750
    settings=json.loads(Path('config/vision_motion.json').read_text());geometry=json.loads(Path('config/demo.json').read_text())
    heads={};meta={}
    for method,mode in [('medium','balanced'),('large','performance')]:heads[method],meta[method]=load_head(mode)
    if meta['medium']['sha256']!='5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c':raise ValueError('Baseline model changed')
    files=sorted(Path('pose_sim').glob('*.py'))+[Path(__file__),Path('config/vision_motion.json'),Path('config/demo.json')]
    fingerprint=hashlib.sha256(b''.join(str(p).encode()+p.read_bytes() for p in files)).hexdigest()
    freeze=dict(code_sha256=fingerprint,models=meta,split=split,frames_per_clip=limit,
        protocol='Development S1/S2 first500; freeze before known heldout S4/S6 first750. Same cached person selection and details; body models both run fresh RGB. Reference never used for inference.',
        acceptance='All-target <=100mm and current coverage nondecreasing; mean/P95 and >500mm errors nonincreasing. Inspect per-clip errors, recovery, pose processing cost. No tuning after development.',inputs={})
    if expected_freeze:
        expected=json.loads(Path(expected_freeze).read_text())
        if expected['code_sha256']!=fingerprint or expected['models']!=json.loads(json.dumps(meta)):raise ValueError('Post-freeze implementation/model change')
    for cid in ids:
        folder=dataset/'inputs'/cid;src=json.loads((folder/'source.json').read_text())
        for v in src['videos'].values():
            if sha(v['path'])!=v['sha256']:raise ValueError('Video hash changed')
        freeze['inputs'][cid]={str(p):sha(p) for p in [folder/'source.json',folder/'cameras.json',cached/cid/'detections.jsonl',proposals/(cid+'.jsonl')]}
    (output/'freeze.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8')
    cases=[];combined={m:[] for m in heads};truth_all=[]
    for cid in ids:
        folder=dataset/'inputs'/cid;cal=json.loads((folder/'cameras.json').read_text());ds=list(read_jsonl(cached/cid/'detections.jsonl'))[:limit]
        proposal={r['frame_id']:r for r in read_jsonl(proposals/(cid+'.jsonl'))};selector=CrossViewPersonSelector(cal['cameras'],geometry)
        pipes={m:(MultiViewTemporal(cal,geometry,structured=False,observation_consistency=True),HumanMotionEstimator(settings['state'])) for m in heads}
        results={m:[] for m in heads};timings={m:[] for m in heads};backend={m:[] for m in heads};parity=[];pixel={m:[] for m in heads};scores={m:[] for m in heads};stamps=[]
        iterator=packets(folder)
        try:
            for i,(packet,raw) in enumerate(zip(itertools.islice(iterator,limit),ds)):
                if (packet.frame_id,packet.timestamp_ns)!=(raw['frame_id'],raw['timestamp_ns']):raise ValueError('Exposure mismatch')
                d=copy.deepcopy(raw)
                if d['frame_id'] in proposal:
                    prop=proposal[d['frame_id']]
                    if prop['timestamp_ns']!=d['timestamp_ns']:raise ValueError('Candidate exposure mismatch')
                    for c,v in prop['views'].items():d['views'][c]['person_candidates']=v
                    d=selector.update(d)
                # Alternate order to avoid systematic warm/cache timing advantage.
                for method in (list(heads) if i%2==0 else list(heads)[::-1]):
                    predicted={};begin=time.perf_counter()
                    for c,view in d['views'].items():
                        if view.get('keypoints_px') is None:continue
                        box=np.asarray(view['person_bbox_xyxy_px'],np.float32)
                        k,s=heads[method](packet.frames[c].image_bgr,bboxes=[box]);predicted[c]=(k[0],s[0])
                    timings[method].append((time.perf_counter()-begin)*1000)
                    changed=replace_body(d,predicted,cal['cameras'],meta[method]['sha256'])
                    if method=='medium':
                        for c in predicted:
                            parity.append(float(np.max(np.abs(np.asarray(changed['views'][c]['keypoints_px'][:17])-np.asarray(d['views'][c]['keypoints_px'][:17])))))
                    pixel[method].append([predicted[c][0].tolist() if c in predicted else np.full((17,2),np.nan).tolist() for c in cal['cameras']])
                    scores[method].append([predicted[c][1].tolist() if c in predicted else np.full(17,np.nan).tolist() for c in cal['cameras']])
                    fusion,state=pipes[method];begin=time.perf_counter();pose,assigned=fusion.update(changed,results[method][-1] if results[method] else None)
                    results[method].append(state.update(pose,cal['cameras'],assigned));backend[method].append((time.perf_counter()-begin)*1000)
                stamps.append(packet.timestamp_ns)
                if (i+1)%100==0:print(cid,i+1,'body_head_ms',{m:round(np.mean(timings[m]),2) for m in heads},flush=True)
        finally:iterator.close()
        # Validate replay before reference access; numerical tolerance only.
        if max(parity,default=0)>.05:raise ValueError('Fresh medium head differs from cached crop baseline')
        truth=list(read_jsonl(dataset/'references'/(cid+'.jsonl')))[:limit]
        for m in heads:
            if [(r['frame_id'],r['timestamp_ns']) for r in results[m]]!=[(r['frame_id'],r['timestamp_ns']) for r in truth]:raise ValueError('Scoring mismatch')
            np.savez_compressed(output/(cid+'_'+m+'_body.npz'),keypoints_px=np.array(pixel[m],np.float32),scores=np.array(scores[m],np.float32),timestamp_ns=stamps,camera_ids=list(cal['cameras']))
            combined[m].extend({**r,'frame_id':cid+'/'+str(r['frame_id'])} for r in results[m])
        truth_all.extend({**r,'frame_id':cid+'/'+str(r['frame_id'])} for r in truth)
        case=dict(id=cid,frames=len(truth),paired=compare_methods(results,truth),
            diagnostics={m:dict(kinematics=extra_metrics(results[m],truth),recovery=transitions(results[m],truth)) for m in heads},
            head_ms={m:statistics(timings[m]) for m in heads},backend_ms={m:statistics(backend[m]) for m in heads},medium_cache_max_pixel_difference=max(parity,default=0))
        cases.append(case);print('SCORED',cid,{m:case['paired']['methods'][m]['common_position_error_mm'] for m in heads},flush=True)
    result=dict(freeze=freeze,cases=cases,paired=compare_methods(combined,truth_all),limitation='Offline head-only comparison, not total online latency; cached medium person selection; no full detail/rotation/contact truth.')
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--split',choices=['development','heldout'],default='development');p.add_argument('--expected-freeze',type=Path)
    a=p.parse_args();run(a.output,a.split,a.expected_freeze)
