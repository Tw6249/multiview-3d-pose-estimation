"""GPU interface audit on development RGB; no reference annotations or scoring.

Retain only summary statistics, hashes, and timings. This is not a MEOM/ProbPose
replication and does not establish any 3D accuracy improvement.
"""
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from pose_sim.body_head_experiment import load_head
from pose_sim.geometry import JOINTS
from pose_sim.inference import select_landmarks
from pose_sim.pipeline import read_jsonl
from pose_sim.person_selection import CrossViewPersonSelector
from pose_sim.spatial_evidence import capture_simcc
from scripts.evaluate_mpi3dhp import packets
from scripts.benchmark_joint_vision import sha


def run(output,frames=32):
    if not 1<=frames<=500:raise ValueError('Development prefix only: 1..500 frames')
    if output.exists():raise FileExistsError(output)
    cv2.setNumThreads(1)
    model,meta=load_head('balanced')
    inputs={};cases=[]
    for cid in ('s1_long','s2_long'):
        folder=ROOT/'outputs/public_datasets/mpi_long/inputs'/cid
        cache=ROOT/'outputs/public_datasets/mpi_long_detections'/cid/'detections.jsonl'
        proposal_path=ROOT/'outputs/public_datasets/mpi_long_person_candidates'/(cid+'.jsonl')
        source=json.loads((folder/'source.json').read_text())
        cal=json.loads((folder/'cameras.json').read_text())
        for v in source['videos'].values():
            if sha(v['path'])!=v['sha256']:raise ValueError('Changed input RGB')
        inputs[cid]={str(p.relative_to(ROOT)):sha(p) for p in (folder/'source.json',folder/'cameras.json',cache,proposal_path)}
        geometry=json.loads((ROOT/'config/demo.json').read_text())
        selector=CrossViewPersonSelector(cal['cameras'],geometry)
        proposals={r['frame_id']:r for r in read_jsonl(proposal_path)}
        rows=list(itertools.islice(read_jsonl(cache),frames))
        timings=[];sizes=[];parity=[];score_parity=[];affine_errors=[]
        sums=[];negative=0;values=0;digest=hashlib.sha256();processed=0
        iterator=packets(folder)
        try:
            for packet,raw in zip(itertools.islice(iterator,frames),rows):
                if (packet.frame_id,packet.timestamp_ns)!=(raw['frame_id'],raw['timestamp_ns']):
                    raise ValueError('Exposure mismatch')
                row=copy.deepcopy(raw)
                if row['frame_id'] in proposals:
                    prop=proposals[row['frame_id']]
                    if prop['timestamp_ns']!=row['timestamp_ns']:raise ValueError('Proposal exposure mismatch')
                    for c,v in prop['views'].items():row['views'][c]['person_candidates']=v
                    row=selector.update(row)
                group_bytes=0
                for c,v in row['views'].items():
                    if v.get('keypoints_px') is None:continue
                    start=time.perf_counter()
                    k,s,e=capture_simcc(model,packet.frames[c],v['person_bbox_xyxy_px'],JOINTS,meta['sha256'])
                    timings.append((time.perf_counter()-start)*1000)
                    group_bytes+=e.response_bytes
                    k,s,_=select_landmarks(k,s,list(range(17)),cal['cameras'][c]['width'],cal['cameras'][c]['height'])
                    parity.append(float(np.max(np.abs(k-np.asarray(v['keypoints_px'][:17])))))
                    score_parity.append(float(np.max(np.abs(s-np.asarray(v['scores'][:17])))))
                    bins=np.stack([e.x_response.argmax(-1),e.y_response.argmax(-1)],axis=-1).astype(float)
                    valid=(e.x_response.max(-1)+e.y_response.max(-1))>0
                    if valid.any():affine_errors.append(float(np.max(np.abs(e.bins_to_pixels(bins)[valid]-k[valid]))))
                    if e.exposure_timestamp_ns!=packet.frames[c].exposure_timestamp_ns:raise ValueError('Retimestamped evidence')
                    for a in (e.x_response,e.y_response):
                        digest.update(a.tobytes());negative+=int((a<0).sum());values+=a.size;sums.extend(a.sum(-1).tolist())
                sizes.append(group_bytes);processed+=1
        finally:iterator.close()
        if processed!=frames:raise ValueError('Incomplete replay')
        if max(parity,default=0)>.05 or max(score_parity,default=0)>1e-6:raise ValueError('Cached baseline parity failed')
        if max(affine_errors,default=0)>1e-4:raise ValueError('Incorrect bin-to-image transform')
        cases.append(dict(id=cid,frames=processed,crops=len(timings),max_pixel_difference=max(parity),
            max_score_difference=max(score_parity),max_affine_pixel_error=max(affine_errors),
            raw_negative_fraction=negative/values,raw_axis_sum_min=min(sums),raw_axis_sum_max=max(sums),
            raw_response_sha256=digest.hexdigest(),max_response_bytes_per_group=max(sizes),
            infer_and_capture_ms=dict(mean=float(np.mean(timings)),p95=float(np.percentile(timings,95)))))
    report=dict(schema_version=1,model=meta,cases=cases,inputs=inputs,
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'pose_sim/spatial_evidence.py',ROOT/'pose_sim/body_head_experiment.py']},
        evidence_semantics=e.semantics,reference_annotations_read=False,
        conclusion='Interface parity only. Raw SimCC responses are not probabilities. No fusion or accuracy claim; defaults unchanged.',
        scope='First N development frames only, four selected crops when available; GPU model inference plus CPU evidence copy. Timings exclude video decode and are not exposure-to-output latency.')
    output.mkdir(parents=True)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(cases,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--frames',type=int,default=32)
    args=p.parse_args();run(args.output.resolve(),args.frames)
