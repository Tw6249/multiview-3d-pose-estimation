"""Sparse CUDA candidates and paired, reference-isolated person selection audit."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pose_sim.pipeline import read_jsonl,write_jsonl
from pose_sim.streaming_vision import StreamingVision
from pose_sim.person_selection import CrossViewPersonSelector
from pose_sim.multiview_temporal import MultiViewTemporal
from pose_sim.human_motion import HumanMotionEstimator
from pose_sim.motion_evaluation import compare_methods,statistics
from scripts.evaluate_mpi3dhp import packets,failure_audit,extra_metrics
from scripts.benchmark_recovery import output_steps
from scripts.import_mpi3dhp import sha256
import hashlib


def fingerprint():
    paths=sorted((ROOT/'pose_sim').glob('*.py'))+[Path(__file__).resolve(),ROOT/'config/vision_motion.json',ROOT/'config/demo.json']
    return hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in paths)).hexdigest()


def cache(dataset,detections,output):
    if output.exists():raise FileExistsError('Preserve candidate evidence')
    output.mkdir(parents=True);processor=None;clips=[];cv2.setNumThreads(1)
    geometry=json.loads((ROOT/'config/demo.json').read_text());profile=json.loads((ROOT/'config/vision.json').read_text())
    try:
        for folder in sorted((dataset/'inputs').iterdir()):
            base=detections/folder.name/'detections.jsonl';rows=list(read_jsonl(base))
            cal=json.loads((folder/'cameras.json').read_text());source=json.loads((folder/'source.json').read_text())
            for video in source['videos'].values():
                if sha256(video['path'])!=video['sha256']:raise ValueError('Video changed')
            if processor is None:processor=StreamingVision(cal,geometry,profile)
            processor.cameras=cal['cameras'];result=[];times=[];frames=0
            for packet,row in zip(packets(folder),rows):
                frames+=1
                if (packet.frame_id,packet.timestamp_ns)!=(row['frame_id'],row['timestamp_ns']):raise ValueError('Exposure mismatch')
                selected={c:v for c,v in row['views'].items() if 2<=v['num_people_detected']<=4}
                if not selected:continue
                start=time.perf_counter();views={}
                for c,view in selected.items():
                    image=packet.frames[c].image_bgr;boxes=processor.body.det_model(image)
                    if len(boxes)!=view['num_people_detected']:raise ValueError('Detector is not reproducible')
                    views[c]=processor.infer_candidates(c,image,boxes)
                result.append(dict(frame_id=row['frame_id'],timestamp_ns=row['timestamp_ns'],views=views))
                times.append((time.perf_counter()-start)*1000)
            if frames!=len(rows) or frames!=source['frames']:raise ValueError('Incomplete RGB or cache')
            write_jsonl(output/(folder.name+'.jsonl'),result)
            clips.append(dict(id=folder.name,frames=frames,candidate_frames=len(result),candidate_views=sum(len(r['views']) for r in result),
                source_sha256=sha256(base),candidates_sha256=sha256(output/(folder.name+'.jsonl')),extra_inference_ms_per_ambiguous_frame=statistics(times)))
            print(folder.name,clips[-1]['candidate_frames'],'ambiguous frames',flush=True)
        (output/'cache.json').write_text(json.dumps(dict(clips=clips,runtime=processor.runtime,reference_access=False),indent=2),encoding='utf-8')
    finally:
        if processor:processor.close()


def replay(dataset,detections,candidates,output,split=None,expected_hash=None):
    code=fingerprint()
    if expected_hash and expected_hash!=code:raise ValueError('Implementation changed after freeze')
    if output.exists():raise FileExistsError('Preserve evaluation evidence')
    output.mkdir(parents=True)
    manifest=json.loads((dataset/'manifest.json').read_text());cache_meta=json.loads((candidates/'cache.json').read_text())
    settings=json.loads((ROOT/'config/vision_motion.json').read_text());cfg=json.loads((ROOT/'config/demo.json').read_text())
    report=dict(implementation_sha256=code,settings=settings,split=split,clips=[]);combined={m:[] for m in ('baseline','selected')};truth_all=[]
    for clip in manifest['clips']:
        if split and clip.get('split')!=split:continue
        cid=clip['id'];cal=json.loads((dataset/'inputs'/cid/'cameras.json').read_text());base=detections/cid/'detections.jsonl';sparse=candidates/(cid+'.jsonl')
        meta=next(c for c in cache_meta['clips'] if c['id']==cid)
        if sha256(base)!=meta['source_sha256'] or sha256(sparse)!=meta['candidates_sha256']:raise ValueError('Cache modified')
        ds=list(read_jsonl(base));proposal={r['frame_id']:r for r in read_jsonl(sparse)};rows={};raw=None;selection=[];times={}
        for method in combined:
            fusion=MultiViewTemporal(cal,cfg,structured=False,observation_consistency=True)
            state=HumanMotionEstimator({**settings['state'],'state_reacquisition':False})
            selector=CrossViewPersonSelector(cal['cameras'],cfg);previous=None;values=[];observed=[];elapsed=[]
            for detection in ds:
                begin=time.perf_counter();d=copy.deepcopy(detection)
                if method=='selected' and d['frame_id'] in proposal:
                    extra=proposal[d['frame_id']]
                    if extra['timestamp_ns']!=d['timestamp_ns']:raise ValueError('Candidate exposure mismatch')
                    for c,ps in extra['views'].items():d['views'][c]['person_candidates']=ps
                    d=selector.update(d)
                    selection.extend(dict(frame_id=d['frame_id'],**e) for e in d['person_selection']['evidence'])
                pose,assigned=fusion.update(d,previous);previous=state.update(pose,cal['cameras'],assigned)
                previous['uncertainty_calibration']=dict(valid_for_domain=False,domain='mpi3dhp_real_rgb',guarantee_claimed=False)
                values.append(previous);observed.append(pose);elapsed.append((time.perf_counter()-begin)*1000)
            rows[method]=values;times[method]=statistics(elapsed)
            if method=='baseline':raw=observed
        truth=list(read_jsonl(dataset/'references'/(cid+'.jsonl')))
        keys=[(r['frame_id'],r['timestamp_ns']) for r in truth]
        if len(ds)!=clip['frames'] or any([(r['frame_id'],r['timestamp_ns']) for r in v]!=keys for v in rows.values()):raise ValueError('Coverage mismatch')
        paired=compare_methods(rows,truth)
        report['clips'].append(dict(id=cid,paired=paired,selection=selection,processing_ms=times,
            details={m:dict(failures=failure_audit(v,raw,truth),output_steps=output_steps(v,truth),extra=extra_metrics(v,truth)) for m,v in rows.items()}))
        for m,rs in rows.items():combined[m].extend({**r,'frame_id':cid+'/'+str(r['frame_id'])} for r in rs)
        truth_all.extend({**r,'frame_id':cid+'/'+str(r['frame_id'])} for r in truth)
        print(cid,{m:dict(mean_mm=v['common_position_error_mm']['mean'],coverage=v['coverage'],success=v['all_target_success']) for m,v in paired['methods'].items()},flush=True)
    if not report['clips']:raise ValueError('No clips selected; this dataset may not define the requested split')
    report['paired']=compare_methods(combined,truth_all)
    report['aggregate_failures']={m:{k:sum(c['details'][m]['failures'][k] for c in report['clips']) for k in ('outputs_over_500mm','current_observations_over_500mm')} for m in combined}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('cache','replay'));p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--detections',type=Path,required=True);p.add_argument('--candidates',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split');p.add_argument('--expected-hash');a=p.parse_args()
    if a.mode=='cache':cache(a.dataset,a.detections,a.output)
    else:replay(a.dataset,a.detections,a.candidates,a.output,a.split,a.expected_hash)
