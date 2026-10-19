"""Isolated current-observation recovery benchmark with reusable RGB detections.

Cache never reads references. Replay scores only after inference, and reports
paired errors, failures, phase and recovery jumps, including missing targets.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pose_sim.pipeline import read_jsonl,write_jsonl
from pose_sim.streaming_vision import StreamingVision
from pose_sim.multiview_temporal import MultiViewTemporal
from pose_sim.human_motion import HumanMotionEstimator
from pose_sim.motion_evaluation import compare_methods,statistics
from scripts.evaluate_mpi3dhp import packets,failure_audit,extra_metrics
from scripts.evaluate_observation_recovery import transitions
from scripts.import_mpi3dhp import sha256


def fingerprint():
    paths=sorted((ROOT/'pose_sim').glob('*.py'))+[Path(__file__).resolve(),
        ROOT/'scripts/evaluate_mpi3dhp.py',ROOT/'scripts/evaluate_observation_recovery.py']
    return hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in paths)).hexdigest()


def output_steps(rows, truth):
    """Status-independent jumps, including reset frames and missing intervals."""
    last={};steps=[];across_gaps=[];resets=[]
    previous_time=None
    for row,gt in zip(rows,truth):
        target=dict(zip(gt['joint_names'],gt['joints_world_m']))
        for n,p,s in zip(row['joint_names'],row['joints_world_m'],row['joint_states']):
            if p is None or target.get(n) is None:continue
            p=np.asarray(p);g=np.asarray(target[n])
            if n in last:
                old_t,old_p,old_g=last[n]
                error=float(np.linalg.norm((p-old_p)-(g-old_g))*1000)
                (steps if old_t==previous_time else across_gaps).append(error)
                if s.get('track_reinitialized'):resets.append(error)
            last[n]=(row['timestamp_ns'],p,g)
        previous_time=row['timestamp_ns']
    return dict(adjacent_output_step_error_mm=statistics(steps),gap_output_step_error_mm=statistics(across_gaps),
                reset_output_step_error_mm=statistics(resets),
                definition='Output displacement minus reference displacement; includes all statuses. Not labelled physical occlusion recovery.')


def cache(dataset,output):
    if output.exists():raise FileExistsError('Preserve existing cached detections')
    output.mkdir(parents=True)
    profile=json.loads((ROOT/'config/vision.json').read_text())
    geometry=json.loads((ROOT/'config/demo.json').read_text())
    cv2.setNumThreads(1);processor=None;clips=[]
    try:
        for folder in sorted((dataset/'inputs').iterdir()):
            cal=json.loads((folder/'cameras.json').read_text())
            source=json.loads((folder/'source.json').read_text())
            for v in source['videos'].values():
                if sha256(v['path'])!=v['sha256']:raise ValueError('Video changed since import')
            if processor is None:processor=StreamingVision(cal,geometry,profile)
            processor.cameras=cal['cameras'];rows=[];elapsed=[]
            for packet in packets(folder):
                start=time.perf_counter();rows.append(processor.detect(packet));elapsed.append((time.perf_counter()-start)*1000)
                if len(rows)%150==0:print(folder.name,len(rows),'/',source['frames'],flush=True)
            dest=output/folder.name;dest.mkdir();write_jsonl(dest/'detections.jsonl',rows)
            clips.append(dict(id=folder.name,frames=len(rows),detections_sha256=sha256(dest/'detections.jsonl'),detector_pose_ms=statistics(elapsed)))
        (output/'cache.json').write_text(json.dumps(dict(profile=profile,geometry=geometry,runtime=processor.runtime,
            clips=clips,reference_access=False,measurement='offline CUDA detection; excludes decode and physical capture latency'),indent=2),encoding='utf-8')
    finally:
        if processor:processor.close()


def replay(dataset,cache_dir,output,split=None,expected_hash=None):
    if output.exists():raise FileExistsError('Preserve existing benchmark evidence')
    code=fingerprint()
    if expected_hash and code!=expected_hash:raise ValueError('Candidate changed after freeze')
    output.mkdir(parents=True)
    settings=json.loads((ROOT/'config/vision_motion.json').read_text())
    geometry=json.loads((ROOT/'config/demo.json').read_text())
    manifest=json.loads((dataset/'manifest.json').read_text())
    cache_manifest=json.loads((cache_dir/'cache.json').read_text()) if (cache_dir/'cache.json').exists() else None
    report=dict(implementation_sha256=code,dataset_sha256=sha256(dataset/'manifest.json'),split=split,
                settings=settings,clips=[],limitations=['Markerless reference, not independent Vicon.','Offline compute timings, no live exposure latency.','S4/S6 are previously used held-out people; do not tune against their results.'])
    combined={m:[] for m in ('baseline','recovery')};truth_all=[]
    for clip in manifest['clips']:
        # import_mpi3dhp preserves the explicit split in each clip entry.
        if split and clip.get('split')!=split:continue
        cid=clip['id'];cal=json.loads((dataset/'inputs'/cid/'cameras.json').read_text())
        detection_path=cache_dir/cid/'detections.jsonl';detections=list(read_jsonl(detection_path))
        if cache_manifest and sha256(detection_path)!=next(c['detections_sha256'] for c in cache_manifest['clips'] if c['id']==cid):
            raise ValueError('Detection cache changed since inference')
        if len(detections)!=clip['frames']:raise ValueError('Incomplete detection cache')
        rows={};times={};raw=None
        for method in combined:
            fusion=MultiViewTemporal(cal,geometry,structured=False,observation_consistency=True)
            state=HumanMotionEstimator({**settings['state'],'state_reacquisition':method=='recovery'})
            previous=None;values=[];observed=[];elapsed=[]
            for detection in detections:
                start=time.perf_counter();pose,assigned=fusion.update(detection,previous)
                previous=state.update(pose,cal['cameras'],assigned)
                previous['uncertainty_calibration']=dict(valid_for_domain=False,domain='mpi3dhp_real_rgb',guarantee_claimed=False)
                elapsed.append((time.perf_counter()-start)*1000);values.append(previous);observed.append(pose)
            rows[method]=values;times[method]=statistics(elapsed)
            if method=='baseline':raw=observed
        truth=list(read_jsonl(dataset/'references'/(cid+'.jsonl')))
        keys=[(r['frame_id'],r['timestamp_ns']) for r in truth]
        if any([(r['frame_id'],r['timestamp_ns']) for r in v]!=keys for v in rows.values()):raise ValueError('Exposure keys mismatch')
        result=dict(id=cid,paired=compare_methods(rows,truth),processing_ms=times,
            details={m:dict(failures=failure_audit(v,raw,truth),transitions=transitions(v,truth),
                output_steps=output_steps(v,truth),extra=extra_metrics(v,truth)) for m,v in rows.items()},
            detections_sha256=sha256(detection_path))
        report['clips'].append(result)
        for method,values in rows.items():combined[method].extend({**r,'frame_id':cid+'/'+str(r['frame_id'])} for r in values)
        truth_all.extend({**r,'frame_id':cid+'/'+str(r['frame_id'])} for r in truth)
        print(cid,{m:dict(mean_mm=v['common_position_error_mm']['mean'],success=v['all_target_success'],coverage=v['coverage']) for m,v in result['paired']['methods'].items()},flush=True)
    if not report['clips']:raise ValueError('No clips selected')
    report['paired']=compare_methods(combined,truth_all)
    report['aggregate_failures']={m:{k:sum(c['details'][m]['failures'][k] for c in report['clips']) for k in ('outputs_over_500mm','current_observations_over_500mm','current_over_500mm_while_raw_under_150mm')} for m in combined}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('cache','replay'))
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path);p.add_argument('--split');p.add_argument('--expected-hash')
    a=p.parse_args()
    if a.mode=='cache':cache(a.dataset,a.output)
    else:
        if a.cache is None:p.error('--cache required for replay')
        replay(a.dataset,a.cache,a.output,a.split,a.expected_hash)
