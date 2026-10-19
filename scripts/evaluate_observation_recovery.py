"""Replay immutable detections through modular ablations; score only afterwards.

No rendering or duplicate RGB copies. Preserve candidate states and compact
ablation evidence; baseline states are referenced by path and SHA-256.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pose_sim.pipeline import read_jsonl, write_jsonl
from pose_sim.multiview_temporal import MultiViewTemporal
from pose_sim.human_motion import HumanMotionEstimator
from pose_sim.motion_evaluation import compare_methods, statistics
from scripts.evaluate_mpi3dhp import failure_audit, extra_metrics
from scripts.import_mpi3dhp import sha256


def implementation_hash():
    paths = sorted((ROOT/'pose_sim').glob('*.py'))+[Path(__file__).resolve()]
    return hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in paths)).hexdigest()


def run_rendered(baseline, output, selected_method='candidate'):
    """Regress the existing rendered RGB baseline without duplicating artifacts."""
    if output.exists():
        raise FileExistsError('Preserve existing report')
    manifest=json.loads((baseline/'manifest.json').read_text(encoding='utf-8'))
    settings=json.loads((ROOT/'config/vision_motion.json').read_text())
    combined={'baseline':[],'candidate':[]};truth_all=[];clips=[]
    for clip in manifest['scenes']:
        source=Path(clip['source']);cal=json.loads((source/'cameras.json').read_text())
        cfg=json.loads((source/'config.json').read_text())
        fusion=MultiViewTemporal(cal,cfg,structured=False,observation_consistency=True)
        state=HumanMotionEstimator({**settings['state'],'state_reacquisition':selected_method=='candidate'})
        previous=None;candidate=[]
        for detection in read_jsonl(clip['detections']):
            pose,assigned=fusion.update(detection,previous)
            previous=state.update(pose,cal['cameras'],assigned);candidate.append(previous)
        old=list(read_jsonl(baseline/clip['result']/'states3d.jsonl'))
        truth=list(read_jsonl(source/'ground_truth.jsonl'))
        if not len(old)==len(candidate)==len(truth)==clip['frames']:
            raise ValueError('Rendered regression coverage mismatch')
        rows={'baseline':old,'candidate':candidate}
        clips.append(dict(id=clip['id'],paired=compare_methods(rows,truth),
                          detections_sha256=sha256(clip['detections'])))
        for m,rs in rows.items():
            combined[m].extend({**r,'frame_id':clip['id']+'/'+str(r['frame_id'])} for r in rs)
        truth_all.extend({**r,'frame_id':clip['id']+'/'+str(r['frame_id'])} for r in truth)
        print('rendered',clip['id'],flush=True)
    report=dict(implementation_sha256=implementation_hash(),selected_method=selected_method,clips=clips,paired=compare_methods(combined,truth_all),
                limitation='Previously inspected rendered RGB, 12 rig body proxies; not new real-human validation.')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def transitions(rows, truth):
    """Recovery is estimator-status based, not a labelled physical occlusion event."""
    excess=[]; jumps=[]; resets=0; disagreements=[]; gap_excess=[]; last_available={}
    previous=None
    for row,gt in zip(rows,truth):
        current=dict(zip(row['joint_names'],row['joints_world_m']))
        states=dict(zip(row['joint_names'],row['joint_states']))
        target=dict(zip(gt['joint_names'],gt['joints_world_m']))
        for n,s in states.items():
            if n not in target:
                continue
            resets+=int(s.get('track_reinitialized',False))
            if current.get(n) is not None:
                if n in last_available:
                    old_t,old_p,old_gt=last_available[n]
                    if old_t != (previous[3] if previous else None):
                        gap_excess.append(float(np.linalg.norm((np.array(current[n])-old_p)-
                            (np.array(target[n])-old_gt))*1000))
                last_available[n]=(row['timestamp_ns'],current[n],target[n])
            if not previous or n not in target or current.get(n) is None:
                continue
            old,oldstates,oldgt,_=previous
            if old.get(n) is None or n not in oldgt:
                continue
            if s['status']=='filtered' and oldstates[n]['status']!='filtered':
                step=np.array(current[n])-old[n]
                excess.append(float(np.linalg.norm(step-(np.array(target[n])-oldgt[n]))*1000))
                jumps.append(float(np.linalg.norm(step)*1000))
            if s.get('track_reinitialized'):
                disagreements.append(float(np.linalg.norm(np.array(current[n])-old[n])*1000))
        previous=(current,states,target,row['timestamp_ns'])
    return dict(recovery_excess_step_mm=statistics(excess), recovery_step_mm=statistics(jumps),
                track_reinitializations=resets,reinitialization_displacement_mm=statistics(disagreements),
                gap_reappearance_excess_step_mm=statistics(gap_excess),
                definition='Transition from non-filtered to filtered; no physical occlusion labels available.')


def run(dataset, baseline, output, expected_hash=None, selected_method='candidate'):
    fingerprint=implementation_hash()
    if expected_hash and expected_hash != fingerprint:
        raise ValueError('Candidate changed after freeze')
    if output.exists():
        raise FileExistsError('Preserve evidence; output already exists')
    manifest=json.loads((dataset/'manifest.json').read_text())
    inference=json.loads((baseline/'inference.json').read_text())
    settings=copy.deepcopy(inference['motion_settings'])
    geometry=inference['geometry_settings']
    output.mkdir(parents=True)
    methods=('baseline','consistency','recovery','candidate')
    report=dict(schema_version=1,implementation_sha256=fingerprint,selected_method=selected_method,
                dataset_manifest_sha256=sha256(dataset/'manifest.json'),
                baseline_inference_sha256=sha256(baseline/'inference.json'),settings=settings,
                ablations={'baseline': [False,False], 'consistency':[True,False],
                           'recovery':[False,True], 'candidate':[True,True]},clips=[],
                limitations=['Project subset, not official benchmark; pretrained data overlap unknown.',
                             'Markerless reference, no independent Vicon truth.',
                             'Heuristic uncertainty is not calibrated; no physical latency or robot validation.'])
    (output/'freeze.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    combined={m:[] for m in methods};all_truth=[];elapsed={m:[] for m in methods if m!='baseline'}
    audits={m:dict(outputs_over_500mm=0,current_observations_over_500mm=0,
                   current_over_500mm_while_raw_under_150mm=0) for m in methods}
    for clip in manifest['clips']:
        cid=clip['id'];source=baseline/cid;cal=json.loads((dataset/'inputs'/cid/'cameras.json').read_text())
        observations=list(read_jsonl(source/'detections.jsonl'))
        if len(observations)!=clip['frames']:
            raise ValueError('Detection coverage mismatch')
        rows={'baseline':list(read_jsonl(source/'current.jsonl'))}
        durations={}
        for method in methods[1:]:
            consistency,recovery=report['ablations'][method]
            fusion=MultiViewTemporal(cal,geometry,association=settings['association'],
                structured=settings['structured_reprojection'],auxiliary=settings['auxiliary_association'],
                history=settings['association_history'],observation_consistency=consistency)
            state=HumanMotionEstimator({**settings['state'],'state_reacquisition':recovery})
            previous=None;values=[];times=[]
            for observation in observations:
                start=time.perf_counter()
                pose,assigned=fusion.update(observation,previous)
                previous=state.update(pose,cal['cameras'],assigned)
                previous['uncertainty_calibration']=dict(valid_for_domain=False,domain='mpi3dhp_real_rgb',guarantee_claimed=False)
                times.append((time.perf_counter()-start)*1000);values.append(previous)
            rows[method]=values;durations[method]=statistics(times);elapsed[method].extend(times)
        dest=output/cid;dest.mkdir()
        write_jsonl(dest/(selected_method+'.jsonl'),rows[selected_method])
        # First label access for this clip occurs after all ablation inference.
        truth=list(read_jsonl(dataset/'references'/(cid+'.jsonl')))
        if any([(r['frame_id'],r['timestamp_ns']) for r in values] !=
               [(r['frame_id'],r['timestamp_ns']) for r in truth] for values in rows.values()):
            raise ValueError('Reference/estimate exposure keys differ')
        raw=list(read_jsonl(source/'raw_geometry.jsonl'))
        details={}
        for method,values in rows.items():
            audit=failure_audit(values,raw,truth)
            for k in audits[method]:audits[method][k]+=audit[k]
            details[method]=dict(audit=audit,transitions=transitions(values,truth),
                                extra_metrics=extra_metrics(values,truth))
            combined[method].extend({**r,'frame_id':cid+'/'+str(r['frame_id'])} for r in values)
        all_truth.extend({**r,'frame_id':cid+'/'+str(r['frame_id'])} for r in truth)
        result=dict(id=cid,paired=compare_methods(rows,truth),
                    pairwise={m:compare_methods({k:rows[k] for k in ('baseline',m)},truth) for m in methods[1:]},
                    details=details,processing_ms=durations,
                    input_hashes={name:sha256(source/name) for name in ('detections.jsonl','current.jsonl')},
                    selected_states_file=selected_method+'.jsonl',
                    selected_states_sha256=sha256(dest/(selected_method+'.jsonl')))
        report['clips'].append(result)
        print(cid,{m:round(v['common_position_error_mm']['mean'],3) for m,v in result['paired']['methods'].items()},flush=True)
    report['paired']=compare_methods(combined,all_truth)
    report['pairwise']={m:compare_methods({k:combined[k] for k in ('baseline',m)},all_truth) for m in methods[1:]}
    report['failure_counts']=audits
    report['processing_ms']={m:statistics(v) for m,v in elapsed.items()}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({m:dict(mean_mm=v['common_position_error_mm']['mean'],coverage=v['coverage'],
          success=v['all_target_success'],large_errors=audits[m]['outputs_over_500mm'])
          for m,v in report['paired']['methods'].items()},indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-hash')
    parser.add_argument('--rendered',action='store_true')
    parser.add_argument('--selected-method',choices=('consistency','candidate'),default='candidate')
    args=parser.parse_args()
    if args.rendered:
        run_rendered(args.baseline,args.output,args.selected_method)
    else:
        if args.dataset is None:parser.error('--dataset required for public RGB evaluation')
        run(args.dataset,args.baseline,args.output,args.expected_hash,args.selected_method)
