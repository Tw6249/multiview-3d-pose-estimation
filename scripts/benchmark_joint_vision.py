"""Frozen shared-detection visual ablation. Reference files are scoring-only."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from pose_sim.pipeline import read_jsonl
from pose_sim.multiview_temporal import MultiViewTemporal
from pose_sim.human_motion import HumanMotionEstimator
from pose_sim.person_selection import CrossViewPersonSelector
from pose_sim.joint_window import load_config
from pose_sim.motion_evaluation import compare_methods,statistics
from scripts.evaluate_mpi3dhp import extra_metrics
from scripts.evaluate_observation_recovery import transitions


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(output,split='development',methods=('baseline','single_frame','joint_window'),expected_freeze=None):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve prior evaluation')
    output.mkdir(parents=True)
    dataset=Path('outputs/public_datasets/mpi_long');detections=Path('outputs/public_datasets/mpi_long_detections');candidates=Path('outputs/public_datasets/mpi_long_person_candidates')
    ids=['s1_long','s2_long'] if split in ('development','calibration') else ['s4_long','s6_long']
    settings=json.loads(Path('config/vision_motion.json').read_text());geometry=json.loads(Path('config/demo.json').read_text());wc=load_config()
    files=sorted(Path('pose_sim').glob('*.py'))+[Path('config/vision_joint_window.json'),Path(__file__)]
    if expected_freeze:files=[Path(p) for p in json.loads(Path(expected_freeze).read_text())['code_files']]
    fingerprint=hashlib.sha256(b''.join(str(p).encode()+p.read_bytes() for p in files)).hexdigest()
    freeze=dict(implementation_sha256=fingerprint,code_files=[str(p) for p in files],configuration=wc,state=settings['state'],methods=methods,split=split,
        acceptance='Require nondecreasing all-target/current success and nonincreasing >500mm errors, mean/P95 on development and frozen heldout; inspect each clip and latency. No ground-truth constraint in estimation.',
        protocol='S1/S2 first 500 frames development; last 250 calibration only after choice. S4/S6 750 frames each post-freeze assessment. Known people/dataset, no unseen/SOTA claim.',
        inputs={str(p):sha(p) for cid in ids for p in (detections/cid/'detections.jsonl',candidates/(cid+'.jsonl'),dataset/'inputs'/cid/'cameras.json')})
    if expected_freeze and json.loads(Path(expected_freeze).read_text())['implementation_sha256']!=fingerprint:raise ValueError('Implementation changed after selection')
    (output/'freeze.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8');cases=[];combined={m:[] for m in methods};truth_all=[]
    for cid in ids:
        cal=json.loads((dataset/'inputs'/cid/'cameras.json').read_text());ds=list(read_jsonl(detections/cid/'detections.jsonl'))
        proposals={r['frame_id']:r for r in read_jsonl(candidates/(cid+'.jsonl'))}
        if split=='development':ds=ds[:500]
        selector=CrossViewPersonSelector(cal['cameras'],geometry);selected=[]
        for row in ds:
            d=copy.deepcopy(row)
            if d['frame_id'] in proposals:
                proposal=proposals[d['frame_id']]
                if proposal['timestamp_ns']!=d['timestamp_ns']:raise ValueError('Candidate clock mismatch')
                for c,v in proposal['views'].items():d['views'][c]['person_candidates']=v
                d=selector.update(d)
            selected.append(d)
        results={};times={};diagnostics={}
        for method in methods:
            fusion=MultiViewTemporal(cal,geometry,structured=method=='single_frame',observation_consistency=True,window_config=wc if method=='joint_window' else None)
            state=HumanMotionEstimator(settings['state']);previous=None;rows=[];elapsed=[];fit_counts=[]
            for i,d in enumerate(selected):
                begin=time.perf_counter();pose,assigned=fusion.update(d,previous);previous=state.update(pose,cal['cameras'],assigned)
                elapsed.append((time.perf_counter()-begin)*1000);rows.append(previous)
                if 'joint_window' in pose:fit_counts.append(pose['joint_window'])
                if (i+1)%100==0:print(cid,method,i+1,'mean_ms',round(np.mean(elapsed),2),flush=True)
            results[method]=rows[500:] if split=='calibration' else rows;times[method]=statistics(elapsed)
            diagnostics[method]=dict(fit_frames=sum(x['status']=='fitted' for x in fit_counts),accepted_joint_corrections=sum(x['accepted_joints'] for x in fit_counts),rejected_joint_corrections=sum(x['rejected_joints'] for x in fit_counts))
        # First reference access after every method has completed inference.
        truth=list(read_jsonl(dataset/'references'/(cid+'.jsonl')))
        truth=truth[:500] if split=='development' else truth[500:] if split=='calibration' else truth
        paired=compare_methods(results,truth);details={}
        for method,rows in results.items():
            if [(r['frame_id'],r['timestamp_ns']) for r in rows]!=[(r['frame_id'],r['timestamp_ns']) for r in truth]:raise ValueError('Scoring alignment mismatch')
            details[method]=dict(kinematics=extra_metrics(rows,truth),recovery=transitions(rows,truth))
            errors=[];sigmas=[];current=[];velocity_errors=[];velocity_sigmas=[];stamps=[]
            for i,(r,g) in enumerate(zip(rows,truth)):
                p=dict(zip(r['joint_names'],r['joints_world_m']));s=dict(zip(r['joint_names'],r['joint_states']));v=dict(zip(r['joint_names'],r['velocities_world_m_s']))
                for n,target in zip(g['joint_names'],g['joints_world_m']):
                    valid=p.get(n) is not None;errors.append(float(np.linalg.norm(np.array(p[n])-target)) if valid else np.nan)
                    sigmas.append(s.get(n,{}).get('max_std_m',np.nan));current.append(s.get(n,{}).get('status')=='filtered');stamps.append(g['timestamp_ns'])
                    ve=vs=np.nan
                    if valid and v.get(n) is not None and 0<i<len(truth)-1:
                        lo,hi=truth[i-1],truth[i+1];a=dict(zip(lo['joint_names'],lo['joints_world_m']))[n];b=dict(zip(hi['joint_names'],hi['joints_world_m']))[n]
                        truth_v=(np.array(b)-a)/((hi['timestamp_ns']-lo['timestamp_ns'])/1e9);ve=float(np.linalg.norm(np.array(v[n])-truth_v))
                        cov=s[n].get('velocity_covariance_m2_s2');vs=float(np.sqrt(np.linalg.eigvalsh(cov).max())) if cov is not None else np.nan
                    velocity_errors.append(ve);velocity_sigmas.append(vs)
            np.savez_compressed(output/(cid+'_'+method+'_scores.npz'),error_m=errors,sigma_m=sigmas,current=current,timestamp_ns=stamps,velocity_error_m_s=velocity_errors,velocity_sigma_m_s=velocity_sigmas)
            details[method]['large_errors_over_500mm']=int(np.sum(np.array(errors)>.5))
            combined[method].extend({**r,'frame_id':cid+'/'+str(r['frame_id'])} for r in rows)
        truth_all.extend({**g,'frame_id':cid+'/'+str(g['frame_id'])} for g in truth)
        cases.append(dict(id=cid,frames=len(truth),paired=paired,details=details,processing_ms=times,fit_diagnostics=diagnostics))
        print('SCORED',cid,{m:(paired['methods'][m]['common_position_error_mm'],details[m]['large_errors_over_500mm']) for m in methods},flush=True)
    result=dict(freeze=freeze,cases=cases,paired=compare_methods(combined,truth_all),
        aggregate_large_errors={m:sum(c['details'][m]['large_errors_over_500mm'] for c in cases) for m in methods})
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--split',choices=['development','calibration','heldout'],default='development')
    p.add_argument('--methods',nargs='+',default=['baseline','single_frame','joint_window']);p.add_argument('--expected-freeze',type=Path)
    a=p.parse_args();run(a.output,a.split,tuple(a.methods),a.expected_freeze)
