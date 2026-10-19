"""Rotation observability and velocity candidate on existing real RGB caches.

Frozen analytical design; no parameters fit to labels. Uses historical cached
raw/current pipelines for this separate state experiment, not the newly chosen
person-selection baseline. References are opened after all candidate updates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation
from pose_sim.pipeline import read_jsonl
from pose_sim.kinematic_observability import KinematicObservability,torso_frame,hinge_flexion
from pose_sim.landmark_velocity import LandmarkVelocity
from pose_sim.motion_evaluation import statistics


def run(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    sets=[('development',Path('outputs/public_datasets/mpi_benchmark'),Path('outputs/public_datasets/mpi_results')),
          ('heldout',Path('outputs/public_datasets/mpi_heldout'),Path('outputs/public_datasets/mpi_heldout_baseline'))]
    freeze=dict(code_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['pose_sim/kinematic_observability.py','pose_sim/landmark_velocity.py','scripts/benchmark_kinematic_state.py']},
        velocity_configuration=dict(frames=9,minimum_samples=5,noise_floor_m=.018,maximum_gap_s=.12,maximum_residual_m=.04),
        scope='Known cached raw/current 1800-frame subset, historical perception configuration. Neither estimator consumes references. No algorithm tuning between subjects or splits.')
    (output/'freeze.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8');cases=[];times=[]
    for split,dataset,cache in sets:
        manifest=json.loads((dataset/'manifest.json').read_text())
        for clip in manifest['clips']:
            cid=clip['id'];raw=list(read_jsonl(cache/cid/'raw_geometry.jsonl'));current=list(read_jsonl(cache/cid/'current.jsonl'))
            kin=KinematicObservability();vel=LandmarkVelocity();ks=[];vs=[]
            for a,b in zip(raw,current):
                if (a['frame_id'],a['timestamp_ns'])!=(b['frame_id'],b['timestamp_ns']):raise ValueError('State alignment mismatch')
                begin=time.perf_counter();ks.append(kin.update(b));vs.append(vel.update(a));times.append((time.perf_counter()-begin)*1000)
            truth=list(read_jsonl(dataset/'references'/(cid+'.jsonl')))
            if len(truth)!=len(current) or len(raw)!=len(current):raise ValueError('Frame count mismatch')
            angle_errors=[];angle_std=[];hinge_errors=[];hinge_std=[];before=[];after=[];common_before=[];common_after=[];all_targets=0;foot_known={s:0 for s in ('left','right')}
            for i,(r,g,k,v) in enumerate(zip(current,truth,ks,vs)):
                if r['timestamp_ns']!=g['timestamp_ns']:raise ValueError('Reference timestamp mismatch')
                p={n:np.array(x) for n,x in zip(g['joint_names'],g['joints_world_m'])};torso=k['landmark_frames']['torso']
                for side in foot_known:foot_known[side]+=k['landmark_frames'][side+'_foot']['rotation_world'] is not None
                R=torso_frame(np.array([p[n] for n in ('left_hip','right_hip','left_shoulder','right_shoulder')]))
                if R is not None and torso['rotation_world'] is not None:
                    angle_errors.append(float(np.linalg.norm(Rotation.from_matrix(np.array(torso['rotation_world'])@R.T).as_rotvec())));angle_std.append(torso['maximum_rotation_std_rad'])
                for side in ('left','right'):
                    for j,parts in [('knee',('hip','knee','ankle')),('elbow',('shoulder','elbow','wrist'))]:
                        target=hinge_flexion(np.array([p[side+'_'+n] for n in parts]));estimate=k['hinges'][side+'_'+j]
                        if target is not None and estimate['flexion_rad'] is not None:
                            hinge_errors.append(abs(target-estimate['flexion_rad']));hinge_std.append(estimate['std_rad'])
                if not 0<i<len(truth)-1:continue
                a,b=truth[i-1],truth[i+1];refv=(np.array(b['joints_world_m'])-a['joints_world_m'])/((b['timestamp_ns']-a['timestamp_ns'])/1e9)
                states=dict(zip(r['joint_names'],r['joint_states']));vv=dict(zip(r['joint_names'],r['velocities_world_m_s']))
                for n,target in zip(g['joint_names'],refv):
                    all_targets+=1;x=vv.get(n) if states.get(n,{}).get('status')=='filtered' else None;y=v['joints'].get(n,{}).get('velocity_world_m_s')
                    xerr=float(np.linalg.norm(np.array(x)-target)) if x is not None else None;yerr=float(np.linalg.norm(np.array(y)-target)) if y is not None else None
                    if xerr is not None:before.append(xerr)
                    if yerr is not None:after.append(yerr)
                    if xerr is not None and yerr is not None:common_before.append(xerr);common_after.append(yerr)
            cases.append(dict(id=cid,split=split,frames=len(truth),input_sha256={n:hashlib.sha256((cache/cid/n).read_bytes()).hexdigest() for n in ('raw_geometry.jsonl','current.jsonl')},
                torso_rotation_error_deg=statistics(np.rad2deg(angle_errors).tolist()),torso_local_std_deg=statistics(np.rad2deg(angle_std).tolist()),
                torso_empirical_three_std_coverage=float(np.mean(np.array(angle_errors)<=3*np.array(angle_std))) if angle_errors else None,
                hinge_error_deg=statistics(np.rad2deg(hinge_errors).tolist()),hinge_three_std_coverage=float(np.mean(np.array(hinge_errors)<=3*np.array(hinge_std))) if hinge_errors else None,
                foot_rotation_available=foot_known,foot_rotation_accuracy=None,foot_contact_accuracy=None,
                velocity=dict(targets=all_targets,baseline=statistics(before),candidate=statistics(after),common_baseline=statistics(common_before),common_candidate=statistics(common_after),
                    baseline_coverage=len(before)/all_targets,candidate_coverage=len(after)/all_targets,
                    baseline_all_target_within_point3_m_s=sum(x<=.3 for x in before)/all_targets,candidate_all_target_within_point3_m_s=sum(x<=.3 for x in after)/all_targets)))
            print(cid,'velocity common',np.mean(common_before),np.mean(common_after),'coverage',len(before)/all_targets,len(after)/all_targets,flush=True)
    result=dict(freeze=freeze,cases=cases,module_cpu_ms=statistics(times),scope='Derived landmark/hinge reference, differentiated reference velocity. No anatomical rotation or force labels, foot availability is not accuracy; unknown remains unknown.')
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
