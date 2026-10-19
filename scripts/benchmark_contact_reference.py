"""Source-only contact projection and paired G1 simulation; no labels are read."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from pose_sim.pipeline import read_jsonl,write_jsonl
from pose_sim.support import AnkleSupportEstimator
from motion_control.contact_reference import ContactReferenceProjector,load_config
from motion_control.humanoid import NAMES
from motion_control.root_path import ScaledRootPath
from motion_control.humanoid_replay import replay


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized_rows(source):
    path=ScaledRootPath();rows=[]
    for original in read_jsonl(source/'humanoid_reference.jsonl'):
        row=copy.deepcopy(original)
        if row['accepted'] and 'root_path_frame' not in row:
            xy=path.update(np.array(row['root_position_m'][:2])/row['scale_to_target'],row['scale_to_target'],row['timestamp_ns'])
            row['root_position_m'][:2]=xy.tolist();row['root_path_frame']='source_world_xy_anchored_at_first_accepted_pelvis; target_height_z'
        rows.append(row)
    return rows


def stats(values):
    a=np.asarray(values,float)
    return dict(mean=float(np.mean(a)),p95=float(np.percentile(a,95)),maximum=float(np.max(a)),count=len(a)) if len(a) else None


def kinematic_comparison(base,candidate,supports,projector):
    heights={m:[] for m in ('baseline','contact')};slip={m:[] for m in heights};pivot={m:[] for m in heights};previous=None;paired=0
    for a,b,support in zip(base,candidate,supports):
        if not a['accepted'] or not b['accepted']:previous=None;continue
        paired+=1;geometry={m:projector.geometry(row) for m,row in (('baseline',a),('contact',b))}
        for m,g in geometry.items():heights[m].append(max(0.,projector.cfg['ground_height_m']-min(v['points'][:,2].min() for v in g.values())))
        if previous:
            t,old,old_support,old_anchors=previous;dt=(a['timestamp_ns']-t)/1e9
            if 0<dt<=projector.cfg['support']['maximum_gap_s']:
                for side,state in support['feet'].items():
                    if state['status']=='support_candidate' and old_support['feet'][side]['status']=='support_candidate':
                        for m in geometry:slip[m].append(float(np.linalg.norm(geometry[m][side]['centre'][:2]-old[m][side]['centre'][:2])/dt))
                        anchor=b.get('contact_projection',{}).get('anchor_geometry',{}).get(side)
                        prior=old_anchors.get(side)
                        if anchor and prior and anchor['since_ns']==prior['since_ns'] and anchor['point']==prior['point']:
                            for m in geometry:
                                j=anchor['point'];pivot[m].append(float(np.linalg.norm(geometry[m][side]['points'][j]-old[m][side]['points'][j])/dt))
        previous=(a['timestamp_ns'],geometry,support,b.get('contact_projection',{}).get('anchor_geometry',{}))
    return dict(common_accepted_frames=paired,baseline_accepted=sum(r['accepted'] for r in base),candidate_accepted=sum(r['accepted'] for r in candidate),
                floor_penetration_m={m:stats(v) for m,v in heights.items()},same_support_pair_sole_center_speed_m_s={m:stats(v) for m,v in slip.items()},
                same_anchor_point_speed_m_s={m:stats(v) for m,v in pivot.items()},
                scope='Target reference FK and estimated support hypotheses; not measured human contact or dynamic robot slip')


def run(sources,output,config=None,simulation=True):
    if output.exists():raise FileExistsError('Preserve contact experiment evidence')
    output.mkdir(parents=True);cfg=copy.deepcopy(config or load_config())
    freeze=dict(config=cfg,source_sha256={str(p):sha(p/'humanoid_reference.jsonl') for p in sources},
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'pose_sim/support.py',ROOT/'motion_control/contact_reference.py',ROOT/'motion_control/humanoid_replay.py',ROOT/'motion_control/tracking_metrics.py',ROOT/'motion_control/root_path.py',ROOT/'motion_control/twist.py',Path(__file__).resolve()]},
        asset_config_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'config/g1_twist.json',ROOT/'assets/humanoid/manifest.json']},
        decision='Experimental; require no lower acceptance or worse dynamic balance/limb tracking before default promotion',ground_truth_used=False,hardware_enabled=False)
    (output/'freeze.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8');cases=[]
    for source in sources:
        case=output/source.parent.name if source.name=='integrated_displacement' else output/source.name
        case.mkdir();base=normalized_rows(source);estimator=AnkleSupportEstimator(cfg['support']);projector=ContactReferenceProjector(cfg)
        candidate=[];supports=[];elapsed=[]
        for row in base:
            points={n:p for n,p in zip(NAMES,row.get('source_body_points_m',[]))}
            support=estimator.update(row['timestamp_ns'],points,row.get('source_leg_length_m',1.),current=row['accepted'])
            start=time.perf_counter();candidate.append(projector.update(row,support));elapsed.append((time.perf_counter()-start)*1000);supports.append(support)
        kin=kinematic_comparison(base,candidate,supports,projector);results={}
        for method,rows in (('baseline',base),('contact',candidate)):
            folder=case/method;folder.mkdir();write_jsonl(folder/'humanoid_reference.jsonl',rows)
            (folder/'retargeting.json').write_text(json.dumps(dict(source=str(source),source_sha256=freeze['source_sha256'][str(source)],method=method,ground_truth_used=False),indent=2),encoding='utf-8')
            if simulation:results[method]=replay(folder,video=False,output_name='simulation',save_logs=False)
        counts={state:sum(v['status']==state for s in supports for v in s['feet'].values()) for state in ('unknown','support_candidate','swing')}
        cases.append(dict(id=case.name,kinematics=kin,support_counts=counts,projection_ms=stats(elapsed),methods=results,
            rejection_reasons={reason:sum(r.get('reason')==reason for r in candidate) for reason in set(r.get('reason') for r in candidate if not r['accepted'])}))
        print(case.name,kin,counts,{m:{k:r.get(k) for k in ('passed','balance_passed','source_reference_leg_rmse_rad','root_path_error_p95_m')} for m,r in results.items()},flush=True)
    report=dict(freeze=freeze,cases=cases)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8');return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sources',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--no-simulation',action='store_true');a=p.parse_args();run(a.sources,a.output,simulation=not a.no_simulation)
