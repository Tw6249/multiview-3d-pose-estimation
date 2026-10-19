"""Frozen support-observation and source/robot phase audit; zero actuation changes."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation
from pose_sim.foot_support import FootSupportEstimator,load_config
from pose_sim.support import AnkleSupportEstimator
from pose_sim.source_phase import SourceLandingWindow
from pose_sim.geometry import JOINTS
from motion_control.humanoid import NAMES
from motion_control.contact_reference import load_config as ankle_config
from motion_control.landing_planner import load_config as landing_config
from motion_control.support_feedback import SimulationSupportFeedback
from motion_control.contact_metrics import FootLoadEvents,PredictionRuns,match_landing_windows
from motion_control.humanoid_replay import replay


def legacy_support(p,v,states,t,memory):
    """Frozen previous HumanMotionEstimator._support, valid attached foot/ground=0."""
    result={}
    for side in ('left','right'):
        ns=[side+'_'+n for n in ('heel','big_toe','small_toe')]
        sigma=max(states[n]['max_std_m'] for n in ns)
        score=max(float(np.exp(-.5*(p[n][2]/.035)**2-.5*(np.linalg.norm(v[n])/.2)**2)) for n in ns)
        old=memory.get(side,dict(state='unknown',candidate=None,count=0,last=t))
        target='contact' if score>.65 and sigma<.03 else 'swing' if score<.25 else old['state']
        count=old['count']+1 if old['candidate']==target and t-old['last']<=150_000_000 else 1
        state=target if count>=3 else old['state']
        memory[side]=dict(state=state,candidate=target,count=count,last=t);result[side]=state
    return result


def analytical_audit():
    # Deterministic kinematic evidence fixtures, NOT RGB/force truth. Frozen
    # before scoring. No threshold fitting and no claim of unseen validation.
    results=[]
    for scenario in ('stationary','toe_pivot','lifted','slide','ambiguous_after_support','uncertain_position','uncertain_velocity'):
        old={};new=FootSupportEstimator();counts={m:Counter() for m in ('previous','current')}
        for i in range(30):
            t=i*40_000_000;p={s+'_'+n:np.array([x,y,.005]) for s,y in [('left',.1),('right',-.1)] for n,x in [('heel',-.06),('big_toe',.16),('small_toe',.14)]}
            v={n:np.zeros(3) for n in p};states={n:dict(status='filtered',last_observed_timestamp_ns=t,max_std_m=.01,velocity_covariance_m2_s2=(np.eye(3)*.01).tolist()) for n in p}
            label='contact'
            if scenario=='toe_pivot':
                for n in p:
                    if n.endswith('heel'):p[n][2]=.15;v[n]=np.array([.5,0,.3])
            if scenario=='lifted':
                for n in p:p[n][2]=.2
                label='swing'
            if scenario=='slide':
                for n in p:v[n][0]=1.
                label='unknown'
            if scenario=='ambiguous_after_support' and i>=10:
                for n in p:v[n][0]=.25
                label='unknown'
            if scenario=='uncertain_position':
                for n in p:p[n][2]=.2;states[n]['max_std_m']=.2
                label='unknown'
            if scenario=='uncertain_velocity':
                for n in p:states[n]['velocity_covariance_m2_s2']=(np.eye(3)*4).tolist()
                label='unknown'
            outputs=dict(previous=legacy_support(p,v,states,t,old),current={s:r['state'] for s,r in new.update(t,p,v,states,dict(left=True,right=True),0.)['feet'].items()})
            if i>=2:
                for m,out in outputs.items():
                    for value in out.values():counts[m][label+'->'+value]+=1
        results.append(dict(scenario=scenario,scored_foot_frames=56,confusion={m:dict(c) for m,c in counts.items()}))
    return results


def rendered_reference_audit():
    result=[]
    for path in sorted(Path('outputs/vision_showcase/sources').glob('*/ground_truth.jsonl')):
        rows=[json.loads(s) for s in path.read_text().splitlines()];idx=[JOINTS.index(s+'_ankle') for s in ('left','right')]
        xyz=np.array([[r['joints_world_m'][j] for j in idx] for r in rows]);t=np.array([r['timestamp_ns'] for r in rows],dtype=np.int64)
        speed=np.linalg.norm(np.diff(xyz,axis=0),axis=2)/(np.diff(t)/1e9)[:,None]
        low=xyz[1:,:,2]-np.percentile(xyz[:,:,2],5)<.05
        result.append(dict(id=path.parent.name,input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            low_ankle_targets=int(low.sum()),low_ankle_speed_m_s=dict(zip(('p10','median','p90'),np.percentile(speed[low],[10,50,90]).tolist())),
            scope='Evaluation-only rig joint centers, no force/sole/contact labels. Moving low ankles do not alone prove sliding.'))
    return result


def run(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve prior contact evidence')
    output.mkdir(parents=True)
    prior_root=Path('outputs/centroidal_revision/root_frame');prior=json.loads((prior_root/'report.json').read_text())
    files=['pose_sim/foot_support.py','pose_sim/human_motion.py','motion_control/contact_metrics.py','scripts/benchmark_contact_observation.py']
    freeze=dict(foot_configuration=load_config(),source_sha256=prior['freeze']['sources_sha256'],
        code_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},
        load_event_config=dict(enter_weight_fraction=.05,exit_weight_fraction=.025,confirmation_frames=2,maximum_gap_s=.05),
        clock_mapping='robot_ns = source_ns - first_source_ns + 2000000000; injected perception delay 120000000 ns',
        source_window_matching='First prediction per contiguous run; no widening; robot touchdown must follow receipt. Separate delayed-source comparison adds only the fixed 120 ms.',
        tilt_stratum='Before/after first |roll or pitch| >= 0.5 rad. Height omitted to avoid labeling crouch as fall. Not a complete balance/fall certificate.',
        scope='Known-case observer regression and analytic evidence fixtures; no independent physical human contact labels or tuning.')
    (output/'freeze.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8');cases=[]
    analytical=analytical_audit();rendered=rendered_reference_audit()
    (output/'source_evidence.json').write_text(json.dumps(dict(analytical=analytical,rendered_reference=rendered),indent=2),encoding='utf-8')
    for path,sha in freeze['source_sha256'].items():
        source=Path(path);name=source.parent.name if source.name in ('baseline','integrated_displacement') else source.name
        if hashlib.sha256((source/'humanoid_reference.jsonl').read_bytes()).hexdigest()!=sha:raise ValueError('Source changed')
        rows=[json.loads(s) for s in (source/'humanoid_reference.jsonl').read_text().splitlines()]
        ankle=AnkleSupportEstimator(ankle_config()['support']);phase=SourceLandingWindow(landing_config()['source_phase']);runs=PredictionRuns()
        annotations={};source_counts=Counter()
        for row in rows:
            t=row['timestamp_ns'];points=dict(zip(NAMES,row.get('source_body_points_m',[])));length=row.get('source_leg_length_m',1.)
            support=ankle.update(t,points,length,current=row['accepted']);window=phase.update(t,points,length,support,current=row['accepted'])
            rid=runs.update(t,window['event']);annotations[t]=dict(support=support,window=window,run_id=rid)
            for side in ('left','right'):source_counts[support['feet'][side]['status']]+=1
        load=FootLoadEvents();offset=2_000_000_000-rows[0]['timestamp_ns'];delay=120_000_000
        cursor=-1;seen=set();received={};updates=[];correspondence=Counter();tilt_first=None;unique_source=0
        original=SimulationSupportFeedback.read
        def observe(adapter,now_ns):
            nonlocal cursor,tilt_first,unique_source
            robot=original(adapter,now_ns);rpy=Rotation.from_quat(adapter.data.qpos[[4,5,6,3]]).as_euler('xyz')
            if tilt_first is None and max(abs(rpy[:2]))>=.5:tilt_first=now_ns
            status=load.update(now_ns,robot['normal_force_n'],robot['mass_kg']*robot['gravity_m_s2'])
            available=now_ns-offset-delay
            while cursor+1<len(rows) and rows[cursor+1]['timestamp_ns']<=available:cursor+=1
            if cursor<0 or now_ns>rows[-1]['timestamp_ns']+offset+delay:return robot
            row=rows[cursor];stamp=row['timestamp_ns']
            if stamp in seen:return robot
            seen.add(stamp);unique_source+=1
            a=annotations[stamp];stratum='before_tilt_violation' if tilt_first is None else 'after_tilt_violation'
            for side,v in a['support']['feet'].items():correspondence[stratum+':'+v['status']+':'+status[side]]+=1
            e=a['window']['event']
            if e is not None:
                rid=a['run_id'];stance=e['stance_side'];update=dict(frame_id=row['frame_id'],run_id=rid,side=e['side'],
                    source_timestamp_ns=stamp,received_robot_ns=now_ns,stratum=stratum,robot_stance_load=status[stance],
                    robot_stance_force_n=robot['normal_force_n'][stance],robot_root_height_m=robot['root_height_m'])
                updates.append(update)
                if rid not in received:
                    received[rid]=dict(run_id=rid,side=e['side'],received_ns=now_ns,stratum=stratum,
                        earliest_ns=e['earliest_source_ns']+offset,latest_ns=e['latest_source_ns']+offset,nominal_ns=e['nominal_source_ns']+offset)
            return robot
        with patch.object(SimulationSupportFeedback,'read',observe):
            report=replay(source,video=False,save_logs=False,output_name=str(output/name),transition_mode='support_aware')
        old=json.loads((prior_root/name/'source_heading/report.json').read_text())
        for key in ('source_reference_leg_rmse_rad','root_path_error_p95_m','standing_stop_audit','tracking_ticks','accepted_frames'):
            if report[key]!=old[key]:raise AssertionError('Observer changed controller '+key)
        # Only source-tracking epoch events; initialization and terminal stopping
        # remain outside the correspondence denominator.
        robot_events=[t for t in load.events if rows[0]['timestamp_ns']+offset+delay<=t['timestamp_ns']<=rows[-1]['timestamp_ns']+offset+delay]
        windows=list(received.values());delayed=[{**w,**{k:w[k]+delay for k in ('earliest_ns','latest_ns','nominal_ns')}} for w in windows]
        cases.append(dict(id=name,source_frames=len(rows),source_foot_status=dict(source_counts),observed_source_frames=unique_source,
            prediction_updates=len(updates),prediction_runs=len(runs.runs),received_prediction_runs=len(windows),runs=runs.runs,
            first_tilt_violation_robot_ns=tilt_first,source_robot_foot_correspondence=dict(correspondence),prediction_update_audit=updates,
            robot_touchdowns=robot_events,first_prediction_windows=windows,
            physical_source_time_matching=match_landing_windows(windows,robot_events),
            delayed_source_time_matching=match_landing_windows(delayed,robot_events),
            diagnostic_did_not_change_execution=True,whole_imitation_passed=report['passed']))
        print(name,'windows',len(updates),'runs',len(windows),'touchdowns',len(robot_events),'tilt',tilt_first,flush=True)
    result=dict(freeze=freeze,analytical=analytical,rendered_reference=rendered,cases=cases)
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
