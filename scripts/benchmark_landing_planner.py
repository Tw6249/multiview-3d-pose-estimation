"""Causal source-phase and robot-feedback planning audit, with zero actuation."""
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from pose_sim.source_phase import SourceLandingWindow
from pose_sim.support import AnkleSupportEstimator
from motion_control.humanoid import NAMES
from motion_control.contact_reference import ContactReferenceProjector,load_config as contact_config
from motion_control.landing_planner import LandingPlanner,load_config
from motion_control.root_tracking import RootTracker,planar_rotation,wrap
from motion_control.support_feedback import SimulationSupportFeedback
from motion_control.stop_transition import load_config as stop_config
from motion_control.twist import TwistPlant
from motion_control.humanoid_replay import replay


def run(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve prior phase/centroidal evidence')
    output.mkdir(parents=True);reference=Path('outputs/centroidal_revision/root_frame')
    baseline=json.loads((reference/'report.json').read_text());cfg=load_config();cases=[]
    files=['motion_control/landing_planner.py','motion_control/support_feedback.py','pose_sim/source_phase.py','motion_control/root_tracking.py','motion_control/humanoid_replay.py','scripts/benchmark_landing_planner.py']
    freeze=dict(configuration=cfg,source_sha256=baseline['freeze']['sources_sha256'],code_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},
        scope='Planning diagnostics only; fixed/position/joint ablation, no controller output and no future source input. Known data, not unseen validation.')
    (output/'freeze.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8')
    for path,sha in freeze['source_sha256'].items():
        source=Path(path);name=source.parent.name if source.name in ('baseline','integrated_displacement') else source.name
        if hashlib.sha256((source/'humanoid_reference.jsonl').read_bytes()).hexdigest()!=sha:raise ValueError('Input changed')
        rows=[json.loads(x) for x in (source/'humanoid_reference.jsonl').read_text().splitlines()]
        estimator=AnkleSupportEstimator(contact_config()['support']);phase=SourceLandingWindow(cfg['source_phase']);annotations={};source_counts=Counter()
        for row in rows:
            points=dict(zip(NAMES,row.get('source_body_points_m',[])));length=row.get('source_leg_length_m',1.);t=row['timestamp_ns']
            support=estimator.update(t,points,length,current=row['accepted'])
            annotation=phase.update(t,points,length,support,current=row['accepted']);annotations[t]=annotation
            source_counts[annotation['status']+':'+annotation.get('reason','')]+=1
        current={};events=[];seen=set();projector=ContactReferenceProjector();clock_offset=2_000_000_000-rows[0]['timestamp_ns']
        original=RootTracker.correct
        def factory():
            plant=TwistPlant();current.update(plant=plant,feedback=SimulationSupportFeedback(plant,stop_config()),planner=LandingPlanner(plant.sha256,cfg));return plant
        def observe(tracker,row,ref,position,heading):
            result=original(tracker,row,ref,position,heading);stamp=row['timestamp_ns']
            if stamp in seen:return result
            seen.add(stamp);annotation=annotations[stamp]
            if annotation['event'] is None:return result
            source_event=annotation['event'];side=source_event['side'];now_ns=round(current['plant'].data.time*1e9)
            event=copy.deepcopy(source_event);event['source_timestamp_ns']=stamp;event['observation_robot_ns']=stamp+clock_offset
            for key in ('earliest','latest','nominal'):event[key+'_robot_ns']=event[key+'_source_ns']+clock_offset
            geometry=projector.geometry(row);foot_now=tracker.origin+tracker.alignment@(geometry[side]['centre'][:2]-tracker.source_origin)
            remaining=(event['nominal_source_ns']-stamp)/1e9
            foot_velocity=tracker.alignment@(np.asarray(event['ankle_velocity_source_world_m_s'][:2])*row['scale_to_target'])
            event['nominal_landing_world_xy_m']=(foot_now+remaining*foot_velocity).tolist()
            target_heading=tracker.heading+wrap(row['root_heading_rad']-tracker.source_heading)
            event['target_heading_rad']=target_heading;event['desired_world_velocity_m_s']=(planar_rotation(target_heading)@row['reference'][:2]).tolist()
            robot=current['feedback'].read(now_ns);methods={}
            for mode in ('fixed','position','joint'):
                planned=current['planner'].solve(robot,event,now_ns,mode)
                methods[mode]={k:v for k,v in planned.items() if k!='attempts'}
                if 'attempts' in planned:methods[mode]['attempt_count']=len(planned['attempts'])
            events.append(dict(frame_id=row['frame_id'],source_timestamp_ns=stamp,perception_age_ms=(now_ns-event['observation_robot_ns'])/1e6,
                event=event,robot_com_height_m=robot['com_world_m'][2],methods=methods))
            return result
        with patch('motion_control.humanoid_replay.TwistPlant',factory),patch.object(RootTracker,'correct',observe):
            report=replay(source,video=False,save_logs=False,output_name=str(output/name),transition_mode='support_aware')
        prior=json.loads((reference/name/'source_heading/report.json').read_text())
        for key in ('source_reference_leg_rmse_rad','root_path_error_p95_m','standing_stop_audit','accepted_frames','tracking_ticks'):
            if report[key]!=prior[key]:raise AssertionError('Diagnostic planner changed execution: '+key)
        counts={m:dict(Counter(e['methods'][m]['status']+':'+e['methods'][m].get('reason','') for e in events)) for m in ('fixed','position','joint')}
        cases.append(dict(id=name,frames=len(rows),source_phase_counts=dict(source_counts),controller_unique_source_frames=len(seen),
            diagnostic_did_not_change_execution=True,planning_counts=counts,events=events))
        print(name,'source phases',source_counts.get('hypothesis:',0),'planning',counts,flush=True)
    result=dict(freeze=freeze,cases=cases)
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
