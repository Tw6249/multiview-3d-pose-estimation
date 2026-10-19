"""Factorial replay of fixed detections; proxy truth is evaluation-only."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pose_sim.pipeline import read_jsonl,write_jsonl
from pose_sim.human_state import HumanStateEstimator
from pose_sim.human_motion import HumanMotionEstimator
from pose_sim.multiview_temporal import MultiViewTemporal
from pose_sim.vision_metrics import summarize
from pose_sim.motion_evaluation import compare_methods


def run(source,baseline,output):
    source,baseline,output=map(Path,(source,baseline,output))
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    cal=json.loads((source/'cameras.json').read_text());cfg=json.loads((source/'config.json').read_text())
    poses=list(read_jsonl(baseline/'poses3d.jsonl'));detections=list(read_jsonl(baseline/'detections.jsonl'))
    if len(poses)!=len(detections):raise ValueError('Mismatched baseline frames')
    results={};associations={}
    for label,associate,modern in [('baseline',False,False),('association_only',True,False),
                                   ('state_only',False,True),('joint',True,True)]:
        state=HumanMotionEstimator() if modern else HumanStateEstimator()
        fusion=MultiViewTemporal(cal,cfg,association=associate,structured=False)
        previous=None;rows=[];new_poses=[];assigned_rows=[];corrections=ambiguities=0
        for raw,detection in zip(poses,detections):
            if raw['frame_id']!=detection['frame_id']:raise ValueError('Mismatched frame identity')
            pose,assigned=fusion.update(detection,previous) if associate else (copy.deepcopy(raw),detection)
            previous=state.update(pose,cal['cameras'],assigned)
            previous['fusion_calibration_sha256']=pose.get('fusion_calibration_sha256')
            rows.append(previous);new_poses.append(pose);assigned_rows.append(assigned)
            corrections+=len(pose.get('association',{}).get('corrections',[]))
            ambiguities+=len(pose.get('association',{}).get('ambiguities',[]))
        results[label]=rows;associations[label]=dict(corrected_pairs=corrections,ambiguous_pairs=ambiguities)
        if label=='joint':
            write_jsonl(output/'poses3d.jsonl',new_poses)
            sha=hashlib.sha256((output/'poses3d.jsonl').read_bytes()).hexdigest()
            for r in rows:
                r['source_poses_sha256']=sha
                r['uncertainty_calibration']=dict(valid_for_domain=False,domain='unvalidated_rgb',guarantee_claimed=False)
            write_jsonl(output/'states3d.jsonl',rows);write_jsonl(output/'assigned_detections.jsonl',assigned_rows)
    truth=list(read_jsonl(source/'ground_truth.jsonl'))
    report=dict(scope='Reused rendered RGB development diagnostic, not independent real-human testing',
        methods={m:summarize(rows,truth) for m,rows in results.items()},paired=compare_methods(results,truth),
        association=associations,input_sha256={n:hashlib.sha256((baseline/n).read_bytes()).hexdigest()
            for n in ('poses3d.jsonl','detections.jsonl')},
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'pose_sim').glob('*.py'))})
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({m:dict(mean_mm=v['proxy_error_mm']['mean'],body_states=v['body12_state_counts'],
        common_mean_mm=report['paired']['methods'][m]['common_position_error_mm']['mean'],
        current_success=report['paired']['methods'][m]['current_all_target_success']) for m,v in report['methods'].items()},indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.source,a.baseline,a.output)
