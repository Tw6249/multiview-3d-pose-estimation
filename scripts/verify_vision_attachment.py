"""Paired rendered-RGB regression, with coverage and consistency denominators.

Reads reference skeletons only for evaluation, after inference files exist.
Render references have 12 body proxies, no ground truth for palms or feet.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pose_sim.pipeline import read_jsonl
from pose_sim.auxiliary import attachment_checks
from pose_sim.motion_evaluation import statistics,compare_methods


def diagnostics(rows):
    wrist=[];palm=[];foot=[];counts=dict(detail_targets=0,current_details=0,available_details=0,
        attachment_group_targets=0,attachment_supported=0,attachment_rejected=0,attachment_unknown=0,
        body_all_current_frames=0,orientation_targets=0,orientation_available=0)
    for row in rows:
        names=row['joint_names'];ss=dict(zip(names,row['joint_states']))
        allp={n:np.array(p) for n,p in zip(names,row['joints_world_m']) if p is not None}
        current={n:p for n,p in allp.items() if ss[n]['status']=='filtered'}
        sigma={n:ss[n]['max_std_m'] for n in current}
        checks,_=attachment_checks(current,sigma)
        counts['attachment_group_targets']+=4
        for d in checks.values():counts['attachment_'+d['status']]+=1
        detail=names[17:];counts['detail_targets']+=len(detail)
        counts['current_details']+=sum(n in current for n in detail)
        counts['available_details']+=sum(n in allp for n in detail)
        counts['body_all_current_frames']+=all(n in current for n in names[5:17])
        for s in ('left','right'):
            a=s+'_wrist';b=s+'_hand_wrist'
            if a in current and b in current:wrist.append(float(np.linalg.norm(current[a]-current[b])*1000))
            for n in (s+'_hand_'+v for v in ('thumb_mcp','index_mcp','middle_mcp','pinky_mcp')):
                if a in current and n in current:palm.append(float(np.linalg.norm(current[a]-current[n])*1000))
            a=s+'_ankle'
            for n in (s+'_'+v for v in ('heel','big_toe','small_toe')):
                if a in current and n in current:foot.append(float(np.linalg.norm(current[a]-current[n])*1000))
            for k in ('palm','foot'):
                counts['orientation_targets']+=1
                counts['orientation_available']+=row.get('orientations',{}).get(s+'_'+k) is not None
    return dict(**counts,frames=len(rows),current_detail_coverage=counts['current_details']/counts['detail_targets'],
        available_detail_coverage=counts['available_details']/counts['detail_targets'],
        wrist_duplicate_gap_mm=statistics(wrist),wrist_gap_over_250mm=sum(v>250 for v in wrist),
        wrist_to_mcp_mm=statistics(palm),ankle_to_foot_point_mm=statistics(foot),
        note='Attachment distances are consistency diagnostics, not ground-truth accuracy.')


def verify(baseline,candidate,output):
    baseline=Path(baseline);candidate=Path(candidate)
    manifest=json.loads((baseline/'manifest.json').read_text(encoding='utf-8'))
    updated=json.loads((candidate/'manifest.json').read_text(encoding='utf-8'))
    if [s['id'] for s in manifest['scenes']]!=[s['id'] for s in updated['scenes']]:
        raise ValueError('Scene list mismatch; refusing zip truncation')
    report=dict(scenes=[],limitations=['Reused rendered development scenes; one walk asset, not independent real humans.',
        'Reference labels cover 12 body rig proxies only; no rendered foot or palm rotation accuracy claim.',
        'Detector/fusion inference ran with other local CPU benchmarks; timings here are not isolated latency measurements.'])
    allstates={'legacy':[],'candidate':[]};alltruth=[];next_id=0
    for old,new in zip(manifest['scenes'],updated['scenes']):
        if old['id']!=new['id']:raise ValueError('Scene mismatch')
        a=baseline/old['result'];b=candidate/new['result']
        detections=[list(read_jsonl(a/'detections.jsonl')),list(read_jsonl(new.get('detections',b/'detections.jsonl')))]
        maximum=0.;scoremax=0.
        if len(detections[0])!=len(detections[1]):raise ValueError('Frame count mismatch')
        for r,s in zip(*detections):
            if (r['frame_id'],r['timestamp_ns'])!=(s['frame_id'],s['timestamp_ns']):raise ValueError('Clock mismatch')
            for c,v in r['views'].items():
                w=s['views'][c]
                if (v['keypoints_px'] is None)!=(w['keypoints_px'] is None):raise ValueError('Detection availability changed')
                if v['keypoints_px'] is not None:
                    maximum=max(maximum,float(np.max(np.abs(np.array(v['keypoints_px'])-w['keypoints_px']))))
                    scoremax=max(scoremax,float(np.max(np.abs(np.array(v['scores'])-w['scores']))))
        states={k:list(read_jsonl(d/'states3d.jsonl')) for k,d in [('legacy',a),('candidate',b)]}
        truth=list(read_jsonl(baseline/old['source']/'ground_truth.jsonl'))
        expected=[(r['frame_id'],r['timestamp_ns']) for r in detections[0]]
        if len(expected)!=old['frames'] or any([(r['frame_id'],r['timestamp_ns']) for r in rows]!=expected
                                             for rows in [*states.values(),truth]):
            raise ValueError('Incomplete or mismatched state/reference timestamps')
        scene=dict(id=old['id'],detector_coordinate_max_difference_px=maximum,detector_score_max_difference=scoremax,
            methods={k:diagnostics(v) for k,v in states.items()},body_proxy_accuracy=compare_methods(states,truth),
            candidate_states_sha256=hashlib.sha256((b/'states3d.jsonl').read_bytes()).hexdigest())
        report['scenes'].append(scene)
        for i,t in enumerate(truth):
            alltruth.append({**t,'frame_id':next_id+i})
            for k in states:allstates[k].append({**states[k][i],'frame_id':next_id+i})
        next_id+=len(truth)
    report['aggregate']={k:diagnostics(v) for k,v in allstates.items()}
    report['body_proxy_accuracy']=compare_methods(allstates,alltruth)
    Path(output).write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(aggregate=report['aggregate'],body_proxy_accuracy=report['body_proxy_accuracy']['methods']),indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(a.baseline,a.candidate,a.output)
