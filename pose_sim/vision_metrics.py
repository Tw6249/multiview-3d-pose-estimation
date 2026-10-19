"""Independent visual quality evaluation; only this module reads proxy truth."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from .geometry import JOINTS
from .landmarks import LIMB_BONES
from .pipeline import read_jsonl


def summarize(rows, truth=None):
    truth = {r['frame_id']:r for r in truth or []}
    errors, errors_observed, possible, valid_body, total_body = [], [], 0, 0, 0
    lengths = {a+'--'+b:[] for a,b in LIMB_BONES}
    states, body_states, reasons, speeds, contacts, orientations = Counter(),Counter(),Counter(),[],Counter(),Counter()
    previous = None
    for row in rows:
        p = dict(zip(row['joint_names'], row['joints_world_m']))
        for name in JOINTS[5:]:
            total_body += 1; valid_body += int(p.get(name) is not None)
        for a,b in LIMB_BONES:
            if p.get(a) is not None and p.get(b) is not None:
                lengths[a+'--'+b].append(float(np.linalg.norm(np.asarray(p[a])-p[b])))
        quality = dict(zip(row['joint_names'], row.get('joint_states',[])))
        body_states.update(quality[n]['status'] for n in JOINTS[5:] if n in quality)
        for s in quality.values():
            states[s['status']] += 1
            if s.get('reason'): reasons[s['reason']] += 1
        for name,v in row.get('orientations',{}).items():
            orientations[name] += int(v is not None)
        for name,v in row.get('contacts',{}).items(): contacts[name] += int(v is not None)
        gt = truth.get(row['frame_id'])
        if gt:
            for name,target in zip(gt.get('joint_names',JOINTS),gt['joints_world_m']):
                if target is None: continue
                possible += 1
                if p.get(name) is not None:
                    error=float(np.linalg.norm(np.asarray(p[name])-target)*1000)
                    errors.append(error)
                    if quality.get(name,{}).get('status') != 'predicted': errors_observed.append(error)
        if previous:
            dt=(row['timestamp_ns']-previous[0])/1e9
            if dt<=0: raise ValueError('Nonmonotonic evaluation sequence')
            for n in JOINTS[5:]:
                if p.get(n) is not None and previous[1].get(n) is not None:
                    speeds.append(float(np.linalg.norm(np.asarray(p[n])-previous[1][n])/dt))
        previous=row['timestamp_ns'],p
    def summary(values):
        return dict(mean=float(np.mean(values)),p95=float(np.percentile(values,95)),maximum=max(values)) if values else None
    return dict(frames=len(rows),body12_coverage=valid_body/total_body,
        proxy_error_mm=summary(errors),proxy_current_observation_error_mm=summary(errors_observed),
        proxy_coverage=len(errors)/possible if possible else None,
        bone_length_cv={k:float(np.std(v)/np.mean(v)) if len(v)>1 else None for k,v in lengths.items()},
        body_step_speed_m_s=summary(speeds),joint_state_counts=dict(states),body12_state_counts=dict(body_states),rejection_reasons=dict(reasons),
        orientation_frame_counts=dict(orientations),contact_frame_counts=dict(contacts))


def evaluate(session, output=None):
    session=Path(session);output=Path(output) if output else session
    truth=list(read_jsonl(session/'ground_truth.jsonl')) if (session/'ground_truth.jsonl').exists() else []
    result={name:summarize(list(read_jsonl(output/(name+'.jsonl'))),truth)
            for name in ('poses3d','states3d') if (output/(name+'.jsonl')).exists()}
    for label,file in [('inference','inference_timing.json'),('triangulation','reconstruction_timing.json'),('state','state_timing.json')]:
        if (output/file).exists():result[label]=json.loads((output/file).read_text())
    result['limitations']=['Proxy rig centres differ from anatomical landmarks; no real-human accuracy claim.',
        'Estimated and predicted coverage is separate from current image observations.',
        'No ground truth for hands, feet orientation or contact; availability is not accuracy.',
        'Offline processing durations exclude camera capture and robot execution.']
    (output/'vision_metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('session');p.add_argument('--output')
    a=p.parse_args(); print(json.dumps(evaluate(a.session,a.output),indent=2))
