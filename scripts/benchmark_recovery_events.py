"""Labelled recovery-event benchmark on cached detections.

Each method runs once on clean detections and once per seed on detections with
injected, labelled faults (pose_sim.fault_injection). Faulted outputs are scored
against the paired clean run of the same method, so a cost is attributed to the
fault and not to the method's baseline error. Natural events are drops of the
usable-view count, or of the reference in-frustum camera count, below two.
Accepted means status 'filtered' with an observation at this exposure, which is
the only state downstream gates may treat as new motion.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pose_sim.pipeline import read_jsonl
from pose_sim.multiview_temporal import MultiViewTemporal
from pose_sim.human_motion import HumanMotionEstimator
from pose_sim.motion_evaluation import statistics
from pose_sim.fault_injection import BODY12,FAULT_TYPES,mirror,plan,inject
from pose_sim.geometry import project
from scripts.benchmark_recovery import fingerprint
from scripts.import_mpi3dhp import sha256

METHODS=dict(default=dict(track_confirmation_frames=1),confirm2=dict(track_confirmation_frames=2),
             confirm3=dict(track_confirmation_frames=3))
TAU_MM=150.;GOOD_RUN=3;RECOVERY_CAP_S=1.;CLEAN_READY_FRAMES=5;MAX_LAG=5
SINGLE_VIEW=('view_dropout','view_outlier','view_lr_swap','view_time_offset')
MIRROR=[BODY12.index(mirror(n)) for n in BODY12]


def run(detections,cal,geometry,state_cfg):
    """Return body12 positions (T,12,3) m, present and accepted masks (T,12)."""
    fusion=MultiViewTemporal(cal,geometry,structured=False,observation_consistency=True)
    state=HumanMotionEstimator(state_cfg);previous=None
    pos=np.full((len(detections),12,3),np.nan);accepted=np.zeros((len(detections),12),bool)
    for f,detection in enumerate(detections):
        pose,assigned=fusion.update(detection,previous)
        previous=state.update(pose,cal['cameras'],assigned)
        index={n:i for i,n in enumerate(previous['joint_names'])}
        for j,n in enumerate(BODY12):
            i=index[n];p=previous['joints_world_m'][i];s=previous['joint_states'][i]
            if p is None:continue
            pos[f,j]=p
            accepted[f,j]=s['status']=='filtered' and s['last_observed_timestamp_ns']==previous['timestamp_ns']
    return dict(pos=pos,present=np.isfinite(pos[...,0]),accepted=accepted)


def derived(r,g):
    err=np.linalg.norm(r['pos']-g,axis=-1)*1000
    mirror_err=np.linalg.norm(r['pos']-g[:,MIRROR],axis=-1)*1000
    with np.errstate(invalid='ignore'):
        bad=r['accepted']&(err>TAU_MM)
        good=r['accepted']&(err<=TAU_MM)
        swap=r['accepted']&(mirror_err<err)&(err>TAU_MM)
    return dict(err=err,bad=bad,good=good,swap=swap)


def first_good_run(good,start,stop):
    for f in range(start,max(start,stop-GOOD_RUN+1)):
        if good[f:f+GOOD_RUN].all():return f
    return None


def max_steps(r,g,j,frames,accepted_only):
    """Largest ||Δoutput-Δreference|| (mm) within frames; accepted_only compares
    each accepted frame with the previous accepted frame, i.e. what a gate that
    holds the last accepted value would emit on resumption."""
    last=None;worst=0.
    for f in range(frames[0]-1,frames[-1]+1):
        ok=r['accepted'][f,j] if accepted_only else r['present'][f,j]
        if not ok:
            if not accepted_only:last=None
            continue
        if last is not None and f>=frames[0]:
            worst=max(worst,float(np.linalg.norm((r['pos'][f,j]-r['pos'][last,j])-(g[f,j]-g[last,j]))*1000))
        last=f
    return worst


def last_accepted_before(r,j,f):
    hits=np.flatnonzero(r['accepted'][:f,j])
    return hits[-1] if len(hits) else None


def control_step(r,g,j,frames):
    first=frames[0];last=last_accepted_before(r,j,first);worst=0.
    for f in frames:
        if not r['accepted'][f,j]:continue
        if last is not None:
            worst=max(worst,float(np.linalg.norm((r['pos'][f,j]-r['pos'][last,j])-(g[f,j]-g[last,j]))*1000))
        last=f
    return worst


def best_lag(r,g,j,frames):
    f=np.array([x for x in frames if r['accepted'][x,j]])
    if len(f)<10:return None
    scores=[]
    for lag in range(-MAX_LAG,MAX_LAG+1):
        src=f-lag;ok=(src>=0)&(src<len(g))
        if ok.sum()<10:scores.append(np.inf);continue
        d=r['pos'][f[ok],j]-g[src[ok],j];d-=d.mean(axis=0)
        scores.append(float((d*d).sum(axis=1).mean()))
    return int(np.argmin(scores))-MAX_LAG


def score_event(event,fault,clean,dc,df,g,fps):
    """clean=None scores a natural event in absolute terms (no paired control)."""
    s,e=event['start_frame'],event['end_frame'];post_end=min(len(g),e+int(RECOVERY_CAP_S*fps))
    J=[BODY12.index(n) for n in event['affected_joints']]
    window=slice(s,post_end);during=slice(s,e);post=slice(e,post_end)
    paired=clean is not None
    if not paired:clean,dc=fault,{k:np.zeros_like(v) for k,v in df.items()}
    attributable=lambda key,sl:int((df[key][sl][:,J]&~dc[key][sl][:,J]).sum())
    frames=list(range(s,post_end));recovery=[];unrecovered=0;eligible=0;lag=[]
    for j in J:
        ready=first_good_run(dc['good'][:,j],e,post_end)
        if not paired or (ready is not None and ready-e<=CLEAN_READY_FRAMES):
            eligible+=1;hit=first_good_run(df['good'][:,j],e,post_end)
            if hit is None:unrecovered+=1
            else:recovery.append((hit-e)/fps*1000)
        if not paired:continue
        lag_window=list(range(max(1,s-int(.5*fps)),post_end))
        a,b=best_lag(fault,g,j,lag_window),best_lag(clean,g,j,lag_window)
        if a is not None and b is not None:lag.append(abs(a)-abs(b))
    jumps=[max_steps(fault,g,j,frames,False) for j in J];clean_jumps=[max_steps(clean,g,j,frames,False) for j in J]
    control=[control_step(fault,g,j,frames) for j in J];clean_control=[control_step(clean,g,j,frames) for j in J]
    clean_accepted=int(clean['accepted'][window][:,J].sum()) if paired else 0
    lost=int((clean['accepted'][window][:,J]&~fault['accepted'][window][:,J]).sum()) if paired else 0
    return dict(id=event['id'],type=event['type'],duration_ms=event['duration_ms'],applied_frames=event['applied_frames'],
        joint_frames_during=len(J)*(e-s),joint_frames_post=len(J)*(post_end-e),
        false_accepts_during=attributable('bad',during),false_accepts_post=attributable('bad',post),
        identity_swaps=attributable('swap',window),clean_accepted=clean_accepted,availability_lost=lost,
        recovery_eligible=eligible,recovery_ms=recovery,unrecovered=unrecovered,
        max_output_step_mm=max(jumps),clean_max_output_step_mm=max(clean_jumps),
        max_control_step_mm=max(control),clean_max_control_step_mm=max(clean_control),
        phase_lag_excess_frames=lag)


def natural_events(mask,fps,min_frames=2):
    """Runs where mask is True for >=min_frames, then False again; tail must fit."""
    events=[];f=0
    while f<len(mask):
        if mask[f]:
            e=f
            while e<len(mask) and mask[e]:e+=1
            if e-f>=min_frames and e<len(mask):
                events.append(dict(id=f'n{f}',type='natural',duration_ms=round((e-f)/fps*1000),start_frame=f,end_frame=e,
                                   affected_joints=list(BODY12),applied_frames=e-f))
            f=e
        else:f+=1
    return events


def in_frustum_counts(g,cal):
    counts=np.zeros(g.shape[:2],int)
    flat=np.nan_to_num(g.reshape(-1,3))
    for cam in cal['cameras'].values():
        depth=(flat@np.asarray(cam['R_world_to_camera']).T+np.asarray(cam['t_world_to_camera_m']))[:,2]
        uv=project(flat,cam)
        inside=((depth>.1)&(uv[:,0]>=0)&(uv[:,0]<cam['width'])&(uv[:,1]>=0)&(uv[:,1]<cam['height'])
                &np.isfinite(g.reshape(-1,3)).all(axis=1))
        counts+=inside.reshape(g.shape[:2])
    return counts


def summarise(events):
    def group(items):
        if not items:return None
        fd=sum(x['joint_frames_during'] for x in items);fp=sum(x['joint_frames_post'] for x in items)
        rec=[v for x in items for v in x['recovery_ms']];elig=sum(x['recovery_eligible'] for x in items)
        ca=sum(x['clean_accepted'] for x in items)
        return dict(events=len(items),applied_frames=sum(x['applied_frames'] for x in items),
            false_accept_rate=(sum(x['false_accepts_during']+x['false_accepts_post'] for x in items))/max(1,fd+fp),
            false_accepts=sum(x['false_accepts_during']+x['false_accepts_post'] for x in items),
            false_accepts_post=sum(x['false_accepts_post'] for x in items),
            identity_swaps=sum(x['identity_swaps'] for x in items),
            availability_loss=sum(x['availability_lost'] for x in items)/max(1,ca),
            recovery_ms=statistics(rec),unrecovered_fraction=sum(x['unrecovered'] for x in items)/max(1,elig),
            recovery_eligible=elig,
            output_step_mm=statistics([x['max_output_step_mm'] for x in items]),
            control_step_mm=statistics([x['max_control_step_mm'] for x in items]),
            control_step_excess_mm=statistics([x['max_control_step_mm']-x['clean_max_control_step_mm'] for x in items]),
            phase_lag_excess_frames=statistics([v for x in items for v in x['phase_lag_excess_frames']]))
    out=dict(all=group(events),single_view=group([x for x in events if x['type'] in SINGLE_VIEW]))
    for kind in sorted({x['type'] for x in events}):
        out[kind]=group([x for x in events if x['type']==kind])
        for ms in sorted({x['duration_ms'] for x in events if x['type']==kind}):
            if not kind.startswith('natural'):
                out[f'{kind}@{ms}ms']=group([x for x in events if x['type']==kind and x['duration_ms']==ms])
    return out


def whole_run(r,d):
    with np.errstate(invalid='ignore'):
        return dict(coverage=float(r['accepted'].mean()),outputs_over_500mm=int((d['err']>500).sum()),
                    accepted_over_500mm=int((r['accepted']&(d['err']>500)).sum()),
                    accepted_over_150mm=int(d['bad'].sum()),accepted_identity_swaps=int(d['swap'].sum()))


def main(a):
    if a.output.exists():raise FileExistsError('Preserve existing benchmark evidence')
    code=fingerprint()
    if a.expected_hash and code!=a.expected_hash:raise ValueError('Candidate changed after freeze')
    settings=json.loads((ROOT/'config/vision_motion.json').read_text())
    geometry=json.loads((ROOT/'config/demo.json').read_text())
    manifest=json.loads((a.dataset/'manifest.json').read_text())
    fps=manifest['configuration']['fps'];methods={m:METHODS[m] for m in a.methods}
    report=dict(implementation_sha256=code,dataset_sha256=sha256(a.dataset/'manifest.json'),split=a.split,seeds=a.seeds,
        methods=methods,state_settings=settings['state'],tau_mm=TAU_MM,fps=fps,clips=[],
        definitions=dict(accepted="status 'filtered' and last observation at this exposure",
            false_accept='accepted with error >tau in the faulted run while the paired clean run is not, during the event or the 1 s after it',
            availability_loss='clean-accepted affected joint-frames that the faulted run does not accept (event + 1 s)',
            recovery='first start of 3 consecutive accepted frames within tau after event end; scored only if the clean run does so within 5 frames; cap 1 s',
            control_step='largest ||Δoutput-Δreference|| between consecutive accepted frames, gaps bridged to the last accepted value',
            identity_swap='accepted, error >tau and nearer the mirror reference joint',
            phase_lag_excess='|best lag| faulted minus clean (frames, ±5) over accepted frames from 0.5 s before to 1 s after'),
        limitations=['Injected faults alter detector output only; they are not physical occlusions.',
            'MPI-INF-3DHP reference is markerless multi-view capture, not Vicon.',
            'In-frustum counts do not imply visibility; body-level minimum over joints.',
            'Offline replay; no live exposure latency or hardware.'])
    events={m:[] for m in methods};natural={m:[] for m in methods};runs={m:[] for m in methods}
    for clip in manifest['clips']:
        if a.split and clip.get('split')!=a.split:continue
        if a.clips and clip['id'] not in a.clips:continue
        cid=clip['id'];cal=json.loads((a.dataset/'inputs'/cid/'cameras.json').read_text())
        detections=list(read_jsonl(a.cache/cid/'detections.jsonl'))
        truth=list(read_jsonl(a.dataset/'references'/(cid+'.jsonl')))
        if [d['timestamp_ns'] for d in detections]!=[t['timestamp_ns'] for t in truth]:raise ValueError('Exposure keys mismatch')
        g=np.array([[p if (p:=dict(zip(t['joint_names'],t['joints_world_m'])).get(n)) is not None else [np.nan]*3
                     for n in BODY12] for t in truth],float)
        usable=np.array([sum(v.get('keypoints_px') is not None for v in d['views'].values()) for d in detections])
        frustum=in_frustum_counts(g,cal).min(axis=1)
        natural_sets=dict(usable_views=natural_events(usable<2,fps),in_frustum=natural_events(frustum<2,fps))
        plans={s:inject(detections,plan(detections,s,fps)) for s in a.seeds}
        clip_report=dict(id=cid,detections_sha256=sha256(a.cache/cid/'detections.jsonl'),
            natural_event_counts={k:len(v) for k,v in natural_sets.items()},
            labels={s:p[1] for s,p in plans.items()},methods={})
        for m,cfg in methods.items():
            state_cfg={**settings['state'],**cfg}
            clean=run(detections,cal,geometry,state_cfg);dc=derived(clean,g)
            entry=dict(clean=whole_run(clean,dc),faulted={})
            for kind,items in natural_sets.items():
                natural[m].extend({**score_event(e,clean,None,None,dc,g,fps),'type':'natural_'+kind,'clip':cid}
                                  for e in items)
            for s,(faulted,labels) in plans.items():
                fr=run(faulted,cal,geometry,state_cfg);df=derived(fr,g)
                entry['faulted'][s]=whole_run(fr,df)
                events[m].extend({**score_event(e,fr,clean,dc,df,g,fps),'clip':cid,'seed':s} for e in labels)
            clip_report['methods'][m]=entry
            print(cid,m,'clean',entry['clean'],'faulted',[v['coverage'] for v in entry['faulted'].values()],flush=True)
        report['clips'].append(clip_report)
    if not report['clips']:raise ValueError('No clips selected')
    report['summary']={m:dict(injected=summarise(events[m]),natural=summarise(natural[m])) for m in methods}
    report['events']=events;report['natural_events']=natural
    a.output.mkdir(parents=True)
    (a.output/'report.json').write_text(json.dumps(report,indent=1,allow_nan=False),encoding='utf-8')
    for m in methods:
        s=report['summary'][m]['injected']['all']
        print(m,{k:s[k] for k in ('false_accept_rate','false_accepts','identity_swaps','availability_loss','unrecovered_fraction')},
              'recovery',s['recovery_ms'],'control_step',s['control_step_mm'])
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--split');p.add_argument('--clips',nargs='*')
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--methods',nargs='+',default=list(METHODS),choices=list(METHODS))
    p.add_argument('--expected-hash')
    main(p.parse_args())
