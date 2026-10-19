"""Independent body/detail-head association and conservative attachment checks.

Inputs: named pixel observations, or calibrated world points in metres with
triangulation quality. Timestamps and world axes are inherited, never changed.
Only whole detail groups are relabelled in 2D; 3D coordinates are never snapped.
Unknown anchors yield unknown attachment. Bounds are plausibility heuristics,
not anatomical measurements, identity guarantees, or calibrated probabilities.
"""
import copy
import numpy as np


DEFAULTS = dict(wrist_offset_m=.025, wrist_forearm_ratio=.15,
                palm_forearm_ratio=.85, foot_shank_ratio=.80,
                extent_offset_m=.025, uncertainty_multiplier=2.,
                pixel_margin=5., pixel_floor=8., pixel_limb_ratio=.25)


def groups():
    for side in ('left', 'right'):
        yield side+'_palm', side+'_wrist', side+'_elbow', [
            side+'_hand_'+n for n in ('wrist','thumb_mcp','index_mcp','middle_mcp','pinky_mcp')]
        yield side+'_foot', side+'_ankle', side+'_knee', [
            side+'_'+n for n in ('big_toe','small_toe','heel')]


def associate_details(row, confidence_threshold=.35, config=None):
    """Resolve independently swapped detail heads using two visible body anchors.

    Foot reassignment requires the original whole-body model's ankle anchors;
    older recordings without them remain usable, without invented foot anchors.
    """
    cfg={**DEFAULTS, **(config or {})}; out=copy.deepcopy(row)
    names=row['joint_names']; index={n:i for i,n in enumerate(names)}; changes=[]
    for cid,view in out['views'].items():
        if view.get('keypoints_px') is None:continue
        xy=np.asarray(view['keypoints_px'],float); scores=np.asarray(view['scores'],float)
        mapping=list(range(len(names)))
        for kind,anchor,proximal,parts in (
                ('palm','wrist','elbow',('hand_wrist','hand_thumb_mcp','hand_index_mcp','hand_middle_mcp','hand_pinky_mcp')),
                ('foot','ankle','knee',('big_toe','small_toe','heel'))):
            ns=[s+'_'+anchor for s in ('left','right')]
            if not all(n in index and scores[index[n]]>=confidence_threshold for n in ns):continue
            detail=[[index.get(s+'_'+p) for p in parts] for s in ('left','right')]
            if any(j is None for js in detail for j in js):continue
            body=xy[[index[n] for n in ns]]; roots=[]; lengths=[]
            for side,js in zip(('left','right'),detail):
                if kind=='palm':
                    if scores[js[0]]<confidence_threshold:break
                    roots.append(xy[js[0]])
                else:
                    a=view.get('auxiliary_model_anchors',{}).get(side+'_ankle',{})
                    if a.get('score',0)<confidence_threshold:break
                    roots.append(np.asarray(a['keypoint_px'],float))
                p=index.get(side+'_'+proximal)
                lengths.append(np.linalg.norm(body[len(lengths)]-xy[p]) if p is not None and scores[p]>=confidence_threshold else 0.)
            if len(roots)!=2 or not np.isfinite(roots).all():continue
            roots=np.asarray(roots); tolerance=cfg['pixel_floor']+cfg['pixel_limb_ratio']*np.asarray(lengths)
            original=np.linalg.norm(roots-body,axis=1); swapped=np.linalg.norm(roots[::-1]-body,axis=1)
            # Close/crossing anchors cannot establish semantic identity in one view.
            if (np.linalg.norm(body[0]-body[1])>2*max(tolerance)
                    and np.all(swapped<tolerance)
                    and np.all(original-swapped>cfg['pixel_margin'])):
                for a,b in zip(*detail):mapping[a],mapping[b]=b,a
                changes.append(dict(camera=cid,group=kind,minimum_gain_px=float(min(original-swapped))))
                if kind=='foot':
                    anchors=view['auxiliary_model_anchors']
                    anchors['left_ankle'],anchors['right_ankle']=anchors['right_ankle'],anchors['left_ankle']
        view['keypoints_px']=[view['keypoints_px'][j] for j in mapping]
        view['scores']=[view['scores'][j] for j in mapping]
        old=view.get('assigned_source_indices',list(range(len(names))))
        view['assigned_source_indices']=[old[j] for j in mapping]
    out['auxiliary_association']=dict(corrections=changes,identity_certified=False)
    return out


def attachment_checks(points, sigmas, config=None):
    """Return group diagnostics and invalid point names, without mutating input.

    Missing body/scale anchors do not delete detail observations. They disable
    derived orientation/support certification in the state consumer instead.
    """
    cfg={**DEFAULTS, **(config or {})}; diagnostics={}; rejected=set()
    for group,anchor,proximal,detail in groups():
        available=[n for n in detail if n in points]
        info=dict(status='unknown',rejected=[],reason='missing current body, scale or detail anchor')
        diagnostics[group]=info
        if anchor not in points or proximal not in points or not available:continue
        length=float(np.linalg.norm(points[anchor]-points[proximal]))
        if not np.isfinite(length) or length<.08:continue
        checks=[]
        for n in available:
            ratio=cfg['foot_shank_ratio'] if group.endswith('_foot') else cfg['palm_forearm_ratio']
            offset=cfg['extent_offset_m']
            if n.endswith('hand_wrist'):ratio=cfg['wrist_forearm_ratio'];offset=cfg['wrist_offset_m']
            uncertainty=cfg['uncertainty_multiplier']*np.hypot(sigmas.get(n,0.),sigmas.get(anchor,0.))
            bound=offset+ratio*length+uncertainty
            distance=float(np.linalg.norm(points[n]-points[anchor]))
            checks.append(dict(joint=n,distance_m=distance,bound_m=float(bound),passed=bool(distance<=bound)))
        bad=[c['joint'] for c in checks if not c['passed']]
        # A detached duplicate wrist invalidates its entire palm frame. A single
        # bad MCP/toe only invalidates that point (and any frame requiring it).
        if any(n.endswith('hand_wrist') for n in bad):bad=available
        rejected.update(bad)
        info.update(status='rejected' if bad else 'supported',rejected=bad,checks=checks,
                    reason='scale-aware attachment plausibility; not proof of identity')
    return diagnostics,rejected


def filter_measurements(row, config=None):
    out=copy.deepcopy(row); points={};sigmas={}
    for n,p,q in zip(row['joint_names'],row['joints_world_m'],row['quality']):
        if (p is not None and q and np.isfinite(p).all() and q.get('linearized_max_std_m',np.inf)<=.08
                and q.get('ray_angle_deg',0)>=2 and q.get('mean_reprojection_px',np.inf)<=8
                and len(set(q.get('inlier_cameras',[])))>=2):
            points[n]=np.asarray(p);sigmas[n]=q['linearized_max_std_m']
    diagnostic,bad=attachment_checks(points,sigmas,config)
    for i,n in enumerate(row['joint_names']):
        if n in bad:out['joints_world_m'][i]=None;out['quality'][i]=None
    out['attachment_measurements']=diagnostic
    return out
