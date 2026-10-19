"""Independent, coverage-aware motion metrics and held-out uncertainty scaling.

Only evaluation/calibration code may consume reference labels. Metrics use the
world frame without rigid/procrustes alignment, and count absent outputs.
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import JOINTS


def statistics(x):
    x=np.asarray(x,float)
    return dict(mean=float(np.mean(x)),p95=float(np.percentile(x,95)),maximum=float(np.max(x)),count=len(x)) if len(x) else None


def compare_methods(methods, truth, threshold_m=.1):
    """Paired world-frame errors and all-target success; never reward dropping points.

    Input: method -> timestamped named estimates, independent reference rows.
    Output: mm errors on identical targets, and coverage including all missing
    frames/joints. Predictions and current observations have separate denominators.
    Reference landmark rotations are compared only when every method emits one.
    """
    indices={m:{r['frame_id']:r for r in rows} for m,rows in methods.items()}
    if not indices:raise ValueError('At least one method is required')
    common={m:[] for m in methods};observed_common={m:[] for m in methods}
    rotations={m:[] for m in methods};per_joint={m:{} for m in methods}
    counts={m:dict(available=0,current_observations=0,predictions=0,transitions=0,within_threshold=0,
                  current_within_threshold=0) for m in methods};targets=0
    for gt in truth:
        reference_names=gt.get('joint_names',JOINTS)
        rows={m:index.get(gt['frame_id']) for m,index in indices.items()}
        points={m:dict(zip(r['joint_names'],r['joints_world_m'])) if r else {} for m,r in rows.items()}
        status={m:dict(zip(r['joint_names'],r.get('joint_states',[]))) if r else {} for m,r in rows.items()}
        for m,r in rows.items():
            if r and r['timestamp_ns']!=gt['timestamp_ns']:raise ValueError('Comparison requires matching exposure times')
        for n,target in zip(reference_names,gt['joints_world_m']):
            if target is None:continue
            targets+=1;errors={};current={}
            for m in methods:
                p=points[m].get(n)
                if p is None:continue
                error=float(np.linalg.norm(np.array(p)-target));errors[m]=error*1000
                current[m]=status[m].get(n,{}).get('status','filtered')=='filtered'
                counts[m]['available']+=1;counts[m]['within_threshold']+=int(error<=threshold_m)
                counts[m]['current_observations']+=int(current[m])
                counts[m]['predictions']+=int(status[m].get(n,{}).get('status')=='predicted')
                counts[m]['transitions']+=int(not current[m] and status[m].get(n,{}).get('status')!='predicted')
                counts[m]['current_within_threshold']+=int(current[m] and error<=threshold_m)
            if len(errors)==len(methods):
                for m,e in errors.items():
                    common[m].append(e);per_joint[m].setdefault(n,[]).append(e)
                if all(current.values()):
                    for m,e in errors.items():observed_common[m].append(e)
        gt_points={n:np.array(p) for n,p in zip(reference_names,gt['joints_world_m']) if p is not None}
        for side in ('left','right'):
            for kind in ('foot','palm'):
                name=side+'_'+kind
                rr={m:r.get('orientations',{}).get(name) if r else None for m,r in rows.items()}
                if any(v is None for v in rr.values()):continue
                try:target=frame_rotation(gt_points,side,kind)
                except KeyError:continue
                for m,value in rr.items():
                    rotations[m].append(float(np.linalg.norm(Rotation.from_matrix(np.array(value)@target.T).as_rotvec())*180/np.pi))
    return dict(position_targets=targets,threshold_m=threshold_m,methods={m:dict(
        **counts[m],coverage=counts[m]['available']/targets if targets else None,
        all_target_success=counts[m]['within_threshold']/targets if targets else None,
        current_all_target_success=counts[m]['current_within_threshold']/targets if targets else None,
        common_position_error_mm=statistics(common[m]),common_current_error_mm=statistics(observed_common[m]),
        common_rotation_error_deg=statistics(rotations[m]),
        common_joint_errors_mm={n:statistics(v) for n,v in per_joint[m].items()}) for m in methods})


def frame_rotation(p,side,kind):
    if kind=='foot':
        a,b,c=(p[side+'_'+n] for n in ('heel','big_toe','small_toe'))
        x=(b+c)/2-a;y=c-b if side=='left' else b-c
    else:
        a,b,c,d=(p[side+'_hand_'+n] for n in ('wrist','middle_mcp','index_mcp','pinky_mcp'))
        x=b-a;y=c-d
    x=x/np.linalg.norm(x);y=y-x*(x@y);y=y/np.linalg.norm(y)
    return np.column_stack((x,y,np.cross(x,y)))


def phase_lag(rows,truth,max_lag=8):
    # A descriptive delay estimate, only for a sufficiently moving wrist/ankle.
    result={};index={r['frame_id']:r for r in rows}
    for joint in ('left_wrist','right_wrist','left_ankle','right_ankle'):
        y=[];x=[];ts=[]
        for gt in truth:
            r=index.get(gt['frame_id'])
            if r is None:continue
            i=gt['joint_names'].index(joint);j=r['joint_names'].index(joint)
            y.append(gt['joints_world_m'][i]);x.append(r['joints_world_m'][j] or [np.nan]*3);ts.append(gt['timestamp_ns'])
        if len(x)<max_lag*3:continue
        x=np.array(x);y=np.array(y);axis=int(np.argmax(np.var(y,axis=0)))
        if np.std(y[:,axis])<.025:continue
        costs=[]
        for lag in range(-max_lag,max_lag+1):
            # positive lag means the estimator follows the source late
            lo=max(0,lag);hi=min(len(x),len(x)+lag)
            a=x[lo:hi,axis];b=y[lo-lag:hi-lag,axis];valid=np.isfinite(a)&np.isfinite(b)
            if valid.sum()<len(x)*.7:continue
            a=a[valid];b=b[valid];costs.append((float(np.mean(((a-a.mean())-(b-b.mean()))**2)),lag))
        if costs:result[joint]=min(costs)[1]*float(np.median(np.diff(ts)))/1e6
    return result


def evaluate_motion(rows,truth):
    index={r['frame_id']:r for r in rows};errors=[];velocity=[];rotations=[];radii=[];normalized=[]
    observed_errors=[];possible=0;within=0;shape_errors=[];recover=[];slide=[];contact=[0,0,0,0];contact_known=0;torso_rotations=[]
    previous=None;predicted=0;observed=0
    for gt in truth:
        r=index.get(gt['frame_id']);possible+=len(gt['joint_names'])
        if r is None:continue
        points=dict(zip(r['joint_names'],r['joints_world_m']));pgt={n:np.array(p) for n,p in zip(gt['joint_names'],gt['joints_world_m'])}
        states=dict(zip(r['joint_names'],r.get('joint_states',[])))
        vv=dict(zip(r['joint_names'],r.get('velocities_world_m_s',[])))
        for i,n in enumerate(gt['joint_names']):
            if points.get(n) is None:continue
            e=float(np.linalg.norm(np.array(points[n])-pgt[n]));errors.append(e*1000);within+=e<=.1
            s=states.get(n,{})
            if s.get('status')=='predicted':predicted+=1
            else:observed_errors.append(e*1000);observed+=1
            if vv.get(n) is not None and gt.get('velocities_world_m_s') is not None:
                velocity.append(float(np.linalg.norm(np.array(vv[n])-gt['velocities_world_m_s'][i])))
            if s.get('max_std_m') is not None:normalized.append(e/max(.005,s['max_std_m']))
            radius=s.get('error_radius_m')
            if radius is not None:radii.append((e<=radius,radius))
            if previous and not gt['events']['occlusion'] and previous[0]['events']['occlusion']:
                oldp=dict(zip(previous[1]['joint_names'],previous[1]['joints_world_m'])).get(n)
                oldgt=dict(zip(previous[0]['joint_names'],previous[0]['joints_world_m'])).get(n)
                if oldp is not None and oldgt is not None:
                    recover.append(float(np.linalg.norm((np.array(points[n])-oldp)-(pgt[n]-oldgt))*1000))
        for name,rotation in r.get('orientations',{}).items():
            if rotation is None:continue
            target=gt['root_rotation_world'] if name=='torso' else frame_rotation(pgt,*name.split('_'))
            error=float(np.linalg.norm(Rotation.from_matrix(np.array(rotation)@np.array(target).T).as_rotvec())*180/np.pi)
            (torso_rotations if name=='torso' else rotations).append(error)
        for k,v in r.get('shape',{}).items():
            if v['status']=='stable' and k in gt['bone_lengths_m']:shape_errors.append(abs(v['length_m']-gt['bone_lengths_m'][k])*1000)
        for side in ('left','right'):
            pred=r.get('support',{}).get('feet',{}).get(side,{}).get('state','unknown');actual=gt['contacts'][side]
            if 'support' not in r and r.get('contacts',{}).get(side) is not None:
                pred='contact' if r['contacts'][side]['stationary_contact_score']>.65 else 'swing'
            if pred!='unknown':
                contact_known+=1;contact[0 if pred=='contact' and actual else 1 if pred=='contact' else 2 if actual else 3]+=1
            elif actual:contact[2]+=1
            if actual and previous:
                old=dict(zip(previous[1]['joint_names'],previous[1]['joints_world_m']));dt=(r['timestamp_ns']-previous[1]['timestamp_ns'])/1e9
                n=side+'_heel'
                if points.get(n) is not None and old.get(n) is not None and dt>0:
                    slide.append(float(np.linalg.norm((np.array(points[n])-old[n])[:2])/dt))
        previous=(gt,r)
    tp,fp,fn,tn=contact
    return dict(position_error_mm=statistics(errors),current_observation_error_mm=statistics(observed_errors),
        coverage=len(errors)/possible,within_100mm_all_targets=within/possible,observed_points=observed,predicted_points=predicted,
        velocity_error_m_s=statistics(velocity),landmark_rotation_error_deg=statistics(rotations),torso_rotation_error_deg=statistics(torso_rotations),shape_error_mm=statistics(shape_errors),
        occlusion_recovery_excess_step_mm=statistics(recover),foot_slide_when_true_contact_m_s=statistics(slide),
        contact_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,contact_confusion=dict(tp=tp,fp=fp,fn=fn,tn=tn),contact_known_fraction=contact_known/(len(truth)*2),
        phase_lag_ms=phase_lag(rows,truth),normalized_error_max=max(normalized) if normalized else None,
        uncertainty_coverage=float(np.mean([v[0] for v in radii])) if radii else None,
        uncertainty_radius_m=statistics([v[1] for v in radii]),position_targets=possible)


def fit_uncertainty(sequence_scores,domain,configuration_sha256,coverage=.90):
    scores=sorted(float(v) for v in sequence_scores if v is not None and np.isfinite(v))
    rank=math.ceil((len(scores)+1)*coverage)
    if rank>len(scores) or len(scores)<10:raise ValueError('Insufficient calibration sequences for requested quantile')
    return dict(schema_version=1,domain=domain,configuration_sha256=configuration_sha256,
        multiplier=scores[rank-1],nominal_coverage=coverage,calibration_sequences=len(scores),
        calibration_unit='whole-sequence maximum normalized joint error',guarantee_claimed=False,
        limitation='Empirical block calibration; shared procedural motion/subjects violate independent real-world exchangeability.')


def apply_uncertainty(row,calibration,domain,configuration_sha256):
    valid=bool(calibration and calibration['domain']==domain and calibration['configuration_sha256']==configuration_sha256)
    row['uncertainty_calibration']=dict(valid_for_domain=valid,domain=domain,guarantee_claimed=False)
    for state in row['joint_states']:
        std=state.get('max_std_m')
        state['error_radius_m']=max(.005,std)*calibration['multiplier'] if valid and std is not None else None
    return row
