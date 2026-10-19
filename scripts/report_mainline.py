"""Score completed module audits and render a local illustrated Markdown report.

Scoring only: reference labels never enter a perception or control estimator.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from scipy.stats import chi2
from pose_sim.pipeline import read_jsonl
from pose_sim.geometry import JOINTS, project, triangulate
from scripts.evaluate_mpi3dhp import extra_metrics, packets, EDGES
from scripts.validate_mainline import load, save, sha

BODY = JOINTS[5:]
CN = ['左肩','右肩','左肘','右肘','左腕','右腕','左髋','右髋','左膝','右膝','左踝','右踝']
COLORS = ['#4773ac','#e67d32','#269589']
METHODS = ['poses3d','fused','states3d']
LABELS = ['原始几何','观测校验后','时序状态']
CLIPS = dict(s1_crouch='S1 蹲伸',s1_exercise='S1 运动',s1_walk='S1 行走',
             s2_floor='S2 地面',s2_misc='S2 综合',s2_sports='S2 体育',synthetic_walk='合成行走')


def stats(values):
    x=np.asarray(values,float).ravel();x=x[np.isfinite(x)]
    return dict(count=len(x),mean=float(x.mean()),p50=float(np.median(x)),p95=float(np.percentile(x,95)),maximum=float(x.max())) if len(x) else None


def arr(rows,names=BODY):
    return np.array([[dict(zip(r.get('joint_names',JOINTS),r['joints_world_m'])).get(n) or [np.nan]*3 for n in names] for r in rows],float)


def error_summary(errors,current=None):
    good=np.isfinite(errors)
    result=dict(targets=int(errors.size),valid=int(good.sum()),coverage=float(good.mean()),error_mm=stats(errors),
                all_target_within_100mm=float((errors<=100).mean()),over_500mm=int((errors>500).sum()))
    if current is not None:
        result.update(current_coverage=float(current.mean()),predicted_or_missing_fraction=float((~current).mean()),
                      current_error_mm=stats(errors[current]),current_over_150mm=int(((errors>150)&current).sum()))
    return result


def current_mask(rows):
    return np.array([[dict(zip(r['joint_names'],r.get('joint_states',[]))).get(n,{}).get('status')=='filtered' for n in BODY] for r in rows])


def assert_alignment(rows,truth):
    if [(r['frame_id'],r['timestamp_ns']) for r in rows]!=[(r['frame_id'],r['timestamp_ns']) for r in truth]:
        raise ValueError('Exact frame/timestamp alignment required; no interpolation or time fitting')


def score(output):
    manifest=load(output/'manifest.json'); report=dict(clips={},reference_scope='MPI markerless multiview reference; synthetic skeleton proxy; no independent anatomical/contact truth')
    all_errors={m:[] for m in METHODS};all_cur=[];all_2d=[];all_mahal=[];data={}
    for clip in manifest['clips']:
        cid=clip['id'];folder=output/'inference'/cid
        gtpath=ROOT/'outputs/human_walk_g1/ground_truth.jsonl' if clip['synthetic'] else ROOT/'outputs/public_datasets/mpi_benchmark/references'/(cid+'.jsonl')
        truth=list(read_jsonl(gtpath))
        rows={m:list(read_jsonl(folder/(m+'.jsonl'))) for m in METHODS}
        detections=list(read_jsonl(folder/'detections.jsonl'));cal=load(folder/'cameras.json')
        for seq in [*rows.values(),detections]:assert_alignment(seq,truth)
        target=arr(truth);points={m:arr(rs) for m,rs in rows.items()}
        errors={m:np.linalg.norm(p-target,axis=-1)*1000 for m,p in points.items()}
        current=current_mask(rows['states3d'])
        cameras=list(cal['cameras']);err2=np.full((len(truth),len(cameras),12),np.nan)
        conf=np.full_like(err2,np.nan);eligible=np.zeros_like(err2,bool);insidebox=np.zeros_like(err2,bool)
        detect_counts=Counter();person_diag=Counter();corrections=ambiguities=0;mahal=[];support=Counter();shape=Counter()
        reproj=[];rays=[];inliers=Counter()
        for i,(det,gt,state,fused) in enumerate(zip(detections,truth,rows['states3d'],rows['fused'])):
            index={n:j for j,n in enumerate(det['joint_names'])}
            for c,cam in enumerate(cameras):
                view=det['views'][cam];detect_counts[str(view['num_people_detected'])]+=1
                xy=project(target[i],cal['cameras'][cam]);eligible[i,c]=np.isfinite(xy).all(1)&(xy[:,0]>=0)&(xy[:,1]>=0)&(xy[:,0]<cal['cameras'][cam]['width'])&(xy[:,1]<cal['cameras'][cam]['height'])
                box=view.get('person_bbox_xyxy_px')
                if box is not None:
                    insidebox[i,c]=(xy[:,0]>=box[0])&(xy[:,0]<=box[2])&(xy[:,1]>=box[1])&(xy[:,1]<=box[3])&eligible[i,c]
                if view.get('keypoints_px') is not None:
                    for j,n in enumerate(BODY):
                        k=index[n];confidence=view['scores'][k];conf[i,c,j]=confidence
                        if confidence>=load(folder/'config.json')['confidence_threshold'] and eligible[i,c,j]:
                            err2[i,c,j]=np.linalg.norm(np.array(view['keypoints_px'][k])-xy[j])
            person_diag[str(det.get('person_selection',{}).get('status','no_status'))]+=1
            corrections+=len(fused.get('association',{}).get('corrections',[]))
            ambiguities+=len(fused.get('association',{}).get('ambiguities',[]))
            support[state.get('support',{}).get('phase','unknown')]+=1
            shape.update(v['status'] for v in state.get('shape',{}).values())
            smap=dict(zip(state['joint_names'],state['joint_states']))
            for j,n in enumerate(BODY):
                cov=smap[n].get('position_covariance_m2')
                if cov is not None and np.isfinite(points['states3d'][i,j]).all():
                    delta=points['states3d'][i,j]-target[i,j]
                    mahal.append(float(delta@np.linalg.pinv(cov)@delta))
            quality=dict(zip(fused['joint_names'],fused['quality']))
            for n in BODY:
                q=quality.get(n)
                if q:
                    reproj.append(q['mean_reprojection_px']);rays.append(q['ray_angle_deg']);inliers[len(q['inlier_cameras'])]+=1
        metrics={m:error_summary(e,current if m=='states3d' else None) for m,e in errors.items()}
        common=np.logical_and.reduce([np.isfinite(e) for e in errors.values()])
        for m in metrics:metrics[m]['common_target_error_mm']=stats(errors[m][common])
        metric2=dict(eligible_targets=int(eligible.sum()),available=int(np.isfinite(err2).sum()),
            coverage=float(np.isfinite(err2).sum()/eligible.sum()),error_px=stats(err2),
            all_target_within_20px=float(np.sum(err2<=20)/eligible.sum()),
            box_encloses_reference_fraction=float(insidebox.sum()/eligible.sum()),
            detection_count_distribution=dict(detect_counts),per_camera={c:stats(err2[:,j]) for j,c in enumerate(cameras)},
            per_joint={n:stats(err2[:,:,j]) for j,n in enumerate(BODY)})
        truthbody=[dict(frame_id=g['frame_id'],timestamp_ns=g['timestamp_ns'],joint_names=BODY,joints_world_m=target[i].tolist()) for i,g in enumerate(truth)]
        kinematics=extra_metrics(rows['states3d'],truthbody)
        timings=list(read_jsonl(folder/'timings.jsonl'))
        report['clips'][cid]=dict(synthetic=clip['synthetic'],frames=len(truth),two_d=metric2,methods=metrics,
            association=dict(corrected_camera_groups=corrections,ambiguous_camera_groups=ambiguities,person_selection=dict(person_diag)),
            geometry=dict(reprojection_px=stats(reproj),ray_angle_deg=stats(rays),inlier_count=dict(inliers)),
            kinematics=kinematics,support_phase_counts=dict(support),shape_status_counts=dict(shape),
            timing_ms={k:stats([r[k] for r in timings]) for k in timings[0]},reference_sha256=sha(gtpath))
        if not clip['synthetic']:
            for m in METHODS:all_errors[m].append(errors[m])
            all_cur.append(current);all_2d.append(err2);all_mahal.extend(mahal)
        data[cid]=dict(truth=truth,rows=rows,points=points,target=target,errors=errors,current=current,err2=err2,
            detection=detections,cal=cal,timings=timings,mahal=mahal,conf=conf)
    cat={m:np.concatenate(es) for m,es in all_errors.items()};cur=np.concatenate(all_cur)
    report['aggregate_real']={m:error_summary(e,cur if m=='states3d' else None) for m,e in cat.items()}
    common=np.logical_and.reduce([np.isfinite(e) for e in cat.values()])
    for m in METHODS:report['aggregate_real'][m]['common_target_error_mm']=stats(cat[m][common])
    nominal=np.array([.5,.8,.9,.95,.99]); observed=[float(np.mean(np.array(all_mahal)<=chi2.ppf(p,3))) for p in nominal]
    report['uncertainty']=dict(nominal=nominal.tolist(),empirical=observed,samples=len(all_mahal),
        interpretation='Gaussian 3D ellipsoid diagnostic only; temporally correlated samples, label/model/calibration bias included; no calibrated confidence guarantee')
    report['two_d_error_px']=stats(np.concatenate(all_2d))
    report['faults']=score_faults(output,data['s1_walk'])
    report['geometry_oracle']=oracle()
    save(output/'metrics.json',report)
    return report,data


def recovery_delay(e,cur,start,end,joints):
    good=np.all((e[:,joints]<=150)&cur[:,joints],axis=1)
    for i in range(end,min(len(good)-2,end+26)):
        if good[i:i+3].all():return (i-end)*40
    return None


def score_faults(output,clean):
    result=[]
    for event in load(output/'faults/events.json'):
        rows=list(read_jsonl(output/'faults'/(event['type']+'.jsonl')));assert_alignment(rows,clean['truth'])
        p=arr(rows);e=np.linalg.norm(p-clean['target'],axis=-1)*1000;cur=current_mask(rows)
        start,end=event['start_frame'],event['end_frame'];js=[BODY.index(n) for n in event['affected_joints']]
        sl=np.s_[start:end,js];post=np.s_[end:end+26,js]
        jumps=[]
        for j in js:
            last=None
            for i in range(start-1,min(len(rows),end+26)):
                if np.isfinite(p[i,j]).all():
                    if last is not None and i>=end:
                        jumps.append(np.linalg.norm((p[i,j]-p[last,j])-(clean['target'][i,j]-clean['target'][last,j]))*1000)
                    last=i
        result.append(dict(**event,current_coverage_during=float(cur[sl].mean()),
            false_current_over_150mm_during=int(((e[sl]>150)&cur[sl]).sum()),
            clean_false_current_over_150mm_during=int(((clean['errors']['states3d'][sl]>150)&clean['current'][sl]).sum()),
            false_current_over_150mm_after=int(((e[post]>150)&cur[post]).sum()),
            recovery_ms=recovery_delay(e,cur,start,end,js),
            clean_recovery_ms=recovery_delay(clean['errors']['states3d'],clean['current'],start,end,js),
            recovery_step_excess_mm=stats(jumps),maximum_prediction_age_ms=max((s.get('observation_age_ns',0) or 0)/1e6
                for r in rows for s in r['joint_states'] if s['status']=='predicted') if any(s['status']=='predicted' for r in rows for s in r['joint_states']) else 0))
    return result


def oracle():
    from tests.test_geometry import camera
    cams=[camera(-.6,distortion=[.1,-.03,.001,-.002,0]),camera(.6),camera(0,.5),camera(0,-.5)]
    rng=np.random.default_rng(19);truth=rng.uniform([-.3,-.3,2.5],[.3,.3,4.5],(50,3));errors=[];outliers=[]
    for p in truth:
        xy=np.array([project(p,c)[0] for c in cams])
        estimate=triangulate(cams,xy,[1.]*4);errors.append(np.linalg.norm(np.array(estimate['xyz_m'])-p)*1000)
        xy[-1]+=[140,-90];estimate=triangulate(cams,xy,[.9]*4)
        outliers.append(np.linalg.norm(np.array(estimate['xyz_m'])-p)*1000)
    small=[camera(0),camera(.0001)];p=[0,0,5]
    return dict(samples=50,ideal_error_mm=stats(errors),one_outlier_error_mm=stats(outliers),
                low_parallax_rejected=triangulate(small,[project(p,c)[0] for c in small],[1,1]) is None,
                scope='Geometry unit experiment only, no detector, no claim of real image accuracy')


def style():
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,
        'font.size':11,'axes.titlesize':14,'axes.labelcolor':'#26364a','text.color':'#26364a',
        'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white',
        'savefig.facecolor':'white','axes.grid':False})


def figsave(output,name,fig):
    fig.savefig(output/'figures'/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(fig)


def plots(output,r,data):
    style();(output/'figures').mkdir(exist_ok=True)
    ids=[c for c in data if c!='synthetic_walk'];x=np.arange(len(ids))
    # Each node has a data figure, even where the available evidence is only a contract check.
    fig,ax=plt.subplots(figsize=(15,5));ax.set(xlim=(-.15,15.15),ylim=(0,5));ax.axis('off')
    blocks=['01 RGB / 同步','02 人体检测框','03 二维关键点','04 关联与校验','05 三维重建',
            '06 时序 / 尺寸','07 旋转 / 支撑','08 可信度门控','09 全身重定向','10 跟踪 / 平衡','11 指令接收','12 在线时延']
    for i,label in enumerate(blocks):
        row=i//6;col=i%6;xx=col*2.5;yy=3.2-row*2.25
        ax.add_patch(FancyBboxPatch((xx+.05,yy),2.25,1,boxstyle='round,pad=.08',edgecolor=COLORS[row],facecolor='#f0f4fa',linewidth=1.5))
        ax.text(xx+1.18,yy+.5,label,ha='center',va='center',fontsize=12)
        if col<5:ax.annotate('',(xx+2.5,yy+.5),(xx+2.3,yy+.5),arrowprops=dict(arrowstyle='->',color='#718097'))
    ax.text(.1,2.6,'增强视觉链路：RGB → 检测 / 重建 → 人体状态；08 质量结果尚未完整接入默认 G1 示例',fontsize=12)
    ax.text(.1,.1,'G1 实测分支：原始三维观测 → 重定向 → MuJoCo / TWIST2 → 模拟指令接收；本轮覆盖两个分支',fontsize=12)
    figsave(output,'01_mainline',fig)
    fig,axs=plt.subplots(1,2,figsize=(13,4))
    for cid in ids:
        ts=np.array([d['timestamp_ns'] for d in data[cid]['detection']],dtype=np.int64)
        axs[0].plot(np.diff(ts)/1e6,label=CLIPS[cid],alpha=.65)
    axs[0].set(xlabel='帧间隔编号',ylabel='ms',title='源文件时间戳间隔');axs[0].legend(ncol=2,fontsize=8)
    axs[1].bar([CLIPS[c] for c in ids],[r['clips'][c]['timing_ms']['skew_ms']['maximum'] for c in ids],color=COLORS[0])
    axs[1].set(ylabel='ms',title='包内最大时间戳偏差（均为数据集时间）',ylim=(0,1));axs[1].tick_params(axis='x',rotation=20)
    figsave(output,'02_input_clock',fig)
    examples(output,data)
    e2=np.concatenate([data[c]['err2'] for c in ids])
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    heat=np.nanmean(e2,axis=0).T;im=axs[0].imshow(heat,aspect='auto',cmap='YlOrRd')
    axs[0].set(xticks=range(4),xticklabels=list(data[ids[0]]['cal']['cameras']),yticks=range(12),yticklabels=CN,
               xlabel='相机',title='二维点平均误差（px）');fig.colorbar(im,ax=axs[0],label='px')
    vals=np.sort(e2[np.isfinite(e2)]);axs[1].plot(vals,np.arange(1,len(vals)+1)/len(vals),color=COLORS[0]);axs[1].set(xlim=(0,150),ylim=(0,1),xlabel='与三维参考投影的距离 / px',ylabel='有效二维点累计比例',title='尾部错误不能由平均值代表')
    axs[1].axvline(20,color=COLORS[1],ls='--',label='20 px 描述阈值');axs[1].legend()
    figsave(output,'04_keypoint_accuracy',fig)
    fig,axs=plt.subplots(1,2,figsize=(13,4))
    axs[0].bar(x-.18,[r['clips'][c]['association']['corrected_camera_groups'] for c in ids],.36,label='更正左右标签',color=COLORS[2])
    axs[0].bar(x+.18,[r['clips'][c]['association']['ambiguous_camera_groups'] for c in ids],.36,label='因歧义撤回',color=COLORS[1])
    axs[0].set(xticks=x,xticklabels=[CLIPS[c] for c in ids],ylabel='相机 × 关节组事件数',title='关联模块实际触发情况');axs[0].legend(fontsize=9)
    for m,l,col in zip(METHODS,LABELS,COLORS):axs[1].plot(x,[r['clips'][c]['methods'][m]['error_mm']['mean'] for c in ids],'o-',label=l,color=col)
    axs[1].set(xticks=x,xticklabels=[CLIPS[c] for c in ids],ylabel='mm',title='同一帧通过三层处理后的误差');axs[1].legend(fontsize=9)
    for a in axs:a.tick_params(axis='x',rotation=20)
    figsave(output,'05_association',fig)
    fig,axs=plt.subplots(1,3,figsize=(15,4))
    for k,(field,title) in enumerate([('error_mm','平均位置误差'),('common_target_error_mm','三方法共同有效点误差')]):
        axs[k].bar(LABELS,[r['aggregate_real'][m][field]['mean'] for m in METHODS],color=COLORS);axs[k].set(ylabel='mm',title=title)
    axs[2].bar(LABELS,[r['aggregate_real'][m]['coverage']*100 for m in METHODS],color=COLORS);axs[2].set(ylabel='%',title='有效输出 / 全部目标',ylim=(0,100))
    figsave(output,'06_reconstruction',fig)
    fig=plt.figure(figsize=(15,5));fig.subplots_adjust(left=.03,right=.92,wspace=.25)
    sample=data['s1_walk'];index=75
    for i,(m,title) in enumerate([('poses3d','原始三角化'),('fused','观测校验后'),('states3d','时序人体状态')]):
        ax=fig.add_subplot(1,3,i+1,projection='3d')
        for points,color,label in [(sample['target'][index],'#b2b9c2','参考'),(sample['points'][m][index],COLORS[i],title)]:
            ax.scatter(points[:,0],points[:,1],points[:,2],color=color,s=12,label=label)
            for a,b in EDGES:
                seg=points[[BODY.index(a),BODY.index(b)]];ax.plot(seg[:,0],seg[:,1],seg[:,2],color=color)
        center=np.nanmean(sample['target'][index],axis=0)
        ax.set(xlim=(center[0]-.8,center[0]+.8),ylim=(center[1]-.8,center[1]+.8),
               zlim=(np.min(sample['target'][index,:,2])-.15,np.max(sample['target'][index,:,2])+.15),xlabel='X / m',ylabel='Y / m',zlabel='Z / m',title=title)
        ax.set_box_aspect((1.6,1.6,2));ax.legend(fontsize=8)
    figsave(output,'06b_skeleton3d',fig)
    d=data['s1_walk'];t=(np.array([g['timestamp_ns'] for g in d['truth']])-d['truth'][0]['timestamp_ns'])/1e9
    fig,axs=plt.subplots(2,1,figsize=(13,7),sharex=True)
    j=BODY.index('left_ankle');axs[0].plot(t,d['target'][:,j,2],color='#424b5b',lw=2,label='参考左踝 Z')
    for m,l,col in zip(METHODS,LABELS,COLORS):axs[0].plot(t,d['points'][m][:,j,2],color=col,label=l,alpha=.8)
    axs[0].set(ylabel='世界 Z / m',title='真实行走：位置、平滑与延迟共同观察');axs[0].legend(ncol=4,fontsize=9)
    a,b=BODY.index('right_knee'),BODY.index('right_ankle')
    for m,l,col in [('target','参考段长','#424b5b'),('poses3d','原始段长',COLORS[0]),('states3d','状态段长',COLORS[2])]:
        p=d['target'] if m=='target' else d['points'][m];axs[1].plot(t,np.linalg.norm(p[:,a]-p[:,b],axis=1),label=l,color=col)
    sh=[row.get('shape',{}).get('right_knee--right_ankle',{}).get('length_m',np.nan) for row in d['rows']['states3d']]
    axs[1].plot(t,sh,'--',color=COLORS[1],label='在线尺寸估计');axs[1].set(xlabel='片段时间 / s',ylabel='右小腿段长 / m',title='段长稳定不等于骨长无偏');axs[1].legend(ncol=4,fontsize=9)
    figsave(output,'07_temporal_shape',fig)
    fault_names=dict(view_dropout='单视角丢失',limb_dropout='单腿跨视角丢失',body_gap='全身丢失',view_outlier='高分错点',view_lr_swap='单视角左右互换',all_lr_swap='全视角左右互换',view_time_offset='错帧但时间戳正常')
    fig,axs=plt.subplots(7,1,figsize=(13,14),sharex=True)
    for a,f in zip(axs,r['faults']):
        rows=list(read_jsonl(output/'faults'/(f['type']+'.jsonl')));err=np.linalg.norm(arr(rows)-d['target'],axis=-1)*1000
        js=[BODY.index(n) for n in f['affected_joints']]
        def finite_mean(z):
            count=np.isfinite(z).sum(1);return np.divide(np.nansum(z,axis=1),count,out=np.full(len(z),np.nan),where=count>0)
        a.plot(t,finite_mean(d['errors']['states3d'][:,js]),color='#acb7c2',label='无注入')
        a.plot(t,finite_mean(err[:,js]),color=COLORS[0],label='注入后');a.axvspan(2.4,2.72,color=COLORS[1],alpha=.2)
        a.set(ylabel='mm',title=fault_names[f['type']],ylim=(0,max(200,np.nanmax(err[:,js])*1.05)))
    axs[0].legend(ncol=2,fontsize=9);axs[-1].set_xlabel('片段时间 / s；橙色为 320 ms 注入区间，断线表示无输出')
    fig.tight_layout();figsave(output,'08_fault_recovery',fig)
    fig,axs=plt.subplots(1,3,figsize=(15,4))
    fields=['torso_landmark_frame_error_deg','hinge_flexion_error_deg','differentiated_reference_velocity_error_m_s']
    for a,f,title,unit in zip(axs,fields,['躯干坐标系旋转误差','肘 / 膝屈曲误差','速度误差（参考位置求导）'],['deg','deg','m/s']):
        a.bar(x,[r['clips'][c]['kinematics'][f]['mean'] if r['clips'][c]['kinematics'][f] else np.nan for c in ids],color=COLORS[0]);a.set(xticks=x,xticklabels=[CLIPS[c] for c in ids],ylabel=unit,title=title);a.tick_params(axis='x',rotation=30)
    figsave(output,'09_human_state',fig)
    fig,axs=plt.subplots(1,2,figsize=(13,4))
    u=r['uncertainty'];axs[0].plot([0,1],[0,1],'--',color='#aeb7c1',label='理想匹配');axs[0].plot(u['nominal'],u['empirical'],'o-',color=COLORS[1],label='本轮真实视频');axs[0].set(xlabel='模型假设的置信水平',ylabel='实际落入椭球的比例',xlim=(0,1),ylim=(0,1),title='位置协方差校准诊断');axs[0].legend()
    known=[sum(v for k,v in r['clips'][c]['support_phase_counts'].items() if k!='unknown')/150 for c in ids]
    axs[1].bar(x,known,color=COLORS[2]);axs[1].set(xticks=x,xticklabels=[CLIPS[c] for c in ids],ylim=(0,1),ylabel='比例',title='人体支撑阶段非 unknown 的帧');axs[1].tick_params(axis='x',rotation=20)
    figsave(output,'10_uncertainty_support',fig)
    robotplots(output,r)
    runtimeplots(output,data)


def examples(output,data):
    chosen=['s1_walk','s1_crouch','s2_floor'];fig,axs=plt.subplots(3,2,figsize=(12,17))
    for row,cid in enumerate(chosen):
        d=data[cid];it=iter(packets(ROOT/'outputs/public_datasets/mpi_benchmark/inputs'/cid))
        packet=None
        for _ in range(76):packet=next(it)
        it.close();det=d['detection'][75]
        for ax,cam in zip(axs[row],list(d['cal']['cameras'])[:2]):
            image=packet.frames[cam].image_bgr;ax.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));view=det['views'][cam]
            bbox=view.get('person_bbox_xyxy_px')
            if bbox:
                from matplotlib.patches import Rectangle
                ax.add_patch(Rectangle((bbox[0],bbox[1]),bbox[2]-bbox[0],bbox[3]-bbox[1],fill=False,edgecolor='#00c9ff',linewidth=1.5))
            ref=project(d['target'][75],d['cal']['cameras'][cam]);ax.scatter(ref[:,0],ref[:,1],s=10,color='#45e58c',label='参考投影')
            if view.get('keypoints_px') is not None:
                xy=np.array(view['keypoints_px']);ix={n:i for i,n in enumerate(det['joint_names'])}
                for a,b in EDGES:
                    j,k=ix[a],ix[b]
                    if min(view['scores'][j],view['scores'][k])>=.3:ax.plot(xy[[j,k],0],xy[[j,k],1],color='#ff9b38',lw=1.5)
                ax.scatter(xy[:17,0],xy[:17,1],s=8,color='#ff9b38',label='网络预测')
            ax.set(xlim=(-.5,image.shape[1]-.5),ylim=(image.shape[0]-.5,-.5))
            ax.set_title(f'{CLIPS[cid]} · 帧ID75 · 相机{cam}');ax.axis('off');ax.legend(loc='lower left',fontsize=9)
            if view.get('keypoints_px') is None:
                ax.text(.03,.06,'该视角无可用人体观测',transform=ax.transAxes,color='white',
                        bbox=dict(facecolor='#26364a',alpha=.8,edgecolor='none'),fontsize=10)
    fig.suptitle('原始 RGB 上的真实模型输出：青框=人体框，橙点=二维预测，绿点=参考投影',fontsize=14)
    fig.tight_layout();figsave(output,'03_rgb_detections',fig)


def robotplots(output,r):
    folder=output/'inference/synthetic_walk'
    refs=list(read_jsonl(folder/'humanoid_reference.jsonl'));valid=[x for x in refs if x['accepted']]
    fig,axs=plt.subplots(1,2,figsize=(13,4))
    errs=[e for x in valid for e in x['limb_direction_error_deg'] if e is not None]
    axs[0].hist(errs,bins=24,color=COLORS[0]);axs[0].axvline(35,color=COLORS[1],ls='--');axs[0].set(xlabel='肢段方向误差 / deg',ylabel='肢段帧数',title='源人体 → G1 几何重定向')
    times=[(x['timestamp_ns']-refs[0]['timestamp_ns'])/1e9 for x in valid]
    for j,name,col in [(3,'左膝',COLORS[0]),(9,'右膝',COLORS[1])]:axs[1].plot(times,[x['reference'][6+j] for x in valid],label=name,color=col)
    axs[1].set(xlabel='源时间 / s',ylabel='目标角 / rad',title=f'接受 {len(valid)}/{len(refs)} 帧；左右膝逐一对应');axs[1].legend()
    figsave(output,'11_retarget',fig)
    fb=list(read_jsonl(folder/'imitation/feedback.jsonl'));t=np.array([x['time_s'] for x in fb]);track=[i for i,x in enumerate(fb) if x['source_reference'] is not None]
    fig,axs=plt.subplots(2,2,figsize=(14,9))
    for j,n,col in [(3,'左膝',COLORS[0]),(9,'右膝',COLORS[1])]:
        axs[0,0].plot(t[track],[fb[i]['source_reference'][6+j] for i in track],'--',color=col,label=n+'源目标')
        axs[0,0].plot(t,[x['position_rad'][j] for x in fb],color=col,label=n+'执行')
    axs[0,0].set(xlabel='仿真时间 / s',ylabel='rad',title='动作目标与物理反馈');axs[0,0].legend(ncol=2,fontsize=9)
    rbt=load(output/'robot.json');labels=['正常输入','丢失400ms']
    axs[0,1].bar(labels,[rbt[k]['root_path_error_p95_m'] for k in rbt],color=COLORS[:2]);axs[0,1].axhline(.25,color='#bd4040',ls='--',label='原有门槛0.25m');axs[0,1].set(ylabel='m',title='骨盆路径跟踪 P95');axs[0,1].legend(fontsize=9)
    axs[1,0].plot(t,[x['root_position_m'][2] for x in fb],color=COLORS[0],label='基座高度');axs[1,0].axhline(.5,color=COLORS[1],ls='--',label='历史检查下限');axs[1,0].set(xlabel='仿真时间 / s',ylabel='m',title='站立 / 跟踪 / 退出全过程');axs[1,0].legend(fontsize=9)
    slip=[max([v for foot in x['foot_contacts'].values() for v in foot['tangent_speeds_m_s']],default=np.nan) for x in fb]
    axs[1,1].plot(t,slip,color=COLORS[1]);axs[1,1].set(xlabel='仿真时间 / s',ylabel='m/s',title='模拟触地点切向速度（脚滑指标）')
    fig.tight_layout();figsave(output,'12_robot_tracking',fig)
    checks=load(output/'command_boundary.json');fig,ax=plt.subplots(figsize=(12,4))
    ax.barh([x['case'] for x in checks],[1]*len(checks),color=[COLORS[2] if x['passed'] else '#bc4242' for x in checks]);ax.invert_yaxis();ax.set(xlim=(0,1.35),xticks=[],title='G1 模拟指令接收边界：逐项验证')
    for i,c in enumerate(checks):ax.text(.02,i,'正常接受' if not c['rejected'] else '按预期拒绝',va='center',color='white')
    figsave(output,'13_command_boundary',fig)
    cap=cv2.VideoCapture(str(folder/'imitation/preview.mp4'));cap.set(cv2.CAP_PROP_POS_MSEC,4000);ok,im=cap.read();cap.release()
    if ok:cv2.imencode('.png',im)[1].tofile(output/'figures/12b_robot_frame.png')


def runtimeplots(output,data):
    stream=load(output/'online/timing.json');fig,axs=plt.subplots(1,2,figsize=(13,4))
    stage=stream['stage_processing_ms'];names=['detector_ms','pose_ms','person_selection_ms','fusion_ms','state_ms']
    axs[0].barh(['人体检测','二维姿态','人物选择','联合重建','人体状态'],[stage[n]['mean'] for n in names],color=COLORS[0]);axs[0].invert_yaxis();axs[0].set(xlabel='平均处理时间 / ms',title='本轮 RTX 4090 在线文件回放分解')
    rows=list(read_jsonl(output/'online/states3d.jsonl'))
    # The aggregate contains measured latency, including producer decoding and scheduling.
    axs[1].bar(['平均','P95'],[stream['latency_ms']['mean'],stream['latency_ms']['p95']],color=COLORS[:2]);axs[1].axhline(stream['output_deadline_ms'],color='#bd4040',ls='--',label='输出期限');axs[1].set(ylabel='ms',title=f"24 Hz 输入 → {stream['state_rate_hz']:.2f} Hz 状态输出");axs[1].legend()
    figsave(output,'14_online',fig)
    reports=[load(output/'tests'/f'{e}.json') for e in ('vision','robot')]
    fig,ax=plt.subplots(figsize=(11,4));x=np.arange(2);bottom=np.zeros(2)
    for k,col,lab in [('passed',COLORS[2],'通过'),('skipped','#aeb7c1','跳过'),('failed','#bc4242','失败'),('error','#913d7c','异常')]:
        values=np.array([r['counts'].get(k,0) for r in reports]);ax.barh(x,values,left=bottom,color=col,label=lab)
        for i,v in enumerate(values):
            if v:ax.text(bottom[i]+v/2,i,str(v),ha='center',va='center',color='white')
        bottom+=values
    ax.set(yticks=x,yticklabels=['视觉环境','机器人环境'],xlabel='测试项',title='本轮完整回归：跳过项单独计数');ax.legend(ncol=4,loc='lower right')
    figsave(output,'15_regression',fig)


def write_report(output,r,data):
    manifest=load(output/'manifest.json');online=load(output/'online/timing.json');robot=load(output/'robot.json')
    retarget=load(output/'retarget.json');boundary=load(output/'command_boundary.json')
    regression={e:load(output/'tests'/f'{e}.json') for e in ('vision','robot')}
    real=[c for c in r['clips'] if not r['clips'][c]['synthetic']]
    def image(name,alt):return f'![{alt}]({(output/"figures"/(name+".png")).as_posix()})'
    def link(path,label=None):return f'[{label or Path(path).name}]({Path(path).resolve().as_posix()})'
    def fmt(value):return '无有效样本' if value is None else f'{value:.2f}'
    def status(value):return '通过' if value else '未通过'
    def table(headers,rows):
        return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
    groups=[
        ('01 输入、标定与时间','契约通过；硬件未验证',['test_sensors','test_import_real','test_real_contract']),
        ('02 人体检测框','真实 RGB 已测；无独立框真值',['test_runtime','test_vision_inference']),
        ('03 二维关键点','真实 RGB 已测；有长尾错误',['test_vision_inference','test_markerless']),
        ('04 人物关联与观测校验','全视角左右互换仍失败',['test_person_selection','test_motion_revision','test_motion_frontend']),
        ('05 三维重建','理想几何通过；真实尾差仍大',['test_geometry','test_spatial_evidence']),
        ('06 时序恢复与尺寸','有限改善；恢复跳变未解决',['test_human_state','test_observation_recovery','test_fault_injection','test_motion_revision']),
        ('07 旋转、速度与支撑','部分量可测；支撑仍未知',['test_kinematic_observability','test_foot_support','test_support']),
        ('08 不确定性与模仿准入','未达到可信准入条件',['test_empirical_bounds','test_motion_control','test_online']),
        ('09 尺寸映射与重定向','几何功能已验证；86/96 帧接受',['test_humanoid','test_root_path']),
        ('10 轨迹、平衡与停止','完整模仿未通过',['test_reference_trajectory','test_root_tracking','test_stop_transition','test_humanoid']),
        ('11 G1 指令边界','模拟接收通过；实机未验证',['test_humanoid','test_replay_deadline']),
        ('12 在线处理与过期丢弃','丢弃机制工作；未达到24Hz',['test_online','test_replay_deadline','test_motion_revision'])]
    rows=[]
    for name,verdict,modules in groups:
        counts=[]
        for env in ('vision','robot'):
            rec=[x for x in regression[env]['records'] if x['id'].split('.')[0] in modules]
            counts.append(f"{sum(x['status']=='passed' for x in rec)}通过 / {sum(x['status']=='skipped' for x in rec)}跳过")
        rows.append([name,verdict,*counts])
    raw,state=r['aggregate_real']['poses3d'],r['aggregate_real']['states3d']
    paragraphs=[
        '# 主线逐环节测试与可视化报告',
        '谭为 · 2026-10-19 · 本机 RTX 4090 · 固定视觉 / 机器人运行环境',
        '## 先看结论',
        '本轮重新运行了 **900 帧真人多视角视频（3,600 张 RGB）+ 96 帧合成人体视频（384 张 RGB）**，并执行七类观测故障注入、两次 G1 动力学回放、一次在线回放和两个环境的完整回归。',
        f'人体主要关节三维状态平均误差 **{state["error_mm"]["mean"]:.2f} mm**，P95 **{state["error_mm"]["p95"]:.2f} mm**，输出覆盖率 **{state["coverage"]:.2%}**（包含短时预测）。但仍有 **{state["over_500mm"]} 个超过500 mm的关节帧**，全视角左右互换仍能骗过系统。可见，平均效果可用，极端错误尚未可靠受控。',
        f'不确定性和控制是更大的缺口：高斯假设下的95%位置区域仅覆盖 **{r["uncertainty"]["empirical"][3]:.1%}** 参考点；在线状态输出 **{online["state_rate_hz"]:.2f} Hz**；正常输入下 G1 骨盆路径 P95 误差 **{robot["imitation"]["root_path_error_p95_m"]:.3f} m**，超过原有0.25 m门槛。**完整模仿仍未通过。**',
        image('01_mainline','主线模块与实际接入关系'),
        '**关键接入事实：**增强视觉路径产生的体型、旋转、支撑和不确定性，尚未完整进入默认 G1 控制。当前机器人示例仍从原始 `poses3d.jsonl` 生成参考轨迹。因此两个分支分别通过，也不能自动证明“可靠人体状态 → 可靠机器人动作”已打通。',
        '## 测试范围与读图方法',
        '真人数据使用已有 MPI-INF-3DHP 开发集：S1/S2 两人、六段各150帧、四相机、25Hz。动作标签是原始时间块的近似描述。没有修改算法参数，也没有重新消费冻结测试集。它是开发集复验，不是新人物泛化或论文排行榜结果。',
        '参考来自该数据集的无标记多视角动捕；合成数据参考是动画骨骼代理点。二者都不是完整独立的人体解剖真值。评分前严格核对帧号和纳秒时间戳；感知阶段不读取参考标签。三维误差在原世界坐标直接计算，不做尺度、刚体或时间对齐。',
        '平均误差只描述有效点，必须与覆盖率一起读。P95表示95%的有效点误差不超过它。一个“关节帧”是一帧中的一个关节。回归测试通过只证明所列软件行为，不等于真实精度、平衡或实机验证通过。',
        table(['节点','结论','视觉环境相关回归','机器人环境相关回归'],rows),
        '表中的回归按节点关联测试文件，同一测试可能关联多个节点，不能把各行相加作为总测试量。',
        '## 01｜RGB输入、标定与时间匹配',
        '**作用：**把不同相机同一时刻的图像放在一起，并说明像素如何对应到米制世界。输入为RGB、相机内外参和曝光时间；输出为同一时间基准的多视角包。图像接口内部采用BGR uint8；世界坐标为米、右手系Z向上；相机坐标为X右、Y下、Z前。',
        '**本轮测试：**校验文件哈希、图像尺寸、标定矩阵、单调时间及每包偏差；重跑最近未使用帧匹配、整数纳秒、漂移映射与过期映射回归。真人片段帧间隔40ms，包内时间戳偏差为0ms。',
        image('02_input_clock','输入帧间隔及同步时间戳'),
        '**结论：**文件及接口一致性通过；0ms来自数据集帧时钟，不是实测物理同步。尚未测试相机触发、真实曝光、卷帘快门和现场标定误差。失败帧必须拒绝，不能补造曝光时间。',
        f'代码：{link(ROOT/"pose_sim/sensors.py")}、{link(ROOT/"scripts/import_real.py")}。',
        '## 02｜深度网络先找到人：人体检测框',
        '**作用：**YOLOX在整张RGB图中给出人体候选框，后续姿态网络在框内定位关节。输出是像素框和候选数量；没有可用人框时不给出新关节观测。',
        table(['片段','0个框 / 1个框 / 多个框','参考投影落在所选框内'],[[CLIPS[c],
            ' / '.join(str(sum(v for k,v in r['clips'][c]['two_d']['detection_count_distribution'].items() if (int(k)==bucket if bucket<2 else int(k)>=2))) for bucket in (0,1,2)),
            f"{r['clips'][c]['two_d']['box_encloses_reference_fraction']:.2%}"] for c in real]),
        image('03_rgb_detections','真实视频上的人体框、网络关键点与参考投影'),
        '**读图：**青色矩形是检测框，橙色是本轮网络预测，绿色是独立评分时投影的三维参考。固定展示每段帧ID75（从0计数），不按效果挑最好的一帧。',
        '**边界：**这里没有独立逐帧人体框标注，不能把“有一个框”称为检测召回率，也不能宣称检测mAP已通过。多个框可能是背景误检或其他人物，需要后续关联。',
        f'代码：{link(ROOT/"pose_sim/runtime.py")}、{link(ROOT/"pose_sim/inference.py")}。',
        '## 03｜深度网络提取二维关节点',
        '**作用：**RTMPose提供身体点，RTMW提供手足细节，接口输出33个具名点及分数。单位为原图像素。网络分数用于筛选，不等于“这个点有多少概率正确”。',
        f'本轮真人身体12点有 **{r["two_d_error_px"]["count"]:,}** 个满足分数门限的有效二维点。相对三维参考投影，平均误差 **{r["two_d_error_px"]["mean"]:.2f}px**，P95 **{r["two_d_error_px"]["p95"]:.2f}px**，最大 **{r["two_d_error_px"]["maximum"]:.2f}px**。',
        image('04_keypoint_accuracy','按关节相机分解的二维误差与累计分布'),
        '**结论：**网络已经能从真人RGB输出主要关节，但仍有数百像素的尾部错点。二维参考是三维动捕参考的投影，包含关节定义和标定偏差；不是独立人工二维标注。无真值的手指/脚掌细节不计入这12点精度，不能据此声称33点均准确。',
        f'代码：{link(ROOT/"pose_sim/streaming_vision.py")}、{link(ROOT/"pose_sim/landmarks.py")}。',
        '## 04｜跨视角人物选择、左右对应与观测校验',
        '**作用：**判断不同相机是否看的是同一个人、同一侧肢体；用几何和历史发现不一致。输出修正或撤回的二维观测，再送三角化。历史预测不算额外相机证据。',
        image('05_association','观测更正撤回次数及各场景误差'),
        f'原始几何中超过500mm的关节帧为 **{raw["over_500mm"]}**，观测校验后为 **{r["aggregate_real"]["fused"]["over_500mm"]}**。这是整套关联和校验的合计对照，不能把收益单独归因于其中一个规则。',
        '**失败方式：**所有相机同时把左腿看成右腿时，视角间仍可能几何一致。本轮专门注入这一错误，后文显示它仍会被接受。',
        f'代码：{link(ROOT/"pose_sim/person_selection.py")}、{link(ROOT/"pose_sim/multiview_temporal.py")}、{link(ROOT/"pose_sim/observation_consistency.py")}。',
        '## 05｜多视角三维重建',
        '**作用：**把多个相机里的同名二维点还原成世界坐标。采用外点筛选、加权三角化和重投影优化；视角不足、低视差或不合法深度时返回未知。',
        '**几何隔离测试：**固定随机种子产生50个理想3D点，经过带畸变投影，再重建；另向一个视角加入[140,-90]px错误。两组最大误差均小于1e-6mm，低视差案例正确拒绝。它仅验证数学实现，不能当成RGB精度。',
        table(['输出层','平均 / P95误差(mm)','覆盖率','全部目标中≤100mm','>500mm关节帧'],[[l,
            f"{r['aggregate_real'][m]['error_mm']['mean']:.2f} / {r['aggregate_real'][m]['error_mm']['p95']:.2f}",
            f"{r['aggregate_real'][m]['coverage']:.2%}",f"{r['aggregate_real'][m]['all_target_within_100mm']:.2%}",r['aggregate_real'][m]['over_500mm']] for m,l in zip(METHODS,LABELS)]),
        image('06_reconstruction','三层输出的误差与覆盖率'),image('06b_skeleton3d','同一帧的三维重建对照'),
        '**结论：**理想几何求解工作正常；真实数据瓶颈集中在观测误差、对应错误及不完全的误差建模。较小重投影残差并不保证左右语义正确。',
        f'代码：{link(ROOT/"pose_sim/geometry.py")}、{link(ROOT/"pose_sim/pipeline.py")}。',
        '## 06｜因果时序恢复与人体尺寸',
        '**作用：**综合当前测量和过去运动减小抖动，短时缺失时显式预测，逐步修订肢段尺寸。状态区分当前观测、预测和缺失；最多预测100ms。稳定段长是约束，不能自动视作真实骨长。',
        image('07_temporal_shape','左踝轨迹与右小腿尺寸变化'),
        f'三层共同有效的 **{state["common_target_error_mm"]["count"]:,}** 个关节帧上，平均误差由 **{raw["common_target_error_mm"]["mean"]:.2f} → {state["common_target_error_mm"]["mean"]:.2f}mm**；当前观测覆盖率 **{state["current_coverage"]:.2%}**，包含预测的输出覆盖率 **{state["coverage"]:.2%}**。不把预测当成重新看到了人。',
        '**七类故障注入：**对同一段S1行走的二维观测，在第60–67帧注入320ms错误；每类独立重跑150帧，其他输入相同。这是模块敏感性测试，不是真实遮挡视频，也不是统计充分的多事件基准。',
        table(['故障','故障中当前覆盖','误接受关节帧(>150mm)','无注入同窗误接受','稳定恢复(ms)','恢复步进偏差最大(mm)'],[[f['type'],f"{f['current_coverage_during']:.1%}",f['false_current_over_150mm_during'],f['clean_false_current_over_150mm_during'],f['recovery_ms'] if f['recovery_ms'] is not None else '窗口内未恢复',fmt(f['recovery_step_excess_mm']['maximum'])] for f in r['faults']]),
        image('08_fault_recovery','七类故障发生及恢复过程'),
        '**指标定义：**误接受指输出仍标为当前观测且偏差>150mm。稳定恢复要求故障结束后连续3帧，受影响关节均为当前观测且误差≤150mm；0ms表示第一帧已满足，并等待后两帧作评分确认。它不是0ms计算延迟。恢复步进偏差是估计位移减去相同两帧参考位移的模长，取恢复后26帧窗口；包含重现时的大跨度。',
        '**结论：**单视角错点和缺失可由其他视角补偿；全视角左右互换仍会形成错误轨迹并在恢复时突变。下一步应解决身份假设和重接续，而不是单纯增加平滑。',
        f'代码：{link(ROOT/"pose_sim/human_motion.py")}、{link(ROOT/"pose_sim/body_shape.py")}、{link(ROOT/"pose_sim/fault_injection.py")}。',
        '## 07｜旋转、速度与支撑状态',
        '**作用：**把位置升级为运动状态：躯干坐标系、关节屈曲、速度和足部支撑。位置看起来正确，不保证速度、旋转和落脚时刻正确。',
        table(['片段','躯干旋转均误差(deg)','屈曲均误差(deg)','速度均误差(m/s)','支撑未知帧'],[[CLIPS[c],
            fmt(r['clips'][c]['kinematics']['torso_landmark_frame_error_deg']['mean']),fmt(r['clips'][c]['kinematics']['hinge_flexion_error_deg']['mean']),
            fmt(r['clips'][c]['kinematics']['differentiated_reference_velocity_error_m_s']['mean']),r['clips'][c]['support_phase_counts'].get('unknown',0)] for c in real]),
        image('09_human_state','躯干朝向关节屈曲和速度误差'),
        '**结论：**全部900帧真人支撑阶段仍为unknown，当前地面高度配置为空；这符合未知保留原则，也表明触地链路没有完成。足底三维朝向、关节轴向扭转、真实接触力没有独立真值。表中旋转来自参考关键点构造的躯干坐标系，屈曲仅是一自由度；速度由参考位置中心差分，包含标签噪声。',
        '动作相位用手腕/脚踝轨迹互相关作描述性检查，时间分辨率40ms；本轮最大偏移80ms（S1蹲伸右踝）。0ms只表示离散搜索未找到更好偏移，不能解释为物理链路无延迟。',
        f'代码：{link(ROOT/"pose_sim/human_motion.py")}、{link(ROOT/"pose_sim/foot_support.py")}。',
        '## 08｜不确定性与“是否允许模仿”',
        '**作用：**给出位置误差范围、是否新鲜、是否为当前观测，决定跟随、过渡或停止。置信度与协方差必须在目标数据上校准。',
        image('10_uncertainty_support','协方差覆盖率与支撑未知比例'),
        f'本轮用每个输出的3×3位置协方差构造高斯椭球，对 **{r["uncertainty"]["samples"]:,}** 个输出检查参考是否落入。名义95%仅实际覆盖 **{r["uncertainty"]["empirical"][3]:.2%}**，明显过于乐观。该图只是模型诊断：样本存在时序相关，参考也有误差，不能据此宣称已得到独立置信保证。',
        f'在线回放 **{online["processed"]}** 个输出中，**{online["valid_deliveries"]}** 个按时交付，但规划门控 follow 次数为 **{online["planner_follow_count"]}**。原因包括RGB域尚未校准、腿部没有合格的误差半径；“按时输出”不等于“允许控制”。',
        f'代码：{link(ROOT/"pose_sim/uncertainty.py")}、{link(ROOT/"motion_control/perception_gate.py")}。',
        '## 09｜人体大小与G1尺寸映射、全身重定向',
        '**作用：**用人体肢段方向和骨盆运动，求G1限位内的29关节参考；人和机器人大小不同，左右对应和腿部动作来源必须保留。',
        f'合成行走96帧重新从RGB推理，**{retarget["accepted"]}/{retarget["frames"]}** 帧进入重定向；拒绝原因包括腿部缺失、骨长突变和骨盆朝向不可观测。重定向回归覆盖缩放不变性、左右腿对应、抬臂不触发行走、平移不影响路径和失败回滚。',
        image('11_retarget','重定向肢段误差及左右膝参考'),
        '**边界：**水平脚掌和中立手腕仍是先验，不能称为视觉实测。局部方向匹配通过，也不保证落脚时机、地面接触和动力学可执行。',
        f'代码：{link(ROOT/"motion_control/humanoid.py")}、{link(ROOT/"motion_control/root_path.py")}。',
        '## 10｜轨迹约束、全身跟踪、平衡和停止',
        '**作用：**重定向目标经过速率限制后，由TWIST2策略驱动自由基座G1，在MuJoCo中验证实际动作。控制50Hz、物理1000Hz；120ms感知延迟是注入参数。',
        table(['条件','平衡检查','腿部模仿','停止检查','骨盆P95(m)','腿部RMSE(rad)','完整验收'],[[
            '正常输入' if k=='imitation' else '中间丢失400ms',status(v['balance_passed']),status(v['limb_imitation_passed']),status(v['stopping_passed']),fmt(v['root_path_error_p95_m']),fmt(v['source_reference_leg_rmse_rad']),status(v['passed'])] for k,v in robot.items()]),
        image('12_robot_tracking','机器人参考执行与路径脚滑结果'),image('12b_robot_frame','本轮四相机输入重定向与G1仿真画面'),
        f'正常输入左膝相关系数 **{robot["imitation"]["motion_correlations"]["left_knee_joint"]:.3f}**，低于原有0.7门槛；骨盆P95 **{robot["imitation"]["root_path_error_p95_m"]:.3f}m**。模拟触地点切向速度P95 **{robot["imitation"]["simulated_contact_tangent_speed_m_s"]["p95"]:.3f}m/s**，表明“站住”与“无脚滑地模仿”不同。',
        '**结论：**两段均未通过完整模仿，原始目标、执行反馈和脚滑数据已保留。平衡检查仍为既有高度/倾角阈值，只是有限验收条件，不是完整稳定性证明。本轮没跑新的10段全场景控制基准，不沿用旧10段数据作为本轮结果。',
        f'仿真视频：{link(output/"inference/synthetic_walk/imitation/preview.mp4","本轮完整G1回放")}。代码：{link(ROOT/"motion_control/humanoid_replay.py")}。',
        '## 11｜具名G1指令与模拟接收边界',
        '**作用：**将关节目标封装为位置rad、速度rad/s、力矩上限Nm、模型哈希、具名顺序、序号、有效期和时钟域；模拟接收器逐项验证。',
        image('13_command_boundary','正常指令与八种错误指令检查'),
        f'正常指令接受，加上过期、模型不符、错误硬件开关、关节错序、越界、未来时间、控制参数不符及重放，共 **{sum(c["passed"] for c in boundary)}/{len(boundary)}** 项符合预期。',
        '**结论：**模拟协议边界通过；没有接入G1的真实DDS/CRC、编码器反馈、里程计或独立停止。模拟接收器明确拒绝硬件使能。本轮没有向物理机器人发送指令。',
        f'代码：{link(ROOT/"motion_control/g1_command.py")}。',
        '## 12｜在线性能、队列积压与过期处理',
        image('14_online','在线各阶段开销及输出延迟'),
        f'以24Hz回放96个四视角包，实测持续 **{online["wall_seconds"]:.2f}s**：处理 **{online["processed"]}** 包、被新帧替换 **{online["replaced"]}** 包，输出过期 **{online["stale_after"]}** 包。队列容量为1。状态吞吐 **{online["state_rate_hz"]:.2f}Hz**；从软件计划曝光时刻到输出平均 **{online["latency_ms"]["mean"]:.2f}ms**，P95 **{online["latency_ms"]["p95"]:.2f}ms**，超过150ms输出期限。',
        '**结论：**丢旧保新机制工作，但处理速度尚不满足24Hz。检测/姿态网络实际使用CUDA，低维几何和状态在CPU运行。这是约4秒文件回放的计算与调度测量，不能当作长时稳定性或真实曝光到状态输出验收。',
        f'代码：{link(ROOT/"scripts/stream_vision.py")}、{link(ROOT/"pose_sim/online.py")}。',
        '## 回归、复现与后续优先级',image('15_regression','两个环境的完整回归结果'),
        f'视觉环境：运行 **{regression["vision"]["tests_run"]}** 项，**{regression["vision"]["counts"].get("passed",0)}** 通过、**{regression["vision"]["counts"].get("skipped",0)}** 条件跳过；机器人环境：运行 **{regression["robot"]["tests_run"]}** 项，**{regression["robot"]["counts"].get("passed",0)}** 通过。跳过项没有计作通过。',
        '测试过程中修复了一个展示层问题：Windows中文绝对路径导致OpenCV预览读图失败。现在先读字节再解码，并加入中文路径回归；感知与控制算法参数保持原样。',
        '接下来按优先级处理：①全视角左右身份错误及恢复连续性；②RGB域误差范围校准；③地面/足底支持状态及其独立真值；④完整人体状态进入控制接口；⑤G1落脚与骨盆路径跟踪；⑥24Hz持续在线性能。先修风险最大的失效，再追求平均误差小幅下降。',
        f'本轮以提交 `{manifest["git_head"][:7]}` 为基线；{link(output/"manifest.json","感知输入/代码/模型指纹")} 保留推理时指纹。{link(output/"metrics.json","逐环节数值")}、{link(output/"robot.json","仿真结果")}、{link(output/"command_boundary.json","指令故障结果")}、{link(output/"tests/vision.json","视觉测试明细")}、{link(output/"tests/robot.json","机器人测试明细")} 可追溯。',
        '### 重跑方法',
        '使用新的输出目录，保留当前证据。依次执行：',
        '```powershell\n$v = (Get-Content .runtime.json -Raw | ConvertFrom-Json).python\n$r = (Get-Content .robot-runtime.json -Raw | ConvertFrom-Json).python\n$out = "outputs/mainline_validation_next"\n& $v scripts/validate_mainline.py infer --output $out\n& $v scripts/validate_mainline.py faults --output $out\n& $r scripts/validate_mainline.py robot --output $out\n& $v scripts/stream_vision.py outputs/human_walk_g1 --output "$out/online" --detector-workers 4 --pose-workers 4 --fps 24\n& $v scripts/validate_mainline.py tests --environment vision --output $out\n& $r scripts/validate_mainline.py tests --environment robot --output $out\n& $v scripts/report_mainline.py --output $out\n```',
        '报告中的真人截图仅作本地研究查看；按原数据许可保留，不把数据和派生视频上传公开仓库。生成产物均在Git忽略目录，脚本和简短验收记录可入库。']
    (output/'逐环节验证报告.md').write_text('\n\n'.join(paragraphs)+'\n',encoding='utf-8')
    concise=dict(date='2026-10-19',base_commit=manifest['git_head'],new_real_frames=900,new_synthetic_frames=96,
        total_rgb_images=3984,real_data='development only; frozen test not reused',metrics=r['aggregate_real'],
        two_d_px=r['two_d_error_px'],gaussian_nominal95_empirical=r['uncertainty']['empirical'][3],
        fault_cases=r['faults'],robot={k:{n:v[n] for n in ('passed','balance_passed','limb_imitation_passed','root_path_error_p95_m','stopping_passed')} for k,v in robot.items()},
        online={k:online[k] for k in ('processed','replaced','stale_after','state_rate_hz','latency_ms','planner_follow_count')},
        regression={e:dict(tests_run=v['tests_run'],counts=v['counts']) for e,v in regression.items()},
        figure_count=len(list((output/'figures').glob('*.png'))),report=(output/'逐环节验证报告.md').relative_to(ROOT).as_posix(),
        integration_gap='Enhanced full human state is not yet connected to the default G1 raw-pose controller',
        hardware_verified=False,algorithm_parameters_changed=False)
    save(ROOT/'reports/mainline-validation-2026-10-19.json',concise)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r,data=score(a.output.resolve());plots(a.output.resolve(),r,data);write_report(a.output.resolve(),r,data)
    print(json.dumps(r['aggregate_real'],indent=2),flush=True)
