"""Causal visual-reference replay into a free-base G1 full-body tracking policy."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .twist import TwistPolicy,TwistPlant,DEFAULT_REF,KP,KD
from .unitree import SDK_JOINT_NAMES
from .root_tracking import RootTracker
from .g1_command import G1SimulationSink
from .reference_trajectory import BoundedReferenceTrajectory,SourceReferenceDerivative,load_config
from .tracking_metrics import phase_lag,foot_contacts,SourceLegAudit,standing_stop_audit


class ReferenceGate:
    def __init__(self,max_age_ns=250_000_000):
        self.max_age_ns=max_age_ns;self.latest=None;self.last_seen=None

    def receive(self,row,now_ns):
        stamp=row['timestamp_ns']
        if not isinstance(stamp,int) or stamp>now_ns or (self.last_seen is not None and stamp<=self.last_seen):
            raise ValueError('Future/nonmonotonic reference')
        self.last_seen=stamp;self.latest=None
        if row['accepted']:
            ref=np.asarray(row['reference'])
            if ref.shape!=(35,) or not np.isfinite(ref).all():raise ValueError('Invalid reference vector')
            self.latest=row
        else:self.latest=None

    def get(self,now_ns):
        if self.latest is None:return None
        age=now_ns-self.latest['timestamp_ns']
        if (self.latest.get('reference_schema_id')=='continuous_simulation_reference_v1' and
                (self.latest.get('hardware_enabled') is not False or
                 now_ns>self.latest.get('prediction_valid_until_ns',-1))):
            return None
        return self.latest if 0<=age<=self.max_age_ns else None


def read_preview_image(path):
    """Decode via bytes because OpenCV imread loses Windows Unicode paths."""
    import cv2
    path=Path(path)
    return cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_COLOR) if path.is_file() else None


def replay(session,video=True,delay_ms=120,dropout=None,output_name='imitation',control_dt=.02,
           reference_mode='rate_limit',trajectory_config=None,save_logs=True,root_tracking_config=None):
    session=Path(session);out=session/output_name
    if out.exists():raise FileExistsError('Output exists; explicitly clean generated replay before rerun')
    out.mkdir()
    rows=[json.loads(x) for x in (session/'humanoid_reference.jsonl').read_text().splitlines()]
    valid=[x for x in rows if x['accepted']]
    if not valid:raise ValueError('No usable full-body observations')
    if not np.isfinite(delay_ms) or not 0<=delay_ms<250:raise ValueError('Invalid perception delay')
    start_ns=rows[0]['timestamp_ns'];end_ns=rows[-1]['timestamp_ns']
    duration=(end_ns-start_ns)/1e9
    source_start=2.;dt=control_dt;delay_ns=round(delay_ms*1e6)
    plant=TwistPlant();policy=TwistPolicy();gate=ReferenceGate();sink=G1SimulationSink(plant)
    leg_audit=SourceLegAudit(plant)
    if reference_mode not in ('rate_limit','jerk_limited','arms_jerk_limited','recovery_jerk_limited'):raise ValueError('Unknown reference trajectory mode')
    trajectory_settings=trajectory_config or load_config()
    trajectory=BoundedReferenceTrajectory(DEFAULT_REF,plant.limits,plant.sha256,trajectory_settings,dt) if reference_mode!='rate_limit' else None
    if reference_mode=='recovery_jerk_limited':
        from .continuous_reference import RecoveryReferenceTrajectory
        trajectory=RecoveryReferenceTrajectory(DEFAULT_REF,plant.limits,plant.sha256,trajectory_settings,dt)
    derivative=SourceReferenceDerivative(trajectory_settings)
    ref=DEFAULT_REF.copy();first=np.array(valid[0]['reference']);index=0;last_frame=rows[0]['frame_id']
    states=[];commands=[];latencies=[];aligned=[];root_errors=[];root_tracker=None
    renderer=writer=reference_data=None
    if video:
        import cv2
        mj=plant.mj
        plant.model.vis.global_.offwidth=640;plant.model.vis.global_.offheight=640
        renderer=mj.Renderer(plant.model,640,640)
        reference_data=mj.MjData(plant.model)
        writer=cv2.VideoWriter(str(out/'preview.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),25,(1920,720))
        if not writer.isOpened():raise RuntimeError('Video encoder failed')
        cam=mj.MjvCamera();cam.distance=2.5;cam.azimuth=110;cam.elevation=-12
    begin=time.perf_counter()
    try:
        for i in range(int((source_start+duration+4.)/dt)+1):
            now=i*dt;source_ns=start_ns+round((now-source_start)*1e9)
            state='standing';requested=DEFAULT_REF.copy();row=None
            if 1<=now<source_start:
                state='initial_alignment';alpha=now-1;requested=(1-alpha)*DEFAULT_REF+alpha*first
            elif source_start<=now<=source_start+duration+delay_ms/1000:
                yaw=Rotation.from_quat(plant.data.qpos[[4,5,6,3]]).as_euler('xyz')[2]
                if root_tracker is None:root_tracker=RootTracker(valid[0],plant.data.qpos[:3],yaw,root_tracking_config)
                while index<len(rows) and rows[index]['timestamp_ns']+delay_ns<=source_ns:
                    candidate=rows[index];last_frame=candidate['frame_id'];index+=1
                    relative_s=(candidate['timestamp_ns']-start_ns)/1e9
                    if dropout is None or not dropout[0]<=relative_s<dropout[1]:gate.receive(candidate,source_ns)
                row=gate.get(source_ns)
                if row is not None:
                    requested,root_error=root_tracker.correct(row,row['reference'],plant.data.qpos[:3],yaw)
                    root_errors.append(root_error);state='tracking'
                else:state='vision_unavailable'
            elif now>source_start+duration:state='return_to_stand'
            # Bounded causal transition; no future frame interpolation or frozen walking target.
            max_delta=np.r_[.04,.04,.012,.03,.03,.06,np.full(29,.12)]*(dt/.02)
            if trajectory:
                rate=derivative.update(row)
                horizon=(source_ns-row['timestamp_ns'])/1e9 if row is not None else 0.
                if trajectory_settings.get('use_source_derivative',True) is False:
                    rate=np.zeros(35);horizon=0.
                curve_target=requested.copy();old_ref=ref.copy()
                if reference_mode=='arms_jerk_limited':
                    curve_target[:21]=trajectory.position[:21];rate[:21]=0.
                ref,trajectory_diagnostic=trajectory.step(curve_target,rate,horizon)
                if reference_mode=='arms_jerk_limited':
                    ref[:21]=old_ref[:21]+np.clip(requested[:21]-old_ref[:21],-max_delta[:21],max_delta[:21])
                    trajectory_diagnostic['constrained_reference_axes']=list(range(21,35))
            else:
                ref+=np.clip(requested-ref,-max_delta,max_delta)
                trajectory_diagnostic=None
            tick=time.perf_counter();target=policy.step(ref,*plant.state());latencies.append((time.perf_counter()-tick)*1000)
            target=np.clip(target,plant.limits[:,0],plant.limits[:,1])
            command=dict(sequence=i,timestamp_ns=round(now*1e9),valid_until_ns=round((now+.06)*1e9),
                clock_domain='offline_simulation',hardware_enabled=False,joint_names=SDK_JOINT_NAMES,
                target_model_sha256=plant.sha256,
                position_rad=target.tolist(),velocity_rad_s=[0.]*29,kp=KP.tolist(),kd=KD.tolist(),
                feedforward_torque_nm=[0.]*29,effort_limit_nm=plant.effort.tolist(),
                source_timestamp_ns=None if row is None else row['timestamp_ns'],state=state)
            validated=sink.validate(command,round(now*1e9))
            plant.step(validated,dt)
            q,dq,quat,omega=plant.state()
            rpy=Rotation.from_quat(quat[[1,2,3,0]]).as_euler('xyz')
            commands.append(command)
            states.append(dict(time_s=now,state=state,source_frame_id=last_frame,
                root_position_m=plant.data.qpos[:3].tolist(),root_rpy_rad=rpy.tolist(),
                position_rad=q.tolist(),velocity_rad_s=dq.tolist(),reference=ref.tolist(),
                source_reference=None if row is None else row.get('source_reference',row['reference']),
                foot_contacts=foot_contacts(plant),source_leg_direction_error_deg=leg_audit.update(row),trajectory=trajectory_diagnostic))
            if state=='tracking':aligned.append((q.copy(),np.array(row.get('source_reference',row['reference'])[6:]),ref[6:].copy()))
            if writer is not None and i%round(.04/dt)==0:
                canvas=np.full((720,1920,3),245,np.uint8)
                for k in range(4):
                    path=session/'images'/f'cam{k:02d}'/f'{last_frame:06d}.png'
                    image=read_preview_image(path)
                    if image is not None:
                        y=80+(k//2)*320;x=(k%2)*320
                        canvas[y:y+320,x:x+320]=cv2.resize(image,(320,320))
                reference_data.qpos[:]=np.r_[0,0,ref[2],1,0,0,0,ref[6:]]
                plant.mj.mj_forward(plant.model,reference_data)
                for x,data in ((640,reference_data),(1280,plant.data)):
                    cam.lookat[:]=[data.qpos[0],data.qpos[1],.65]
                    renderer.update_scene(data,camera=cam)
                    canvas[80:720,x:x+640]=renderer.render()[:,:,::-1]
                for x,title in ((10,'4 CAMERA RGB / MARKERLESS HUMAN'),(650,'SIZE-RETARGETED MOTION REFERENCE'),(1290,'G1 / FEEDBACK PHYSICS TRACKING')):
                    cv2.putText(canvas,title,(x,28),cv2.FONT_HERSHEY_SIMPLEX,.6,(30,30,30),1,cv2.LINE_AA)
                cv2.putText(canvas,f't={now:.2f}s  {state}  source frame={last_frame}  delay={delay_ms:g}ms',
                    (10,60),cv2.FONT_HERSHEY_SIMPLEX,.65,(40,40,160),1,cv2.LINE_AA)
                writer.write(canvas)
    except Exception as error:
        (out/'failure.json').write_text(json.dumps(dict(error=str(error),reference_mode=reference_mode,
            trajectory_configuration=trajectory_settings,last_completed_tick=len(states)-1,
            hardware_enabled=False,behavior='Simulation aborted; no motor or network command sent'),indent=2),encoding='utf-8')
        raise
    finally:
        if writer is not None:writer.release()
        if renderer is not None:renderer.close()
    if save_logs:
        for name,data in [('g1_commands.jsonl',commands),('feedback.jsonl',states)]:
            (out/name).write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in data),encoding='utf-8')
    actual,source,bounded=np.array(aligned).transpose(1,0,2) if aligned else (np.empty((0,29)),)*3
    correlations={}
    for j in (0,3,6,9,15,18,22,25):
        correlations[SDK_JOINT_NAMES[j]]=float(np.corrcoef(actual[:,j],source[:,j])[0,1]) if len(actual)>2 and np.std(source[:,j])>.03 else None
    heights=np.array([x['root_position_m'][2] for x in states]);tilts=np.array([x['root_rpy_rad'][:2] for x in states])
    stop_audit=standing_stop_audit(states,dt);final_speed=stop_audit['mean_base_speed_m_s']
    report=dict(kind='markerless_human_full_body_imitation_in_simulation',hardware_tested=False,
        reference_mode=reference_mode,trajectory_configuration=trajectory_settings if trajectory else None,
        source_frames=len(rows),accepted_frames=len(valid),acceptance_fraction=len(valid)/len(rows),
        duration_s=duration,control_frequency_hz=1/dt,physics_frequency_hz=1/plant.model.opt.timestep,
        perception_delay_ms=delay_ms,dropout=dropout,tracking_ticks=len(actual),
        root_min_height_m=float(heights.min()),max_abs_roll_pitch_rad=float(np.abs(tilts).max()),
        end_root_displacement_m=float(np.linalg.norm(plant.data.qpos[:2])),
        root_tracking_uses='MuJoCo world odometry; real robot requires calibrated odometry feedback',
        root_path_error_mean_m=float(np.mean([x['source_position_error_m'] for x in root_errors])) if root_errors else None,
        root_path_error_p95_m=float(np.percentile([x['source_position_error_m'] for x in root_errors],95)) if root_errors else None,
        final_base_speed_m_s=final_speed,initial_alignment_s=2.,return_and_settle_s=4.,
        source_reference_joint_rmse_rad=float(np.sqrt(np.mean((actual-source)**2))) if len(actual) else None,
        source_reference_leg_rmse_rad=float(np.sqrt(np.mean((actual[:,:12]-source[:,:12])**2))) if len(actual) else None,
        source_reference_hip_knee_rmse_rad=float(np.sqrt(np.mean((actual[:,[0,1,2,3,6,7,8,9]]-source[:,[0,1,2,3,6,7,8,9]])**2))) if len(actual) else None,
        bounded_reference_joint_rmse_rad=float(np.sqrt(np.mean((actual-bounded)**2))) if len(actual) else None,
        motion_correlations=correlations,policy_mean_ms=float(np.mean(latencies)),policy_p95_ms=float(np.percentile(latencies,95)),
        wall_seconds=time.perf_counter()-begin,command_count=len(commands),
        priors=valid[0]['priors'],unobserved=valid[0]['unobserved'],
        inputs_sha256={name:hashlib.sha256((session/name).read_bytes()).hexdigest() for name in ('poses3d.jsonl','humanoid_reference.jsonl','retargeting.json') if (session/name).exists()})
    def derivative_audit(values):
        values=np.asarray(values)
        result={}
        for label in ('velocity','acceleration','jerk'):
            values=np.diff(values,axis=0)/dt
            result[label]=dict(maximum_abs=float(np.abs(values).max()),p95_abs=float(np.percentile(np.abs(values),95)))
        return result
    report['reference_joint_discrete_derivatives']=derivative_audit([s['reference'][6:] for s in states])
    report['reference_arm_discrete_derivatives']=derivative_audit([s['reference'][21:] for s in states])
    report['policy_target_joint_discrete_derivatives']=derivative_audit([c['position_rad'] for c in commands])
    report['trajectory_scope']='Reference continuity only; the tracking policy output is not jerk constrained, and foot contact/balance are not guaranteed by this generator.'
    report['phase_lag_from_delayed_original_source']=phase_lag([s['position_rad'] for s in states],
        [s['source_reference'][6:] if s['source_reference'] is not None else [np.nan]*29 for s in states],dt,(0,3,6,9))
    slip=[v for s in states for f in s['foot_contacts'].values() for v in f['tangent_speeds_m_s']]
    report['simulated_contact_tangent_speed_m_s']=dict(mean=float(np.mean(slip)),p95=float(np.percentile(slip,95)),maximum=float(max(slip)),count=len(slip)) if slip else None
    direction_errors=[e for s in states if s['source_leg_direction_error_deg'] is not None for e in s['source_leg_direction_error_deg']]
    report['estimated_source_leg_direction_error_deg']=dict(mean=float(np.mean(direction_errors)),p95=float(np.percentile(direction_errors,95)),count=len(direction_errors)) if direction_errors else None
    report['balance_by_stage']={stage:dict(frames=len(samples),minimum_height_m=min(s['root_position_m'][2] for s in samples),
        minimum_height_minus_reference_m=min(s['root_position_m'][2]-s['reference'][2] for s in samples),
        maximum_abs_roll_pitch_rad=max(max(abs(v) for v in s['root_rpy_rad'][:2]) for s in samples),
        first_violation_s=next((s['time_s'] for s in samples if s['root_position_m'][2]<=.5 or max(abs(v) for v in s['root_rpy_rad'][:2])>=.5),None))
        for stage in sorted({s['state'] for s in states}) if (samples:=[s for s in states if s['state']==stage])}
    report['balance_check_scope']='Historical fixed height >0.5 m and |roll/pitch|<0.5 rad are retained for comparison. Height-only failure can be intended deep crouching, not a fall; inspect stage height relative to requested reference. This is not a complete balance certificate.'
    report['root_tracking_configuration']=root_tracker.cfg if root_tracker else None
    report['root_feedforward_frame_error_m_s']=dict(mean=float(np.mean([e['feedforward_world_frame_error_m_s'] for e in root_errors])),
        maximum=float(max(e['feedforward_world_frame_error_m_s'] for e in root_errors))) if root_errors else None
    report['evaluation_target']='Original source joint/path reference retained before any offline regularization. Phase lag includes controller response relative to delayed available source, not physical capture latency.'
    report['balance_passed']=bool(heights.min()>.5 and np.abs(tilts).max()<.5)
    report['limb_imitation_passed']=bool(len(actual)>50 and report['acceptance_fraction']>=.7 and
        report['source_reference_leg_rmse_rad']<.35 and all(correlations[SDK_JOINT_NAMES[j]] is not None and
        correlations[SDK_JOINT_NAMES[j]]>.7 for j in (0,3,6,9)))
    report['root_path_passed']=bool(root_errors and report['root_path_error_p95_m']<.25)
    report['standing_stop_audit']=stop_audit
    report['legacy_speed_only_stopping_passed']=stop_audit['speed_only_passed']
    report['stopping_passed']=stop_audit['passed']
    report['imitation_passed']=report['limb_imitation_passed'] and report['root_path_passed']
    report['passed']=report['balance_passed'] and report['imitation_passed'] and report['stopping_passed']
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('session',type=Path)
    parser.add_argument('--no-video',action='store_true');parser.add_argument('--delay-ms',type=float,default=120)
    parser.add_argument('--reference-mode',choices=['rate_limit','jerk_limited','arms_jerk_limited','recovery_jerk_limited'],default='rate_limit')
    parser.add_argument('--output-name',default='imitation')
    parser.add_argument('--no-logs',action='store_true')
    args=parser.parse_args();result=replay(args.session,not args.no_video,args.delay_ms,
        output_name=args.output_name,reference_mode=args.reference_mode,save_logs=not args.no_logs)
    print(json.dumps(result,indent=2));raise SystemExit(0 if result['passed'] else 1)
