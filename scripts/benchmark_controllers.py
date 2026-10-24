"""Paired G1 controller comparison with explicit causal reference availability.

All runs use the identical pinned G1 XML, initial state, torque envelope,
source trajectory, reference rate limiter, 50 Hz control and 1000 Hz physics.
Controller-specific trained PD gains are part of each controller, not retuned.
0 ms: newest cached reference, no injected transport delay or preview buffer;
this is causal simulation, not a measurement of live camera-to-motor latency.
120 ms: only the latest received reference, repeated in SONIC's future slots.
300 ms: all controllers track the same delayed reference; SONIC's 180 ms
window is available after the same 120 ms transport delay. No future leakage.
900 ms: no injected transport; common 900 ms buffer. SONIC v1.1 uses ten
samples at 0..900 ms; low_latency uses 0..180 ms; single-step policies use
the same window anchor. A hold ablation repeats the anchor with equal delay.
Malformed references abort; rejected visual frames generate standing withdrawal,
are excluded from tracking scores and never counted as successful imitation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation
from motion_control.twist import TwistPlant,DEFAULT_Q,DEFAULT_REF,CONFIG as TWIST_CONFIG
from motion_control.comparison_policies import FACTORIES,Reference,asset,CONFIG
from motion_control.tracking_metrics import foot_contacts
from pose_sim.robot_assets import require_asset
from scripts.benchmark_twist_policy import VISUAL,MOTION,save,sha,arrays,metrics

DT=.02
NAMES={'twist':'TWIST2','sonic':'SONIC 低延迟版','sonic_v11':'SONIC v1.1','beyond':'BeyondMimic 舞蹈专家*'}
DEFAULT_CONTROLLERS=('twist','sonic','beyond')
CASES={'standing':'站立','push':'站立 + 外力','official':'TWIST2 官方行走',
       'visual':'真人视觉行走参考','dance':'BeyondMimic 自带舞蹈前 12 秒'}


def native_dance():
    """Extract the public checkpoint's embedded reference, never its actions."""
    import onnx
    from onnx import numpy_helper
    model=onnx.load(asset(CONFIG['beyond']['policy_asset']))
    constants={}
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type==onnx.AttributeProto.TENSOR and attr.t.dims and attr.t.dims[0]==6574:
                constants[node.name]=numpy_helper.to_array(attr.t).copy()
    names=CONFIG['beyond']['policy_joint_names']
    order=[names.index(n) for n in CONFIG['joint_names']]
    return dict(fps=50.,root_pos=constants['/Constant_3'][:,0],
                root_rot=constants['/Constant_4'][:,0][:,[1,2,3,0]],
                dof_pos=constants['/Constant_1'][:,order])


def source(name):
    if name in ('standing','push'):
        return None,0.
    if name=='visual':
        rows=[json.loads(x) for x in VISUAL.read_text().splitlines()]
        stamps=np.array([(r['timestamp_ns']-rows[0]['timestamp_ns'])/1e9 for r in rows])
        first=next(r for r in rows if r['accepted'])
        origin=np.array(first['root_position_m'])
        align=Rotation.from_euler('z',-first['root_heading_rad'])
        q=[];xyz=[];quat=[];accepted=[]
        for row in rows:
            valid=row['accepted'];accepted.append(valid)
            if valid:
                ref=np.array(row['reference']);pos=align.apply(np.array(row['root_position_m'])-origin)
                pos[2]=ref[2]
                rot=Rotation.from_euler('xyz',[ref[3],ref[4],row['root_heading_rad']-first['root_heading_rad']])
                q.append(ref[6:]);xyz.append(pos);quat.append(rot.as_quat())
            else:
                q.append(DEFAULT_Q.copy());xyz.append([0,0,DEFAULT_REF[2]]);quat.append([0,0,0,1])
        return dict(times=stamps,q=np.array(q),xyz=np.array(xyz),quat=np.array(quat),valid=np.array(accepted)),float(stamps[-1])
    data=pickle.loads(require_asset(MOTION).read_bytes()) if name=='official' else native_dance()
    count=len(data['dof_pos']) if name=='official' else 601
    pos=np.array(data['root_pos'][:count],float);rot=Rotation.from_quat(data['root_rot'][:count])
    align=Rotation.from_euler('z',-rot[0].as_euler('xyz')[2])
    xyz=align.apply(pos-pos[0]);xyz[:,2]=pos[:,2]
    times=np.arange(count)/data['fps']
    return dict(times=times,q=np.array(data['dof_pos'][:count],float),xyz=xyz,quat=(align*rot).as_quat(),valid=np.ones(count,bool)),float(times[-1])


def prepare(name):
    """Freeze common source-to-command stream; no robot state is consulted."""
    data,duration=source(name)
    total=8. if data is None else 3+duration+3
    count=math.ceil(total/DT)
    original=[];bounded=[];active=[];stages=[];source_ids=[]
    q=DEFAULT_Q.copy();rpy=np.zeros(3);height=DEFAULT_REF[2];last_xy=np.zeros(2)
    for i in range(count):
        t=i*DT;pose=np.r_[0,0,DEFAULT_REF[2],1,0,0,0,DEFAULT_Q]
        valid=False;stage='standing';idx=-1
        if data is not None:
            if 2<=t<3:
                idx=int(np.flatnonzero(data['valid'])[0]);u=t-2
                pose[7:]=(1-u)*DEFAULT_Q+u*data['q'][idx]
                pose[2]=(1-u)*DEFAULT_REF[2]+u*data['xyz'][idx,2]
                stage='aligning'
            elif 3<=t<=3+duration+1e-9:
                idx=int(np.searchsorted(data['times'],t-3+1e-9,side='right')-1)
                valid=bool(data['valid'][idx]);stage='tracking' if valid else 'reference_unavailable'
                if valid:
                    pose=np.r_[data['xyz'][idx],data['quat'][idx][[3,0,1,2]],data['q'][idx]]
                    last_xy=pose[:2].copy()
                else:
                    pose[:2]=last_xy
            elif t>3+duration:
                pose[:2]=last_xy;stage='stopping'
        else:
            valid=t>=2
        q+=np.clip(pose[7:]-q,-.12,.12)
        goal_rpy=Rotation.from_quat(pose[[4,5,6,3]]).as_euler('xyz')
        delta=np.arctan2(np.sin(goal_rpy-rpy),np.cos(goal_rpy-rpy))
        rpy+=np.clip(delta,[-.03,-.03,-.06],[.03,.03,.06])
        height+=np.clip(pose[2]-height,-.012,.012)
        rotation=Rotation.from_euler('xyz',rpy)
        command=np.r_[pose[:2],height,rotation.as_quat()[[3,0,1,2]],q]
        original.append(pose);bounded.append(command);active.append(valid);stages.append(stage);source_ids.append(idx)
    original,bounded=np.array(original),np.array(bounded)
    # Backward differences only. Rate-limited commands can have finite velocity
    # during withdrawal; it is a generated safety transition, not a new source pose.
    dq=np.vstack((np.zeros(29),np.diff(bounded[:,7:],axis=0)/DT))
    velocity=np.vstack((np.zeros(3),np.diff(bounded[:,:3],axis=0)/DT))
    yaw=np.unwrap(Rotation.from_quat(bounded[:,[4,5,6,3]]).as_euler('xyz')[:,2])
    yaw_rate=np.r_[0,np.diff(yaw)/DT]
    return dict(original=original,bounded=bounded,dq=dq,velocity=velocity,yaw_rate=yaw_rate,
                active=np.array(active),stage=np.array(stages),source_ids=np.array(source_ids),time_s=np.arange(count)*DT)


def window_indices(tick,delay_ticks,length):
    """Only received samples: 0/6 ticks hold; 15 ticks adds a 9-tick buffer."""
    if delay_ticks not in (0,6,15) or length<1 or tick<0:
        raise ValueError('Expected 0/120/300 ms protocol and a nonempty stream')
    current=tick-delay_ticks
    offsets=np.arange(10) if delay_ticks==15 else np.zeros(10,dtype=int)
    indices=np.clip(current+offsets,0,length-1)
    if int(indices.max())>available_index(tick,delay_ticks):
        raise AssertionError('Future source leaked into policy')
    return indices.astype(int)


def available_index(tick,delay_ticks):
    return max(0,tick-(0 if delay_ticks==0 else 6))


def native_window_indices(name,tick,length,mode='native'):
    """At tick t, execute t-45 with an entirely received native preview.

    Missing negative-time samples are the known initial standing state.
    Beyond the finite clip, hold its terminal withdrawal command.
    """
    if name not in FACTORIES or tick<0 or length<1 or mode not in ('native','hold'):
        raise ValueError('Invalid native-window schedule')
    step=CONFIG[name].get('reference_step',1) if name in ('sonic','sonic_v11') and mode=='native' else 0
    indices=np.clip(tick-45+np.arange(10)*step,0,length-1).astype(int)
    if int(indices.max())>tick:
        raise AssertionError('Native window contains a source sample not yet received')
    return indices


def packet(stream,index):
    pose=stream['bounded'][index]
    return Reference(pose[7:].copy(),stream['dq'][index].copy(),pose[:3].copy(),
                     Rotation.from_quat(pose[[4,5,6,3]]),stream['velocity'][index].copy(),stream['yaw_rate'][index])


class Plant(TwistPlant):
    def __init__(self):
        super().__init__()
        self.observer=self.mj.MjData(self.model)

    def advance(self,target,kp,kd):
        target=np.clip(target,self.limits[:,0],self.limits[:,1])
        sat=0;limit=0.;ratio=0.
        for _ in range(20):
            self.data.ctrl[:]=np.clip(kp*(target-self.data.qpos[7:])-kd*self.data.qvel[6:],-self.effort,self.effort)
            sat+=np.count_nonzero(np.abs(self.data.ctrl)>=self.effort-1e-8)
            ratio=max(ratio,float(np.max(np.abs(self.data.ctrl)/self.effort)))
            self.mj.mj_step(self.model,self.data)
            limit=max(limit,float(np.maximum(self.limits[:,0]-self.data.qpos[7:],self.data.qpos[7:]-self.limits[:,1]).max()))
        if not np.isfinite(np.r_[self.data.qpos,self.data.qvel]).all():
            raise FloatingPointError('Nonfinite physics state')
        self.observer.qpos[:]=self.data.qpos;self.observer.qvel[:]=self.data.qvel;self.observer.ctrl[:]=self.data.ctrl
        self.mj.mj_forward(self.model,self.observer)
        shadow=object.__new__(TwistPlant);shadow.mj=self.mj;shadow.model=self.model;shadow.data=self.observer
        contacts=foot_contacts(shadow)
        slip=[v for f in contacts.values() for v in f['tangent_speeds_m_s']]
        return dict(qpos=self.data.qpos.copy(),qvel=self.data.qvel.copy(),time_s=float(self.data.time),target=target.copy(),
                    saturation_fraction=sat/(20*29),max_torque_ratio=ratio,joint_limit_violation_rad=limit,
                    slip_mean=float(np.mean(slip)) if slip else np.nan)


def rollout(name,case,stream,delay_ms,reference_mode='native'):
    if delay_ms not in (0,120,300,900):
        raise ValueError('Supported reference delays are 0, 120, 300 and 900 ms')
    if reference_mode not in ('native','hold') or (reference_mode=='hold' and delay_ms!=900):
        raise ValueError('Explicit hold ablation is supported only by the 900 ms protocol')
    if name=='sonic_v11' and delay_ms==300:
        raise ValueError('v1.1 G1 needs 900 ms native preview; 300 ms protocol provides only 180 ms. Use current-reference 0/120 ms.')
    plant=Plant();policy=FACTORIES[name](plant);records=[]
    delay_ticks=round(delay_ms/20);length=len(stream['time_s'])
    for tick in range(length+delay_ticks):
        ix=(native_window_indices(name,tick,length,reference_mode) if delay_ms==900
            else window_indices(tick,delay_ticks,length))
        window=[packet(stream,j) for j in ix]
        index=int(ix[0]);ref=window[0]
        plant.data.xfrc_applied[:]=0
        if case=='push' and 4<=tick*DT<4.2:
            plant.data.xfrc_applied[plant.model.body('pelvis').id,0]=25.
        start=time.perf_counter();target=policy.step(window,plant);ms=(time.perf_counter()-start)*1000
        record=plant.advance(target,policy.kp,policy.kd)
        rpy=ref.rotation.as_euler('xyz')
        record.update(policy_ms=ms,reference=np.r_[ref.velocity[:2],ref.xyz[2],rpy[:2],ref.yaw_rate,ref.q],
                      desired_qpos=stream['original'][index].copy(),command_qpos=stream['bounded'][index].copy(),
                      active=bool(stream['active'][index] and tick>=delay_ticks),stage=stream['stage'][index],
                      reference_index=index,newest_reference_index=int(ix[-1]),
                      available_reference_index=tick if delay_ms==900 else available_index(tick,delay_ticks))
        if delay_ms==900:
            record['reference_window_indices']=ix.copy()
        records.append(record)
    a=arrays(records);result=metrics(a)
    from scripts.benchmark_twist_policy import balance_violation
    safe=~balance_violation(a['qpos']);mask=a['active'];upright=mask&safe
    result['active_upright_fraction']=float(np.mean(safe[mask]))
    result['joint_rmse_while_upright_deg']=float(np.degrees(np.sqrt(np.mean((a['qpos'][upright,7:]-a['desired_qpos'][upright,7:])**2)))) if upright.any() else None
    result['reference_active_fraction']=float(mask.mean())
    # Body positions are derived only for scoring / robot-side kinematics.
    # They never replace missing visual measurements.
    body_ids=[plant.model.body(n).id for n in CONFIG['beyond']['body_names']]
    actual_data=plant.mj.MjData(plant.model);desired_data=plant.mj.MjData(plant.model)
    local_errors=[]
    for i in np.flatnonzero(mask):
        actual_data.qpos[:]=a['qpos'][i];desired_data.qpos[:]=a['desired_qpos'][i]
        plant.mj.mj_kinematics(plant.model,actual_data);plant.mj.mj_kinematics(plant.model,desired_data)
        actual_R=Rotation.from_quat(a['qpos'][i,[4,5,6,3]])
        desired_R=Rotation.from_quat(a['desired_qpos'][i,[4,5,6,3]])
        actual_local=actual_R.inv().apply(actual_data.xpos[body_ids]-a['qpos'][i,:3])
        desired_local=desired_R.inv().apply(desired_data.xpos[body_ids]-a['desired_qpos'][i,:3])
        local_errors.extend(np.linalg.norm(actual_local-desired_local,axis=1))
    result['root_local_body_position_mean_mm']=float(np.mean(local_errors)*1000)
    result['root_local_body_position_p95_mm']=float(np.percentile(local_errors,95)*1000)
    result['pd_gains']='original trained/deployment gains, no tuning; common motor torque envelope'
    result['delay_ms']=delay_ms;result['controller']=name;result['case']=case
    result['checkpoint_scope']='motion-specific dance expert; OOD on other references' if name=='beyond' else 'general motion tracker'
    if delay_ms==900:result['reference_mode']=reference_mode
    return a,result


def simulate(out,delays=(120,300),cases=None,controllers=DEFAULT_CONTROLLERS):
    if 'sonic_v11' in controllers and 300 in delays:
        raise ValueError('v1.1 supports direct/hold 0 or 120 ms here; its 900 ms native preview cannot use the 300 ms protocol.')
    out.mkdir(parents=True,exist_ok=False)
    inputs=[Path(__file__),ROOT/'motion_control/comparison_policies.py',ROOT/'config/g1_controller_comparison.json',
            ROOT/'assets/humanoid/controller_comparison_manifest.json',VISUAL,require_asset(MOTION),
            require_asset(TWIST_CONFIG['model_asset']),require_asset(TWIST_CONFIG['policy_asset']),ROOT/'motion_control/twist.py',
            ROOT/'.robot-runtime.json',ROOT/'requirements-robot-lock.txt']
    fingerprints={str(p):sha(p) for p in inputs}
    streams={case:prepare(case) for case in (CASES if cases is None else cases)}
    for case,stream in streams.items():np.savez_compressed(out/(case+'_reference.npz'),**stream)
    protocol=dict(hardware_enabled=False,simulation_only=True,controllers={n:NAMES[n] for n in controllers},cases={c:CASES[c] for c in streams},
        protocol_revision='v4: add official SONIC v1.1, heading-normalized G1 input and documented ankle pitch gain scale 1.5; existing controller outputs regression-checked.',
        sonic_v11='1751 encoder inputs; native G1 samples at 0..900ms (step5). Direct/120ms repeats current reference in all slots. Kp/Kd motors 4,10 multiplied by official 1.5; action scale unchanged. No native-preview claim.',
        selected_delay_ms=list(delays),
        comparison='Pinned pretrained artifacts on one fixed G1 plant; not a paper leaderboard or equal-training-budget comparison',
        delay_conditions={'0':'No injected transport or preview-buffer delay. Newest cached reference repeated in SONIC future slots; no prediction. Not a live camera latency measurement.',
                          '120':'120ms transport; repeated newest received reference in SONIC future slots; no predicted/future observations',
                          '300':'120ms transport + common 180ms buffer; all track the same 300ms-old target; SONIC uses available offsets 0..180ms'},
        model_sha256=sha(require_asset(TWIST_CONFIG['model_asset'])),initial_state='Common production default qpos, zero qvel/history action; no source-state reset',
        physics_dt_s=.001,control_dt_s=.02,reference_processing='ZOH source sampling, causal rate limits, backward derivatives, invalid observations withdraw to standing; no reference uses robot truth',
        actuators='All share original G1 torque envelope. Native trained Kp/Kd and action normalization retained, not tuned. No model mass/contact/inertia/armature changes.',
        path_interfaces='Common aligned root trajectory. TWIST2 uses existing bounded position/heading correction; SONIC consumes joint trajectory and root orientation; BeyondMimic wose consumes joint trajectory and torso orientation. Last two have no explicit XY command channel.',
        beyond_scope='RoboJuDo public Dance_wose checkpoint, native dance first 12 seconds included; other clips test out-of-distribution behavior. No claim of official generalist BeyondMimic weights.',
        scoring='All post-failure samples retained; active valid source ticks scored; no time warp or post-hoc position alignment. Balance envelope height>0.5m and tilt<0.5rad is diagnostic, not proof.',
        inputs_sha256=fingerprints)
    save(out/'protocol.json',protocol)
    results={}
    for delay in delays:
        for case,stream in streams.items():
            if delay==300 and case in ('standing','push'):continue
            for name in controllers:
                key=f'{case}_{delay}_{name}'
                a,result=rollout(name,case,stream,delay)
                np.savez_compressed(out/(key+'.npz'),**a);results[key]=result
                save(out/'results.json',results)
                print(key,json.dumps({k:result[k] for k in ['balance_envelope_passed','joint_rmse_deg','path_error_p95_m','first_balance_violation_s']}),flush=True)
    for p,digest in fingerprints.items():
        if sha(p)!=digest:raise RuntimeError('Input changed during evaluation: '+p)
    save(out/'integrity.json',dict(inputs_unchanged=True,reference_identical_across_controllers=True,inputs_sha256=fingerprints))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/controller_compare_20261020/evaluation_v2')
    p.add_argument('--delays',nargs='+',type=int,choices=(0,120,300),default=[120,300])
    p.add_argument('--cases',nargs='+',choices=tuple(CASES))
    p.add_argument('--controllers',nargs='+',choices=tuple(NAMES),default=list(DEFAULT_CONTROLLERS))
    args=p.parse_args();simulate(args.output.resolve(),tuple(dict.fromkeys(args.delays)),args.cases,tuple(dict.fromkeys(args.controllers)))


if __name__=='__main__':main()
