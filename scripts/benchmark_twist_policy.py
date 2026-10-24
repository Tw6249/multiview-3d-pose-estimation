"""Reproducible, simulation-only audit of the pinned TWIST2 controller.

Inputs: pinned offline reference asset, named cached visual references, and a
frozen test protocol. SI units; world Z up, quaternions XYZW in source assets
and WXYZ in MuJoCo. Integer source ns remain owned by the existing replay.
Outputs: measured dynamics, reference/feedback plots, video and JSON evidence.
No source perception is synthesized from robot state. Analytic excitations and
the upstream motion are controller tests, not markerless imitation evidence.
Failures are retained. No tuning, hardware I/O, or controller changes occur.
"""
import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from motion_control.twist import TwistPlant, TwistPolicy, DEFAULT_REF, DEFAULT_Q, CONFIG, KP, KD
from motion_control.tracking_metrics import foot_contacts
from pose_sim.robot_assets import require_asset

DT = .02
MOTION = 'twist2/assets/example_motions/0807_yanjie_walk_001.pkl'
VISUAL = ROOT/'outputs/foot_contact_baseline_20261019/accepted_baseline/s1_walk/baseline/humanoid_reference.jsonl'
TITLES = {
    'standing': '静止站立', 'push': '站立 + 已知外力',
    'upper_body': '双臂与腰部协调', 'official_twist': '官方行走参考 + TWIST2',
    'official_pd': '同一行走参考 + 直接 PD', 'visual': '真人视觉参考 + 当前完整控制链',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def percent(values, p=95):
    return float(np.percentile(values, p)) if len(values) else None


class AuditPlant(TwistPlant):
    """Call production step one physical tick at a time; only observe it.

    This preserves the applied PD law and saturation. The equivalence regression
    compares complete qpos/qvel against the uninstrumented production plant.
    """
    def __init__(self):
        super().__init__()
        self.records = []
        self.request = DEFAULT_REF.copy()
        self.policy_ms = 0.

    def step(self, target, period=DT):
        saturated = total = 0
        max_limit_violation = 0.
        maximum_torque_ratio = 0.
        for _ in range(round(period/self.model.opt.timestep)):
            applied = super().step(target, self.model.opt.timestep)
            saturated += int(np.count_nonzero(np.abs(self.data.ctrl) >= self.effort-1e-8))
            total += len(self.effort)
            maximum_torque_ratio = max(maximum_torque_ratio, float(np.max(np.abs(self.data.ctrl)/self.effort)))
            q = self.data.qpos[7:]
            max_limit_violation = max(max_limit_violation, float(np.max(np.maximum(self.limits[:, 0]-q, q-self.limits[:, 1]))))
        # Refresh derived positions/contact quantities for the end-of-step state
        # on a private copy, so observation cannot alter the live dynamics.
        observed = object.__new__(TwistPlant)
        observed.mj, observed.model = self.mj, self.model
        observed.data = self.mj.MjData(self.model)
        observed.data.qpos[:] = self.data.qpos
        observed.data.qvel[:] = self.data.qvel
        observed.data.ctrl[:] = self.data.ctrl
        observed.data.xfrc_applied[:] = self.data.xfrc_applied
        self.mj.mj_forward(self.model, observed.data)
        contacts = foot_contacts(observed)
        slip = [v for f in contacts.values() for v in f['tangent_speeds_m_s']]
        self.records.append(dict(
            time_s=float(self.data.time), qpos=self.data.qpos.copy(), qvel=self.data.qvel.copy(),
            reference=self.request.copy(), target=np.array(applied), policy_ms=self.policy_ms,
            saturation_fraction=saturated/total, max_torque_ratio=maximum_torque_ratio,
            joint_limit_violation_rad=max_limit_violation,
            foot_force_n=[contacts[s]['normal_force_n'] for s in ('left', 'right')],
            slip_mean=float(np.mean(slip)) if slip else np.nan,
            slip_max=max(slip) if slip else np.nan))
        return applied


class OfflineMotion:
    """Single-clip offline adapter matching upstream MotionLib rate semantics.

    Root XYZ is linearly interpolated; rotation uses SLERP. Translation and
    world rotation rates use central differences and one-sided endpoints.
    Rates are held at the left sample, as in the pinned MotionLib implementation.
    This uses recorded future frames OFFLINE, never in a visual online test.
    Joint order is the pinned G1 asset order. Source full rotation maps velocity
    into the root-local frame, matching the upstream reference packet contract.
    """
    def __init__(self, data):
        self.fps = float(data['fps'])
        self.pos = np.asarray(data['root_pos'], float)
        self.rot = Rotation.from_quat(np.asarray(data['root_rot'], float))
        self.joints = np.asarray(data['dof_pos'], float)
        if self.fps <= 0 or self.pos.shape != (len(self.joints), 3) or self.joints.shape[1] != 29:
            raise ValueError('Invalid upstream reference shape/rate')
        if len(self.pos) < 2 or not np.isfinite(np.r_[self.pos.ravel(), self.joints.ravel()]).all():
            raise ValueError('Invalid upstream reference values')
        self.times = np.arange(len(self.pos))/self.fps
        self.duration = float(self.times[-1])
        self.slerp = Slerp(self.times, self.rot)
        self.velocity = np.gradient(self.pos, 1/self.fps, axis=0)
        boundary = (self.rot[1:]*self.rot[:-1].inv()).as_rotvec()*self.fps
        interior = (self.rot[2:]*self.rot[:-2].inv()).as_rotvec()*(self.fps/2)
        self.omega = np.vstack((boundary[0], interior, boundary[-1]))

    def sample(self, t):
        t = float(np.clip(t, 0, self.duration))
        index = min(int(t*self.fps), len(self.times)-2)
        a = (t-self.times[index])*self.fps
        interp = lambda x: (1-a)*x[index]+a*x[index+1]
        pos, q, rotation = interp(self.pos), interp(self.joints), self.slerp(t)
        rate_index = min(int(t*self.fps), len(self.times)-1)
        velocity = rotation.inv().apply(self.velocity[rate_index])
        omega = rotation.inv().apply(self.omega[rate_index])
        rpy = rotation.as_euler('xyz')
        return np.r_[velocity[:2], pos[2], rpy[:2], omega[2], q], pos, rotation


def alignment_reference(nominal, fraction):
    """Blend a frozen initial pose; do not repeat its motion feedforward."""
    reference = (1-fraction)*DEFAULT_REF+fraction*nominal
    reference[[0,1,5]] = 0.
    return reference


def arrays(records):
    return {k: np.asarray([r[k] for r in records]) for k in records[0]}


def balance_violation(qpos):
    rpy = Rotation.from_quat(qpos[:, [4, 5, 6, 3]]).as_euler('xyz')
    return (qpos[:, 2] <= .5) | (np.abs(rpy[:, :2]).max(axis=1) >= .5)


def metrics(a):
    active = a['active'].astype(bool)
    joint_error = a['qpos'][:, 7:]-a['desired_qpos'][:, 7:]
    bound_error = a['qpos'][:, 7:]-a['reference'][:, 6:]
    path_error = np.linalg.norm(a['qpos'][:, :2]-a['desired_qpos'][:, :2], axis=1)
    rpy = Rotation.from_quat(a['qpos'][:, [4, 5, 6, 3]]).as_euler('xyz')
    violation = balance_violation(a['qpos'])
    final = a['time_s'] >= a['time_s'][-1]-.2-1e-8
    slip = a['slip_mean'][np.isfinite(a['slip_mean']) & active]
    active_violation = np.flatnonzero(violation & active)
    tail_speed = np.linalg.norm(a['qvel'][final, :3], axis=1)
    valid = np.flatnonzero(active)
    result = dict(
        duration_s=float(a['time_s'][-1]), active_samples=int(active.sum()),
        balance_envelope_passed=bool(not violation.any()),
        active_balance_envelope_passed=bool(not len(active_violation)),
        first_balance_violation_s=float(a['time_s'][np.flatnonzero(violation)[0]]) if violation.any() else None,
        minimum_root_height_m=float(a['qpos'][:, 2].min()),
        maximum_abs_roll_pitch_deg=float(np.degrees(np.abs(rpy[:, :2]).max())),
        joint_rmse_deg=float(np.degrees(np.sqrt(np.mean(joint_error[active]**2)))),
        leg_rmse_deg=float(np.degrees(np.sqrt(np.mean(joint_error[active, :12]**2)))),
        bounded_joint_rmse_deg=float(np.degrees(np.sqrt(np.mean(bound_error[active]**2)))),
        path_error_p95_m=percent(path_error[active]),
        final_root_xy_m=a['qpos'][-1, :2].tolist(),
        saturation_fraction=float(a['saturation_fraction'].mean()),
        maximum_torque_limit_ratio=float(a['max_torque_ratio'].max()),
        maximum_joint_limit_violation_deg=float(np.degrees(a['joint_limit_violation_rad'].max())),
        loaded_contact_tick_mean_slip_m_s=float(np.mean(slip)) if len(slip) else None,
        policy_cpu_p50_ms=percent(a['policy_ms'][1:], 50),
        policy_cpu_p95_ms=percent(a['policy_ms'][1:]),
        policy_cpu_p99_ms=percent(a['policy_ms'][1:], 99),
        final_base_speed_m_s=float(tail_speed.mean()),
        standing_stop_passed=bool(not violation[final].any() and tail_speed.mean() < .05),
        metric_scope='End-of-command-interval state vs held reference; no phase realignment. Balance envelope is a diagnostic, not a stability proof; all post-failure samples retained.',
    )
    correlations = {}
    for j in (0, 3, 6, 9):
        x, y = a['desired_qpos'][active, 7+j], a['qpos'][active, 7+j]
        correlations[str(j)] = float(np.corrcoef(x, y)[0, 1]) if min(np.std(x), np.std(y)) > .03 else None
    result['hip_knee_zero_lag_correlation'] = correlations
    if not len(valid):
        raise ValueError('No active samples: cannot score controller')
    return result


def run_isolated(name, motion):
    plant, policy = AuditPlant(), TwistPolicy() if name != 'official_pd' else None
    walking = name.startswith('official')
    total = 3+motion.duration+4 if walking else 15. if name == 'upper_body' else 12.
    ref = DEFAULT_REF.copy()
    desired, active, stages = [], [], []
    alignment = None
    for i in range(math.ceil(total/DT)):
        t = i*DT
        request = DEFAULT_REF.copy()
        pose = np.r_[0., 0., DEFAULT_REF[2], 1., 0., 0., 0., DEFAULT_Q]
        phase = 'standing'
        used = t >= 2
        if walking:
            source_t = np.clip(t-3, 0, motion.duration)
            nominal, pos, rot = motion.sample(source_t)
            if 2 <= t < 3:
                request = alignment_reference(nominal, t-2)
                phase = 'aligning'
            elif 3 <= t < 3+motion.duration:
                request = nominal
                phase = 'tracking'
            elif t >= 3+motion.duration:
                phase = 'stopping'
            if t >= 3 and alignment is None:
                actual_yaw = Rotation.from_quat(plant.data.qpos[[4, 5, 6, 3]]).as_euler('xyz')[2]
                alignment = Rotation.from_euler('z', actual_yaw-motion.rot[0].as_euler('xyz')[2])
                origin = plant.data.qpos[:3].copy()
            if alignment is not None:
                mapped = alignment.apply(pos-motion.pos[0])+origin
                mapped[2] = pos[2]
                r = (alignment*rot).as_quat()
                pose = np.r_[mapped, r[[3, 0, 1, 2]], nominal[6:]]
                if phase != 'tracking':
                    pose[2] = request[2]
                    pose[7:] = request[6:]
                    final_yaw = (alignment*rot).as_euler('xyz')[2]
                    quat = Rotation.from_euler('xyz', [request[3],request[4],final_yaw]).as_quat()
                    pose[3:7] = quat[[3,0,1,2]]
            else:
                pose[2] = request[2]
                pose[7:] = request[6:]
            used = phase == 'tracking'
        elif name == 'upper_body' and 3 <= t < 11:
            u = (t-3)/8
            envelope = np.sin(np.pi*u)**2
            request[6+15] -= .65*envelope
            request[6+22] -= .45*envelope
            request[6+18] += .30*envelope*np.sin(4*np.pi*u)
            request[6+25] -= .25*envelope*np.sin(4*np.pi*u)
            request[6+12] += .20*envelope*np.sin(2*np.pi*u)
            phase = 'tracking'
        elif name == 'upper_body':
            phase = 'stopping' if t >= 11 else 'standing'
        if not walking:
            pose[7:] = request[6:]
            used = 3 <= t < 11 if name == 'upper_body' else t >= 2
        plant.data.xfrc_applied[:] = 0
        if name == 'push' and 6 <= t < 6.2:
            plant.data.xfrc_applied[plant.model.body('pelvis').id, 0] = 25.
            phase = 'push_25N'
        ref += np.clip(request-ref, -np.r_[.04,.04,.012,.03,.03,.06,np.full(29,.12)],
                       np.r_[.04,.04,.012,.03,.03,.06,np.full(29,.12)])
        plant.request = ref.copy()
        begin = time.perf_counter()
        target = policy.step(ref, *plant.state()) if policy else ref[6:].copy()
        plant.policy_ms = (time.perf_counter()-begin)*1000 if policy else 0.
        plant.step(target)
        desired.append(pose)
        active.append(used)
        stages.append(phase)
    a = arrays(plant.records)
    a.update(desired_qpos=np.asarray(desired), active=np.asarray(active), stage=np.asarray(stages))
    return a, metrics(a)


def run_visual(out):
    """Run the EXISTING visual replay/gate/root tracker without code changes."""
    from motion_control import humanoid_replay as module
    session = out/'visual_session'
    session.mkdir()
    shutil.copyfile(VISUAL, session/'humanoid_reference.jsonl')
    instances = []
    class Plant(AuditPlant):
        def __init__(self):
            super().__init__()
            instances.append(self)
    class Policy(TwistPolicy):
        def step(self, reference, *state):
            instances[-1].request = np.array(reference)
            begin = time.perf_counter()
            target = super().step(reference, *state)
            instances[-1].policy_ms = (time.perf_counter()-begin)*1000
            return target
    original_plant, original_policy = module.TwistPlant, module.TwistPolicy
    module.TwistPlant, module.TwistPolicy = Plant, Policy
    try:
        original_report = module.replay(session, video=False, output_name='replay',
                                        delay_ms=120, save_logs=True, transition_mode='legacy')
    finally:
        module.TwistPlant, module.TwistPolicy = original_plant, original_policy
    feedback = [json.loads(x) for x in (session/'replay/feedback.jsonl').read_text().splitlines()]
    rows = [json.loads(x) for x in VISUAL.read_text().splitlines()]
    rows_by_frame = {r['frame_id']: r for r in rows}
    first = next(r for r in rows if r['accepted'])
    a = arrays(instances[0].records)
    poses, active, stages = [], [], []
    # RootTracker anchors at t=2 before that control step, including the delay gap.
    anchor_i = round(2/DT)-1
    origin = a['qpos'][anchor_i, :3]
    yaw0 = Rotation.from_quat(a['qpos'][anchor_i, [4,5,6,3]]).as_euler('xyz')[2]
    align = Rotation.from_euler('z', yaw0-first['root_heading_rad'])
    for i, f in enumerate(feedback):
        row = rows_by_frame[f['source_frame_id']]
        valid = f['state'] == 'tracking' and row['accepted']
        ref = np.asarray(row['reference']) if valid else a['reference'][i]
        if valid:
            xyz = align.apply(np.asarray(row['root_position_m'])-np.asarray(first['root_position_m']))+origin
            xyz[2] = row['root_position_m'][2]
            yaw = yaw0+row['root_heading_rad']-first['root_heading_rad']
        else:
            xyz = a['qpos'][i, :3].copy()
            xyz[2] = ref[2]
            yaw = Rotation.from_quat(a['qpos'][i,[4,5,6,3]]).as_euler('xyz')[2]
        quat = Rotation.from_euler('xyz', [ref[3],ref[4],yaw]).as_quat()
        poses.append(np.r_[xyz,quat[[3,0,1,2]],ref[6:]])
        active.append(valid)
        stages.append(f['state'])
    a.update(desired_qpos=np.asarray(poses), active=np.asarray(active), stage=np.asarray(stages))
    result = metrics(a)
    result['production_replay'] = original_report
    result['acceptance_fraction'] = original_report['acceptance_fraction']
    return a, result


def simulate(out):
    out.mkdir(parents=True, exist_ok=False)
    input_paths = [require_asset(MOTION), require_asset(CONFIG['policy_asset']), require_asset(CONFIG['model_asset']), VISUAL,
                   ROOT/'motion_control/twist.py', ROOT/'config/g1_twist.json', ROOT/'motion_control/humanoid_replay.py',
                   ROOT/'motion_control/root_tracking.py', Path(__file__)]
    hashes = {str(p): sha(p) for p in input_paths}
    protocol = dict(date_local='2026-10-20', hardware_enabled=False,
        cases=list(TITLES), control_dt_s=DT, physics_dt_s=.001,
        controller='Pinned TWIST2 ONNX, CPU; no model retraining or controller/config changes',
        initial_state='Fresh production default floating-base G1 state and zero policy history for each case',
        official_adapter='Offline XYZ linear / quaternion SLERP; central differences with one-sided endpoints and left-sample rate hold, following pinned MotionLib. Full original duration, speed 1x; no frame trimming.',
        official_schedule='2 s standing, 1 s first-pose blend with zero velocity/yaw-rate feedforward, original motion, 4 s standing withdrawal; current reference rate limiter',
        protocol_revision='v2: initial v1 incorrectly repeated moving first-frame feedforward during frozen pose alignment. Original first yaw-rate is approximately -16 rad/s. v1 retained separately as adapter failure evidence; no controller tuning.',
        official_path='First source pose rigidly aligned to actual robot pose at motion start; no XY feedback correction',
        direct_pd='Identical reference schedule, model, PD gains, rate limits; reference joint angles directly feed PD, neural policy omitted',
        push='25 N world +X on pelvis from 6.0 to 6.2 s, impulse 5 N s; one diagnostic condition only',
        visual='Existing replay with original gate, root tracker, rate limiter, 120 ms simulated delay, legacy withdrawal; cached real S1 walking, no new perception',
        assessment='No time warping. Full trials including failures. Active-reference error and entire-trial balance reported separately.',
        balance_envelope='root height > 0.5 m and abs roll/pitch < 0.5 rad; appropriate for these standing/walking clips, not general fall classification',
        inputs_sha256=hashes)
    save(out/'protocol.json', protocol)
    motion = OfflineMotion(pickle.loads(require_asset(MOTION).read_bytes()))
    results = {}
    for name in TITLES:
        begin = time.perf_counter()
        a, result = run_visual(out) if name == 'visual' else run_isolated(name, motion)
        np.savez_compressed(out/f'{name}.npz', **a)
        result.update(title=TITLES[name], wall_seconds=time.perf_counter()-begin)
        results[name] = result
        save(out/'results.json', results)
        print(name, json.dumps({k:result[k] for k in ('balance_envelope_passed','joint_rmse_deg','path_error_p95_m','standing_stop_passed')}, ensure_ascii=False), flush=True)
    if any(sha(p) != digest for p,digest in hashes.items()):
        raise RuntimeError('Source/configuration changed during benchmark')
    save(out/'integrity.json', dict(inputs_unchanged=True, inputs_sha256=hashes))


def plots(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'], 'axes.unicode_minus':False,
                         'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    results = json.loads((out/'results.json').read_text(encoding='utf-8'))
    fig, axs = plt.subplots(2,3,figsize=(16,9),layout='constrained')
    for ax,(name,result) in zip(axs.flat,results.items()):
        a = np.load(out/f'{name}.npz')
        actual, desired = a['qpos'], a['desired_qpos']
        mask = a['active'].astype(bool)
        ax.plot(desired[mask,0],desired[mask,1],color='#d69725',lw=2.4,label='参考根部路径')
        ax.plot(actual[:,0],actual[:,1],color='#176fbd',lw=1.8,label='实际动力学轨迹')
        ax.scatter(actual[0,0],actual[0,1],color='#176fbd',marker='o',s=30)
        ax.scatter(actual[-1,0],actual[-1,1],color='#b52c3a',marker='x',s=50)
        ax.set_title(f'{TITLES[name]}\n关节 RMSE {result["joint_rmse_deg"]:.1f}° · 路径 P95 {result["path_error_p95_m"]:.3f} m')
        ax.set(xlabel='世界 X (m)',ylabel='世界 Y (m)')
        ax.axis('equal'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('固定 G1 / TWIST2 控制测试：参考与实际根部轨迹\n源动作、视觉参考与解析激励分别评测；仅动力学仿真',fontsize=17)
    fig.savefig(out/'summary.png',dpi=140);plt.close(fig)
    for name in results:
        a=np.load(out/f'{name}.npz'); t=a['time_s']; actual=a['qpos']; desired=a['desired_qpos']
        fig,axs=plt.subplots(3,2,figsize=(14,9),layout='constrained')
        for ax,j,label in zip(axs.flat[:4],(3,9,15,12),('左膝','右膝','左肩俯仰','腰部偏航')):
            ax.plot(t,np.degrees(desired[:,7+j]),color='#d69725',lw=1.8,label='原始动作参考')
            ax.plot(t,np.degrees(a['reference'][:,6+j]),color='#777777',lw=1,ls=':',label='实际送入策略的参考')
            ax.plot(t,np.degrees(actual[:,7+j]),color='#176fbd',lw=1.3,label='物理反馈')
            ax.set(title=label,ylabel='角度 (°)');ax.grid(alpha=.2)
        axs[0,0].legend(fontsize=8)
        axs[2,0].plot(t,actual[:,2],color='#176fbd',label='实际骨盆高度')
        axs[2,0].plot(t,desired[:,2],color='#d69725',label='参考高度')
        axs[2,0].axhline(.5,color='#b52c3a',ls='--',lw=1)
        axs[2,0].set(ylabel='高度 (m)',xlabel='仿真时间 (s)');axs[2,0].legend(fontsize=8)
        tilt=np.degrees(Rotation.from_quat(actual[:,[4,5,6,3]]).as_euler('xyz')[:,:2])
        axs[2,1].plot(t,tilt[:,0],label='roll');axs[2,1].plot(t,tilt[:,1],label='pitch')
        for bound in (-np.degrees(.5),np.degrees(.5)):axs[2,1].axhline(bound,color='#b52c3a',ls='--',lw=1)
        axs[2,1].set(ylabel='机身倾角 (°)',xlabel='仿真时间 (s)');axs[2,1].legend(fontsize=8)
        for ax in axs.flat:
            ax.grid(alpha=.2)
            if name=='push':ax.axvspan(6,6.2,color='#b52c3a',alpha=.15)
        fig.suptitle(TITLES[name]+'：原始参考 / 受限参考 / 实际反馈（未事后对齐）',fontsize=16)
        fig.savefig(out/f'{name}_tracking.png',dpi=130);plt.close(fig)


def render(out):
    from PIL import Image, ImageDraw, ImageFont
    import cv2
    results=json.loads((out/'results.json').read_text(encoding='utf-8'))
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:raise RuntimeError('ffmpeg missing; keep simulation evidence')
    font=lambda size,bold=False:ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc' if bold else 'C:/Windows/Fonts/msyh.ttc',size)
    fonts={size:font(size) for size in (16,18,22,26,32)}
    plant=TwistPlant();mj=plant.mj
    plant.model.vis.global_.offwidth=600;plant.model.vis.global_.offheight=500
    renderer=mj.Renderer(plant.model,500,600)
    actual=mj.MjData(plant.model);reference=mj.MjData(plant.model)
    camera=mj.MjvCamera();camera.azimuth=135;camera.elevation=-15;camera.distance=2.8
    output=out/'control_comparison.mp4'
    command=[ffmpeg,'-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','25','-i','-',
             '-an','-c:v','libx264','-preset','fast','-crf','21','-threads','4','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    encoder=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    timestamps={};total=0;snapshots=[]
    stage_names={'standing':'站立','aligning':'初始过渡','tracking':'动作跟踪','stopping':'结束停止',
                 'push_25N':'25 N 外力','initial_alignment':'初始过渡','vision_unavailable':'视觉参考不可用','return_to_stand':'结束停止'}
    try:
        for name,result in results.items():
            a=np.load(out/f'{name}.npz'); timestamps[name]=total/25
            active=np.flatnonzero(a['active'])
            snapshot_index=int(active[len(active)//2])
            for frame in range(math.ceil(a['time_s'][-1]*25)):
                idx=min(int(round((frame/25)/DT)),len(a['time_s'])-1)
                canvas=Image.new('RGB',(1280,720),'#f4f6f8');draw=ImageDraw.Draw(canvas)
                draw.text((30,16),TITLES[name],font=fonts[32],fill='#17283c')
                draw.text((30,61),'左：动作参考（仅运动学显示）',font=fonts[22],fill='#94620d')
                draw.text((670,61),'右：实际机器人（自由基座动力学）',font=fonts[22],fill='#176fbd')
                for x,data,pose in ((25,reference,a['desired_qpos'][idx]),(655,actual,a['qpos'][idx])):
                    data.qpos[:]=pose;mj.mj_forward(plant.model,data)
                    camera.lookat[:]=[pose[0],pose[1],.65]
                    renderer.update_scene(data,camera=camera)
                    canvas.paste(Image.fromarray(renderer.render()),(x,100))
                draw=ImageDraw.Draw(canvas)
                stage=stage_names.get(str(a['stage'][idx]),str(a['stage'][idx]))
                draw.text((30,615),f't = {a["time_s"][idx]:5.2f} s   阶段：{stage}   实际高度：{a["qpos"][idx,2]:.3f} m',font=fonts[22],fill='#17283c')
                draw.text((30,652),f'活动段关节 RMSE：{result["joint_rmse_deg"]:.1f}°    根部路径 P95：{result["path_error_p95_m"]:.3f} m',font=fonts[18],fill='#17283c')
                draw.text((30,685),'原速播放 · 相机分别跟随机器人，位移偏差请看路径图 · 仅仿真，不代表实机能力',font=fonts[16],fill='#536679')
                if name=='push' and 6<=a['time_s'][idx]<6.22:
                    draw.text((900,615),'外力 +X：25 N',font=fonts[22],fill='#b52c3a')
                encoder.stdin.write(np.asarray(canvas).tobytes());total+=1
                if abs(idx-snapshot_index)<=1 and not any(s[0]==name for s in snapshots):
                    snapshot=canvas.copy();snapshot.save(out/f'{name}_snapshot.png');snapshots.append((name,snapshot))
            print('rendered',name,flush=True)
    finally:
        renderer.close()
        encoder.stdin.close()
        error=encoder.stderr.read().decode('utf-8',errors='replace');code=encoder.wait()
    if code:raise RuntimeError('Video encoder failed: '+error)
    subprocess.run([ffmpeg,'-v','error','-xerror','-i',str(output),'-f','null','-'],check=True,capture_output=True)
    cap=cv2.VideoCapture(str(output));decoded=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        if frame.shape!=(720,1280,3):raise RuntimeError('Unexpected rendered frame size')
        decoded+=1
    cap.release()
    if decoded!=total:raise RuntimeError('Video frame count mismatch')
    save(out/'video_validation.json',dict(frames=total,decoded_frames=decoded,fps=25,duration_s=total/25,case_start_seconds=timestamps,ffmpeg_full_decode_passed=True))
    grid=Image.new('RGB',(1280,3*360),'white')
    for i,(name,snapshot) in enumerate(snapshots):grid.paste(snapshot.resize((640,360)),((i%2)*640,(i//2)*360))
    grid.save(out/'snapshot_grid.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/twist_policy_eval_20261020')
    parser.add_argument('--stage',choices=('simulate','visualize','all'),default='all')
    args=parser.parse_args();out=args.output.resolve()
    if args.stage in ('simulate','all'):simulate(out)
    if args.stage in ('visualize','all'):plots(out);render(out)


if __name__=='__main__':main()
