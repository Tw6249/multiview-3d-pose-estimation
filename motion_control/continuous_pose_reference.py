"""Continuous 25 Hz pose retargeting and 50 Hz C2 controller reference stream.

Contract
--------
- Inputs:
  * `rows`: sequence of causal 3D human state dicts (`state_schema_id ==
    'continuous_human_state_v1'`, `hardware_enabled is False`) at 25 Hz with
    monotonic `timestamp_ns` (int, nanoseconds), 12 body landmarks in source
    world coordinates (meters, +Z up), and per-joint `control_permission` /
    `prediction_age_ns`.
  * `config`: dict from `config/continuous_prediction.json`.
- Outputs:
  * `retarget_continuous_pose`: list of 25 Hz reference dicts with
    `reference_schema_id == 'continuous_simulation_reference_v2'`, 35-element
    `reference` (`[vx, vy, z, roll, pitch, yaw_rate, q_0..q_28]` in SI units:
    m/s, m, rad, rad/s), `root_position_m` (m), `root_heading_rad` (rad), and
    explicit `short_gap_held` flag.
  * `build_c2_reference_stream`: 50 Hz (`dt = 0.02 s`) stream dict compatible
    with `scripts/benchmark_controllers.py` (`original`, `bounded`, `dq`,
    `ddq`, `jerk`, `velocity`, `yaw_rate`, `active`, `stage`, `source_ids`,
    `time_s`), with analytical C1 velocities and C2 bounded accelerations/jerks.
- Coordinate frames & time base:
  * World frame: +Z up, XY anchored and heading-aligned at first accepted
    pelvis observation.
  * Time base: integer nanoseconds (`timestamp_ns`) for 25 Hz source frames;
    fixed 50 Hz (`0.02 s`) simulation control clock with 2 s standing, 1 s
    alignment, clip duration tracking, and 3 s terminal standing stop.
- Failure behavior:
  * Nonmonotonic timestamps, altered source identity/calibration, nonfinite or
    out-of-bound trajectory states raise `ValueError`.
  * Short permission dips (`<= max_hold_s`, default `0.40 s`) hold the last
    valid retargeted pose with decaying root velocity (`short_gap_held=True`)
    and are excluded from active tracking error scores (`active=False`).
  * Longer dropouts (`> max_hold_s`) set `accepted=False` and transition
    smoothly via C2 trajectory replanning while preserving current heading.
  * No upstream balance/CBF filter (`reference_safety.py`) is applied; physical
    balance, stability, and safety are handled by the downstream controller.
"""
import copy
import json
import math
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .continuous_reference import RecoveryReferenceTrajectory, permitted_body
from .humanoid import BONES, DEFAULT_Q, FullBodyRetargeter, NAMES, SDK_JOINT_NAMES
from .pose import OneEuro, unit
from .twist import DEFAULT_REF, TwistPlant

ROOT = Path(__file__).resolve().parents[1]


def load_recovery_config():
    return json.loads((ROOT / 'config/g1_continuous_recovery.json').read_text(encoding='utf-8'))


def _compute_expiry_ns(row, config, timestamp_ns):
    states = dict(zip(row['joint_names'], row['joint_states']))
    a = 0.5 * config['uncertainty_acceleration_m_s2']
    b = config['uncertainty_velocity_m_s']
    expiry = []
    for name in NAMES[6:]:
        s = states[name]
        age = s['prediction_age_ns'] / 1e9
        base_scale = s['uncertainty_scale_m'] - b * age - a * age * age
        c = base_scale - config['permission_uncertainty_m']
        horizon = max(0.0, min(config['max_prediction_age_s'],
                               (-b + math.sqrt(max(0.0, b * b - 4.0 * a * c))) / (2.0 * a)))
        expiry.append(s['last_observed_timestamp_ns'] + round(horizon * 1e9))
    return max(int(timestamp_ns), int(min(expiry)))


def retarget_continuous_pose(
    rows,
    config,
    max_joint_vel_rad_s=6.0,
    min_height_m=0.48,
    max_height_m=0.95,
    max_root_speed_m_s=1.5,
    max_yaw_rate_rad_s=2.5,
    max_hold_s=0.40,
    gimbal_barrier=False,
):
    """Retarget 25 Hz continuous 3D human states without turn lockouts or crouch resets.

    `gimbal_barrier=True` (opt-in, default keeps the 2026-10-21 behaviour) adds a
    soft residual against the shoulder pitch/roll branch where |pitch| > 1.15 rad
    and roll is beyond 1.15 rad toward its large-range side; joint limits and the
    per-step velocity trust region are unchanged.
    """
    if max_joint_vel_rad_s <= 0 or min_height_m >= max_height_m or max_hold_s < 0:
        raise ValueError('Invalid continuous retargeting bounds')
    engine = FullBodyRetargeter(input_filtered=True)
    results = []
    last_time = None
    signature = None
    last_valid_result = None
    for row in rows:
        t = row['timestamp_ns']
        if type(t) is not int:
            raise ValueError('Integer timestamp_ns required')
        current = tuple(
            row.get(k)
            for k in ('source_id', 'landmark_schema_id', 'calibration_sha256', 'perception_signature')
        )
        if last_time is not None and (t <= last_time or current != signature):
            raise ValueError('Source identity or clock changed')
        last_time, signature = t, current
        result = dict(
            frame_id=row['frame_id'],
            timestamp_ns=t,
            accepted=False,
            reference_schema_id='continuous_simulation_reference_v2',
            hardware_enabled=False,
        )
        try:
            points = permitted_body(row)
            p = np.asarray(points, float)
            if p.shape != (12, 3) or not np.isfinite(p[6:]).all():
                raise ValueError('Invalid body geometry')
            observed = np.isfinite(p).all(axis=1)
            dt = (t - engine.last_time) / 1e9 if engine.last_time is not None else 0.04
            if engine.last_time is not None and dt <= 0:
                raise ValueError('Nonmonotonic observation')
            dt_eff = max(0.02, min(dt, 0.20)) if engine.last_time is not None else 0.04
            lengths = np.array([np.linalg.norm(p[b] - p[a]) for a, b in BONES])
            if np.any(lengths < 0.1) or np.any(lengths > 0.85):
                raise ValueError('Implausible human limb length')
            if engine.lengths is not None:
                bad = np.maximum(lengths / engine.lengths, engine.lengths / lengths) > 1.65
                if bad[4:].any():
                    raise ValueError('Leg bone length discontinuity')
                for k in np.where(bad[:4])[0]:
                    side = 0 if k < 2 else 1
                    p[[side, side + 2, side + 4]] = np.nan
                    observed[[side, side + 2, side + 4]] = False
                lengths = np.array([np.linalg.norm(p[b] - p[a]) for a, b in BONES])
            torso_observed = bool(observed[:2].all())
            lateral = 0.3 * (p[6] - p[7]) + 0.7 * (p[0] - p[1]) if torso_observed else p[6] - p[7]
            lateral[2] = 0.0
            if np.linalg.norm(lateral) < 0.08:
                raise ValueError('Unobservable pelvis heading')
            if engine.last_time is not None and dt > 0.5:
                engine.filter = OneEuro(dict(min_cutoff_hz=2.5, beta=0.2, derivative_cutoff_hz=1))
                engine.root_filter = OneEuro(dict(min_cutoff_hz=0.8, beta=0.05, derivative_cutoff_hz=1))
            raw = p.copy()
            filled = p.copy()
            for k in np.where(~observed)[0]:
                filled[k] = (
                    engine.filter.filtered[k]
                    if engine.filter.filtered is not None
                    else p[6 + k % 2] + [0.0, 0.0, 0.5]
                )
            p = filled if engine.input_filtered else engine.filter.update(filled, dt)
            lateral = 0.3 * (p[6] - p[7]) + 0.7 * (p[0] - p[1]) if torso_observed else p[6] - p[7]
            lateral[2] = 0.0
            root = engine.root_filter.update(np.array([(p[6] + p[7]) / 2.0, unit(lateral)]), dt)
            lateral = unit(root[1])
            forward = np.cross(lateral, [0.0, 0.0, 1.0])
            raw_heading = float(np.arctan2(forward[1], forward[0]))
            origin = root[0].copy()

            # Continuous yaw-rate clamping prevents turn-lockout rollback cascades.
            if engine.last_heading is not None and 0.0 < dt <= 0.5:
                dyaw = float(
                    np.arctan2(
                        np.sin(raw_heading - engine.last_heading),
                        np.cos(raw_heading - engine.last_heading),
                    )
                )
                max_dyaw = max_yaw_rate_rad_s * dt
                yaw_clamped = abs(dyaw) > max_dyaw
                if yaw_clamped:
                    dyaw = math.copysign(max_dyaw, dyaw)
                heading = float(
                    np.arctan2(np.sin(engine.last_heading + dyaw), np.cos(engine.last_heading + dyaw))
                )
                yaw_rate = dyaw / dt
            else:
                heading = raw_heading
                yaw_rate = 0.0
                yaw_clamped = False

            R = Rotation.from_euler('z', heading).as_matrix()
            bone_observed = np.array([observed[a] and observed[b] for a, b in BONES])
            directions = np.array(
                [unit(p[b] - p[a]) @ R if good else np.zeros(3) for (a, b), good in zip(BONES, bone_observed)]
            )
            torso_up = unit((p[0] + p[1]) / 2.0 - origin) @ R if torso_observed else np.array([0.0, 0.0, 1.0])
            shoulder_left = unit(p[0] - p[1]) @ R if torso_observed else np.array([0.0, 1.0, 0.0])
            previous = engine.q.copy()
            active = engine.active
            neutral_target = previous.copy()
            for indices, points_idx in ((np.arange(15, 19), (0, 2, 4)), (np.arange(22, 26), (1, 3, 5))):
                if not observed[list(points_idx)].all():
                    neutral_target[indices] += np.clip(DEFAULT_Q[indices] - previous[indices], -0.12, 0.12)

            def residual(x):
                engine.data.qpos[7:] = previous
                engine.data.qpos[7 + active] = x
                engine.mj.mj_kinematics(engine.model, engine.data)
                actual = np.array([unit(engine.data.xpos[b] - engine.data.xpos[a]) for a, b in engine.ids])
                torso = engine.data.body('torso_link').xmat.reshape(3, 3)
                foot_rot = []
                for side in ('left', 'right'):
                    foot = engine.data.body(side + '_ankle_roll_link').xmat.reshape(3, 3)
                    foot_rot.extend(Rotation.from_matrix(foot).as_rotvec() * 0.45)
                barrier = []
                if gimbal_barrier:
                    # Active indices 15/16 and 19/20 are shoulder pitch/roll of left/right arm
                    # (active == arange(23) up to 18, then 22..25). Penalize the doubly
                    # inverted (|pitch| and |roll| both large) IK branch.
                    barrier = [
                        2.5 * max(0.0, abs(x[15]) - 1.15) * max(0.0, x[16] - 1.15),
                        2.5 * max(0.0, abs(x[19]) - 1.15) * max(0.0, -x[20] - 1.15),
                    ]
                return np.r_[
                    ((actual - directions) * bone_observed[:, None]).ravel(),
                    (torso[:, 2] - torso_up) * 1.3,
                    (torso[:, 1] - shoulder_left) * 0.5,
                    foot_rot,
                    barrier,
                    0.15 * (x - neutral_target[active]),
                    0.025 * (x - DEFAULT_Q[active]),
                ]

            if engine.last_time is None:
                lo = engine.plant.limits[active, 0] + 1e-5
                hi = engine.plant.limits[active, 1] - 1e-5
            else:
                max_step = max_joint_vel_rad_s * dt_eff
                lo = np.maximum(engine.plant.limits[active, 0] + 1e-5, previous[active] - max_step)
                hi = np.minimum(engine.plant.limits[active, 1] - 1e-5, previous[active] + max_step)
            x0 = np.clip(previous[active], lo, hi)
            fit = least_squares(residual, x0, bounds=(lo, hi), max_nfev=45, ftol=1e-5, xtol=1e-5, gtol=1e-5)
            candidate = previous.copy()
            candidate[active] = np.clip(fit.x, lo, hi)
            residual(candidate[active])
            actual = np.array([unit(engine.data.xpos[b] - engine.data.xpos[a]) for a, b in engine.ids])
            errors = np.degrees(np.arccos(np.clip(np.sum(actual * directions, axis=1), -1.0, 1.0)))
            feet = np.array([engine.data.body(s + '_ankle_roll_link').xpos.copy() for s in ('left', 'right')])
            raw_height = float(-np.min(feet[:, 2]) + 0.035)
            height = float(np.clip(raw_height, min_height_m, max_height_m))
            height_clamped = abs(height - raw_height) > 1e-9

            source_leg = float(np.mean((engine.lengths if engine.lengths is not None else lengths)[4:]) * 2.0)
            scale = float(engine.target_leg_length / source_leg)
            velocity = np.zeros(2)
            speed_clamped = False
            if engine.last_origin is not None and 0.0 < dt <= 0.5:
                disp_world = (origin - engine.last_origin) * scale
                horiz_speed = float(np.linalg.norm(disp_world[:2]) / dt)
                if horiz_speed > max_root_speed_m_s and horiz_speed > 1e-6:
                    disp_world[:2] *= (max_root_speed_m_s * dt) / np.linalg.norm(disp_world[:2])
                    origin[:2] = engine.last_origin[:2] + disp_world[:2] / scale
                    speed_clamped = True
                velocity = ((disp_world / dt) @ R)[:2]
            path = engine.root_path.update(origin[:2], scale, t)
            reference = np.r_[velocity, height, 0.0, 0.0, yaw_rate, candidate]
            engine.q = candidate
            engine.last_points = raw
            engine.last_time = t
            engine.last_origin = origin
            engine.last_heading = heading
            if engine.lengths is None:
                engine.lengths = lengths.copy()
            else:
                engine.lengths = np.where(
                    np.isfinite(lengths),
                    np.where(np.isfinite(engine.lengths), 0.98 * engine.lengths + 0.02 * lengths, lengths),
                    engine.lengths,
                )
            states = dict(zip(row['joint_names'], row['joint_states']))
            measured = [states[n]['last_observed_timestamp_ns'] == t for n in NAMES]
            result.update(
                timestamp_ns=t,
                reference=reference.tolist(),
                joint_names=SDK_JOINT_NAMES,
                root_position_m=np.r_[path, height].tolist(),
                unclamped_base_height_m=raw_height,
                root_heading_rad=float(heading),
                unclamped_root_heading_rad=raw_heading,
                root_path_frame='source_world_xy_anchored_at_first_accepted_pelvis; target_height_z',
                limb_direction_error_deg=[float(e) if valid else None for e, valid in zip(errors, bone_observed)],
                source_leg_length_m=source_leg,
                scale_to_target=scale,
                source_body_points_m=[point.tolist() if valid else None for point, valid in zip(p, observed)],
                estimated_body_points=observed.tolist(),
                observed_body_points=measured,
                arm_observed=[bool(observed[[0, 2, 4]].all()), bool(observed[[1, 3, 5]].all())],
                torso_observed=torso_observed,
                foot_height_above_lowest_m=(feet[:, 2] - feet[:, 2].min()).tolist(),
                source_state_status=[states[n]['estimate_status'] for n in NAMES],
                source_last_observed_ns=[states[n]['last_observed_timestamp_ns'] for n in NAMES],
                prediction_valid_until_ns=_compute_expiry_ns(row, config, t),
                clamped_channels=dict(
                    height=bool(height_clamped),
                    yaw_rate=bool(yaw_clamped),
                    root_speed=bool(speed_clamped),
                ),
                short_gap_held=False,
                priors=[
                    'upright pelvis; torso tilt assigned to waist',
                    'heading from fused hip/shoulder lateral axes',
                    'flat foot heading',
                    'neutral wrists',
                ],
                unobserved=[
                    'pelvis vs waist tilt separation',
                    'foot yaw/roll/pitch',
                    'wrist/hand orientation',
                    'finger joints',
                ],
                accepted=True,
            )
            last_valid_result = copy.deepcopy(result)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            if last_valid_result is not None and (t - last_valid_result['timestamp_ns']) / 1e9 <= max_hold_s:
                hold_age_s = (t - last_valid_result['timestamp_ns']) / 1e9
                held = copy.deepcopy(last_valid_result)
                held['frame_id'] = row['frame_id']
                held['timestamp_ns'] = t
                decay = max(0.0, 1.0 - hold_age_s / max(max_hold_s, 1e-6))
                ref_vec = np.asarray(held['reference'], float)
                ref_vec[0:2] *= decay
                ref_vec[5] *= decay
                held['reference'] = ref_vec.tolist()
                held['accepted'] = True
                held['short_gap_held'] = True
                held['hold_reason'] = str(exc)
                held['prediction_valid_until_ns'] = t + round((max_hold_s - hold_age_s) * 1e9)
                engine.last_time = t
                result = held
            else:
                result['reason'] = str(exc)
        results.append(result)
    return results


def rows_to_source_dict(rows):
    """Extract aligned world trajectory and joint targets from 25 Hz reference rows."""
    if not rows:
        raise ValueError('Empty reference rows')
    stamps = np.array([(r['timestamp_ns'] - rows[0]['timestamp_ns']) / 1e9 for r in rows], float)
    if len(stamps) > 1 and np.any(np.diff(stamps) <= 0):
        raise ValueError('Nonmonotonic reference timestamps')
    valid_rows = [r for r in rows if r.get('accepted')]
    if not valid_rows:
        raise ValueError('No accepted reference frames')
    first = valid_rows[0]
    origin = np.array(first['root_position_m'], float)
    align = Rotation.from_euler('z', -first['root_heading_rad'])
    q, xyz, quat, accepted, held = [], [], [], [], []
    for row in rows:
        valid = bool(row.get('accepted'))
        accepted.append(valid)
        held.append(bool(row.get('short_gap_held', False)))
        if valid:
            ref = np.array(row['reference'], float)
            pos = align.apply(np.array(row['root_position_m'], float) - origin)
            pos[2] = ref[2]
            rot = Rotation.from_euler('xyz', [ref[3], ref[4], row['root_heading_rad'] - first['root_heading_rad']])
            q.append(ref[6:])
            xyz.append(pos)
            quat.append(rot.as_quat())
        else:
            q.append(DEFAULT_Q.copy())
            xyz.append([0.0, 0.0, DEFAULT_REF[2]])
            quat.append([0.0, 0.0, 0.0, 1.0])
    return (
        dict(
            times=stamps,
            q=np.array(q, float),
            xyz=np.array(xyz, float),
            quat=np.array(quat, float),
            valid=np.array(accepted, bool),
            held=np.array(held, bool),
        ),
        float(stamps[-1]),
    )


def build_legacy_reference_stream(rows, dt=0.02):
    """Build the legacy first-order rate-limited 50 Hz stream for paired comparison."""
    data, duration = rows_to_source_dict(rows)
    total = 3.0 + duration + 3.0
    count = math.ceil(total / dt)
    original, bounded, active, stages, source_ids = [], [], [], [], []
    q = DEFAULT_Q.copy()
    rpy = np.zeros(3)
    height = DEFAULT_REF[2]
    last_xy = np.zeros(2)
    for i in range(count):
        t = i * dt
        pose = np.r_[0.0, 0.0, DEFAULT_REF[2], 1.0, 0.0, 0.0, 0.0, DEFAULT_Q]
        valid = False
        stage = 'standing'
        idx = -1
        if 2.0 <= t < 3.0:
            idx = int(np.flatnonzero(data['valid'])[0])
            u = t - 2.0
            pose[7:] = (1.0 - u) * DEFAULT_Q + u * data['q'][idx]
            pose[2] = (1.0 - u) * DEFAULT_REF[2] + u * data['xyz'][idx, 2]
            stage = 'aligning'
        elif 3.0 <= t <= 3.0 + duration + 1e-9:
            idx = int(np.searchsorted(data['times'], t - 3.0 + 1e-9, side='right') - 1)
            valid = bool(data['valid'][idx])
            stage = 'tracking' if valid else 'reference_unavailable'
            if valid:
                pose = np.r_[data['xyz'][idx], data['quat'][idx][[3, 0, 1, 2]], data['q'][idx]]
                last_xy = pose[:2].copy()
            else:
                pose[:2] = last_xy
        elif t > 3.0 + duration:
            pose[:2] = last_xy
            stage = 'stopping'
        q += np.clip(pose[7:] - q, -0.12, 0.12)
        goal_rpy = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_euler('xyz')
        delta = np.arctan2(np.sin(goal_rpy - rpy), np.cos(goal_rpy - rpy))
        rpy += np.clip(delta, [-0.03, -0.03, -0.06], [0.03, 0.03, 0.06])
        height += np.clip(pose[2] - height, -0.012, 0.012)
        rotation = Rotation.from_euler('xyz', rpy)
        command = np.r_[pose[:2], height, rotation.as_quat()[[3, 0, 1, 2]], q]
        original.append(pose)
        bounded.append(command)
        active.append(valid and not bool(data['held'][idx] if idx >= 0 else False))
        stages.append(stage)
        source_ids.append(idx)
    original, bounded = np.array(original), np.array(bounded)
    dq = np.vstack((np.zeros(29), np.diff(bounded[:, 7:], axis=0) / dt))
    ddq = np.vstack((np.zeros(29), np.diff(dq, axis=0) / dt))
    jerk = np.vstack((np.zeros(29), np.diff(ddq, axis=0) / dt))
    velocity = np.vstack((np.zeros(3), np.diff(bounded[:, :3], axis=0) / dt))
    yaw = np.unwrap(Rotation.from_quat(bounded[:, [4, 5, 6, 3]]).as_euler('xyz')[:, 2])
    yaw_rate = np.r_[0.0, np.diff(yaw) / dt]
    return dict(
        original=original,
        bounded=bounded,
        dq=dq,
        ddq=ddq,
        jerk=jerk,
        velocity=velocity,
        yaw_rate=yaw_rate,
        active=np.array(active),
        stage=np.array(stages),
        source_ids=np.array(source_ids),
        time_s=np.arange(count) * dt,
    )


def _per_dof_step(otg, inp_primary, inp_fallback, state, plans, target_p, target_v, limits, dt):
    """One independent jerk-limited Ruckig step per joint toward (p, v).

    Tries `inp_primary` with `(target_v, 0.0)`, then `inp_fallback` with `0.0`,
    and verifies next-tick stopping feasibility under `inp_fallback` (never clipped).
    """
    from ruckig import RuckigError, Trajectory

    n = len(target_p)
    q, v, a = state
    out_p, out_v, out_a = np.empty(n), np.empty(n), np.empty(n)
    candidates = ((inp_primary, True), (inp_primary, False))
    if inp_fallback is not inp_primary:
        candidates = candidates + ((inp_fallback, False),)
    for j in range(n):
        tp_j = float(np.clip(target_p[j], limits[j, 0], limits[j, 1]))
        found = False
        for inp, use_tv in candidates:
            tv = float(target_v[j]) if use_tv else 0.0
            inp.current_position = [float(q[j])]
            inp.current_velocity = [float(v[j])]
            inp.current_acceleration = [float(a[j])]
            inp.target_position = [tp_j]
            inp.target_velocity = [tv]
            inp.target_acceleration = [0.0]
            curve = Trajectory(1)
            try:
                res = otg[j].calculate(inp, curve)
                ok = int(res) >= 0 and curve.position_extrema[0].min >= limits[j, 0] - 1e-8 \
                    and curve.position_extrema[0].max <= limits[j, 1] + 1e-8
            except RuckigError:
                ok = False
            if ok:
                p1, v1, a1 = curve.at_time(dt)
                if tv == 0.0 and inp is inp_fallback:
                    plans[j] = (curve, dt)
                else:
                    inp_fallback.current_position = [float(p1[0])]
                    inp_fallback.current_velocity = [float(v1[0])]
                    inp_fallback.current_acceleration = [float(a1[0])]
                    inp_fallback.target_position = [tp_j]
                    inp_fallback.target_velocity = [0.0]
                    inp_fallback.target_acceleration = [0.0]
                    stop = Trajectory(1)
                    try:
                        rs = otg[j].calculate(inp_fallback, stop)
                        ok = int(rs) >= 0 and stop.position_extrema[0].min >= limits[j, 0] - 1e-8 \
                            and stop.position_extrema[0].max <= limits[j, 1] + 1e-8
                    except RuckigError:
                        ok = False
                    if ok:
                        plans[j] = (stop, 0.0)
            if ok:
                out_p[j], out_v[j], out_a[j] = p1[0], v1[0], a1[0]
                found = True
                break
        if not found:
            if plans[j] is not None:
                plan, elapsed = plans[j]
                elapsed += dt
                p1, v1, a1 = plan.at_time(elapsed)
                plans[j] = (plan, elapsed)
                out_p[j], out_v[j], out_a[j] = p1[0], v1[0], a1[0]
            else:
                raise ValueError('No certified per-joint trajectory')
    return out_p, out_v, out_a


def build_c2_reference_stream(rows, plant=None, traj_cfg=None, dt=0.02, joint_mode='group_zero_v'):
    """Build a 50 Hz C2-continuous reference stream with analytical C1 velocities.

    `joint_mode='monotonic_soft_jerk'` (alias `'per_dof_balanced'`) implements the
    backend of BRM-C2 (Barrier-Regularized Monotonic C2 Retargeting): each joint is
    planned independently with a softened tracking jerk/acceleration profile
    (1500 rad/s^3, 90 rad/s^2) and monotonicity-gated damped terminal velocity
    (0.45x, zeroed at direction reversals).
    `joint_mode='group_zero_v'` uses whole-limb synchronized Ruckig groups with zero
    terminal velocity. `'per_dof_rate'` plans every joint with full-rate causal
    terminal velocity and bounded lead extrapolation (<= 0.04 s).
    """
    from ruckig import InputParameter, Ruckig, Synchronization, Trajectory

    if joint_mode not in ('group_zero_v', 'per_dof_rate', 'monotonic_soft_jerk', 'per_dof_balanced'):
        raise ValueError('Unknown joint_mode')
    if plant is None:
        plant = TwistPlant()
    if traj_cfg is None:
        traj_cfg = load_recovery_config()
    v_max = float(traj_cfg['joint_max_velocity_rad_s'])
    joint_otg = [Ruckig(1) for _ in range(29)]
    joint_inp = InputParameter(1)
    joint_inp.max_velocity = [v_max]
    joint_inp.max_acceleration = [float(traj_cfg['joint_max_acceleration_rad_s2'])]
    joint_inp.max_jerk = [float(traj_cfg['joint_max_jerk_rad_s3'])]
    joint_inp.synchronization = Synchronization.No
    joint_inp_soft = InputParameter(1)
    joint_inp_soft.max_velocity = [v_max]
    joint_inp_soft.max_acceleration = [min(90.0, float(traj_cfg['joint_max_acceleration_rad_s2']))]
    joint_inp_soft.max_jerk = [min(1500.0, float(traj_cfg['joint_max_jerk_rad_s3']))]
    joint_inp_soft.synchronization = Synchronization.No
    joint_state = np.array([DEFAULT_Q.copy(), np.zeros(29), np.zeros(29)])
    joint_plans = [None] * 29
    data, duration = rows_to_source_dict(rows)
    total = 3.0 + duration + 3.0
    count = math.ceil(total / dt)
    traj = RecoveryReferenceTrajectory(DEFAULT_REF, plant.limits, plant.sha256, traj_cfg, dt)
    root_otg = Ruckig(3)
    root_inp = InputParameter(3)
    root_curve = Trajectory(3)
    root_inp.max_velocity = [1.5, 1.5, 2.5]
    root_inp.max_acceleration = [6.0, 6.0, 10.0]
    root_inp.max_jerk = [30.0, 30.0, 50.0]
    root_inp.synchronization = Synchronization.Time
    root_p = np.zeros(3)
    root_v = np.zeros(3)
    root_a = np.zeros(3)
    original, bounded, dq_list, ddq_list, jerk_list = [], [], [], [], []
    vel_list, yaw_rate_list, active, stages, source_ids = [], [], [], [], []
    last_xy = np.zeros(2)
    last_yaw = 0.0
    for i in range(count):
        t = i * dt
        pose = np.r_[0.0, 0.0, DEFAULT_REF[2], 1.0, 0.0, 0.0, 0.0, DEFAULT_Q]
        valid = False
        stage = 'standing'
        idx = -1
        if 2.0 <= t < 3.0:
            idx = int(np.flatnonzero(data['valid'])[0])
            u = t - 2.0
            pose[7:] = (1.0 - u) * DEFAULT_Q + u * data['q'][idx]
            pose[2] = (1.0 - u) * DEFAULT_REF[2] + u * data['xyz'][idx, 2]
            stage = 'aligning'
        elif 3.0 <= t <= 3.0 + duration + 1e-9:
            idx = int(np.searchsorted(data['times'], t - 3.0 + 1e-9, side='right') - 1)
            valid = bool(data['valid'][idx])
            stage = 'tracking' if valid else 'reference_unavailable'
            if valid:
                pose = np.r_[data['xyz'][idx], data['quat'][idx][[3, 0, 1, 2]], data['q'][idx]]
                last_xy = pose[:2].copy()
                goal_yaw = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_euler('xyz')[2]
                last_yaw = last_yaw + float(np.arctan2(np.sin(goal_yaw - last_yaw), np.cos(goal_yaw - last_yaw)))
            else:
                pose[:2] = last_xy
                rot_hold = Rotation.from_euler('xyz', [0.0, 0.0, last_yaw])
                pose[3:7] = rot_hold.as_quat()[[3, 0, 1, 2]]
        elif t > 3.0 + duration:
            pose[:2] = last_xy
            rot_hold = Rotation.from_euler('xyz', [0.0, 0.0, last_yaw])
            pose[3:7] = rot_hold.as_quat()[[3, 0, 1, 2]]
            stage = 'stopping'
        goal_rpy = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_euler('xyz')
        target_35 = np.r_[0.0, 0.0, pose[2], goal_rpy[0], goal_rpy[1], 0.0, pose[7:]]
        ref_35, diag = traj.step(target_35)
        if joint_mode in ('per_dof_rate', 'monotonic_soft_jerk', 'per_dof_balanced'):
            tv = np.zeros(29)
            tp = pose[7:].copy()
            if joint_mode == 'per_dof_rate':
                if stage == 'tracking' and valid and idx >= 1 and data['valid'][idx - 1] and not data['held'][idx - 1]:
                    src_dt = float(data['times'][idx] - data['times'][idx - 1])
                    if 0.0 < src_dt <= 0.12:
                        tv = np.clip((data['q'][idx] - data['q'][idx - 1]) / src_dt, -0.9 * v_max, 0.9 * v_max)
                        lead = float(np.clip(t - 3.0 - data['times'][idx], 0.0, 0.04))
                        tp = np.clip(pose[7:] + tv * lead, plant.limits[:, 0], plant.limits[:, 1])
                inp_pri = joint_inp
            else:
                if (
                    stage == 'tracking'
                    and valid
                    and not data['held'][idx]
                    and idx >= 2
                    and data['valid'][idx - 1]
                    and data['valid'][idx - 2]
                    and not data['held'][idx - 1]
                ):
                    src_dt = float(data['times'][idx] - data['times'][idx - 1])
                    prev_dt = float(data['times'][idx - 1] - data['times'][idx - 2])
                    if 0.0 < src_dt <= 0.08 and 0.0 < prev_dt <= 0.08:
                        dq1 = (data['q'][idx] - data['q'][idx - 1]) / src_dt
                        dq0 = (data['q'][idx - 1] - data['q'][idx - 2]) / prev_dt
                        tv = np.where((dq1 * dq0) > 0.0, np.clip(0.45 * dq1, -0.7 * v_max, 0.7 * v_max), 0.0)
                inp_pri = joint_inp_soft
            q_new, v_new, a_new = _per_dof_step(joint_otg, inp_pri, joint_inp, joint_state, joint_plans, tp, tv, plant.limits, dt)
            ref_35 = ref_35.copy()
            ref_35[6:] = q_new
            diag = dict(
                reference_rate_per_s=np.r_[np.zeros(6), v_new],
                reference_acceleration_per_s2=np.r_[np.zeros(6), a_new],
                interval_mean_jerk_per_s3=np.r_[np.zeros(6), (a_new - joint_state[2]) / dt],
            )
            joint_state[:] = [q_new, v_new, a_new]
        root_inp.current_position = root_p.tolist()
        root_inp.current_velocity = root_v.tolist()
        root_inp.current_acceleration = root_a.tolist()
        root_inp.target_position = [float(pose[0]), float(pose[1]), float(last_yaw)]
        root_inp.target_velocity = [0.0, 0.0, 0.0]
        root_inp.target_acceleration = [0.0, 0.0, 0.0]
        try:
            res = root_otg.calculate(root_inp, root_curve)
            if int(res) >= 0:
                rp, rv, ra = root_curve.at_time(dt)
                root_p, root_v, root_a = np.asarray(rp, float), np.asarray(rv, float), np.asarray(ra, float)
        except Exception:
            pass
        rot_cmd = Rotation.from_euler('xyz', [ref_35[3], ref_35[4], root_p[2]])
        command = np.r_[root_p[0], root_p[1], ref_35[2], rot_cmd.as_quat()[[3, 0, 1, 2]], ref_35[6:]]
        rate_35 = np.asarray(diag['reference_rate_per_s'], float)
        acc_35 = np.asarray(diag['reference_acceleration_per_s2'], float)
        jerk_35 = np.asarray(diag['interval_mean_jerk_per_s3'], float)
        original.append(pose)
        bounded.append(command)
        dq_list.append(rate_35[6:])
        ddq_list.append(acc_35[6:])
        jerk_list.append(jerk_35[6:])
        vel_list.append(np.array([root_v[0], root_v[1], rate_35[2]], float))
        yaw_rate_list.append(float(root_v[2]))
        active.append(valid and not bool(data['held'][idx] if idx >= 0 else False))
        stages.append(stage)
        source_ids.append(idx)
    return dict(
        original=np.array(original),
        bounded=np.array(bounded),
        dq=np.array(dq_list),
        ddq=np.array(ddq_list),
        jerk=np.array(jerk_list),
        velocity=np.array(vel_list),
        yaw_rate=np.array(yaw_rate_list),
        active=np.array(active),
        stage=np.array(stages),
        source_ids=np.array(source_ids),
        time_s=np.arange(count) * dt,
    )


def build_brm_c2_reference_stream(states, cfg=None, plant=None, traj_cfg=None, dt=0.02):
    """Mainline BRM-C2 (Barrier-Regularized Monotonic C2 Retargeting) pipeline.

    Combines singularity-barrier continuous kinematic retargeting (`gimbal_barrier=True`)
    with the monotonicity-gated soft-jerk 50 Hz C2 trajectory generator
    (`joint_mode='monotonic_soft_jerk'`). Returns `(rows_25hz, stream_50hz)`.
    """
    rows = retarget_continuous_pose(states, cfg, gimbal_barrier=True)
    stream = build_c2_reference_stream(rows, plant=plant, traj_cfg=traj_cfg, dt=dt, joint_mode='monotonic_soft_jerk')
    return rows, stream

