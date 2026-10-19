"""Replay observed poses through IK, a guarded controller and simulated feedback."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .adapters import SimulatedRobot, ros2_trajectory
from .controller import MotionController, StopRequired
from .model import RobotModel
from .pose import PoseProcessor
from .retarget import ArmRetargeter

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_rows(path, rows):
    with Path(path).open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False)+'\n')


def run(session, output, model_path, perception_delay_ms=120.0):
    """Virtual-clock replay, with explicit capture-to-arrival delay, never wall time."""
    session, output, model_path = Path(session).resolve(), Path(output).resolve(), Path(model_path).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite control evidence: {output}')
    if not np.isfinite(perception_delay_ms) or perception_delay_ms < 0:
        raise ValueError('Perception delay must be finite and nonnegative')
    model = RobotModel.load(model_path)
    rows = read_rows(session/'poses3d.jsonl')
    if not rows or any(type(r['timestamp_ns']) is not int or r['timestamp_ns'] < 0 for r in rows):
        raise ValueError('Poses require nonnegative integer exposure timestamps')
    if any(b['timestamp_ns'] <= a['timestamp_ns'] for a,b in zip(rows,rows[1:])):
        raise ValueError('Replay input must be strictly ordered')
    processor, retargeter = PoseProcessor(model.config['perception']), ArmRetargeter(model)
    clock = 'replay_monotonic'
    controller, robot = MotionController(model, clock), SimulatedRobot(model, clock)
    delay_ns = round(perception_delay_ms*1e6)
    end = rows[-1]['timestamp_ns']+delay_ns+round(model.config['control']['settle_seconds']*1e9)
    index = 0
    observations, commands, feedback_rows, events, processing_times = [], [], [], [], []
    start_wall = time.perf_counter()
    for now in range(rows[0]['timestamp_ns'], end+1, controller.period_ns):
        # Drop older pending frames, bounding processing latency at the source boundary.
        ready = []
        while index < len(rows) and rows[index]['timestamp_ns']+delay_ns <= now:
            ready.append(rows[index])
            index += 1
        if ready:
            row = ready[-1]
            begin = time.perf_counter()
            observation = processor.process(row)
            target = retargeter.solve(observation) if observation is not None else None
            controller.observe(target)
            processing_times.append(time.perf_counter()-begin)
            entry = {'frame_id': row['frame_id'], 'timestamp_ns': row['timestamp_ns'], 'arrival_timestamp_ns': now,
                     'accepted': bool(target is not None and target.accepted), 'reason': processor.reason,
                     'dropped_pending_frames': len(ready)-1}
            if observation is not None:
                entry.update(observation=observation.to_dict(), target_positions_rad=target.positions_rad.tolist(),
                             ik_max_direction_error_deg=target.max_direction_error_deg)
                if not target.accepted:
                    entry['reason'] = 'IK residual or solver convergence failed'
            observations.append(entry)
        feedback = robot.read_feedback(now)
        feedback_rows.append({'timestamp_ns': now, 'positions_rad': feedback.positions_rad,
                              'velocities_rad_s': feedback.velocities_rad_s, 'enabled': feedback.enabled})
        try:
            command = controller.tick(now, feedback)
            robot.send(command, now)
            commands.append(command)
        except (StopRequired, ValueError) as error:
            robot.stop(str(error))
            events.append({'timestamp_ns': now, 'state': 'STOP', 'reason': str(error)})
            break
    if not commands:
        raise RuntimeError('No valid commands generated')
    q = np.array([c.positions_rad for c in commands])
    v = np.array([c.velocities_rad_s for c in commands])
    dt = controller.period_ns/1e9
    acceleration = np.diff(v, axis=0)/dt
    actual = np.array([f['positions_rad'] for f in feedback_rows[:len(commands)]])
    # Feedback precedes the current send, so compare it to the previous setpoint.
    prior = np.vstack((model.home, q[:-1]))
    tracking_error = actual-prior
    checks = {'position_limits': bool(np.all(q >= model.lower-1e-9) and np.all(q <= model.upper+1e-9)),
              'velocity_limits': bool(np.all(np.abs(v) <= model.velocity+1e-9)),
              'sampled_acceleration_limits': bool(np.all(np.abs(acceleration) <= model.acceleration+1e-9)),
              'position_velocity_consistency': bool(np.allclose(np.diff(q,axis=0)/dt, v[1:], atol=1e-8)),
              'fresh_sources': all(c.source_timestamp_ns is None or 0 <= c.timestamp_ns-c.source_timestamp_ns <= model.config['control']['max_pose_age_ms']*1e6 for c in commands),
              'end_stationary': bool(np.max(np.abs(v[-1])) < 1e-8),
              'motion_generated': any(c.state == 'TRACKING' for c in commands) and float(np.max(np.abs(q-model.home))) > 1e-6,
              'no_adapter_faults': not events}
    report = {'schema_version': 1, 'simulation_only': True, 'checks': checks,
              'passed': all(checks.values()), 'robot_id': model.robot_id, 'model_sha256': model.sha256,
              'input_pose_sha256': hashlib.sha256((session/'poses3d.jsonl').read_bytes()).hexdigest(),
              'clock_domain': clock, 'control_rate_hz': model.config['control']['rate_hz'],
              'assumed_perception_delay_ms': perception_delay_ms,
              'observations': len(observations), 'accepted_observations': sum(o['accepted'] for o in observations),
              'command_count': len(commands), 'states': dict(Counter(c.state for c in commands)),
              'max_ik_direction_error_deg': max((o.get('ik_max_direction_error_deg',0) for o in observations), default=0),
              'max_command_velocity_rad_s': float(np.abs(v).max()),
              'max_command_acceleration_rad_s2': float(np.abs(acceleration).max()) if len(acceleration) else 0,
              'feedback_tracking_rmse_rad': float(np.sqrt(np.mean(tracking_error**2))),
              'feedback_tracking_max_rad': float(np.abs(tracking_error).max()),
              'mean_pose_ik_compute_ms': 1000*float(np.mean(processing_times)) if processing_times else None,
              'p95_pose_ik_compute_ms': 1000*float(np.percentile(processing_times,95)) if processing_times else None,
              'wall_seconds': time.perf_counter()-start_wall,
              'limitations': ['Kinematic first-order servo; no dynamics, contact, collision or balance simulation.',
                             'Virtual 50 Hz schedule is not a hardware real-time guarantee.',
                             'Perception delay is injected, not measured capture-to-command latency.',
                             'COCO17 cannot observe wrist roll, fingers or full segment rotations.',
                             'ROS trajectory is offline simulation data; hardware interpolation requires retiming.']}
    source_files = [*ROOT.glob('motion_control/*.py'), model_path]
    report['source_sha256'] = {p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p):
                               hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    output.mkdir(parents=True)
    write_rows(output/'observations.jsonl', observations)
    write_rows(output/'commands.jsonl', [c.to_dict() for c in commands])
    write_rows(output/'feedback.jsonl', feedback_rows)
    write_rows(output/'events.jsonl', events)
    (output/'robot_model.json').write_text(json.dumps(model.config, indent=2), encoding='utf-8')
    (output/'ros2_trajectory.json').write_text(json.dumps({'simulation_only': True,
        'type': 'trajectory_msgs/msg/JointTrajectory', 'message': ros2_trajectory(commands)}, indent=2), encoding='utf-8')
    (output/'control_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if not report['passed']:
        raise RuntimeError(f'Control replay failed: {checks}; see {output}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--robot', type=Path, default=ROOT/'config/robot_demo.json')
    parser.add_argument('--perception-delay-ms', type=float, default=120)
    args = parser.parse_args()
    report = run(args.session, args.output or args.session/'control', args.robot, args.perception_delay_ms)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
