"""Independent command validation, a kinematic servo, and ROS 2 serialization."""
import numpy as np
from .contracts import JointFeedback, timestamp, vector


class SimulatedRobot:
    """First-order position servo with joint speed clipping, NOT rigid-body dynamics.

    The adapter has its own expiry watchdog, even if the host controller stops.
    read_feedback/advance/send run on the same injected monotonic clock.
    """
    def __init__(self, model, clock_domain):
        if model.config['deployment'] != 'simulation_only':
            raise ValueError('This adapter accepts simulation models only')
        self.model, self.clock = model, clock_domain
        self.q, self.v = model.home.copy(), np.zeros(model.size)
        self.command = None
        self.last_sequence = -1
        self.last_time = None
        self.enabled = True
        self.stop_reason = None

    def stop(self, reason):
        self.enabled, self.stop_reason = False, reason
        self.command = None
        self.v[:] = 0

    def read_feedback(self, now_ns):
        timestamp(now_ns)
        if self.last_time is not None and now_ns < self.last_time:
            self.stop('adapter clock went backwards')
        if self.command is not None and now_ns >= self.command.valid_until_ns:
            self.stop('independent command watchdog expired')
        if self.last_time is not None and self.enabled and self.command is not None:
            dt = (now_ns-self.last_time)/1e9
            delta = np.asarray(self.command.positions_rad)-self.q
            # Exact first-order discretization avoids overshoot at variable dt.
            step = np.clip((1-np.exp(-25*dt))*delta, -self.model.velocity*dt, self.model.velocity*dt)
            self.q += step
            self.v = step/dt if dt else self.v
        self.last_time = now_ns
        return JointFeedback(now_ns, self.clock, self.model.names, tuple(self.q), tuple(self.v), self.enabled)

    def send(self, command, now_ns):
        try:
            self._validate(command, now_ns)
        except ValueError as error:
            self.stop(str(error))
            raise
        self.command = command
        self.last_sequence = command.sequence

    def _validate(self, c, now):
        timestamp(now)
        timestamp(c.timestamp_ns)
        timestamp(c.valid_until_ns)
        if not self.enabled:
            raise ValueError('Adapter is stopped; reconnect/rearm explicitly')
        if c.schema_version != 1 or c.clock_domain != self.clock:
            raise ValueError('Command schema/clock mismatch')
        if c.robot_id != self.model.robot_id or c.model_sha256 != self.model.sha256 or tuple(c.joint_names) != self.model.names:
            raise ValueError('Command robot/model/joint order mismatch')
        if type(c.sequence) is not int or c.sequence <= self.last_sequence:
            raise ValueError('Replayed/out-of-order command')
        if not c.timestamp_ns <= now < c.valid_until_ns or c.valid_until_ns-c.timestamp_ns > self.model.config['control']['command_ttl_ms']*1e6:
            raise ValueError('Command expired/future or excessive lifetime')
        if c.state not in ('TRACKING', 'BRAKING', 'HOLD'):
            raise ValueError('Unsupported command state')
        if c.state == 'TRACKING':
            timestamp(c.source_timestamp_ns)
            if not 0 <= now-c.source_timestamp_ns <= self.model.config['control']['max_pose_age_ms']*1e6:
                raise ValueError('Command has no fresh observation')
        q = vector(c.positions_rad, self.model.size, 'command positions')
        v = vector(c.velocities_rad_s, self.model.size, 'command velocities')
        if np.any(q < self.model.lower) or np.any(q > self.model.upper) or np.any(np.abs(v) > self.model.velocity+1e-9):
            raise ValueError('Command exceeds joint limits')
        if self.command is not None:
            dt = (c.timestamp_ns-self.command.timestamp_ns)/1e9
            previous_q = np.asarray(self.command.positions_rad)
            previous_v = np.asarray(self.command.velocities_rad_s)
            if dt <= 0 or not np.allclose((q-previous_q)/dt,v,atol=1e-8):
                raise ValueError('Command position/velocity discontinuity')
            if np.any(np.abs(v-previous_v)/dt > self.model.acceleration+1e-8):
                raise ValueError('Command acceleration exceeds limits')
        elif not np.allclose(q,self.q,atol=1e-8) or np.max(np.abs(v)) > 1e-8:
            raise ValueError('First command must match stationary device feedback')


def ros2_trajectory(commands):
    """Offline ROS 2 JointTrajectory layout; no ROS dependency or network send.

    Positions only deliberately select linear segments. Hardware must retime
    against its own dynamics; sampled acceleration bounds are not continuous
    acceleration guarantees of the receiving controller's interpolator.
    """
    if not commands:
        raise ValueError('Cannot export empty trajectory')
    start = commands[0].timestamp_ns
    names, model = commands[0].joint_names, commands[0].model_sha256
    points, previous = [], -1
    for c in commands:
        elapsed = c.timestamp_ns-start
        if elapsed <= previous or c.joint_names != names or c.model_sha256 != model:
            raise ValueError('Inconsistent trajectory')
        previous = elapsed
        points.append({'positions': list(c.positions_rad),
                       'time_from_start': {'sec': elapsed//1_000_000_000, 'nanosec': elapsed%1_000_000_000}})
    return {'header': {'stamp': {'sec': 0, 'nanosec': 0}, 'frame_id': ''},
            'joint_names': list(names), 'points': points}
