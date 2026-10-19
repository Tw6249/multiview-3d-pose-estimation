"""Deterministic position governor with freshness checks and latched faults.

This is a host-level command guard, not a certified safety controller. Loss of
trustworthy feedback demands the adapter's independent device stop.
"""
import numpy as np
from .contracts import JointCommand, timestamp, vector


class StopRequired(RuntimeError):
    pass


class MotionController:
    def __init__(self, model, clock_domain):
        self.model, self.clock = model, clock_domain
        self.cfg = model.config['control']
        self.period_ns = round(1e9/self.cfg['rate_hz'])
        self.last_tick = None
        self.q = self.v = None
        self.target = None
        self.consecutive = 0
        self.last_source = None
        self.reason = 'waiting for pose'
        self.fault = None
        self.sequence = 0

    def observe(self, target):
        if target is None or not target.accepted:
            self.target, self.consecutive = None, 0
            self.reason = 'pose or IK rejected'
            return
        timestamp(target.timestamp_ns)
        if self.last_source is not None and target.timestamp_ns <= self.last_source:
            self.target, self.consecutive = None, 0
            self.reason = 'nonmonotonic target'
            return
        try:
            values = vector(target.positions_rad, self.model.size, 'target')
        except ValueError:
            self.target, self.consecutive = None, 0
            self.reason = 'nonfinite or malformed target'
            return
        if np.any(values < self.model.lower) or np.any(values > self.model.upper):
            self.target, self.consecutive = None, 0
            self.reason = 'target outside limits'
            return
        if self.last_source is not None and target.timestamp_ns-self.last_source > self.cfg['max_pose_age_ms']*1e6:
            self.consecutive = 0
        self.last_source = target.timestamp_ns
        self.target = target
        self.consecutive += 1

    def _stop(self, reason):
        self.fault = reason
        raise StopRequired(reason)

    def tick(self, now_ns, feedback, estop=False):
        timestamp(now_ns)
        if estop:
            self._stop('emergency stop requested')
        if self.fault:
            raise StopRequired(self.fault)
        m, cfg = self.model, self.cfg
        interval_ns = self.period_ns if self.last_tick is None else now_ns-self.last_tick
        if interval_ns <= 0 or interval_ns > cfg['max_tick_interval_ms']*1e6:
            self._stop('control clock discontinuity/deadline miss')
        if feedback.clock_domain != self.clock or tuple(feedback.joint_names) != m.names:
            self._stop('feedback clock/joint order mismatch')
        try:
            timestamp(feedback.timestamp_ns)
            actual = vector(feedback.positions_rad, m.size, 'feedback positions')
            speed = vector(feedback.velocities_rad_s, m.size, 'feedback velocities')
        except ValueError as error:
            self._stop(str(error))
        age = now_ns-feedback.timestamp_ns
        if age < 0 or age > cfg['max_feedback_age_ms']*1e6 or not feedback.enabled:
            self._stop('feedback stale, future or device disabled')
        if np.any(actual < m.lower-1e-8) or np.any(actual > m.upper+1e-8) or np.any(np.abs(speed) > m.velocity+1e-8):
            self._stop('feedback outside joint limits')
        initializing = self.q is None
        if initializing:
            if np.max(np.abs(speed)) > 1e-3:
                self._stop('initial feedback must be stationary')
            self.q, self.v = actual.copy(), np.zeros(m.size)
        if np.max(np.abs(actual-self.q)) > cfg['tracking_error_rad']:
            self._stop('robot tracking error')
        fresh = self.target is not None and 0 <= now_ns-self.target.timestamp_ns <= cfg['max_pose_age_ms']*1e6
        active = fresh and self.consecutive >= cfg['recovery_frames'] and not initializing
        dt = interval_ns/1e9
        if active:
            error = self.target.positions_rad-self.q
            wanted = np.sign(error)*np.minimum(cfg['gain_per_s']*np.abs(error), np.sqrt(2*m.acceleration*np.abs(error)))
            self.reason = 'tracking accepted pose'
        else:
            wanted = np.zeros(m.size)
            if self.target is not None:
                self.reason = 'recovering observations' if fresh else 'pose stale or future'
            if not fresh:
                self.consecutive = 0
        # Conservative stopping envelope includes an extra sample of travel.
        cap_up = np.sqrt((m.acceleration*dt)**2 + 2*m.acceleration*np.maximum(0,m.upper-self.q))-m.acceleration*dt
        cap_down = np.sqrt((m.acceleration*dt)**2 + 2*m.acceleration*np.maximum(0,self.q-m.lower))-m.acceleration*dt
        wanted = np.clip(wanted, -np.minimum(m.velocity, cap_down), np.minimum(m.velocity, cap_up))
        new_v = self.v + np.clip(wanted-self.v, -m.acceleration*dt, m.acceleration*dt)
        new_q = self.q + new_v*dt
        if np.any(new_q < m.lower-1e-9) or np.any(new_q > m.upper+1e-9):
            self._stop('no feasible bounded trajectory')
        self.q, self.v = new_q, new_v
        self.last_tick = now_ns
        state = 'TRACKING' if active else ('BRAKING' if np.max(np.abs(new_v)) > 1e-9 else 'HOLD')
        result = JointCommand(1, self.sequence, now_ns, now_ns+round(cfg['command_ttl_ms']*1e6),
                              self.target.timestamp_ns if active else None,
                              self.clock, m.robot_id, m.sha256, m.names,
                              tuple(new_q), tuple(new_v), state, self.reason)
        self.sequence += 1
        return result
