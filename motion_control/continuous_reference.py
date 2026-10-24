"""Explicit simulation adapter for estimated (including predicted) human states.

Does not alter legacy read_body or IK, and never relabels predictions as camera
observations. Requires independent per-point permission and observation age.
"""
import numpy as np
from .humanoid import FullBodyRetargeter, NAMES
from .reference_trajectory import BoundedReferenceTrajectory


class RecoveryReferenceTrajectory(BoundedReferenceTrajectory):
    """C2 replanning with a retained, verified zero-terminal-speed fallback.

    On an infeasible/numerically failed replan continue the previously certified
    trajectory to rest. Never clip the output, reset its derivatives, or publish
    an unverified solver result. There is no inference of physical stability.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plans = [None]*len(self.groups)

    def step(self, target, target_rate=None, prediction_s=0.):
        from ruckig import Trajectory, RuckigError
        target = np.asarray(target, float)
        if target.shape != (35,) or not np.isfinite(target).all():
            raise ValueError('Invalid recovery target')
        bounded = np.clip(target, self.lower, self.upper)
        position, velocity, acceleration = np.empty(35), np.empty(35), np.empty(35)
        fallbacks = []
        for k, (ids, generator, inp, _) in enumerate(self.groups):
            inp.current_position = self.position[ids].tolist()
            inp.current_velocity = self.rate[ids].tolist()
            inp.current_acceleration = self.acceleration[ids].tolist()
            inp.target_position = bounded[ids].tolist()
            inp.target_velocity = [0.]*len(ids)
            inp.target_acceleration = [0.]*len(ids)
            candidate = Trajectory(len(ids))
            try:
                result = generator.calculate(inp, candidate)
                valid = int(result) >= 0 and all(
                    e.min >= lo-1e-8 and e.max <= hi+1e-8
                    for e, lo, hi in zip(candidate.position_extrema, self.lower[ids], self.upper[ids]))
            except RuckigError:
                valid = False
            if valid:
                plan, elapsed = candidate, self.dt
                self.plans[k] = (plan, elapsed)
                p, v, a = plan.at_time(elapsed)
            elif self.plans[k] is not None:
                plan, elapsed = self.plans[k]
                elapsed += self.dt
                p, v, a = plan.at_time(elapsed)
                self.plans[k] = (plan, elapsed)
                fallbacks.append(k)
            elif np.max(np.abs(self.rate[ids])) < 1e-12 and np.max(np.abs(self.acceleration[ids])) < 1e-12:
                p, v, a = self.position[ids], self.rate[ids], self.acceleration[ids]
                fallbacks.append(k)
            else:
                raise ValueError('No certified recovery trajectory available')
            position[ids], velocity[ids], acceleration[ids] = p, v, a
        jerk = (acceleration-self.acceleration)/self.dt
        if (not np.isfinite(np.r_[position, velocity, acceleration]).all() or
                np.any(position < self.lower-1e-8) or np.any(position > self.upper+1e-8) or
                np.any(np.abs(velocity) > self.max_rate+1e-7) or
                np.any(np.abs(acceleration) > self.max_acceleration+1e-7) or
                np.any(np.abs(jerk) > self.max_jerk+1e-5)):
            raise ValueError('Certified recovery trajectory numerical limit failure')
        self.position, self.rate, self.acceleration = position, velocity, acceleration
        return position.copy(), dict(reference_rate_per_s=velocity.tolist(),
            reference_acceleration_per_s2=acceleration.tolist(),
            interval_mean_jerk_per_s3=jerk.tolist(),
            retained_plan_groups=fallbacks, zero_terminal_velocity=True,
            target_model_sha256=self.model_sha256, hardware_enabled=False)


def permitted_body(row):
    if (row.get('state_schema_id') != 'continuous_human_state_v1' or
            row.get('hardware_enabled') is not False or
            row.get('state_timestamp_ns') != row['timestamp_ns']):
        raise ValueError('Explicit continuous simulation state required')
    names = row['joint_names']
    if len(names) != len(set(names)):
        raise ValueError('Duplicate body names')
    points = []
    for name in NAMES:
        i = names.index(name)
        state = row['joint_states'][i]
        p = row['joints_world_m'][i]
        last = state['last_observed_timestamp_ns']
        valid = (state['control_permission'] and type(last) is int and
                 0 <= last <= row['timestamp_ns'] and
                 state['prediction_age_ns'] == row['timestamp_ns']-last and
                 p is not None and np.asarray(p).shape == (3,) and np.isfinite(p).all())
        if not valid:
            if name.endswith(('hip', 'knee', 'ankle')):
                raise ValueError('Prediction permission withdrawn: '+name)
            points.append([np.nan]*3)
        else:
            points.append(p)
    return np.asarray(points)


def retarget_continuous(rows, config):
    engine = FullBodyRetargeter(input_filtered=True)
    results = []
    last_time = None
    signature = None
    for row in rows:
        t = row['timestamp_ns']
        current = tuple(row.get(k) for k in
                        ('source_id', 'landmark_schema_id', 'calibration_sha256', 'perception_signature'))
        if last_time is not None and (t <= last_time or current != signature):
            raise ValueError('Source identity or clock changed')
        last_time, signature = t, current
        result = dict(frame_id=row['frame_id'], timestamp_ns=t, accepted=False,
                      reference_schema_id='continuous_simulation_reference_v1',
                      hardware_enabled=False)
        try:
            points = permitted_body(row)
            result.update(engine.solve(points, t))
            result['accepted'] = True
            states = dict(zip(row['joint_names'], row['joint_states']))
            measured = [states[n]['last_observed_timestamp_ns'] == t for n in NAMES]
            result.update(estimated_body_points= np.isfinite(points).all(1).tolist(),
                          observed_body_points=measured,
                          arm_observed=[all(measured[i] for i in ids) for ids in ((0, 2, 4), (1, 3, 5))],
                          source_state_status=[states[n]['estimate_status'] for n in NAMES],
                          source_last_observed_ns=[states[n]['last_observed_timestamp_ns'] for n in NAMES])
            # Transport delay counts against leg prediction age, not only frame age.
            # Solve scale(age) <= configured permission threshold analytically.
            a = .5*config['uncertainty_acceleration_m_s2']
            b = config['uncertainty_velocity_m_s']
            expiry = []
            for name in NAMES[6:]:
                s = states[name]; age = s['prediction_age_ns']/1e9
                base_scale = s['uncertainty_scale_m']-b*age-a*age*age
                c = base_scale-config['permission_uncertainty_m']
                horizon = max(0., min(config['max_prediction_age_s'],
                                      (-b+np.sqrt(max(0., b*b-4*a*c)))/(2*a)))
                expiry.append(s['last_observed_timestamp_ns']+round(horizon*1e9))
            result['prediction_valid_until_ns'] = min(expiry)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            result['reason'] = str(exc)
        results.append(result)
    return results
