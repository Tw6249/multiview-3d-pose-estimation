"""Opt-in causal robust window estimation and constrained human prediction.

Input: named triangulated points and their ORIGINAL quality, calibrated world
metres, source integer ns. Only the twelve body landmarks are estimated here.
Output: positions/velocities, original observation age, uncertainty *scale* and
per-landmark simulation permission; no fabricated image quality or orientation.

The window jointly estimates p,v with CV process factors and vector Huber IRLS
on actual 3-D observations. Its arrival prior precedes the oldest observation.
There are no future frames, synthetic measurement factors or retroactive outputs.
Endpoint bone projection and explicit prediction bounds are separate from the
window fit: this is not a nonlinear constrained-MHE stability guarantee.
"""
from collections import deque
import copy
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import solve
from scipy.optimize import least_squares

from .landmarks import LIMB_BONES

BODY = tuple(f'{side}_{joint}' for joint in
             ('shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle')
             for side in ('left', 'right'))
STRUCTURE = LIMB_BONES + [('left_hip', 'right_hip'),
                         ('left_shoulder', 'right_shoulder'),
                         ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip')]


def load_config():
    return json.loads((Path(__file__).resolve().parents[1] /
                       'config/continuous_prediction.json').read_text())


class ContinuousHumanState:
    def __init__(self, config=None, window=True):
        self.cfg = {**load_config(), **(config or {})}
        if self.cfg['hardware_enabled'] is not False:
            raise ValueError('Simulation candidate only')
        for key, val in self.cfg.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                if not np.isfinite(val) or val <= 0:
                    raise ValueError('Invalid continuous estimator parameter: '+key)
        for key in ('window_frames', 'irls_iterations', 'recovery_confirmation_frames',
                    'bone_calibration_samples'):
            if type(self.cfg[key]) is not int:
                raise ValueError('Integer estimator parameter required: '+key)
        self.window = window
        self.last_time = self.signature = None
        self.tracks = {}
        self.samples, self.lengths = {}, {}

    def _measurement(self, point, quality):
        if point is None or quality is None:
            return None, None, 'missing'
        p = np.asarray(point, float)
        sigma = quality.get('linearized_max_std_m', np.inf)
        reproj = quality.get('mean_reprojection_px', np.inf)
        if (p.shape != (3,) or not np.isfinite(p).all() or
                not np.isfinite([sigma, reproj]).all() or sigma < 0 or sigma > .25 or
                quality.get('ray_angle_deg', 0) < 2 or
                len(set(quality.get('inlier_cameras', []))) < 2):
            return None, None, 'invalid_geometry'
        # High reprojection error is downweighted continuously, not an 8 px cliff.
        noise = max(sigma, self.cfg['measurement_floor_m']) * max(1., reproj/4.)
        noise *= 1.4 if len(set(quality['inlier_cameras'])) == 2 else 1.
        return p, min(noise, 1.), 'available'

    def _fit(self, history):
        n = len(history)
        rows, values = [], []
        def factor(entries, target, sigma):
            a = np.zeros(2*n)
            for i, value in entries:
                a[i] = value/sigma
            rows.append(a); values.append(np.asarray(target)/sigma)
        # Prior contains only observations preceding the first window node.
        first = history[0]
        factor([(0, 1)], first['prior_p'], first['prior_sigma'])
        factor([(1, 1)], first['prior_v'], .8)
        for k in range(1, n):
            dt = (history[k]['t']-history[k-1]['t'])/1e9
            factor([(2*k, 1), (2*k-2, -1), (2*k-1, -dt)],
                   np.zeros(3), .001 + .5*self.cfg['process_acceleration_m_s2']*dt*dt)
            decay = np.exp(-dt/self.cfg['velocity_decay_s']) if history[k]['z'] is None else 1.
            factor([(2*k+1, 1), (2*k-1, -decay)], np.zeros(3),
                   self.cfg['process_acceleration_m_s2']*dt)
        a, b = np.array(rows), np.array(values)
        h, rhs = a.T@a, a.T@b
        measured = [(k, r) for k, r in enumerate(history) if r['z'] is not None]
        weights = np.ones(len(measured))
        x = None
        for _ in range(self.cfg['irls_iterations']):
            normal, target = h.copy(), rhs.copy()
            for w, (k, rec) in zip(weights, measured):
                precision = w/rec['sigma']**2
                normal[2*k, 2*k] += precision
                target[2*k] += precision*rec['z']
            x = solve(normal, target, assume_a='pos', check_finite=False)
            residual = np.array([np.linalg.norm(x[2*k]-rec['z'])/rec['sigma']
                                 for k, rec in measured])
            weights = np.minimum(1., self.cfg['huber_delta']/np.maximum(residual, 1e-12))
        return x[-2].copy(), x[-1].copy(), (
            float(weights[-1]) if measured and measured[-1][0] == n-1 else None)

    def _project(self, positions, scales):
        names = list(positions)
        index = {name: k for k, name in enumerate(names)}
        bones = [(index[a], index[b], length) for (a, b), length in self.lengths.items()
                 if a in index and b in index]
        if not bones:
            return positions, True, 0.
        target = np.array([positions[n] for n in names])
        sigma = np.array([max(.025, min(scales[n], .2)) for n in names])[:, None]
        def residual(x):
            p = x.reshape(-1, 3)
            return np.r_[((p-target)/sigma).ravel(),
                         [(np.linalg.norm(p[a]-p[b])-length)/self.cfg['bone_sigma_m']
                          for a, b, length in bones]]
        def jacobian(x):
            p = x.reshape(-1, 3)
            jac = np.zeros((target.size+len(bones), target.size))
            jac[:target.size] = np.diag(np.repeat(1/sigma[:, 0], 3))
            for k, (a, b, _) in enumerate(bones):
                delta = p[a]-p[b]
                direction = delta/max(np.linalg.norm(delta), 1e-10)/self.cfg['bone_sigma_m']
                jac[target.size+k, 3*a:3*a+3] = direction
                jac[target.size+k, 3*b:3*b+3] = -direction
            return jac
        fit = least_squares(residual, target.ravel(), jac=jacobian, method='lm',
                            max_nfev=20, ftol=1e-5, xtol=1e-5, gtol=1e-5)
        p = fit.x.reshape(-1, 3)
        error = max(abs(np.linalg.norm(p[a]-p[b])-length) for a, b, length in bones)
        valid = bool(fit.success and np.isfinite(p).all() and error <= self.cfg['bone_tolerance_m'])
        return dict(zip(names, p)), valid, float(error)

    def update(self, row):
        begin = time.perf_counter()
        t, names = row['timestamp_ns'], row['joint_names']
        if type(t) is not int or (self.last_time is not None and t <= self.last_time):
            raise ValueError('Source timestamps must be strictly increasing integer ns')
        if (row.get('landmark_schema_id') not in ('coco17', 'human33') or
                len(set(names)) != len(names) or not set(BODY).issubset(names) or
                len(row['quality']) != len(names) or len(row['joints_world_m']) != len(names)):
            raise ValueError('Invalid named human schema')
        signature = tuple(row.get(k) for k in ('source_id', 'landmark_schema_id',
                          'calibration_sha256', 'perception_signature')) + (tuple(names),)
        dt = 0. if self.last_time is None else (t-self.last_time)/1e9
        reset = self.signature is not None and (signature != self.signature or dt > self.cfg['reset_gap_s'])
        if reset:
            self.tracks, self.samples, self.lengths = {}, {}, {}
            dt = 0.
        self.last_time, self.signature = t, signature
        measurements = {}
        for name in BODY:
            i = names.index(name)
            measurements[name] = self._measurement(row['joints_world_m'][i], row['quality'][i])
        # Lengths freeze after calibration from actual, high-quality measurements.
        for a, b in STRUCTURE:
            pa, sa, _ = measurements[a]; pb, sb, _ = measurements[b]
            if (a, b) in self.lengths or pa is None or pb is None or max(sa, sb) > .06:
                continue
            length = float(np.linalg.norm(pa-pb))
            if .08 < length < .85:
                samples = self.samples.setdefault((a, b), [])
                samples.append(length)
                if len(samples) >= self.cfg['bone_calibration_samples']:
                    self.lengths[(a, b)] = float(np.median(samples))
        positions, scales, states = {}, {}, {}
        for name, (z, sigma, measurement_status) in measurements.items():
            track = self.tracks.get(name)
            new_track = track is None
            if track is None and z is None:
                states[name] = dict(estimate_status='unknown', measurement_status=measurement_status,
                                    control_permission=False, last_observed_timestamp_ns=None,
                                    prediction_age_ns=None, uncertainty_scale_m=None)
                continue
            if track is None:
                track = dict(p=z.copy(), v=np.zeros(3), last=t, anchor=z.copy(), age_before=0.,
                             streak=0, uncertainty=sigma, history=deque(maxlen=self.cfg['window_frames'] if self.window else 1))
                track['last_measurement_sigma'] = sigma
                track['pending'] = deque(maxlen=self.cfg['recovery_confirmation_frames'])
                self.tracks[name] = track
            old_p, old_v = track['p'].copy(), track['v'].copy()
            decay = np.exp(-dt/self.cfg['velocity_decay_s'])
            prior_p = old_p + old_v*self.cfg['velocity_decay_s']*(1-decay)
            prior_v = old_v*decay
            rec = dict(t=t, z=z, sigma=sigma, prior_p=prior_p, prior_v=prior_v,
                       prior_sigma=1e3 if new_track else max(.04, track['uncertainty']))
            track['history'].append(rec)
            solver_ok = True
            try:
                p, velocity, weight = self._fit(track['history'])
                if not self.window and dt > 0 and z is not None:
                    # One-frame ablation has no adjacent-node CV factor. Its
                    # alpha-beta correction must still estimate motion velocity.
                    velocity = prior_v+.25*(p-prior_p)/dt
            except (ValueError, np.linalg.LinAlgError):
                p, velocity, weight = prior_p, prior_v, None
                solver_ok = False
            # Large innovations are evidence to confirm over multiple real frames.
            innovation = np.inf if z is None else np.linalg.norm(z-prior_p)
            trusted = z is not None and solver_ok and (weight or 0.) >= .35
            if z is not None and not trusted:
                track['pending'].append((t, z.copy(), sigma))
                pending = track['pending']
                if len(pending) == pending.maxlen:
                    times = np.array([r[0] for r in pending], float)/1e9
                    points = np.array([r[1] for r in pending])
                    speeds = np.linalg.norm(np.diff(points, axis=0)/np.diff(times)[:, None], axis=1)
                    if np.max(speeds) <= self.cfg['max_speed_m_s'] and max(r[2] for r in pending) < .1:
                        # Confirmed real samples can overrule the old motion prior.
                        # This revises the current posterior, never previously issued states.
                        p = z.copy()
                        velocity = (points[-1]-points[0])/(times[-1]-times[0])
                        track['history'].clear()
                        track['pending'].clear()
                        trusted = True
            else:
                track['pending'].clear()
            if trusted:
                track['last'] = t
                track['anchor'] = z.copy()
                track['streak'] += 1
                track['last_measurement_sigma'] = sigma
            else:
                track['streak'] = 0
            age = (t-track['last'])/1e9
            returning = (track['age_before'] > 0 or innovation > .2 or track.get('recovering', False))
            recovering = returning and track['streak'] < self.cfg['recovery_confirmation_frames']
            track['recovering'] = recovering
            if not trusted:
                # Actual constrained propagation, not unlimited polynomial extrapolation.
                p, velocity = prior_p, prior_v
                offset = p-track['anchor']; distance = np.linalg.norm(offset)
                if distance > self.cfg['max_prediction_displacement_m']:
                    p = track['anchor']+offset*self.cfg['max_prediction_displacement_m']/distance
                    velocity[:] = 0.
            speed = np.linalg.norm(velocity)
            velocity *= min(1., self.cfg['max_speed_m_s']/max(speed, 1e-12))
            scale = (track['last_measurement_sigma'] +
                     self.cfg['uncertainty_velocity_m_s']*age +
                     .5*self.cfg['uncertainty_acceleration_m_s2']*age*age)
            permission = (solver_ok and age <= self.cfg['max_prediction_age_s'] and
                          scale <= self.cfg['permission_uncertainty_m'] and
                          (not recovering or z is None or innovation < .2))
            positions[name], scales[name] = p, scale
            track.update(p=p, v=velocity, uncertainty=scale, age_before=age, previous_p=old_p)
            states[name] = dict(estimate_status='predicted' if not trusted else 'recovering' if recovering else 'updated',
                                measurement_status=measurement_status if trusted else
                                'robust_downweighted' if z is not None else measurement_status,
                                control_permission=bool(permission),
                                last_observed_timestamp_ns=track['last'],
                                prediction_age_ns=t-track['last'], uncertainty_scale_m=float(scale),
                                robust_weight=weight, solver_ok=solver_ok)
        projected, structure_ok, bone_error = self._project(positions, scales)
        for name, p in projected.items():
            track = self.tracks[name]; state = states[name]
            # Projection must not override prediction displacement or create NaNs.
            bounded = (np.isfinite(p).all() and (state['estimate_status'] != 'predicted' or
                       (np.linalg.norm(p-track['anchor']) <= self.cfg['max_prediction_displacement_m']+1e-9 and
                        np.linalg.norm(p-track['previous_p']) <= self.cfg['max_speed_m_s']*dt+1e-9)))
            if structure_ok and bounded:
                track['p'] = p
            else:
                state['control_permission'] = False
            state['structure_ok'] = bool(structure_ok and bounded)
        result = {k: copy.deepcopy(row.get(k)) for k in
                  ('frame_id', 'timestamp_ns', 'source_id', 'landmark_schema_id',
                   'calibration_sha256', 'perception_signature')}
        result.update(state_schema_id='continuous_human_state_v1', joint_names=list(BODY),
                      state_timestamp_ns=t, hardware_enabled=False, reset=reset,
                      joints_world_m=[self.tracks[n]['p'].tolist() if n in self.tracks else None for n in BODY],
                      velocities_world_m_s=[self.tracks[n]['v'].tolist() if n in self.tracks else None for n in BODY],
                      quality=[copy.deepcopy(row['quality'][names.index(n)]) if measurements[n][0] is not None else None for n in BODY],
                      joint_states=[states[n] for n in BODY],
                      constraints=dict(bone_max_error_m=bone_error, structure_ok=structure_ok,
                                       calibrated_bones=len(self.lengths)),
                      uncertainty_kind='uncalibrated_error_scale_not_covariance',
                      orientations=dict(feet=None, hands=None), contacts=None,
                      window_frames=len(next(iter(self.tracks.values()))['history']) if self.tracks else 0,
                      uses_future_frames=False, latency_ms=(time.perf_counter()-begin)*1000)
        return result
