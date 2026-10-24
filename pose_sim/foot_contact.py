"""Causal multiview foot-contact baseline; never measures load or invents pose.

Input: human_state_v2 current heel/toe XYZ (world metres), image quality,
exposure ns, attachment checks and an independently calibrated static plane.
Output: per-region stationary-contact / clearance evidence and foot state at
the unchanged exposure time. Five CURRENT samples + repeated confirmation
introduce onset delay; zero future samples. Missing/expired/ambiguous data
withdraw labels immediately. Window fits are diagnostics, not calibrated CIs.
"""
from collections import deque
import json
from pathlib import Path
import numpy as np
from .ground import validate_ground


def load_config():
    return json.loads((Path(__file__).resolve().parents[1]/'config/foot_contact_baseline.json').read_text())


class MultiViewFootContact:
    def __init__(self, config=None):
        self.cfg = {**load_config(), **(config or {})}
        for key, value in self.cfg.items():
            if key not in ('schema_version', 'status') and (not np.isfinite(value) or value <= 0):
                raise ValueError('Invalid contact setting: '+key)
        for key, minimum in (('window_samples', 3), ('confirmation_frames', 2), ('minimum_views', 2)):
            if type(self.cfg[key]) is not int or self.cfg[key] < minimum:
                raise ValueError('Invalid contact count: '+key)
        if self.cfg['minimum_window_s'] > self.cfg['maximum_window_s']:
            raise ValueError('Invalid temporal window')
        self.history, self.memory = {}, {}
        self.last_time = self.signature = None

    def update(self, row, ground):
        t = row['timestamp_ns']
        if type(t) is not int or (self.last_time is not None and t <= self.last_time):
            self.history.clear(); self.memory.clear()
            raise ValueError('Increasing exposure nanoseconds required')
        signature = (row.get('source_id'), row.get('landmark_schema_id'),
                     row.get('calibration_sha256'), row.get('perception_signature'),
                     json.dumps(ground, sort_keys=True))
        if signature != self.signature or (self.last_time is not None and
                (t-self.last_time)/1e9 > self.cfg['maximum_gap_s']):
            self.history.clear(); self.memory.clear()
        self.signature, self.last_time = signature, t
        try:
            n, d, floor_bound, lo, hi = validate_ground(ground, row.get('calibration_sha256'))
            ground_error = None
        except (ValueError, KeyError, TypeError) as exc:
            ground_error = str(exc)
        names = row['joint_names']
        if len(set(names)) != len(names):
            raise ValueError('Duplicate joint names')
        points = dict(zip(names, row['joints_world_m']))
        states = dict(zip(names, row['joint_states']))
        quality = dict(zip(names, row['quality']))
        feet = {}
        for side in ('left', 'right'):
            foot_names = [side+'_'+part for part in ('heel', 'big_toe', 'small_toe')]
            regions = {}
            attachment = row.get('attachment_state', {}).get(side+'_foot', {}).get('status') == 'supported'
            for name in foot_names:
                reason = ground_error or (None if attachment else 'Foot attachment unsupported')
                p, s, q = points.get(name), states.get(name, {}), quality.get(name)
                try:
                    xyz = np.asarray(p, float)
                    sigma = float(s['max_std_m'])
                    valid = (xyz.shape == (3,) and np.isfinite(xyz).all() and np.isfinite(sigma) and
                        0 <= sigma <= self.cfg['maximum_position_std_m'] and s['status'] == 'filtered' and
                        s['last_observed_timestamp_ns'] == t and q is not None and
                        len(set(q['inlier_cameras'])) >= self.cfg['minimum_views'] and
                        np.isfinite([q['mean_reprojection_px'], q['ray_angle_deg']]).all() and
                        0 <= q['mean_reprojection_px'] <= self.cfg['maximum_reprojection_px'] and
                        q['ray_angle_deg'] >= self.cfg['minimum_ray_angle_deg'])
                except (ValueError, TypeError, KeyError):
                    valid = False
                if not valid:
                    reason = reason or 'Missing, stale or weak multiview landmark'
                if reason is None and (np.any(xyz[:2] < lo-self.cfg['ground_domain_margin_m']) or
                                       np.any(xyz[:2] > hi+self.cfg['ground_domain_margin_m'])):
                    reason = 'Outside calibrated ground domain'
                if reason is not None:
                    self.history.pop(name, None)
                    regions[name] = dict(state='unknown', reason=reason)
                    continue
                height = float(xyz @ n+d-self.cfg['landmark_above_sole_m'])
                history = self.history.setdefault(name, deque(maxlen=self.cfg['window_samples']))
                history.append((t, xyz.copy()))
                while history and (t-history[0][0])/1e9 > self.cfg['maximum_window_s']:
                    history.popleft()
                r = dict(state='unknown', height_above_sole_plane_m=height, position_std_m=sigma,
                         reason='Temporal window warming up', current_multiview=True)
                clear = height-self.cfg['uncertainty_multiplier']*sigma-floor_bound > self.cfg['minimum_swing_clearance_m']
                if clear:
                    r.update(state='clearance', reason='Current geometric clearance')
                elif len(history) == self.cfg['window_samples'] and (t-history[0][0])/1e9 >= self.cfg['minimum_window_s']:
                    ts = np.array([(stamp-t)/1e9 for stamp, _ in history]); ts -= ts.mean()
                    xyzs = np.array([x for _, x in history]); mean = xyzs.mean(axis=0)
                    slope = ts @ xyzs/(ts @ ts)
                    residual = float(np.max(np.linalg.norm(xyzs-mean-ts[:, None]*slope, axis=1)))
                    extent = float(np.max(np.linalg.norm(xyzs-xyzs[-1], axis=1)))
                    near = abs(height) <= self.cfg['near_plane_tolerance_m']+floor_bound
                    stationary = (np.linalg.norm(slope) <= self.cfg['maximum_stationary_speed_m_s'] and
                                  residual <= self.cfg['maximum_fit_residual_m'] and
                                  extent <= self.cfg['maximum_window_extent_m'])
                    r.update(window_speed_m_s=float(np.linalg.norm(slope)), window_extent_m=extent,
                             fit_residual_m=residual, window_s=(t-history[0][0])/1e9,
                             state='stationary_contact' if near and stationary else 'unknown',
                             reason='Near-plane stationary window' if near and stationary else 'Ambiguous height or movement')
                regions[name] = r
            witnesses = [name for name, r in regions.items() if r['state'] == 'stationary_contact']
            candidate = 'contact' if witnesses else 'swing' if all(r['state'] == 'clearance' for r in regions.values()) else 'unknown'
            previous = self.memory.get(side, {})
            count = previous.get('count', 0)+1 if previous.get('candidate') == candidate else 1
            # Contact evidence must persist at the SAME point across confirmation.
            if candidate == 'contact' and not set(witnesses).intersection(previous.get('witnesses', [])):
                count = 1
            if candidate == 'unknown':
                self.memory.pop(side, None); count = 0
            else:
                self.memory[side] = dict(candidate=candidate, count=count, witnesses=witnesses)
            state = candidate if count >= self.cfg['confirmation_frames'] else 'unknown'
            feet[side] = dict(state=state, candidate=candidate, confirmation_frames=count,
                witness_landmarks=witnesses if state == 'contact' else [], regions=regions,
                reason='Confirmed current evidence' if state != 'unknown' else 'Insufficient current evidence',
                calibrated_probability=False)
        labels = [feet[side]['state'] for side in ('left', 'right')]
        phase = ('unknown' if 'unknown' in labels else 'double' if labels == ['contact', 'contact'] else
                 'left' if labels[0] == 'contact' else 'right' if labels[1] == 'contact' else 'feet_clear')
        return dict(schema='multiview_foot_contact_v1', timestamp_ns=t, feet=feet, phase=phase,
            source_id=row.get('source_id'), calibration_sha256=row.get('calibration_sha256'),
            phase_scope='Feet only; other body parts may touch the ground',
            ground=ground, physical_support_verified=False, uncertainty_calibrated=False,
            uses_future=False, basis='Current multiview landmarks and causal positional windows; no force observation')
