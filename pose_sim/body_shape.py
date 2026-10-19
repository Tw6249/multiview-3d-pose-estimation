"""Slow, revisable subject dimensions from current multi-view evidence only.

Lengths are metres, uncertainty is a conservative descriptive scale (not a
calibrated confidence interval). Correlated detector bias remains unidentifiable
without diverse views/poses or an external anthropometric measurement.
"""
from collections import deque
import numpy as np
from .landmarks import BONES


class BodyShape:
    def __init__(self, window=90, minimum_samples=24):
        self.window=window; self.minimum_samples=minimum_samples; self.samples={};self.result={}

    def update(self, row, accepted):
        points=dict(zip(row['joint_names'],row['joints_world_m']))
        quality=dict(zip(row['joint_names'],row['quality']))
        t=row['timestamp_ns']
        for a,b in BONES:
            key=a+'--'+b; history=self.samples.setdefault(key,deque(maxlen=self.window))
            if a in accepted and b in accepted:
                qa,qb=quality[a],quality[b]
                if (qa and qb and min(len(qa['inlier_cameras']),len(qb['inlier_cameras']))>=3
                        and min(qa['ray_angle_deg'],qb['ray_angle_deg'])>=10
                        and max(qa['linearized_max_std_m'],qb['linearized_max_std_m'])<.04):
                    delta=np.array(points[b])-points[a];length=float(np.linalg.norm(delta))
                    if .08<length<.85:
                        direction=delta/length
                        weight=1/max(.005,qa['linearized_max_std_m'],qb['linearized_max_std_m'])**2
                        history.append((t,length,weight,direction))
            if not history:continue
            # Bound age as well as sample count; stale shape evidence cannot silently persist.
            while history and t-history[0][0]>10_000_000_000:history.popleft()
            if not history:
                self.result.pop(key,None);continue
            values=np.array([s[1] for s in history]);weights=np.array([s[2] for s in history])
            order=np.argsort(values); cumulative=np.cumsum(weights[order]);length=float(values[order[np.searchsorted(cumulative,cumulative[-1]/2)]])
            mad=float(np.median(abs(values-length)))*1.4826
            spread=max(np.linalg.norm(s[3]-history[0][3]) for s in history)
            span=(history[-1][0]-history[0][0])/1e9
            # Broad agreement, view redundancy, temporal duration and pose diversity
            # are required before dimensions participate in reconstruction constraints.
            stable=(len(history)>=self.minimum_samples and span>=.7 and mad<.025
                    and (spread>.12 or key=='left_hip--right_hip'))
            self.result[key]=dict(length_m=length,uncertainty_m=max(.008,2*mad),
                status='stable' if stable else 'collecting',samples=len(history),span_seconds=span,
                direction_spread=spread,last_evidence_timestamp_ns=history[-1][0],
                systematic_bias_identifiable=False)
        return self.result
