"""Bounded alternative track from consecutive current multi-view measurements.

World metres, source exposure ns; no prior filter prediction or reference labels.
Unsupported, missing, discontinuous or stale samples reset the candidate. A
confirmed candidate resets estimation, never authorizes immediate robot motion.
Observe all supported samples, including those accepted by the primary filter;
its rejection decisions must not determine when alternative evidence starts.
"""
import numpy as np


def current_structure_support(name, points, quality):
    """Two-view fallback needs a whole current limb sharing camera evidence.

    Broad human segment/torso ratios are plausibility checks, not measured bone
    lengths. Applies to human body endpoints only, never palm/foot orientations.
    """
    torso=[]
    for side in ('left','right'):
        a,b=side+'_shoulder',side+'_hip'
        if a in points and b in points:
            torso.append(np.linalg.norm(points[a]-points[b]))
    if not torso:
        return False
    scale=float(np.median(torso))
    if not .2 < scale < 1.:
        return False
    for side in ('left','right'):
        for chain in (('shoulder','elbow','wrist'),('hip','knee','ankle')):
            names=[side+'_'+n for n in chain]
            if name not in names or not all(n in points for n in names):
                continue
            cameras=set.intersection(*(set(quality[n]['inlier_cameras']) for n in names))
            lengths=[np.linalg.norm(points[b]-points[a]) for a,b in zip(names,names[1:])]
            if (len(cameras)>=2 and all(.2*scale<length<1.2*scale for length in lengths)
                    and max(quality[n]['mean_reprojection_px'] for n in names)<5.):
                return True
    return False


class Reacquisition:
    def __init__(self):
        self.pending = {}

    def observe(self, name, point, timestamp_ns, sigma, supported):
        if not supported:
            self.pending.pop(name, None)
            return None
        history = self.pending.setdefault(name, [])
        if history:
            dt = (timestamp_ns-history[-1][0])/1e9
            if not 0 < dt <= .06 or np.linalg.norm(point-history[-1][1]) > 6.*dt+3*(sigma+history[-1][2]):
                history.clear()
        history.append((timestamp_ns, point.copy(), sigma))
        del history[:-3]
        if len(history) < 3:
            return None
        ts = np.array([(r[0]-timestamp_ns)/1e9 for r in history[-3:]])
        points = np.array([r[1] for r in history[-3:]])
        ts -= ts.mean()
        velocity = ts@points/(ts@ts)
        residual = np.max(np.linalg.norm(points-points.mean(axis=0)-ts[:,None]*velocity, axis=1))
        if residual > max(.04, 3*sigma):
            history.pop(0)
            return None
        return velocity

    def retain(self, names):
        self.pending = {n: v for n, v in self.pending.items() if n in names}
