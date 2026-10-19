"""Domain-bound descriptive calibration against an external reference.

Evaluation-only: blocks of dimensionless error/std ratios -> scale multiplier.
No ground truth enters inference. Adjacent blocks/people are not independent;
finite-rank quantiles here DO NOT assert distribution-free coverage or hardware
safety. Domain and estimator fingerprint must match when requesting a radius.
This does not populate the controller's uncertainty_calibration permission.
"""
import math
import numpy as np


def fit_bounds(block_scores,domain,fingerprint,coverage=.9,std_floor=0.):
    values=np.asarray(block_scores,float)
    if values.ndim!=1 or len(values)<20 or not np.isfinite(values).all() or np.any(values<0) or not 0<coverage<1 or not domain or not fingerprint or not np.isfinite(std_floor) or std_floor<0:raise ValueError('Invalid or insufficient calibration blocks')
    rank=math.ceil((len(values)+1)*coverage)
    if rank>len(values):raise ValueError('Insufficient blocks for quantile')
    return dict(domain=domain,estimator_sha256=fingerprint,nominal_coverage=coverage,blocks=len(values),
        multiplier=float(np.sort(values)[rank-1]),rank=rank,std_floor=std_floor,guarantee_claimed=False,
        reference='Dataset markerless 3D reference; not independent physical metrology',
        assumption_warning='Block dependence and distribution shift invalidate unconditional coverage guarantees')


def radius(calibration,std,domain,fingerprint):
    if calibration['domain']!=domain or calibration['estimator_sha256']!=fingerprint or not np.isfinite(std) or std<0:return None
    return float(max(std,calibration.get('std_floor',0.))*calibration['multiplier'])


def block_maxima(errors,std,timestamps_ns,current,floor,block_ns=1_000_000_000):
    errors=np.asarray(errors,float);std=np.asarray(std,float);t=np.asarray(timestamps_ns);current=np.asarray(current,bool)
    if errors.ndim!=1 or errors.shape!=std.shape or errors.shape!=t.shape or errors.shape!=current.shape or t.dtype.kind not in 'iu' or not np.isfinite(floor) or floor<=0 or type(block_ns) is not int or block_ns<=0:raise ValueError('Invalid aligned score arrays')
    if not len(t):return []
    ids=(t-t.min())//block_ns;result=[]
    for i in np.unique(ids):
        mask=(ids==i)&current&np.isfinite(errors)&np.isfinite(std)&(std>=0)
        if mask.any():result.append(float(np.max(errors[mask]/np.maximum(floor,std[mask]))))
    return result
