"""Scale-independent arm-direction IK with bounds and temporal regularization."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class JointTarget:
    timestamp_ns: int
    positions_rad: np.ndarray
    max_direction_error_deg: float
    accepted: bool


class ArmRetargeter:
    def __init__(self, model):
        self.model = model
        self.previous = model.home.copy()

    def solve(self, observation):
        model, cfg = self.model, self.model.config['retarget']
        def residual(q):
            fk = model.forward(q)
            match = np.concatenate([(fk[s]['directions']-observation.arm_directions_body[s]).ravel()
                                    for s in ('left', 'right')])
            return np.r_[match, np.sqrt(cfg['continuity_weight'])*(q-self.previous)]
        result = least_squares(residual, self.previous, bounds=(model.lower, model.upper),
                               max_nfev=cfg['max_nfev'], ftol=1e-6, xtol=1e-6, gtol=1e-6)
        fk = model.forward(result.x)
        error = max(float(np.degrees(np.arccos(np.clip(np.sum(
            fk[s]['directions']*observation.arm_directions_body[s], axis=1), -1, 1))).max())
            for s in ('left', 'right'))
        accepted = bool(result.success and np.isfinite(result.x).all() and error <= cfg['max_direction_error_deg'])
        if accepted:
            self.previous = result.x.copy()
        return JointTarget(observation.timestamp_ns, result.x, error, accepted)
