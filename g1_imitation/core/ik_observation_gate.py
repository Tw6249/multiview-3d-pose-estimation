"""Opt-in causal arm observation gate; production IK is untouched.

Input: golden-27 points, metres, right-handed Z-up world; capture time seconds.
Output: normal KeypointIKFrame plus per-arm gate reasons and diagnostic bend sine.
Length medians use only the original IK calibration phase, then remain fixed.
Direction-hold reproduces the exploratory missing-point fallback. Arm-freeze
instead zero-constrains all seven G1 arm joint increments inside every IK QP and
removes that arm's observation costs while it has missing/degenerate input.
Frozen arms remain held states, never fresh observations; valid legs/root continue.
No model/physics parameters, measured points or output controller actions change.
Bone-valid directions can still be wrong. No palm/twist measurement is inferred.
Validation: calibration/prefix invariance, outlier/missing/recovery, exact joint
freeze, unchanged valid-input output, then identical-physics saved-clip comparison.
"""
from __future__ import annotations

import mink
import numpy as np

from .constants import KEYPOINT_INDEX_BY_NAME as K
from .s45c_keypoint_ik import KeypointIKRetargeter, SIDES


class ArmObservationGate:
    def __init__(self, cfg):
        self.cfg = cfg
        if not 0 < cfg["length_ratio_min"] < 1 < cfg["length_ratio_max"]:
            raise ValueError("length ratio interval must contain one")
        if cfg["min_calibration_samples"] < 3:
            raise ValueError("at least three calibration samples required")
        self.reset()

    def reset(self):
        self.samples = {(side, bone): [] for side in SIDES for bone in ("upper", "fore")}
        self.lengths = {}
        self.last_time = None

    def inspect(self, points, timestamp_s, calibrating):
        points = np.asarray(points, dtype=float)
        if points.shape != (27, 3) or not np.isfinite(timestamp_s):
            raise ValueError("expected (27,3) points and finite capture time")
        if self.last_time is not None and timestamp_s <= self.last_time:
            raise ValueError("capture time must increase")
        self.last_time = timestamp_s
        result = {}
        for side in SIDES:
            reasons, lengths, vectors = [], {}, []
            for bone, start, end in (("upper", "shoulder", "elbow"), ("fore", "elbow", "wrist")):
                vector = points[K[f"{side}_{end}"]] - points[K[f"{side}_{start}"]]
                length = float(np.linalg.norm(vector))
                lengths[bone] = length
                vectors.append(vector)
                key = side, bone
                if not np.isfinite(length) or length <= 1e-6:
                    reasons.append(f"{bone}_missing_or_degenerate")
                    continue
                if calibrating:
                    self.samples[key].append(length)
                    continue
                if key not in self.lengths and len(self.samples[key]) >= self.cfg["min_calibration_samples"]:
                    self.lengths[key] = float(np.median(self.samples[key]))
                median = self.lengths.get(key)
                if median is None:
                    reasons.append(f"{bone}_uncalibrated")
                elif not self.cfg["length_ratio_min"] * median <= length <= self.cfg["length_ratio_max"] * median:
                    reasons.append(f"{bone}_length_outlier")
            bend_sine = None
            if not reasons and all(np.isfinite(v).all() for v in vectors):
                bend_sine = float(np.linalg.norm(np.cross(*vectors)) / (lengths["upper"] * lengths["fore"]))
            result[side] = dict(reasons=reasons, lengths_m=lengths, bend_sine=bend_sine,
                                swivel_observable=bool(not calibrating and bend_sine is not None
                                                       and bend_sine >= self.cfg["swivel_min_sin_bend"]))
        return result


class GatedKeypointIKRetargeter(KeypointIKRetargeter):
    def __init__(self, gate_cfg, mode="arm_freeze", **kwargs):
        if mode not in ("direction_hold", "arm_freeze"):
            raise ValueError("unknown gate mode")
        self.gate = ArmObservationGate(gate_cfg)
        self.gate_mode = mode
        super().__init__(**kwargs)
        if self.robot != "unitree_g1" or int(self.cfg["wrist_iterations"]) != 0:
            raise ValueError("experiment requires G1 and untracked wrists")
        if int(self.cfg.get("ik_branch_candidates", 1)) != 1:
            raise ValueError("experiment requires one warm-start candidate")
        self.arm_limits = {}
        for side in SIDES:
            names = [f"{side}_{joint}_joint" for joint in
                     ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                      "wrist_roll", "wrist_pitch", "wrist_yaw")]
            self.arm_limits[side] = mink.VelocityLimit(self.model, {name: 0.0 for name in names})

    def reset(self):
        super().reset()
        self.gate.reset()
        self.gate_diagnostics = {}

    def step(self, points, t_s):
        calibrating = self.scale is None
        diagnostics = self.gate.inspect(points, t_s, calibrating)
        self.gate_diagnostics = diagnostics
        rejected = [side for side in SIDES if diagnostics[side]["reasons"] and not calibrating]
        filtered = np.asarray(points, dtype=float).copy()
        previous_limits = self.limits
        original_costs = []
        if self.gate_mode == "direction_hold":
            for side in rejected:
                for bone, endpoint in (("upper", "elbow"), ("fore", "wrist")):
                    if any(reason.startswith(bone) for reason in diagnostics[side]["reasons"]):
                        filtered[K[f"{side}_{endpoint}"]] = np.nan
        else:
            self.limits = previous_limits + [self.arm_limits[side] for side in rejected]
            for side in rejected:
                for task in (self.t_elbow[side], self.t_hand[side]):
                    original_costs.append((task, task.cost.copy()))
                    task.cost[:] = 0.0
        try:
            frame = super().step(filtered, t_s)
        finally:
            self.limits = previous_limits
            for task, cost in original_costs:
                task.cost[:] = cost
        frame.held_groups += tuple(f"{side}_arm_quality_{self.gate_mode}" for side in rejected)
        return frame
