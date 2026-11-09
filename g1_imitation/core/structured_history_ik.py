"""Independent G1 structured geometry / causal history IK experiments.

Input: original golden-27 points (m, right-handed world Z-up), capture seconds.
Output: original G1 q/dq and geometry/constraint diagnostics. Source BVH is absent.
Structured targets use normalized FK limb vectors and bend cosines, not virtual
joint centers; root calibration/trajectory and foot tasks retain baseline behavior.
Only current/past samples are used. A joint box around the last emitted pose
enforces one shared velocity/acceleration budget across every IK iteration.
History state resets during startup, trunk loss and long gaps; these are reported
exceptions to acceleration continuity. Floating-base motion is not rate-limited.
Missing limb evidence removes its geometry task; no inferred twist/palm rotation.
No robot XML, plant limit, physics or controller action is modified.
Validation: analytic Jacobian finite differences, causal prefix/reset/missing data,
whole-frame constraints, then identical-physics saved-vision ablations.
"""
from __future__ import annotations

import mink
import mujoco
import numpy as np
import time

from .constants import KEYPOINT_INDEX_BY_NAME as K
from .retarget_experiments import RelativePositionTask
from .s45c_keypoint_ik import KeypointIKRetargeter, SIDES, _ok


def normalized_segment(configuration, start_id, end_id, min_length):
    delta = configuration.data.xpos[end_id] - configuration.data.xpos[start_id]
    length = float(np.linalg.norm(delta))
    if not np.isfinite(length) or length < min_length:
        raise ValueError("robot segment is degenerate")
    direction = delta / length
    start_jac = np.zeros((3, configuration.model.nv))
    end_jac = np.zeros_like(start_jac)
    mujoco.mj_jacBody(configuration.model, configuration.data, start_jac, None, start_id)
    mujoco.mj_jacBody(configuration.model, configuration.data, end_jac, None, end_id)
    jacobian = (np.eye(3) - np.outer(direction, direction)) @ (end_jac - start_jac) / length
    return direction, jacobian


class SegmentDirectionTask(mink.Task):
    def __init__(self, model, start, end, cost, min_length=1e-5):
        super().__init__(cost=np.full(3, cost), gain=1., lm_damping=1.)
        self.start_id, self.end_id = model.body(start).id, model.body(end).id
        self.min_length = min_length
        self.target = np.array([0., 0., -1.])

    def compute_error(self, configuration):
        direction, _ = normalized_segment(configuration, self.start_id, self.end_id, self.min_length)
        return direction - self.target

    def compute_jacobian(self, configuration):
        _, jacobian = normalized_segment(configuration, self.start_id, self.end_id, self.min_length)
        return jacobian


class BendCosineTask(mink.Task):
    def __init__(self, model, chain, cost, min_length=1e-5):
        super().__init__(cost=np.array([cost]), gain=1., lm_damping=1.)
        self.ids = [model.body(name).id for name in chain]
        self.min_length = min_length
        self.target = 1.

    def compute_error(self, configuration):
        upper, _ = normalized_segment(configuration, *self.ids[:2], self.min_length)
        lower, _ = normalized_segment(configuration, *self.ids[1:], self.min_length)
        return np.array([np.dot(upper, lower) - self.target])

    def compute_jacobian(self, configuration):
        upper, upper_jac = normalized_segment(configuration, *self.ids[:2], self.min_length)
        lower, lower_jac = normalized_segment(configuration, *self.ids[1:], self.min_length)
        return (lower @ upper_jac + upper @ lower_jac)[None]


class FrameJointBox(mink.Limit):
    def __init__(self, model):
        self.projection = np.eye(model.nv)[6:]
        self.lower = None
        self.upper = None

    def set_bounds(self, anchor, previous_velocity, dt, velocity_limit, acceleration_limit, lo, hi):
        anchor = np.asarray(anchor, dtype=float)
        previous_velocity = np.asarray(previous_velocity, dtype=float)
        if not np.isfinite(dt) or dt <= 0 or velocity_limit <= 0 or acceleration_limit <= 0:
            raise ValueError("positive dt and rate limits required")
        stop_upper = np.sqrt((acceleration_limit * dt) ** 2 +
                             2 * acceleration_limit * np.maximum(hi - anchor, 0)) - acceleration_limit * dt
        stop_lower = np.sqrt((acceleration_limit * dt) ** 2 +
                             2 * acceleration_limit * np.maximum(anchor - lo, 0)) - acceleration_limit * dt
        lower_velocity = np.maximum.reduce([np.full_like(anchor, -velocity_limit),
                                           previous_velocity - acceleration_limit * dt, -stop_lower])
        upper_velocity = np.minimum.reduce([np.full_like(anchor, velocity_limit),
                                           previous_velocity + acceleration_limit * dt, stop_upper])
        self.lower = np.maximum(lo, anchor + lower_velocity * dt)
        self.upper = np.minimum(hi, anchor + upper_velocity * dt)
        if np.any(self.lower > self.upper + 1e-8):
            raise RuntimeError("history joint position/rate constraints conflict")

    def compute_qp_inequalities(self, configuration, dt):
        if self.lower is None:
            return mink.Constraint()
        current = configuration.q[7:]
        return mink.Constraint(G=np.vstack([self.projection, -self.projection]),
                               h=np.r_[self.upper - current, current - self.lower])


class StructuredHistoryRetargeter(KeypointIKRetargeter):
    def __init__(self, experiment, *, structured=True, history=False, **kwargs):
        self.experiment = experiment
        self.use_structured, self.use_history = structured, history
        super().__init__(**kwargs)
        if self.robot != "unitree_g1" or self.cfg["wrist_iterations"] != 0 or self.cfg.get("ik_branch_candidates", 1) != 1:
            raise ValueError("experiment requires G1, one seed and untracked wrists")
        geometry = experiment["geometry"]
        self.direction_tasks, self.bend_tasks = {}, {}
        for name, chain in geometry["chains"].items():
            self.direction_tasks[name] = [SegmentDirectionTask(self.model, start, end,
                                          geometry["direction_cost"], geometry["min_segment_m"])
                                          for start, end in zip(chain[:-1], chain[1:])]
            self.bend_tasks[name] = BendCosineTask(self.model, chain, geometry["bend_cosine_cost"],
                                                 geometry["min_segment_m"])
        self.wrist_relation = {side: RelativePositionTask(self.model, self._b_hand[side], self._b_torso,
                               self._b_torso, geometry["wrist_torso_position_cost"]) for side in SIDES}
        self.history_prior = mink.PostureTask(self.model, cost=experiment["history"]["position_prior_cost"])
        self.frame_box = FrameJointBox(self.model)

    def reset(self):
        super().reset()
        self.previous_velocity = np.zeros(self.nj)
        self.history_active = False
        self.experiment_diagnostics = {}

    def _prepare_target_tasks(self, points, t_s, pelvis_rotation, torso_rotation, pelvis_position, relative_targets):
        tasks = []
        if self.use_structured:
            self.current_directions = {}
            for side in SIDES:
                self.t_knee[side].cost[:3] = 0.
                self.t_elbow[side].cost[:3] = 0.
                self.t_hand[side].cost[:3] = 0.
                for limb, joints in (("arm", ("shoulder", "elbow", "wrist")),
                                     ("leg", ("hip", "knee", "ankle"))):
                    name = f"{side}_{limb}"
                    vectors = []
                    for task, start, end in zip(self.direction_tasks[name], joints[:-1], joints[1:]):
                        if not _ok(points, f"{side}_{start}", f"{side}_{end}"):
                            vectors.append(None)
                            continue
                        delta = points[K[f"{side}_{end}"]] - points[K[f"{side}_{start}"]]
                        length = float(np.linalg.norm(delta))
                        if length < self.experiment["geometry"]["min_segment_m"]:
                            vectors.append(None)
                            continue
                        task.target = delta / length
                        vectors.append(task.target.copy())
                        tasks.append(task)
                        self.current_directions[f"{name}_{start}"] = task
                    if all(vector is not None for vector in vectors):
                        self.bend_tasks[name].target = float(np.dot(*vectors))
                        tasks.append(self.bend_tasks[name])
                if _ok(points, f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist") and f"{side}_hand" in relative_targets:
                    relation = self.wrist_relation[side]
                    relation.target = torso_rotation.T @ (relative_targets[f"{side}_hand"] -
                                                          pelvis_rotation @ self.torso_off)
                    tasks.append(relation)
        if self.history_active:
            tasks.append(self.history_prior)
        return pelvis_position, relative_targets, tasks

    def step(self, points, t_s):
        start = time.perf_counter()
        points = np.asarray(points, dtype=float)
        if points.shape != (27, 3) or not np.isfinite(t_s):
            raise ValueError("expected (27,3) and finite capture timestamp")
        dt = None if self._t_prev is None else float(t_s - self._t_prev)
        if dt is not None and dt <= 0:
            raise ValueError("capture timestamps must increase")
        previous = None if self._q_prev is None else self._q_prev.copy()
        trunk_valid = _ok(points, "left_hip", "right_hip", "left_shoulder", "right_shoulder")
        history_cfg = self.experiment["history"]
        after_startup = self.scale is not None and t_s - self._t_calib_end >= self.cfg["startup_blend_s"]
        reset_reason = None
        if not trunk_valid:
            reset_reason = "trunk_missing"
        elif dt is not None and dt > history_cfg["reset_gap_s"]:
            reset_reason = "timestamp_gap"
        elif not after_startup:
            reset_reason = "startup"
        if reset_reason is not None:
            self.previous_velocity[:] = 0.
        self.history_active = bool(self.use_history and after_startup and trunk_valid and previous is not None
                                   and dt is not None and reset_reason is None)
        original_limits = self.limits
        if self.history_active:
            self.configuration.update(previous)
            self.frame_box.set_bounds(previous[7:], self.previous_velocity, dt,
                                      history_cfg["velocity_limit_rad_s"], history_cfg["acceleration_limit_rad_s2"],
                                      self.lo, self.hi)
            target = previous.copy()
            prediction = self.previous_velocity * history_cfg["velocity_decay_tau_s"] * (
                1. - np.exp(-dt / history_cfg["velocity_decay_tau_s"]))
            target[7:] = np.minimum(np.maximum(previous[7:] + prediction, self.frame_box.lower), self.frame_box.upper)
            self.history_prior.set_target(target)
            self.limits = original_limits + [self.frame_box]
        try:
            result = super().step(points, t_s)
        finally:
            self.limits = original_limits
        constraint_error = 0.
        acceleration = None
        if self.history_active:
            constraint_error = float(max(0., np.max(self.frame_box.lower - result.q[7:]),
                                         np.max(result.q[7:] - self.frame_box.upper)))
            acceleration = float(np.max(np.abs((result.dq[6:] - self.previous_velocity) / dt)))
            if constraint_error > 1e-6:
                raise RuntimeError("emitted output violated the shared history frame budget")
            self.previous_velocity = result.dq[6:].copy()
        else:
            self.previous_velocity[:] = 0.
        direction_errors = {}
        if self.use_structured and result.observed:
            solver_q = self.configuration.q.copy()
            self.configuration.update(result.q)
            for name, task in self.current_directions.items():
                direction, _ = normalized_segment(self.configuration, task.start_id, task.end_id, task.min_length)
                direction_errors[name] = float(np.degrees(np.arctan2(np.linalg.norm(np.cross(direction, task.target)),
                                                                                    np.dot(direction, task.target))))
            self.configuration.update(solver_q)
        self.experiment_diagnostics = dict(history_active=self.history_active, reset_reason=reset_reason,
                                           history_acceleration_max_rad_s2=acceleration,
                                           history_box_violation_rad=constraint_error,
                                           direction_error_vs_input_deg=direction_errors,
                                           held_groups=list(result.held_groups))
        result.solve_ms = 1000 * (time.perf_counter() - start)
        return result
