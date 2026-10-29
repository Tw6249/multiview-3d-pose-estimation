"""Unit tests for the AdvancedTake evaluation helpers (synthetic data with known truth)."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from g1_imitation.core import adv_eval as ae  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
from advanced_take_gmr_control import to_50hz  # noqa: E402


def make_pose():
    p = np.zeros((1, 12, 3))
    J = ae.J
    p[0, J["l_hip"]] = [0, 0.1, 0.9]
    p[0, J["r_hip"]] = [0, -0.1, 0.9]
    p[0, J["l_knee"]] = [0.1, 0.1, 0.5]
    p[0, J["r_knee"]] = [0, -0.1, 0.5]
    p[0, J["l_ankle"]] = [0, 0.1, 0.1]
    p[0, J["r_ankle"]] = [0, -0.1, 0.1]
    p[0, J["l_shoulder"]] = [0, 0.18, 1.4]
    p[0, J["r_shoulder"]] = [0, -0.18, 1.4]
    p[0, J["l_elbow"]] = [0, 0.18, 1.1]
    p[0, J["r_elbow"]] = [0, -0.18, 1.1]
    p[0, J["l_wrist"]] = [0.25, 0.18, 1.1]
    p[0, J["r_wrist"]] = [0, -0.18, 0.85]
    return p


def test_end_alignment_maps_last_frame_to_last_bvh_and_rejects_overrun():
    assert ae.end_aligned_index(9 / 30, 1000, 10) == 999
    assert ae.end_aligned_index(8 / 30, 1000, 10) == 995
    assert ae.end_aligned_index(0.0, 10, 10) is None      # pre-roll is dropped, never clamped
    with pytest.raises(ValueError):
        ae.end_aligned_index(float("nan"), 10, 10)


def test_slot_grid_keeps_missing_slots_and_rejects_duplicate_pts():
    pts = {"a": np.arange(5) / 30.0}
    grid, idx = ae.build_slot_grid(pts, 20, None)
    assert len(grid) == 5 and (idx["a"] >= 0).sum() == 5
    pts = {"a": np.array([0.0, 0.0333, 0.0334, 0.1, 0.1333])}
    with pytest.raises(ValueError):
        ae.build_slot_grid(pts, 20, None)
    grid, idx = ae.build_slot_grid({"a": np.arange(3) / 30.0}, 20, None)
    assert (idx["a"] == -1).sum() == len(grid) - 3


def test_umeyama_recovers_similarity_and_is_proper_rotation():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    R0 = Rotation.from_euler("xyz", [0.3, -0.5, 1.0]).as_matrix()
    Y = 2.5 * (R0 @ X.T).T + np.array([1.0, -2.0, 0.5])
    s, R, t = ae.umeyama(X, Y)
    assert s == pytest.approx(2.5, abs=1e-9)
    assert np.allclose(R, R0, atol=1e-9) and np.linalg.det(R) == pytest.approx(1.0)
    assert np.allclose(t, [1.0, -2.0, 0.5], atol=1e-9)


def test_robust_umeyama_ignores_gross_outliers():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 3))
    Y = 1.2 * X + 0.3
    Y[:10] += 5.0
    s, R, t = ae.robust_umeyama(X, Y, trim=0.9)
    assert s == pytest.approx(1.2, abs=1e-3)


def test_segment_direction_and_joint_angle_are_rotation_invariant_for_angles():
    p = make_pose()
    Rg = Rotation.from_euler("z", 0.7).as_matrix()
    q = (Rg @ p[0].T).T[None]
    a, b = ae.interior_angles(p), ae.interior_angles(q)
    for k in a:
        assert a[k][0] == pytest.approx(b[k][0], abs=1e-9)
    assert a["r_knee_flex"][0] == pytest.approx(180.0, abs=1e-6)   # straight right leg
    da, db = ae.segment_directions(p), ae.segment_directions(q)
    assert ae.angle_between_deg(da["l_thigh"], da["l_thigh"])[0] == pytest.approx(0.0, abs=1e-4)  # arccos near 1 loses ~1e-6 deg
    R = ae.rotation_only_align(np.concatenate([da[n] for n in da]), np.concatenate([db[n] for n in db]))
    assert np.allclose(R, Rg, atol=1e-9)


def test_bone_lengths_and_jitter():
    p = make_pose()
    bl = ae.bone_lengths(p)
    assert bl["r_shank"][0] == pytest.approx(0.4)
    e = np.zeros((10, 12, 3))
    assert ae.jitter_rms(e) == 0.0
    e[:, 0, 0] = np.arange(10) * 0.001
    assert ae.jitter_rms(e) == pytest.approx(0.001 / np.sqrt(12.0))


def test_to_50hz_holds_instead_of_bridging_unobserved_slots():
    t = np.arange(10) / 30.0
    q = np.zeros((10, 36))
    q[:, 3] = 1.0
    q[:, 7] = np.arange(10, dtype=float)                 # joint 0 ramps 0..9
    observed = np.ones(10, bool)
    observed[4:6] = False                                # two missing slots
    t50, out, fresh, held = to_50hz(t, q, observed)
    assert held > 0 and held == int((~fresh).sum())
    gap = (t50 > t[3] + 1e-9) & (t50 < t[6] - 1e-9)
    assert np.all(~fresh[gap])
    assert np.allclose(out[gap, 7], out[np.flatnonzero(gap)[0] - 1, 7])   # held, not interpolated toward 6
    ok = t50 < t[3]
    assert np.allclose(out[ok, 7], t50[ok] * 30.0)        # observed span is linearly interpolated
