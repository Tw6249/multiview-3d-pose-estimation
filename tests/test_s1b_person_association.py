"""Tests for Module S1b (multi-person cross-view association & target lock) on synthetic multi-camera scenes.

Synthetic people are exact golden-27 skeletons projected into a 4-camera ring with 1.5 px noise; candidate order is
shuffled per camera. Occlusion is modelled the way YOLOX NMS behaves: when two people's boxes overlap strongly in a
camera, only the nearer one is detected there. These test the association logic, not RTMW or YOLOX.
"""
from __future__ import annotations

import numpy as np
import pytest

from g1_imitation.core.constants import load_config
from g1_imitation.core.s1b_person_association import (
    BODY12, LOST, OCCLUDED, TRACKING, UNSELECTED, MultiPersonTargetTracker, ViewCandidates,
    group_candidates)
from g1_imitation.core.s2_triangulation import create_circular_camera_rig
from g1_imitation.core.synthetic_human import synthesize_human_27_pose

CFG = load_config()["s1b_person_association"]
P4, _ = create_circular_camera_rig(num_cameras=4, radius_m=4.0, camera_height_m=1.5, lookat_xyz_m=(0.0, 0.0, 0.9),
                                   focal_px=900.0)
W, H = 1280, 720
DT = 1.0 / 30.0


def person(xy, yaw=0.0):
    return synthesize_human_27_pose(root_xy_m=np.asarray(xy, float), torso_yaw_rad=yaw)


def project(X, rng, noise_px=1.5):
    """(V, 27, 3) [u, v, score]; points behind a camera or outside its image get score 0."""
    Xh = np.c_[X, np.ones(len(X))]
    out = np.zeros((len(P4), len(X), 3))
    for v, p in enumerate(P4):
        uvw = Xh @ p.T
        z = uvw[:, 2]
        uv = uvw[:, :2] / np.maximum(z, 1e-6)[:, None] + rng.normal(0.0, noise_px, (len(X), 2))
        vis = (z > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        out[v, :, :2] = uv
        out[v, :, 2] = np.where(vis, 0.9, 0.0)
    return out


def box_of(kp):
    ok = kp[:, 2] > 0
    return np.array([kp[ok, 0].min() - 15, kp[ok, 1].min() - 25, kp[ok, 0].max() + 15, kp[ok, 1].max() + 15])


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def frame(people, rng, present=None, nms_iou=None, swap=None):
    """Candidates per camera, shuffled. `present[p][v]` False drops person p in camera v; with `nms_iou`, of two
    people whose boxes overlap more than that only the nearer is kept; `swap=(v, a, b)` replaces person a's
    keypoints in camera v by person b's (a crop that locked onto the wrong person). Returns (candidates, owner)."""
    kp = [project(X, rng) for X in people]
    centres = [-np.linalg.solve(p[:, :3], p[:, 3]) for p in P4]
    cands, owner = [], []
    for v in range(len(P4)):
        rows = [(i, kp[i][v]) for i in range(len(people))
                if (present is None or present[i][v]) and (kp[i][v][BODY12, 2] > 0).sum() >= 8]
        if nms_iou is not None:
            keep = []
            for i, k in rows:
                hidden = any(j != i and iou(box_of(k), box_of(kk)) > nms_iou and
                             np.linalg.norm(people[j][11] - centres[v]) < np.linalg.norm(people[i][11] - centres[v])
                             for j, kk in rows)
                if not hidden:
                    keep.append((i, k))
            rows = keep
        if swap is not None and swap[0] == v:
            rows = [(i, kp[swap[2]][v] if i == swap[1] else k) for i, k in rows]
        order = rng.permutation(len(rows))
        rows = [rows[o] for o in order]
        cands.append(ViewCandidates(box_xyxy=np.array([box_of(k) for _, k in rows]).reshape(-1, 4),
                                    det_score=np.full(len(rows), 0.9),
                                    kp=np.array([k for _, k in rows]).reshape(-1, 27, 3)))
        owner.append([i for i, _ in rows])
    return cands, owner


def target_owner(res, owner, cands):
    """Person index behind the target output (one per used camera), or None when no camera is used."""
    who = set()
    for v in res.target_views:
        i = next(i for i, k in enumerate(cands[v].kp) if np.array_equal(k, res.keypoints_2d[v]))
        who.add(owner[v][i])
    assert len(who) <= 1, f"target mixes people {who}"
    return who.pop() if who else None


def test_grouping_recovers_people_from_shuffled_partial_candidates():
    rng = np.random.default_rng(0)
    people = [person((0.8, 0.3)), person((-0.7, -0.4), 1.0), person((0.1, 1.2), -0.5)]
    present = [[True] * 4, [True, False, True, True], [True] * 4]
    cands, owner = frame(people, rng, present=present)
    # a person-like detection in camera 0 only (cannot be triangulated)
    stray = project(person((-2.0, 0.8)), rng)[0]
    cands[0] = ViewCandidates(np.vstack([cands[0].box_xyxy, box_of(stray)]), np.r_[cands[0].det_score, 0.8],
                              np.concatenate([cands[0].kp, stray[None]]))
    owner[0].append(-1)
    hyps = group_candidates(cands, P4, CFG)
    assert len(hyps) == 3
    found = []
    for h in hyps:
        ids = {owner[v][h.sel[v]] for v in range(4) if h.sel[v] >= 0}
        assert len(ids) == 1 and -1 not in ids
        found.append(ids.pop())
        assert np.nanmedian(np.linalg.norm(h.body_m - people[found[-1]][BODY12], axis=1)) < 0.03
    assert sorted(found) == [0, 1, 2]
    assert next(h for h, f in zip(hyps, found) if f == 1).n_views == 3


def test_wrong_person_crop_is_not_grouped_with_its_box_owner():
    rng = np.random.default_rng(1)
    people = [person((0.6, 0.0)), person((-0.6, 0.2), 2.0)]
    cands, owner = frame(people, rng, swap=(2, 0, 1))      # camera 2: person 0's crop shows person 1
    hyps = group_candidates(cands, P4, CFG)
    ids = [{owner[v][h.sel[v]] for v in range(4) if h.sel[v] >= 0} for h in hyps]
    h0 = next(h for h, s in zip(hyps, ids) if s == {0})
    assert h0.sel[2] < 0 and h0.n_views == 3


def run(seq, rng, designate=None, pixel=None, nms_iou=0.5, present=None):
    trk = MultiPersonTargetTracker(CFG)
    if designate is not None:
        trk.designate_xy(designate)
    if pixel is not None:
        trk.designate_pixel(*pixel)
    out = []
    for k, people in enumerate(seq):
        cands, owner = frame(people, rng, nms_iou=nms_iou, present=None if present is None else present(k))
        res = trk.step(cands, P4, k * DT)
        out.append((res, target_owner(res, owner, cands)))
    return out


def test_designated_target_kept_while_people_cross():
    rng = np.random.default_rng(2)
    T = 60
    seq = [[person((-1.2 + 2.4 * k / T, 0.2), 0.0), person((1.2 - 2.4 * k / T, -0.2), np.pi)] for k in range(T)]
    out = run(seq, rng, designate=(-1.2, 0.2))
    assert all(who in (0, None) for _, who in out)
    assert all(r.valid == (r.status == TRACKING) for r, _ in out)
    assert all(who == 0 for r, who in out if r.valid)
    assert sum(r.valid for r, _ in out) >= 0.8 * T
    assert len({r.target_id for r, _ in out if r.valid}) == 1


def test_target_that_leaves_is_lost_not_replaced_by_the_person_taking_its_place():
    rng = np.random.default_rng(3)
    T = 70
    seq = [[person((0.5, 0.0)), person((-1.5 + 2.0 * min(k / 35, 1.0), 0.0))] for k in range(T)]
    out = run(seq, rng, designate=(0.5, 0.0), present=lambda k: [[k < 20] * 4, [True] * 4])
    st = [r.status for r, _ in out]
    assert st[2:20] == [TRACKING] * 18
    assert all(who != 1 for _, who in out)
    assert st[20] == OCCLUDED
    first_lost = st.index(LOST)
    assert 20 + 14 <= first_lost <= 20 + 17                       # target_lost_s = 0.5 s at 30 Hz
    assert set(st[first_lost:]) == {LOST}                          # a designated target is not re-locked automatically
    assert all(r.target_id is None for r, _ in out[first_lost:])


def test_auto_select_needs_exactly_one_person_in_the_capture_volume():
    rng = np.random.default_rng(4)
    outside = person((0.0, -2.7))       # seen by all 4 cameras, beyond the ring radius (0.6 x 4.0 m): gets a track
    two_view = person((1.9, 1.9))       # seen by cameras 2 and 3 only: grouped, but never starts a track
    out = run([[person((0.3, -0.2)), outside, two_view]] * 20, rng)
    r0 = out[0][0]
    assert r0.status == UNSELECTED and sorted(h.n_views for h in r0.hypotheses) == [2, 4, 4]
    assert [tid for h, tid in zip(r0.hypotheses, r0.hypothesis_track_ids) if h.n_views == 2] == [-1]
    assert all(len(r.track_centroids_m) == 2 for r, _ in out)
    assert all(r.status == TRACKING and who == 0 for r, who in out[2:])
    out2 = run([[person((0.6, 0.0)), person((-0.6, 0.0))]] * 20, rng)
    assert all(r.status == UNSELECTED and not r.valid for r, _ in out2)


def test_pixel_designation_and_ambiguous_xy_designation():
    rng = np.random.default_rng(5)
    people = [person((0.6, 0.6)), person((-0.6, -0.6))]
    c0, owner0 = frame(people, np.random.default_rng(5))
    i1 = owner0[0].index(1)
    u, v = c0[0].box_xyxy[i1, :2] + 0.5 * (c0[0].box_xyxy[i1, 2:] - c0[0].box_xyxy[i1, :2])
    out = run([people] * 10, rng, pixel=(0, u, v))
    assert all(who == 1 for r, who in out[2:]) and out[-1][0].designation == "resolved"
    out2 = run([people] * 80, rng, designate=(0.0, 0.0))           # equidistant: never resolved, then expires
    assert all(not r.valid for r, _ in out2) and out2[-1][0].designation == "expired"


def test_rejects_non_increasing_time():
    rng = np.random.default_rng(6)
    trk = MultiPersonTargetTracker(CFG)
    cands, _ = frame([person((0.0, 0.0))], rng)
    trk.step(cands, P4, 1.0)
    with pytest.raises(ValueError):
        trk.step(cands, P4, 1.0)
