"""Deterministic, labelled fault injection on cached 2D detection rows.

Input: detection rows (timestamp_ns int, views[cid] with keypoints_px [N,2] px,
scores [N], optional auxiliary_model_anchors) in exposure order. Output: new rows
plus event labels; the source rows are never modified. Faults alter detector
evidence only and never read references, so the downstream estimator sees what a
real failure could look like. Injected faults are not physical occlusions: they
lack correlated appearance change, partial visibility and detector score drift.
"""
import copy
import numpy as np

FAULT_TYPES = ('view_dropout', 'limb_dropout', 'body_gap', 'view_outlier',
               'view_lr_swap', 'all_lr_swap', 'view_time_offset')
DURATIONS_MS = (100, 300, 600)
BODY12 = tuple(s+'_'+j for j in ('shoulder','elbow','wrist','hip','knee','ankle') for s in ('left','right'))
LIMBS = dict(arm=('elbow','wrist','hand_'), leg=('knee','ankle','big_toe','small_toe','heel'))
OUTLIER_JOINTS = ('elbow','wrist','knee','ankle')


def mirror(name):
    return ('right_'+name[5:] if name.startswith('left_') else
            'left_'+name[6:] if name.startswith('right_') else name)


def limb_landmarks(names, side, limb):
    return [n for n in names if n.startswith(side+'_') and any(n[len(side)+1:].startswith(p) for p in LIMBS[limb])]


def _usable(view):
    return view is not None and view.get('keypoints_px') is not None


def plan(rows, seed, fps, start_s=2., period_s=3., tail_s=1.):
    """One event per period; every (type, duration) pair recurs evenly over seeds.

    Events end at least tail_s before the clip end so recovery can be scored.
    The camera for single-view faults is the one with the most usable frames in
    the event window, so a fault is not silently placed on an empty view.
    """
    rng = np.random.default_rng(seed)
    combos = [(f, d) for f in FAULT_TYPES for d in DURATIONS_MS]
    order = np.random.default_rng(12345).permutation(len(combos))
    names = rows[0]['joint_names']
    events, slot = [], 0
    while True:
        start = int(round((start_s+slot*period_s)*fps))
        kind, ms = combos[order[(seed*7+slot) % len(combos)]]
        end = start+max(1, int(round(ms*fps/1000)))
        if end+tail_s*fps > len(rows):
            break
        cams = sorted(rows[start]['views'])
        cam = max(cams, key=lambda c: (sum(_usable(r['views'].get(c)) for r in rows[start:end]), -cams.index(c)))
        side = ('left','right')[rng.integers(2)]
        event = dict(id=f'{seed}-{slot}', type=kind, duration_ms=ms, start_frame=start, end_frame=end,
                     start_ns=rows[start]['timestamp_ns'], end_ns=rows[end]['timestamp_ns'], cameras=[cam],
                     affected_joints=list(BODY12), landmarks=None, params={})
        if kind in ('body_gap', 'all_lr_swap', 'limb_dropout'):
            event['cameras'] = cams
        if kind == 'limb_dropout':
            limb = ('arm','leg')[rng.integers(2)]
            event['landmarks'] = limb_landmarks(names, side, limb)
            event['affected_joints'] = [side+'_'+j for j in LIMBS[limb][:2]]
            event['params'] = dict(side=side, limb=limb)
        elif kind == 'view_outlier':
            joint = side+'_'+OUTLIER_JOINTS[rng.integers(len(OUTLIER_JOINTS))]
            event['landmarks'] = [joint]; event['affected_joints'] = [joint]
            event['params'] = dict(offset_px=float(rng.uniform(60, 150)),
                                   direction_rad=float(rng.uniform(0, 2*np.pi)), score=.9)
        elif kind == 'view_time_offset':
            event['params'] = dict(lag_frames=int(rng.integers(2, 4)))
        events.append(event)
        slot += 1
    return events


def _swap(view, names):
    index = {n: i for i, n in enumerate(names)}
    order = [index.get(mirror(n), index[n]) for n in names]
    for key in ('keypoints_px', 'scores'):
        if view.get(key) is not None:
            view[key] = [view[key][j] for j in order]
    anchors = view.get('auxiliary_model_anchors')
    if anchors:
        view['auxiliary_model_anchors'] = {mirror(k): v for k, v in anchors.items()}


def _drop(view):
    view.update(keypoints_px=None, scores=None, num_people_detected=0)
    view.pop('auxiliary_model_anchors', None)
    view.pop('outside_image_landmarks', None)


def inject(rows, events):
    """Return faulted copies and labels with the number of frames actually altered.

    A frame counts as applied only when a targeted view had usable keypoints.
    """
    out = copy.deepcopy(rows)
    names = rows[0]['joint_names']
    index = {n: i for i, n in enumerate(names)}
    labels = []
    for event in events:
        applied = 0
        for f in range(event['start_frame'], event['end_frame']):
            changed = False
            for cam in event['cameras']:
                view = out[f]['views'].get(cam)
                if not _usable(view):
                    continue
                kind = event['type']
                if kind in ('view_dropout', 'body_gap'):
                    _drop(view)
                elif kind == 'limb_dropout':
                    for n in event['landmarks']:
                        view['scores'][index[n]] = 0.
                    if event['params']['limb'] == 'leg':
                        (view.get('auxiliary_model_anchors') or {}).pop(event['params']['side']+'_ankle', None)
                elif kind == 'view_outlier':
                    i = index[event['landmarks'][0]]
                    a, r = event['params']['direction_rad'], event['params']['offset_px']
                    x, y = view['keypoints_px'][i]
                    view['keypoints_px'][i] = [x+r*np.cos(a), y+r*np.sin(a)]
                    view['scores'][i] = max(view['scores'][i], event['params']['score'])
                elif kind in ('view_lr_swap', 'all_lr_swap'):
                    _swap(view, names)
                elif kind == 'view_time_offset':
                    # Stale image content stamped with the current exposure time:
                    # an undetected synchronisation fault.
                    stale = copy.deepcopy(rows[max(0, f-event['params']['lag_frames'])]['views'].get(cam))
                    if stale is None:
                        continue
                    stale['timestamp_ns'] = view['timestamp_ns']
                    out[f]['views'][cam] = stale
                changed = True
            applied += changed
        labels.append({**event, 'applied_frames': applied})
    return out, labels
