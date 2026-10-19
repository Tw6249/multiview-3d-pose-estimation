"""Independent articulated fixtures for geometry/state stress tests, NOT RGB truth.

World X-forward, Y-left, Z-up; metres, seconds, integer exposure nanoseconds.
This module is never imported by the perception or controller paths.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import project
from .landmarks import HUMAN33, BONES


def camera_ring(offset=0.):
    cameras = {}
    for i, a in enumerate(np.deg2rad(np.array([25, 115, 205, 295])+offset)):
        c = np.array([4.5*np.cos(a), 4.5*np.sin(a), 1.65])
        z = np.array([0., 0., .9])-c; z /= np.linalg.norm(z)
        x = np.cross(z, [0, 0, 1]); x /= np.linalg.norm(x)
        r = np.array([x, np.cross(z, x), z])
        cameras[f'cam{i:02d}'] = dict(K=[[850., 0, 480.], [0, 850., 360.], [0, 0, 1]],
            R_world_to_camera=r.tolist(), t_world_to_camera_m=(-r@c).tolist(),
            distortion=[0.]*5, width=960, height=720)
    return cameras


def articulated_motion(subject, action, times, phase=0.):
    h = subject['height_m']; leg = h*subject['leg_ratio']
    thigh, shank = leg*.51, leg*.49
    all_points, rotations = [], []
    for t in times:
        yaw = (1.8*np.sin(.75*t+phase) if action == 'turn' else .15*np.sin(t*.5+phase))
        r = Rotation.from_euler('z', yaw).as_matrix()
        root = np.array([.15*t if action in ('walk', 'turn', 'cross_step') else 0., 0., leg*.93+.08])
        if action == 'squat': root[2] -= .23*(1-np.cos(2.4*t+phase))/2
        p = {}
        for side, sign in [('left', 1), ('right', -1)]:
            hip = root+r@np.array([0, sign*h*.085, 0])
            shoulder = root+r@np.array([0, sign*h*.115, h*.27])
            cycle = 2*np.pi*(.85*t+phase)+(0 if sign == 1 else np.pi)
            moving = action in ('walk', 'turn', 'cross_step', 'occlusion')
            dx = .19*np.sin(cycle) if moving else .025
            dz = .09*max(0., np.cos(cycle)) if moving else 0.
            lateral = sign*h*.085
            if action == 'cross_step': lateral += sign*.19*np.sin(cycle)
            ankle = np.array([root[0], root[1], .08])+r@np.array([dx, lateral, dz])
            d = ankle-hip; distance = np.linalg.norm(d); direction = d/distance
            distance = min(distance, thigh+shank-.001)
            ankle = hip+direction*distance
            along = (thigh**2-shank**2+distance**2)/(2*distance)
            bend = r[:, 0]-direction*np.dot(r[:, 0], direction); bend /= np.linalg.norm(bend)
            knee = hip+direction*along+bend*np.sqrt(max(0, thigh**2-along**2))
            arm_phase = (8.5*t+phase if action == 'fast_arms' else cycle)
            swing = .9*np.sin(arm_phase)
            upper = np.array([np.sin(swing), sign*.12, -np.cos(swing)])
            upper /= np.linalg.norm(upper)
            elbow = shoulder+r@upper*h*.175
            fore = np.array([np.sin(swing+.5), sign*.09, -np.cos(swing+.5)])
            fore /= np.linalg.norm(fore)
            wrist = elbow+r@fore*h*.155
            for n, v in [('hip',hip),('knee',knee),('ankle',ankle),('shoulder',shoulder),('elbow',elbow),('wrist',wrist)]:
                p[f'{side}_{n}'] = v
            for n, v in [('heel',[-.06,0,-.075]),('big_toe',[.16,-sign*.025,-.075]),('small_toe',[.14,sign*.055,-.075])]:
                p[f'{side}_{n}'] = ankle+r@np.array(v)
            for n, v in [('wrist',[0,0,0]),('thumb_mcp',[.04,.05,0]),('index_mcp',[.075,.035,0]),
                         ('middle_mcp',[.09,0,0]),('pinky_mcp',[.065,-.035,0])]:
                p[f'{side}_hand_{n}'] = wrist+r@np.array(v)
        for n, v in [('nose',[.07,0,h*.43]),('left_eye',[.06,.035,h*.46]),('right_eye',[.06,-.035,h*.46]),
                     ('left_ear',[0,.07,h*.44]),('right_ear',[0,-.07,h*.44])]:
            p[n] = root+r@np.array(v)
        all_points.append(np.array([p[n] for n in HUMAN33])); rotations.append(r)
    return np.array(all_points), np.array(rotations)


def validate_splits(cfg):
    seen = set()
    for split in ('tune', 'calibration', 'test'):
        subjects = {s['subject'] for s in cfg['splits'][split]}
        if len(subjects) != len(cfg['splits'][split]) or seen & subjects:
            raise ValueError('Subjects must be unique and disjoint across splits')
        seen |= subjects


def generate(root, cfg):
    from .pipeline import write_jsonl
    root = Path(root)
    if root.exists(): raise FileExistsError('Do not overwrite a benchmark')
    validate_splits(cfg); root.mkdir(parents=True)
    manifest = dict(schema_version=1, domain=cfg['domain'], configuration=cfg, sequences=[])
    case = 0
    for split, subjects in cfg['splits'].items():
        for subject in subjects:
            for action in cfg['actions']:
                rng = np.random.default_rng(cfg['seed']+case*173)
                randomized=cfg.get('corruption_schedule')=='randomized_v2'
                angle_range=45 if randomized else 15
                cameras = camera_ring(float(rng.uniform(-angle_range,angle_range)))
                times = np.arange(cfg['frames'])/cfg['fps']
                xyz, rotations = articulated_motion(subject, action, times, float(rng.uniform(0,.8)))
                velocity = np.gradient(xyz, times, axis=0, edge_order=2)
                sequence = f"{split}/{subject['subject']}/{action}"
                folder = root/sequence; folder.mkdir(parents=True)
                cal = dict(schema_version=1, units='metre', world_axes='right-handed, Z-up', cameras=cameras)
                (folder/'cameras.json').write_text(json.dumps(cal), encoding='utf-8')
                calibration_hash = hashlib.sha256((folder/'cameras.json').read_bytes()).hexdigest()
                opts = dict(max_sync_error_ms=5, confidence_threshold=.35, ransac_threshold_px=8., min_triangulation_angle_deg=1.)
                (folder/'config.json').write_text(json.dumps(opts), encoding='utf-8')
                rows, truth = [], []
                biases = rng.normal(0,1.,(4,len(HUMAN33),2))
                detail_stress=cfg.get('detail_corruption',False)
                detail_swap_start=int(rng.integers(25,32)) if detail_stress else 0
                detail_shift_start=int(rng.integers(60,67)) if detail_stress else 0
                detail_shift=np.array([rng.uniform(.55,.85),rng.uniform(-.3,.3),rng.uniform(.15,.3)]) if detail_stress else None
                # Event timing, affected views and joints are independent of split
                # and unavailable to inference. v1's archived data stays unchanged.
                if randomized:
                    swap_start=int(rng.integers(18,36));swap_end=swap_start+int(rng.integers(8,22))
                    occ_start=int(rng.integers(55,70));occ_end=occ_start+int(rng.integers(8,20))
                    blackout_start=int(rng.integers(92,100));blackout_end=blackout_start+int(rng.integers(5,13))
                    swap_views=set(map(int,rng.choice(4,2,replace=False)));occ_views=set(map(int,rng.choice(4,2,replace=False)))
                    swap_pairs=[[(13,14)],[(9,10),(15,16)],[(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]][case%3]
                    early_bias_end=int(rng.integers(5,22));noise_px=float(rng.uniform(1.,2.8));stale_cam=int(rng.integers(4))
                else:
                    swap_start,swap_end,occ_start,occ_end,blackout_start,blackout_end=22,38,45,60,65,72
                    swap_views,occ_views={0,1},{1,2};swap_pairs=[(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]
                    early_bias_end=10;noise_px=1.6;stale_cam=None
                for f,t in enumerate(times):
                    stamp = round(t*1e9)
                    row = dict(frame_id=f,timestamp_ns=stamp,source_id=subject['subject'],
                        landmark_schema_id='human33',joint_names=HUMAN33,perception_signature='synthetic_observation_corruptor_v2' if randomized else 'synthetic_observation_corruptor_v1',
                        calibration_sha256=calibration_hash,views={})
                    for c,(cid,cam) in enumerate(cameras.items()):
                        observed_frame=max(0,f-1) if randomized and c==stale_cam and 75<=f<85 else f
                        xy = project(xyz[observed_frame],cam)+rng.normal(0,noise_px,(len(HUMAN33),2))+biases[c]
                        scores = np.full(len(HUMAN33),.9)
                        spikes = rng.random(len(HUMAN33))<.015
                        xy[spikes] += rng.normal(0,35,(spikes.sum(),2))
                        auxiliary_anchors={s+'_ankle':dict(keypoint_px=xy[j].tolist(),score=float(scores[j]))
                                           for s,j in [('left',15),('right',16)]} if detail_stress else None
                        # Early systematic detector error; size adaptation must recover later.
                        if f < early_bias_end: xy[[13,14],1] += 14
                        # Incorrect left/right labels in two views, including the highest-score views.
                        if swap_start <= f < swap_end and c in swap_views:
                            for a,b in swap_pairs:
                                xy[[a,b]] = xy[[b,a]]; scores[[a,b]] = .98
                        # A view-specific occlusion is recoverable; global occlusion is not observed.
                        if occ_start <= f < occ_end and c in occ_views: scores[[7,9,13,15]] = .05
                        if action == 'occlusion' and blackout_start <= f < blackout_end: scores[[13,15]] = .02
                        if detail_stress:
                            # Independent detail-head labels, not the body's swap.
                            if detail_swap_start<=f<detail_swap_start+16 and c in (0,2):
                                for a,b in [(17,20),(18,21),(19,22)]+list(zip(range(23,28),range(28,33))):
                                    xy[[a,b]]=xy[[b,a]];scores[[a,b]]=scores[[b,a]]
                                auxiliary_anchors['left_ankle'],auxiliary_anchors['right_ankle']=auxiliary_anchors['right_ankle'],auxiliary_anchors['left_ankle']
                            if detail_shift_start<=f<detail_shift_start+16:
                                # Correlated wrong 3D cluster: passes epipolar
                                # checks in all cameras; attachment must detect it.
                                js=list(range(23,28))+[20,21,22]
                                xy[js]=project(xyz[observed_frame,js]+detail_shift,cam)+rng.normal(0,noise_px,(len(js),2))
                                auxiliary_anchors['right_ankle']['keypoint_px']=project(xyz[observed_frame,16]+detail_shift,cam)[0].tolist()
                        row['views'][cid] = dict(timestamp_ns=round(times[observed_frame]*1e9),keypoints_px=xy.tolist(),scores=scores.tolist(),num_people_detected=1)
                        if auxiliary_anchors:row['views'][cid]['auxiliary_model_anchors']=auxiliary_anchors
                    rows.append(row)
                    p = dict(zip(HUMAN33,xyz[f])); v = dict(zip(HUMAN33,velocity[f]))
                    contacts = {s: any(abs(p[f'{s}_{n}'][2])<.025 and np.linalg.norm(v[f'{s}_{n}'])<.2
                                       for n in ('heel','big_toe','small_toe')) for s in ('left','right')}
                    truth.append(dict(frame_id=f,timestamp_ns=stamp,joint_names=HUMAN33,joints_world_m=xyz[f].tolist(),
                        velocities_world_m_s=velocity[f].tolist(),root_rotation_world=rotations[f].tolist(),contacts=contacts,
                        bone_lengths_m={a+'--'+b:float(np.linalg.norm(p[a]-p[b])) for a,b in BONES},
                        events=dict(detail_swap=detail_stress and detail_swap_start<=f<detail_swap_start+16,
                                    detail_detachment=detail_stress and detail_shift_start<=f<detail_shift_start+16,
                                    swap=swap_start<=f<swap_end,occlusion=occ_start<=f<occ_end or
                                    (action=='occlusion' and blackout_start<=f<blackout_end))))
                write_jsonl(folder/'detections.jsonl',rows); write_jsonl(folder/'ground_truth.jsonl',truth)
                manifest['sequences'].append(dict(id=sequence,split=split,subject=subject['subject'],action=action,
                    frames=len(rows),sha256={n:hashlib.sha256((folder/n).read_bytes()).hexdigest()
                                          for n in ('detections.jsonl','ground_truth.jsonl','cameras.json','config.json')}))
                case += 1
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest
