"""Causal, single-person cross-view label hypotheses and structured reconstruction.

Inspired by geometry/instance consistency in TwinPose and uncertainty-weighted
fitting in MAMMA. This is a small geometric implementation, not either network.
Input is timestamped named 2D observations; output remains current-image 3D
observations. Historical predictions never count as additional camera evidence.
"""
import copy
import hashlib
import itertools
import json
import cv2
import numpy as np
from scipy.optimize import least_squares
from .geometry import arrays, CameraGeometry, project
from .landmarks import BONES
from .pipeline import reconstruct_record
from .sensors import validate_calibration
from .auxiliary import associate_details
from .observation_consistency import ObservationConsistency


def skew(v):
    x,y,z = v
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])


class MultiViewTemporal:
    def __init__(self, calibration, config, association=True, structured=True, solver='newton',
                 auxiliary=True, observation_consistency=False):
        validate_calibration(calibration)
        self.cameras = calibration['cameras']; self.cfg = {**config,'triangulation_solver':solver}
        self.triangulation_geometry={}
        self.association = association; self.structured = structured
        self.auxiliary = auxiliary
        self.consistency = ObservationConsistency(self.cameras, self.cfg) if observation_consistency else None
        self.calibration_sha = hashlib.sha256(json.dumps(calibration,sort_keys=True).encode()).hexdigest()
        self.geometry = {c:CameraGeometry([cam]) for c,cam in self.cameras.items()}
        self.fundamental = {}
        for a,b in itertools.combinations(self.cameras,2):
            ka,ra,ta,_ = arrays(self.cameras[a]); kb,rb,tb,_ = arrays(self.cameras[b])
            r = rb@ra.T; t = tb-r@ta
            self.fundamental[a,b] = np.linalg.inv(kb).T@skew(t)@r@np.linalg.inv(ka)
        self.last_timestamp = None

    def prior(self, state, row):
        if not state or any(state.get(k)!=row.get(k) for k in ('source_id','landmark_schema_id','perception_signature')):
            return {}
        if state.get('fusion_calibration_sha256') != self.calibration_sha:
            return {}
        dt = (row['timestamp_ns']-state['timestamp_ns'])/1e9
        if not 0 < dt <= .15: return {}
        result = {}
        for n,p,v,s in zip(state['joint_names'],state['joints_world_m'],state['velocities_world_m_s'],state['joint_states']):
            age = (row['timestamp_ns']-s['last_observed_timestamp_ns'])/1e9 if s['last_observed_timestamp_ns'] is not None else np.inf
            if p is not None and v is not None and age <= .15 and s.get('status') != 'reacquiring':
                result[n] = (np.array(p)+np.array(v)*dt, max(.06,s['max_std_m']+age*1.5))
        return result

    def associate(self, row, prior):
        out = copy.deepcopy(row); names = row['joint_names']; corrections = []; ambiguities = []; candidates = {}
        ids = [c for c,v in row['views'].items() if c in self.cameras and v.get('keypoints_px') is not None
               and abs(v['timestamp_ns']-row['timestamp_ns'])<=self.cfg['max_sync_error_ms']*1e6]
        if len(ids)>8: raise ValueError('Hypothesis search supports at most eight views')
        xy, confidence, undistorted = {}, {}, {}
        for c in ids:
            v = row['views'][c]
            xy[c] = np.asarray(v['keypoints_px'],float); confidence[c] = np.asarray(v['scores'],float)
            if xy[c].shape!=(len(names),2) or confidence[c].shape!=(len(names),): raise ValueError('Invalid 2D shape')
            k,_,_,d = arrays(self.cameras[c])
            undistorted[c] = cv2.undistortPoints(xy[c].reshape(-1,1,2),k,d,P=k).reshape(-1,2)
        # A knee/wrist detector can swap independently of the hip/shoulder.
        # Whole-limb permutations would relabel otherwise correct root evidence.
        groups = {part:(part,) for part in ('shoulder','elbow','wrist','hip','knee','ankle')}
        prior_names=list(prior)
        projected={c:dict(zip(prior_names,project(np.array([prior[n][0] for n in prior_names]),self.cameras[c])))
                   for c in ids} if prior else {}
        for group, parts in groups.items():
            pairs = [(names.index('left_'+p),names.index('right_'+p)) for p in parts
                     if 'left_'+p in names and 'right_'+p in names]
            if not pairs or len(ids)<2: continue
            original = np.array([i for pair in pairs for i in pair]); flipped = original.reshape(-1,2)[:,::-1].ravel()
            maps = [original,flipped]; unary = np.zeros((len(ids),2)); pair_cost = {}
            for ci,c in enumerate(ids):
                for label in (0,1):
                    residuals=[]
                    for dst,src in zip(original,maps[label]):
                        if names[dst] in prior and confidence[c][src]>=self.cfg['confidence_threshold']:
                            pixels = projected[c][names[dst]]
                            # Bounded temporal influence: geometry can overrule an accelerating prior.
                            residuals.append(min(np.linalg.norm(pixels-xy[c][src]),60.))
                    unary[ci,label] = .12*np.mean(residuals) if residuals else 0.
            for ai,bi in itertools.combinations(range(len(ids)),2):
                a,b = ids[ai],ids[bi]
                f = self.fundamental.get((a,b))
                if f is None: f = self.fundamental[b,a].T
                costs = np.zeros((2,2))
                for la,lb in itertools.product((0,1),repeat=2):
                    ia,ib = maps[la],maps[lb]
                    mask = ((confidence[a][ia]>=self.cfg['confidence_threshold']) &
                            (confidence[b][ib]>=self.cfg['confidence_threshold']) &
                            np.isfinite(undistorted[a][ia]).all(axis=1) & np.isfinite(undistorted[b][ib]).all(axis=1))
                    if mask.sum()<2: continue
                    pa = np.c_[undistorted[a][ia[mask]],np.ones(mask.sum())]
                    pb = np.c_[undistorted[b][ib[mask]],np.ones(mask.sum())]
                    lbp,lap = pa@f.T,pb@f
                    e = np.abs(np.sum(pb*lbp,axis=1))
                    dist = .5*e*(1/np.maximum(np.linalg.norm(lbp[:,:2],axis=1),1e-9)+1/np.maximum(np.linalg.norm(lap[:,:2],axis=1),1e-9))
                    costs[la,lb] = float(np.mean(np.minimum(dist,20.)))
                pair_cost[ai,bi] = costs
            hypotheses=[]
            for bits in itertools.product((0,1),repeat=len(ids)):
                cost=sum(unary[i,b] for i,b in enumerate(bits))
                cost+=sum(c[bits[a],bits[b]] for (a,b),c in pair_cost.items())
                hypotheses.append((float(cost),bits))
            hypotheses.sort()
            best,bits = hypotheses[0]
            candidates[group]=[dict(flipped_cameras=[ids[i] for i,b in enumerate(h) if b],
                relative_cost_px=float(cost-best)) for cost,h in hypotheses[:3]]
            geometry_original=sum(c[0,0] for c in pair_cost.values())
            geometry_best=sum(c[bits[a],bits[b]] for (a,b),c in pair_cost.items())
            global_temporal_gain=float(np.mean(unary[:,0]-unary[:,1]))
            if all(bits) and prior and geometry_original-geometry_best<1.5 and global_temporal_gain>=1.5:
                # A simultaneous global permutation has identical epipolar geometry.
                # History can flag it, but cannot certify a semantic correction.
                for c in ids:
                    for j in original:out['views'][c]['scores'][int(j)]=0.
                ambiguities.append(dict(camera='all',group=group,reason='global label symmetry lacks cross-view evidence'))
                continue
            # Complementary global label permutations cannot be resolved geometrically.
            # With no temporal evidence keep detector semantics, explicitly report that prior.
            if not prior:
                complement = tuple(1-b for b in bits)
                if sum(complement)<sum(bits): bits = complement
            for ci,c in enumerate(ids):
                alternative = min(cost for cost,h in hypotheses if h[ci]!=bits[ci])
                margin = alternative-best
                temporal_gain=float(unary[ci,0]-unary[ci,1])
                temporal_available=bool(prior)
                if bits[ci] and temporal_available and margin>=1.5 and geometry_original-geometry_best>=1.5 and temporal_gain>=1.5:
                    mapping = list(range(len(names)))
                    for dst,src in zip(original,flipped): mapping[dst]=int(src)
                    # Body and foot/palm heads can disagree independently. Never
                    # relabel auxiliary observations based only on body evidence.
                    view=out['views'][c]
                    view['keypoints_px']=[view['keypoints_px'][j] for j in mapping]
                    view['scores']=[view['scores'][j] for j in mapping]
                    provenance=view.get('assigned_source_indices',list(range(len(names))))
                    view['assigned_source_indices']=[provenance[j] for j in mapping]
                    corrections.append(dict(camera=c,group=group,cost_margin_px=float(margin)))
                elif bits[ci] and temporal_available and temporal_gain>=1.5:
                    # No confident identity assignment: remove only affected current observations.
                    for j in original: out['views'][c]['scores'][int(j)]=0.
                    ambiguities.append(dict(camera=c,group=group,reason='label hypotheses indistinguishable'))
        out['association'] = dict(corrections=corrections,ambiguities=ambiguities,candidates=candidates,
            candidate_scores_calibrated=False,history_enabled=False,
            semantic_identity='temporal_and_geometry' if prior else 'unverified_detector_label_prior')
        return out

    def update(self, row, state=None):
        stamp=row['timestamp_ns']
        if type(stamp) is not int or (self.last_timestamp is not None and stamp<=self.last_timestamp):
            raise ValueError('Exposure timestamps must strictly increase')
        self.last_timestamp=stamp
        prior=self.prior(state,row)
        checked = self.consistency.update(row) if self.consistency else row
        assigned=self.associate(checked,prior) if self.association else copy.deepcopy(checked)
        if self.auxiliary:assigned=associate_details(assigned,self.cfg['confidence_threshold'])
        result=reconstruct_record(assigned,self.cameras,self.cfg,self.triangulation_geometry)
        result['calibration_sha256']=self.calibration_sha
        result['fusion_calibration_sha256']=self.calibration_sha
        result['association']=assigned.get('association',{})
        result['auxiliary_association']=assigned.get('auxiliary_association',{})
        result['observation_consistency']=assigned.get('observation_consistency',{})
        if self.structured: self.refine(result,assigned,prior,state.get('shape',{}) if state and prior else {})
        return result,assigned

    def refine(self, row, detection, prior, shape):
        names=row['joint_names']; points=row['joints_world_m']; qualities=row['quality']
        js=[i for i,p in enumerate(points) if p is not None and names[i] in {n for pair in BONES for n in pair}]
        if len(js)<3:return
        index={names[j]:i for i,j in enumerate(js)}; original=np.array([points[j] for j in js])
        obs=[]
        for i,j in enumerate(js):
            for c in qualities[j]['inlier_cameras']:
                v=detection['views'][c]
                obs.append((i,self.geometry[c],np.asarray(v['keypoints_px'][j]),max(.2,float(v['scores'][j]))**.5/2.5))
        bones=[(index[a],index[b],shape[a+'--'+b]['length_m'],max(.015,shape[a+'--'+b]['uncertainty_m']))
               for a,b in BONES if a in index and b in index and shape.get(a+'--'+b,{}).get('status')=='stable']
        priors=[(i,*prior[names[j]]) for i,j in enumerate(js) if names[j] in prior]
        def residual(v):
            p=v.reshape(-1,3)
            return np.r_[np.concatenate([(g.project(p[i])[0]-xy)*w for i,g,xy,w in obs]),
                [(np.linalg.norm(p[a]-p[b])-length)/s for a,b,length,s in bones],
                np.concatenate([(p[i]-target)/sigma for i,target,sigma in priors]) if priors else []]
        def jac(v):
            p=v.reshape(-1,3); mat=np.zeros((len(obs)*2+len(bones)+3*len(priors),len(js)*3)); r=0
            for i,g,_,w in obs: mat[r:r+2,3*i:3*i+3]=g.jacobian(p[i])[0]*w;r+=2
            for a,b,_,s in bones:
                delta=p[a]-p[b];grad=delta/max(np.linalg.norm(delta),1e-9)/s
                mat[r,3*a:3*a+3]=grad;mat[r,3*b:3*b+3]=-grad;r+=1
            for i,_,sigma in priors:mat[r:r+3,3*i:3*i+3]=np.eye(3)/sigma;r+=3
            return mat
        fit=least_squares(residual,original.ravel(),jac=jac,loss='soft_l1',f_scale=2.,max_nfev=8)
        if not np.isfinite(fit.x).all():return
        for i,j in enumerate(js):
            point=fit.x.reshape(-1,3)[i]; q=qualities[j]
            errors=[np.linalg.norm(project(point,self.cameras[c])[0]-detection['views'][c]['keypoints_px'][j]) for c in q['inlier_cameras']]
            if max(errors)>self.cfg['ransac_threshold_px']*1.5:continue
            correction=float(np.linalg.norm(point-original[i])); points[j]=point.tolist()
            q['mean_reprojection_px']=float(np.mean(errors));q['structure_correction_m']=correction
            q['linearized_max_std_m']=max(q['linearized_max_std_m'],correction)
        row['fusion_method']='causal discrete label association + robust reprojection/shape/temporal fitting'
