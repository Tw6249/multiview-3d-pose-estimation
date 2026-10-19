"""Select ambiguous detector instances using independent current camera evidence.

Input: named pixel candidates, calibrated world-to-camera transforms, exposure ns.
Output: copied observations with one candidate selected, or the original missing
view. Never averages people, changes joint labels, uses history or reads truth.
Requires >=4 anchors across upper and lower body, low residual and a unique match.
Pixel thresholds are heuristics, not identity probabilities. Single-person scope:
the other cameras must already contain a unique, geometrically consistent person.
"""
import copy
import numpy as np
from .geometry import CameraGeometry, arrays, project, triangulate


class CrossViewPersonSelector:
    def __init__(self, cameras, config):
        self.cameras, self.cfg = cameras, config
        self.cache = {}

    def update(self, row):
        out=copy.deepcopy(row);names=row['joint_names'];index={n:i for i,n in enumerate(names)}
        threshold=self.cfg['confidence_threshold'];sync=self.cfg['max_sync_error_ms']*1e6
        # Only original unique views are anchors. A newly selected candidate
        # cannot certify another ambiguous view in the same packet.
        anchors={c:v for c,v in row['views'].items() if c in self.cameras
                 and v.get('num_people_detected')==1 and v.get('keypoints_px') is not None
                 and abs(v['timestamp_ns']-row['timestamp_ns'])<=sync}
        evidence=[]
        for cid,view in row['views'].items():
            candidates=view.get('person_candidates',[])
            if (cid not in self.cameras or view.get('num_people_detected',0)<=1
                    or not 2<=len(candidates)<=4 or abs(view['timestamp_ns']-row['timestamp_ns'])>sync):continue
            projections={}
            for name in (s+'_'+p for p in ('shoulder','hip','knee') for s in ('left','right')):
                if name not in index:continue
                j=index[name]
                ids=tuple(c for c,v in anchors.items() if c!=cid and v['scores'][j]>=threshold
                          and np.isfinite(v['keypoints_px'][j]).all())
                if len(ids)<2:continue
                if ids not in self.cache:self.cache[ids]=CameraGeometry([self.cameras[c] for c in ids])
                fit=triangulate([self.cameras[c] for c in ids],[anchors[c]['keypoints_px'][j] for c in ids],
                    [anchors[c]['scores'][j] for c in ids],self.cfg['ransac_threshold_px'],2.,self.cache[ids],'newton')
                if fit is None or fit['linearized_max_std_m']>.06 or fit['mean_reprojection_px']>4:continue
                point=np.array(fit['xyz_m']);_,r,t,_=arrays(self.cameras[cid]);xy=project(point,self.cameras[cid])[0]
                cam=self.cameras[cid]
                if (r@point+t)[2]>0 and 0<=xy[0]<cam['width'] and 0<=xy[1]<cam['height']:
                    projections[name]=xy
            scores=[]
            for ci,candidate in enumerate(candidates):
                xy=candidate.get('keypoints_px');confidence=candidate.get('scores');residuals=[];parts=[]
                if xy is None or confidence is None:continue
                for part in ('shoulder','hip','knee'):
                    pair=[s+'_'+part for s in ('left','right')]
                    if not all(n in projections and confidence[index[n]]>=threshold and
                               np.isfinite(xy[index[n]]).all() for n in pair):continue
                    a=np.array([projections[n] for n in pair]);b=np.array([xy[index[n]] for n in pair])
                    direct=np.linalg.norm(a-b,axis=1);swapped=np.linalg.norm(a-b[::-1],axis=1)
                    residuals.extend(direct if direct.sum()<=swapped.sum() else swapped);parts.append(part)
                if len(residuals)<4 or 'shoulder' not in parts or not ({'hip','knee'} & set(parts)):continue
                scores.append(dict(candidate=ci,median_px=float(np.median(residuals)),
                                   p80_px=float(np.percentile(residuals,80)),anchors=len(residuals)))
            scores.sort(key=lambda v:v['median_px'])
            selected=None
            if scores and scores[0]['median_px']<=20 and scores[0]['p80_px']<=35:
                if len(scores)==1 or scores[1]['median_px']-scores[0]['median_px']>=15:
                    selected=scores[0]['candidate']
                    out['views'][cid].update(copy.deepcopy(candidates[selected]))
            evidence.append(dict(camera=cid,selected_candidate=selected,candidates=scores,
                                 reason='unique cross-view body match' if selected is not None else 'insufficient or ambiguous current evidence'))
        out['person_selection']=dict(evidence=evidence,uses_history=False,calibrated_probability=False)
        return out
