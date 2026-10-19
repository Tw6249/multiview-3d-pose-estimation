"""Bounded causal beam of canonical two-joint image trajectories.

Pixel positions/velocities use each camera's image axes, time is exposure ns.
Keeps alternatives instead of feeding back only the selected 3D state. Costs
are bounded heuristics, not posterior probabilities. No camera is fabricated.
"""
import numpy as np


class AssociationHistory:
    def __init__(self, capacity=4, max_age_ms=150.):
        self.capacity=capacity;self.max_age_ns=int(max_age_ms*1e6)
        self.signature=None;self.last_time=None;self.groups={}

    def begin(self,row,calibration_sha):
        signature=tuple(row.get(k) for k in ('source_id','landmark_schema_id','perception_signature'))+(calibration_sha,tuple(row['joint_names']))
        t=row['timestamp_ns']
        if self.signature!=signature or (self.last_time is not None and t-self.last_time>self.max_age_ns):self.groups.clear()
        self.signature=signature;self.last_time=t
        self.groups={g:h for g,h in self.groups.items() if t-h['timestamp_ns']<=self.max_age_ns}

    def rank(self,group,ids,xy,confidence,original,hypotheses,threshold):
        """Return reranked hypotheses and per-view temporal costs, without commit."""
        h=self.groups.get(group);t=self.last_time;maps=[original,original[::-1]]
        samples={bits:{c:np.where((confidence[c][maps[b]]>=threshold)[:,None],xy[c][maps[b]],np.nan)
                       for c,b in zip(ids,bits)} for _,bits in hypotheses}
        if h is None:return hypotheses,np.zeros((len(ids),2)),samples,False
        dt=(t-h['timestamp_ns'])/1e9;old=h['candidates']
        # Tiny dense arrays on CPU avoid thousands of Python/NumPy dispatches
        # per frame. Axes: previous path, camera, current permutation, joint, xy.
        missing=np.full((2,2),np.nan)
        predicted=np.array([[p['xy'][c]+p['velocity'][c]*dt if c in p['xy'] else missing
                             for c in ids] for p in old])
        current=np.array([[np.where((confidence[c][maps[b]]>=threshold)[:,None],xy[c][maps[b]],np.nan)
                           for b in (0,1)] for c in ids])
        distance=np.minimum(np.linalg.norm(current[None]-predicted[:,:,None],axis=-1),50.)
        valid=np.isfinite(distance).all(axis=-1)
        cost=np.where(valid,.12*np.nan_to_num(distance).mean(axis=-1),np.inf)
        oldcost=np.array([p['cost'] for p in old])
        view_cost=np.min(np.where(valid,np.minimum(6.,cost+.25*oldcost[:,None,None]),np.inf),axis=0)
        view_cost=np.where(np.isfinite(view_cost),view_cost,0.)
        bits=np.array([b for _,b in hypotheses])
        selected=cost[:,np.arange(len(ids))[None,:],bits]  # old path, hypothesis, camera
        mask=np.isfinite(selected);count=mask.sum(axis=-1)
        sums=np.where(mask,selected,0.).sum(axis=-1)
        transitions=np.where(count>0,np.minimum(6.,sums/np.maximum(1,count)+.25*oldcost[:,None]),np.inf)
        penalty=transitions.min(axis=0);penalty=np.where(np.isfinite(penalty),penalty,0.)
        ranked=[(float(cost+penalty[i]),bits) for i,(cost,bits) in enumerate(hypotheses)]
        ranked.sort()
        return ranked,view_cost,samples,h['anchored']

    def commit(self,group,hypotheses,samples,anchored,ambiguous=False):
        # Do not refresh hypotheses with known global semantic ambiguity or
        # missing pairs. Age is measured from real usable image observations.
        if ambiguous:return
        old=self.groups.get(group); candidates=[]
        for cost,bits in hypotheses[:self.capacity]:
            points={c:p for c,p in samples[bits].items() if np.isfinite(p).all()}
            if not points:continue
            velocity={c:np.zeros_like(p) for c,p in points.items()}
            if old:
                dt=(self.last_time-old['timestamp_ns'])/1e9
                for c,p in points.items():
                    previous=[h for h in old['candidates'] if c in h['xy']]
                    if previous and dt>0:
                        match=min(previous,key=lambda h:np.linalg.norm(p-(h['xy'][c]+h['velocity'][c]*dt))+.25*h['cost'])
                        velocity[c]=np.clip((p-match['xy'][c])/dt,-2000.,2000.)
            candidates.append(dict(xy=points,velocity=velocity,cost=min(12.,cost-hypotheses[0][0])))
        if candidates:self.groups[group]=dict(timestamp_ns=self.last_time,candidates=candidates,anchored=bool(anchored))
