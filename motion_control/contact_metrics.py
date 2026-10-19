"""Evaluation-only contact timing; never imported by perception or controllers.

Robot force (N) -> debounced load transitions on integer ROBOT ns. Source
landing predictions -> contiguous prediction runs, not measured human steps.
Offline one-to-one matching uses explicitly mapped robot-clock windows, with
no tolerance widening. Unknown/gaps cannot generate touchdown. Initial loaded
feet are not touchdowns. Event onset and confirmation timestamps are separate.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


class FootLoadEvents:
    def __init__(self,enter_fraction=.05,exit_fraction=.025,confirmation_frames=2,maximum_gap_s=.05):
        if not 0<exit_fraction<enter_fraction<1 or type(confirmation_frames) is not int or confirmation_frames<2 or not np.isfinite(maximum_gap_s) or maximum_gap_s<=0:
            raise ValueError('Invalid load transition parameters')
        self.enter=enter_fraction;self.exit=exit_fraction;self.confirmation=confirmation_frames;self.gap=maximum_gap_s
        self.last=None;self.memory={};self.events=[]

    def update(self,timestamp_ns,forces_n,weight_n):
        if type(timestamp_ns) is not int or (self.last is not None and timestamp_ns<=self.last):raise ValueError('Increasing robot ns required')
        if self.last is not None and (timestamp_ns-self.last)/1e9>self.gap:self.memory.clear()
        self.last=timestamp_ns;out={}
        for side in ('left','right'):
            f=forces_n.get(side)
            if not isinstance(f,(int,float)) or not np.isfinite(f) or f<0 or not np.isfinite(weight_n) or weight_n<=0:
                self.memory.pop(side,None);out[side]='unknown';continue
            old=self.memory.get(side,dict(state='unknown',candidate=None,count=0,since=timestamp_ns))
            candidate='loaded' if f>=self.enter*weight_n else 'unloaded' if f<=self.exit*weight_n else old['state']
            count=old['count']+1 if candidate==old['candidate'] else 1
            since=old['since'] if candidate==old['candidate'] else timestamp_ns
            state=old['state']
            if count>=self.confirmation:
                if state=='unloaded' and candidate=='loaded':
                    self.events.append(dict(side=side,timestamp_ns=since,confirmed_timestamp_ns=timestamp_ns))
                state=candidate
            self.memory[side]=dict(state=state,candidate=candidate,count=count,since=since)
            out[side]=state
        return out


class PredictionRuns:
    """Group consecutive prediction updates; never assert these are gait cycles."""
    def __init__(self,maximum_gap_s=.12):
        if not np.isfinite(maximum_gap_s) or maximum_gap_s<=0:raise ValueError('Invalid source gap')
        self.gap=maximum_gap_s;self.last=None;self.active=None;self.runs=[]

    def update(self,timestamp_ns,event):
        if type(timestamp_ns) is not int or (self.last is not None and timestamp_ns<=self.last):raise ValueError('Increasing source ns required')
        gap=self.last is not None and (timestamp_ns-self.last)/1e9>self.gap;self.last=timestamp_ns
        if event is None:self.active=None;return None
        if event['side'] not in ('left','right'):raise ValueError('Invalid side')
        if self.active is None or gap or self.active['side']!=event['side']:
            self.active=dict(run_id=len(self.runs),side=event['side'],first_source_ns=timestamp_ns,
                last_source_ns=timestamp_ns,updates=0,first_prediction=dict(event))
            self.runs.append(self.active)
        self.active['updates']+=1;self.active['last_source_ns']=timestamp_ns
        return self.active['run_id']


def match_landing_windows(windows,touchdowns):
    """Max-cardinality one-to-one match, then minimum nominal-time distance.

Only same-side touchdowns inside the ORIGINAL first-prediction window and at
or after its receipt can match. No union of later windows, event reuse, nearest
event outside bounds, or replacement of exposure by confirmation timestamp.
All inputs are mapped robot-clock integer ns; a missing match remains missing.
"""
    for w in windows:
        if w['side'] not in ('left','right') or any(type(w[k]) is not int for k in ('earliest_ns','latest_ns','nominal_ns','received_ns')) or not w['earliest_ns']<=w['nominal_ns']<=w['latest_ns']:
            raise ValueError('Invalid mapped source window')
    for t in touchdowns:
        if t['side'] not in ('left','right') or type(t['timestamp_ns']) is not int:raise ValueError('Invalid robot touchdown')
    n,m=len(windows),len(touchdowns);penalty=n+1.;cost=np.full((n,m+n),penalty)
    valid=np.zeros((n,m),bool)
    for i,w in enumerate(windows):
        for j,t in enumerate(touchdowns):
            valid[i,j]=w['side']==t['side'] and max(w['earliest_ns'],w['received_ns'])<=t['timestamp_ns']<=w['latest_ns']
            if valid[i,j]:cost[i,j]=abs(t['timestamp_ns']-w['nominal_ns'])/max(1,w['latest_ns']-w['earliest_ns'])
    matches=[]
    if n:
        ii,jj=linear_sum_assignment(cost)
        for i,j in zip(ii,jj):
            if j<m and valid[i,j]:matches.append(dict(window_index=int(i),touchdown_index=int(j),
                nominal_error_ms=(touchdowns[j]['timestamp_ns']-windows[i]['nominal_ns'])/1e6))
    return dict(windows=n,touchdowns=m,matches=matches,unmatched_windows=n-len(matches),
                unmatched_robot_touchdowns=m-len(matches),scope='Source hypotheses versus simulated robot load; not human contact accuracy')
