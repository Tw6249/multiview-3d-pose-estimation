"""Offline regularization of source-derived references, never perception truth.

Uses future accepted estimates within each uninterrupted segment. Not an online
filter, new observation, contact estimator or hardware command. Original source
references are retained so evaluation cannot improve just by changing its target.
Joint paths use a third-difference penalty; root velocity is differentiated from
the same smoothed pelvis path and heading. Missing/rejected frames remain invalid.
"""
import copy
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.sparse import eye,diags
from scipy.sparse.linalg import spsolve


def retime(rows,scale):
    """Uniform offline time dilation, preserving limb phase and pelvis path.

    The new clock is explicitly a replay clock, not refreshed camera exposure.
    All source-derived translation/yaw velocities scale with the same factor.
    """
    if not np.isfinite(scale) or scale<1:raise ValueError('Offline slowdown factor must be >=1')
    if not rows:return []
    out=copy.deepcopy(rows);start=rows[0]['timestamp_ns']
    for original,row in zip(rows,out):
        row['original_exposure_timestamp_ns']=original.get('original_exposure_timestamp_ns',original['timestamp_ns'])
        row['timestamp_ns']=start+round((original['timestamp_ns']-start)*scale)
        row['reference_clock_domain']='retimed_offline_reference'
        row['offline_time_scale']=scale
        if row['accepted']:
            for key in ('reference','source_reference'):
                if key in row:
                    for j in (0,1,5):row[key][j]/=scale
            if 'offline_reference_rate_per_s' in row:
                row['offline_reference_rate_per_s']=(np.array(row['offline_reference_rate_per_s'])/scale).tolist()
    return out


def regularize(rows,joint_limits,weight=.08,max_gap_s=.12):
    if not np.isfinite(weight) or weight<0:raise ValueError('Invalid smoothing weight')
    out=copy.deepcopy(rows);limits=np.asarray(joint_limits)
    if limits.shape!=(29,2):raise ValueError('Require ordered target joint limits')
    segments=[];current=[];last=None
    for i,row in enumerate(rows):
        t=row['timestamp_ns']
        if type(t) is not int or (last is not None and t<=last):raise ValueError('Nonmonotonic source reference')
        if (not row['accepted']) or (last is not None and (t-last)/1e9>max_gap_s):
            if current:segments.append(current)
            current=[]
        if row['accepted']:current.append(i)
        last=t
    if current:segments.append(current)
    for ids in segments:
        for i in ids:
            out[i]['source_reference']=copy.deepcopy(rows[i].get('source_reference',rows[i]['reference']))
            out[i]['source_root_position_m']=copy.deepcopy(rows[i].get('source_root_position_m',rows[i]['root_position_m']))
            out[i]['source_root_heading_rad']=rows[i].get('source_root_heading_rad',rows[i]['root_heading_rad'])
            out[i]['offline_trajectory']=dict(uses_future_estimates=True,source_time_unchanged=True,weight=weight,
                contact_constraints=False,hardware_enabled=False)
        if len(ids)<5:continue
        stamps=np.array([(rows[i]['timestamp_ns']-rows[ids[0]]['timestamp_ns'])/1e9 for i in ids])
        intervals=np.diff(stamps)
        if not np.allclose(intervals,np.median(intervals),atol=1e-6):raise ValueError('Offline regularizer requires a uniform source clock within segments')
        values=np.array([rows[i]['reference'] for i in ids]);root=np.array([rows[i]['root_position_m'] for i in ids]);heading=np.unwrap([rows[i]['root_heading_rad'] for i in ids])
        n=len(ids);d3=diags([-np.ones(n-3),3*np.ones(n-3),-3*np.ones(n-3),np.ones(n-3)],[0,1,2,3],shape=(n-3,n),format='csc')
        # Scale regularization with source rate so the cutoff is rate independent.
        penalty=weight*(1/(24*np.median(intervals)))**6
        system=eye(n,format='csc')+penalty*(d3.T@d3)
        smooth=spsolve(system,values[:,6:]);path=spsolve(system,root);yaw=spsolve(system,heading)
        # If a regularized path leaves a model limit, retain that original axis.
        bad=((smooth<limits[:,0])|(smooth>limits[:,1])).any(axis=0)
        smooth[:,bad]=values[:,6:][:,bad]
        q_spline=CubicSpline(stamps,smooth);path_spline=CubicSpline(stamps,path);yaw_spline=CubicSpline(stamps,yaw)
        for k,i in enumerate(ids):
            ref=values[k].copy();ref[6:]=smooth[k];world_velocity=path_spline(stamps[k],1)[:2]
            c,s=np.cos(yaw[k]),np.sin(yaw[k]);rotation=np.array([[c,-s],[s,c]])
            ref[:2]=world_velocity@rotation;ref[5]=float(yaw_spline(stamps[k],1))
            if np.linalg.norm(ref[:2])>1.5 or abs(ref[5])>2.5:
                # Do not smuggle an infeasible root request into a new trajectory.
                out[i]['accepted']=False;out[i]['reason']='regularized root exceeds imitation profile';continue
            out[i]['reference']=ref.tolist();out[i]['root_position_m']=path[k].tolist();out[i]['root_heading_rad']=float(yaw[k])
            out[i]['offline_reference_rate_per_s']=np.r_[np.zeros(6),q_spline(stamps[k],1)].tolist()
    return out
