"""Audit recorded imitation discontinuities without changing perception/control."""
import argparse
from collections import Counter
import json
from pathlib import Path
import cv2
import numpy as np


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def diagnose(session):
    references=read_rows(session/'humanoid_reference.jsonl')
    feedback=read_rows(session/'imitation/feedback.jsonl')
    poses=read_rows(session/'poses3d.jsonl')
    timing=json.loads((session/'inference_timing.json').read_text())
    metrics=json.loads((session/'metrics.json').read_text())
    config=json.loads((session/'config.json').read_text())
    names=references[0]['joint_names']
    times=np.array([row['time_s'] for row in feedback])
    tracking=np.array([row['state']=='tracking' for row in feedback])
    first,last=np.flatnonzero(tracking)[[0,-1]]
    interrupted=[]
    for k in range(first,last+1):
        if feedback[k]['state']!='vision_unavailable':continue
        if not interrupted or k!=interrupted[-1]['last_tick']+1:
            interrupted.append(dict(start_s=float(times[k]),end_s=float(times[k]+.02),last_tick=k))
        else:interrupted[-1].update(end_s=float(times[k]+.02),last_tick=k)
    for row in interrupted:row.pop('last_tick')
    valid=[r for r in references if r['accepted']]
    jumps=[]
    for a,b in zip(valid,valid[1:]):
        d=np.abs(np.array(b['reference'][6:])-a['reference'][6:]);j=int(np.argmax(d))
        jumps.append(dict(from_frame=a['frame_id'],to_frame=b['frame_id'],joint=names[j],
            change_deg=float(np.degrees(d[j])),interval_ms=(b['timestamp_ns']-a['timestamp_ns'])/1e6,
            arm_observed_before=a['arm_observed'],arm_observed_after=b['arm_observed']))
    adjacent=[x for x in jumps if x['to_frame']==x['from_frame']+1]
    lengths=[]
    for row in poses:
        a,b=[row['joints_world_m'][row['joint_names'].index('right_'+part)] for part in ('knee','ankle')]
        lengths.append(float(np.linalg.norm(np.array(a)-b)) if a is not None and b is not None else np.nan)
    video=cv2.VideoCapture(str(session/'imitation/preview.mp4'))
    fps=video.get(cv2.CAP_PROP_FPS);pts=[]
    try:
        while video.grab():pts.append(video.get(cv2.CAP_PROP_POS_MSEC))
    finally:video.release()
    intervals=np.diff(pts)
    report=dict(scope='Recorded data audit; does not measure UI playback rendering or establish real-world pose accuracy',
        video=dict(fps=fps,frames=len(pts),timestamp_interval_min_ms=float(intervals.min()),
                   timestamp_interval_max_ms=float(intervals.max()),irregular_timestamp_count=int(np.sum(abs(intervals-1000/fps)>.1))),
        source_fps=config['fps'],serial_vision_packet_ms=timing['mean_seconds_per_packet_with_image_io']*1000,
        serial_vision_groups_per_second=1/timing['mean_seconds_per_packet_with_image_io'],
        accepted_frames=len(valid),source_frames=len(references),
        rejected_reasons=dict(Counter(r['reason'] for r in references if not r['accepted'])),
        within_motion_fallback_windows=interrupted,within_motion_fallback_count=len(interrupted),
        within_motion_fallback_total_s=sum(x['end_s']-x['start_s'] for x in interrupted),
        left_arm_unobserved_valid_frames=sum(not r['arm_observed'][0] for r in valid),
        right_arm_unobserved_valid_frames=sum(not r['arm_observed'][1] for r in valid),
        largest_adjacent_reference_change=max(adjacent,key=lambda r:r['change_deg']),
        right_shank_length_p05_median_p95_m=np.nanpercentile(lengths,[5,50,95]).tolist(),
        right_shank_length_coefficient_of_variation=float(np.nanstd(lengths)/np.nanmean(lengths)),
        body12_proxy_mean_error_mm=metrics['body12_proxy_mpjpe_mm'],
        body12_proxy_p95_error_mm=metrics['body12_proxy_p95_mm'],
        proxy_error_caveat='Stylized render; rig joints differ from COCO landmark definitions. Not real human accuracy.',
        findings=['Encoded frame timestamps are regular; UI playback performance remains unmeasured.',
                  'Invalid frames immediately request default standing, causing repeated withdraw/reacquire transitions.',
                  'Arm observation loss/recovery and ambiguous IK cause large reference angle changes.',
                  'Reference update limits per-tick angle only, not acceleration or jerk continuity.',
                  'Serial GPU perception throughput is below the source rate; offline replay hides this scheduling deficit.',
                  'COCO17 does not observe complete foot/hand orientation or uniquely separate pelvis/waist tilt.'])
    out=session/'imitation'
    (out/'diagnostics.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True,layout='constrained')
    start=2.12
    source_times=np.array([r['timestamp_ns'] for r in references])/1e9+start
    axes[0].plot(source_times,lengths,color='#b53b43',label='Reconstructed right lower-leg length')
    axes[0].axhline(np.nanmedian(lengths),color='gray',ls=':',label='Median (length should remain stable)')
    axes[0].set_ylabel('Length (m)');axes[0].legend(loc='upper left',fontsize=9)
    for axis,joint in zip(axes[1:],('left_shoulder_yaw_joint','left_knee_joint')):
        j=names.index(joint)
        raw=[np.degrees(r['reference'][j+6]) if r['accepted'] else np.nan for r in references]
        axis.plot(source_times,raw,marker='.',markersize=3,lw=1,label='Retargeted observation')
        axis.plot(times,np.degrees([r['reference'][j+6] for r in feedback]),lw=1,label='Bounded policy reference')
        axis.plot(times,np.degrees([r['position_rad'][j] for r in feedback]),lw=1,label='Physics feedback')
        axis.set_ylabel(joint.replace('_joint','').replace('_',' ')+' (deg)')
        axis.legend(loc='upper left',fontsize=9,ncol=3)
    for axis in axes:
        for window in interrupted:axis.axvspan(window['start_s'],window['end_s'],color='red',alpha=.13)
        axis.grid(alpha=.2);axis.set_xlim(2.1,6.1)
    axes[-1].set_xlabel('Video time (s); red bands = visual input withdrawn / transition toward stand')
    fig.suptitle('Imitation stutter audit: vision geometry, retargeting, controller continuity',fontsize=13)
    fig.savefig(out/'diagnostics.png',dpi=150);plt.close(fig)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('session',type=Path)
    report=diagnose(parser.parse_args().session)
    print(json.dumps(report,indent=2))
