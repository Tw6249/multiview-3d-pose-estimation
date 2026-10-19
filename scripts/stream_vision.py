"""Paced RGB replay through the actual online processor, explicitly not camera latency."""
import argparse
from collections import deque
import json
from pathlib import Path
import sys
import threading
import time
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pose_sim.online import ExposurePacket,LatestMailbox,consume_latest
from pose_sim.sensors import ExposureFrame
from pose_sim.streaming_vision import StreamingVision
from pose_sim.pipeline import read_jsonl,write_jsonl
from pose_sim.motion_evaluation import statistics
from motion_control.perception_gate import PerceptionGate


class ReplayDecodeBudget:
    """Skip obsolete file frames before decoding, without changing exposure ns.

    Last decode duration reserves up to half the start-age budget. The cap lets
    the source try fresh frames again after a transient disk stall. The consumer
    still enforces actual age; this estimate never certifies an output as fresh.
    """
    def __init__(self,maximum_age_ms):
        if not np.isfinite(maximum_age_ms) or maximum_age_ms<=0:raise ValueError('Positive source age budget required')
        self.maximum_ns=round(maximum_age_ms*1e6);self.reserve_ns=0

    def should_decode(self,exposure_ns,now_ns):
        return now_ns-exposure_ns+self.reserve_ns<=self.maximum_ns

    def record_decode(self,duration_ns):
        if duration_ns<0:raise ValueError('Negative decode duration')
        self.reserve_ns=min(duration_ns,self.maximum_ns//2)


def run(source,output,workers=None,fps=None,pose_workers=None,legacy_details=False,save_states=True,drop_expired_source=True):
    source=Path(source);output=Path(output)
    if output.exists():raise FileExistsError('Preserve existing evidence')
    output.mkdir(parents=True)
    cal=json.loads((source/'cameras.json').read_text());cfg=json.loads((source/'config.json').read_text())
    profile=json.loads((ROOT/'config/vision.json').read_text())
    settings=json.loads((ROOT/'config/vision_motion.json').read_text())
    workers=settings['online'].get('detector_workers',1) if workers is None else workers
    pose_workers=settings['online'].get('pose_workers',1) if pose_workers is None else pose_workers
    if legacy_details:
        settings['auxiliary_association']=False
        settings['state']['auxiliary_consistency']=False
    processor=StreamingVision(cal,cfg,profile,state_config=settings['state'],detector_workers=workers,pose_workers=pose_workers,fusion_config=settings)
    packets=list(read_jsonl(source/'frames.jsonl'));mailbox=LatestMailbox();errors=[];rows=[];gate=PerceptionGate()
    cv2.setNumThreads(1)
    source_metrics=dict(skipped_before_decode=0,decoded_frames=0,decode_ms=deque(maxlen=4096),start_lateness_ms=deque(maxlen=4096))
    deliveries=dict(fresh=0,follow=0)
    budget=ReplayDecodeBudget(settings['online']['max_start_age_ms'])
    # Disk decoding occurs in the producer. It is part of measured replay latency.
    def produce():
        origin=time.perf_counter_ns()+50_000_000;first=packets[0]['timestamp_ns']
        try:
            for i,row in enumerate(packets):
                target=origin+(round(i/fps*1e9) if fps else row['timestamp_ns']-first)
                remaining=(target-time.perf_counter_ns())/1e9
                if remaining>0:time.sleep(remaining)
                oldest=target+min(v['timestamp_ns']-row['timestamp_ns'] for v in row['views'].values())
                decode_begin=time.perf_counter_ns()
                if drop_expired_source and not budget.should_decode(oldest,decode_begin):
                    source_metrics['skipped_before_decode']+=1;continue
                source_metrics['start_lateness_ms'].append((decode_begin-oldest)/1e6)
                frames={}
                for c,v in row['views'].items():
                    image=cv2.imdecode(np.fromfile(source/v['image'],np.uint8),cv2.IMREAD_COLOR)
                    if image is None:raise ValueError('Unreadable replay RGB')
                    exposure=target+(v['timestamp_ns']-row['timestamp_ns'])
                    frames[c]=ExposureFrame(c,exposure,time.perf_counter_ns(),'host_monotonic',image)
                decoded_ns=time.perf_counter_ns()-decode_begin;budget.record_decode(decoded_ns)
                source_metrics['decode_ms'].append(decoded_ns/1e6)
                source_metrics['decoded_frames']+=1
                packet=ExposurePacket(row['frame_id'],min(v.exposure_timestamp_ns for v in frames.values()),frames)
                mailbox.submit(packet)
        except BaseException as e:errors.append(e)
        finally:mailbox.close()
    def receive(row):
        row['planner_gate']=gate.update(row,time.perf_counter_ns())
        deliveries['fresh']+=bool(row['delivery']['valid_for_new_motion'])
        deliveries['follow']+=row['planner_gate']['mode']=='follow'
        if save_states:rows.append(row)
    producer=threading.Thread(target=produce,name='paced-rgb-source',daemon=True)
    begin=time.perf_counter();producer.start()
    try:stats=consume_latest(mailbox,processor.process,receive,
        max_start_age_ms=settings['online']['max_start_age_ms'],max_output_age_ms=settings['online']['max_output_age_ms'])
    finally:
        mailbox.close();producer.join(timeout=5);processor.close()
    if errors:raise errors[0]
    stats['wall_seconds']=time.perf_counter()-begin
    stats['latency_ms']=statistics(stats['latency_ms']);stats['processing_ms']=statistics(stats['processing_ms'])
    stats['source']=dict(input_frames=len(packets),drop_expired_before_decode=drop_expired_source,
        skipped_before_decode=source_metrics['skipped_before_decode'],
        decoded_frames=source_metrics['decoded_frames'],metrics_scope='last 4096 decoded frames',
        decode_ms=statistics(source_metrics['decode_ms']),start_lateness_ms=statistics(source_metrics['start_lateness_ms']))
    if mailbox.submitted+source_metrics['skipped_before_decode']!=len(packets):raise RuntimeError('Source accounting mismatch')
    stats.update(measurement_kind='paced_file_replay_with_host_scheduled_exposure',physical_exposure_latency_verified=False,
        source_rate_hz=fps or cfg.get('fps'),detector_workers=workers,pose_workers=pose_workers,runtime=processor.runtime,
        legacy_details=legacy_details,association_history=settings.get('association_history',False),
        stage_processing_ms={k:statistics([r[k] for r in processor.stage_timings]) for k in ('detector_ms','pose_ms','person_selection_ms','fusion_ms','state_ms')},
        warmup_seconds=processor.warmup_seconds,queue_capacity_packets=1,
        output_deadline_ms=settings['online']['max_output_age_ms'],
        limitation='Real exposure and synchronization require SDK clock mapping; replay measures actual computation and scheduling only.',
        valid_deliveries=deliveries['fresh'],planner_follow_count=deliveries['follow'],
        state_rate_hz=stats['processed']/stats['wall_seconds'])
    if save_states:write_jsonl(output/'states3d.jsonl',rows)
    stats['states_saved']=save_states
    (output/'timing.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in stats.items() if k!='runtime'},indent=2));return stats


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--detector-workers',type=int,choices=(1,2,4));p.add_argument('--fps',type=float)
    p.add_argument('--pose-workers',type=int,choices=(1,2,4))
    p.add_argument('--legacy-details',action='store_true',help='Disable detail association and attachment checks for paired timing')
    p.add_argument('--no-states',action='store_true',help='Keep compact timing evidence only')
    p.add_argument('--keep-expired-source',action='store_true',help='Historical replay backlog ablation only')
    a=p.parse_args()
    if a.fps is not None and a.fps<=0:p.error('fps must be positive')
    run(a.source,a.output,a.detector_workers,a.fps,a.pose_workers,a.legacy_details,not a.no_states,not a.keep_expired_source)
