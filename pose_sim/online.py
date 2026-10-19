"""Bounded acquisition/scheduling primitives with actual monotonic timestamps.

SDK adapters must map EXPOSURE timestamps to host_monotonic before submit.
Receive timestamps cannot substitute for exposure. A replay clock is labelled
separately by callers and cannot prove physical camera latency.
"""
from collections import deque
from dataclasses import dataclass
from threading import Condition
import time
from .sensors import ExposureFrame


@dataclass(frozen=True)
class ExposurePacket:
    frame_id: int
    timestamp_ns: int
    frames: dict[str,ExposureFrame]
    clock_domain: str = 'host_monotonic'

    def validate(self,max_skew_ns=5_000_000):
        if type(self.timestamp_ns) is not int or len(self.frames)<2:raise ValueError('Need a timestamp and two views')
        stamps=[]
        for cid,frame in self.frames.items():
            frame.validate()
            if cid!=frame.camera_id or frame.clock_domain!=self.clock_domain:raise ValueError('Mixed exposure clock or camera identity')
            stamps.append(frame.exposure_timestamp_ns)
        if min(stamps)!=self.timestamp_ns or max(stamps)-min(stamps)>max_skew_ns:
            raise ValueError('Packet must use earliest exposure and bounded inter-camera skew')


class LatestMailbox:
    """One pending packet; replacing old work bounds memory and queue latency."""
    def __init__(self):
        self.condition=Condition();self.pending=None;self.closed=False;self.replaced=0;self.submitted=0;self.last_stamp=None

    def submit(self,packet):
        packet.validate()
        with self.condition:
            if self.closed:raise RuntimeError('Mailbox closed')
            if self.last_stamp is not None and packet.timestamp_ns<=self.last_stamp:raise ValueError('Packet time must increase')
            self.last_stamp=packet.timestamp_ns;self.submitted+=1
            self.replaced+=self.pending is not None;self.pending=packet;self.condition.notify()

    def get(self,timeout=.1):
        with self.condition:
            if self.pending is None and not self.closed:self.condition.wait(timeout)
            result=self.pending;self.pending=None;return result

    def close(self):
        with self.condition:self.closed=True;self.condition.notify_all()


class ExposureSynchronizer:
    def __init__(self,camera_ids,max_skew_ns=5_000_000,max_age_ns=100_000_000,capacity=3):
        self.buffers={c:deque(maxlen=capacity) for c in camera_ids};self.last={}
        self.max_skew_ns=max_skew_ns;self.max_age_ns=max_age_ns;self.packet_id=0
        self.dropped=0

    def push(self,frame,now_ns):
        frame.validate()
        if frame.clock_domain!='host_monotonic' or frame.camera_id not in self.buffers:raise ValueError('Unmapped or unknown camera')
        if frame.arrival_timestamp_ns>now_ns:raise ValueError('Future arrival clock')
        last=self.last.get(frame.camera_id)
        if last is not None and frame.exposure_timestamp_ns<=last:raise ValueError('Reused/nonmonotonic camera frame')
        self.last[frame.camera_id]=frame.exposure_timestamp_ns
        if now_ns-frame.exposure_timestamp_ns>self.max_age_ns:self.dropped+=1;return None
        buf=self.buffers[frame.camera_id]
        if len(buf)==buf.maxlen:self.dropped+=1
        buf.append(frame)
        for q in self.buffers.values():
            while q and now_ns-q[0].exposure_timestamp_ns>self.max_age_ns:q.popleft();self.dropped+=1
        if any(not q for q in self.buffers.values()):return None
        anchor=min(q[-1].exposure_timestamp_ns for q in self.buffers.values())
        chosen={c:min(q,key=lambda f:abs(f.exposure_timestamp_ns-anchor)) for c,q in self.buffers.items()}
        times=[f.exposure_timestamp_ns for f in chosen.values()]
        if max(times)-min(times)>self.max_skew_ns:return None
        for c,f in chosen.items():
            while self.buffers[c]:
                old=self.buffers[c].popleft()
                if old is f:break
                self.dropped+=1
        packet=ExposurePacket(self.packet_id,min(times),chosen);self.packet_id+=1;packet.validate(self.max_skew_ns)
        return packet


def consume_latest(mailbox,process,on_result,max_start_age_ms=100.,max_output_age_ms=180.,metrics_capacity=4096):
    if type(metrics_capacity) is not int or metrics_capacity<1:raise ValueError('Metrics capacity must be a positive integer')
    stats=dict(processed=0,stale_before=0,stale_after=0,
               latency_ms=deque(maxlen=metrics_capacity),processing_ms=deque(maxlen=metrics_capacity))
    while True:
        packet=mailbox.get()
        if packet is None:
            if mailbox.closed:break
            continue
        begin=time.perf_counter_ns()
        if packet.clock_domain!='host_monotonic':raise ValueError('Output latency requires the host monotonic clock domain')
        age=begin-packet.timestamp_ns
        if age<0:raise ValueError('Future exposure')
        if age>max_start_age_ms*1e6:stats['stale_before']+=1;continue
        result=process(packet);end=time.perf_counter_ns()
        latency=(end-packet.timestamp_ns)/1e6;fresh=latency<=max_output_age_ms
        stats['processed']+=1;stats['stale_after']+=not fresh
        stats['latency_ms'].append(latency);stats['processing_ms'].append((end-begin)/1e6)
        result['delivery']=dict(exposure_timestamp_ns=packet.timestamp_ns,output_timestamp_ns=end,
            exposure_to_output_ms=latency,valid_for_new_motion=fresh,clock_domain=packet.clock_domain)
        on_result(result)
    stats['submitted']=mailbox.submitted;stats['replaced']=mailbox.replaced
    stats['metrics_window']=dict(capacity=metrics_capacity,samples=len(stats['latency_ms']),
                                truncated=stats['processed']>metrics_capacity,scope='most recent processed packets')
    stats['latency_ms']=list(stats['latency_ms']);stats['processing_ms']=list(stats['processing_ms'])
    return stats
