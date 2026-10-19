"""Check complete live CUDA inference against frozen sparse candidate replay."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from pose_sim.pipeline import read_jsonl
from pose_sim.streaming_vision import StreamingVision
from pose_sim.person_selection import CrossViewPersonSelector
from scripts.evaluate_mpi3dhp import packets
from scripts.benchmark_person_selection import fingerprint
from scripts.import_mpi3dhp import sha256


def run(folder,detections,candidates,output,start=64,end=78):
    if output.exists():raise FileExistsError('Preserve integration evidence')
    meta=next(c for c in json.loads((candidates.parent/'cache.json').read_text())['clips'] if c['id']==folder.name)
    if meta['source_sha256']!=sha256(detections) or meta['candidates_sha256']!=sha256(candidates):raise ValueError('Frozen cache changed')
    cal=json.loads((folder/'cameras.json').read_text());cfg=json.loads((ROOT/'config/demo.json').read_text())
    profile=json.loads((ROOT/'config/vision.json').read_text());base={r['frame_id']:r for r in read_jsonl(detections)}
    sparse={r['frame_id']:r for r in read_jsonl(candidates)};cv2.setNumThreads(1)
    processor=StreamingVision(cal,cfg,profile,detector_workers=4,pose_workers=4)
    selector=CrossViewPersonSelector(cal['cameras'],cfg);results=[];samples=[];score_difference=0.;point_difference=0.
    try:
        for packet in packets(folder):
            if packet.frame_id<start:continue
            if packet.frame_id>end:break
            expected=copy.deepcopy(base[packet.frame_id])
            for c,values in sparse.get(packet.frame_id,{}).get('views',{}).items():expected['views'][c]['person_candidates']=values
            expected=selector.update(expected)
            begin=time.perf_counter();actual=processor.detect(packet);elapsed=(time.perf_counter()-begin)*1000
            samples.append(dict(**processor.last_detection_timing,total_ms=elapsed))
            selections=lambda r:{e['camera']:e['selected_candidate'] for e in r['person_selection']['evidence']}
            if selections(expected)!=selections(actual):raise AssertionError('Identity choice differs from frozen cache')
            for c,view in expected['views'].items():
                a=actual['views'][c]
                if (view['keypoints_px'] is None)!=(a['keypoints_px'] is None):raise AssertionError('Visibility changed')
                if view['keypoints_px'] is not None:
                    np.testing.assert_allclose(a['keypoints_px'],view['keypoints_px'],atol=1e-4,rtol=0)
                    np.testing.assert_allclose(a['scores'],view['scores'],atol=1e-6,rtol=0)
                    np.testing.assert_array_equal(np.array(a['scores'])>=cfg['confidence_threshold'],
                                                  np.array(view['scores'])>=cfg['confidence_threshold'])
                    score_difference=max(score_difference,float(np.max(np.abs(np.array(a['scores'])-view['scores']))))
                    point_difference=max(point_difference,float(np.max(np.abs(np.array(a['keypoints_px'])-view['keypoints_px']))))
            results.append(dict(frame_id=packet.frame_id,selection=selections(actual)))
        if len(results)!=end-start+1:raise ValueError('Incomplete integration interval')
        report=dict(passed=True,frames=len(results),selection_keypoint_visibility_parity=True,keypoint_tolerance_px=1e-4,
            maximum_keypoint_difference_px=point_difference,maximum_score_difference=score_difference,
            body_score_tolerance=1e-6,detail_score_tolerance=1e-6,
            fingerprint=fingerprint(),source_detections_sha256=sha256(detections),candidates_sha256=sha256(candidates),
            timing_ms={k:dict(mean=float(np.mean([s[k] for s in samples])),p95=float(np.percentile([s[k] for s in samples],95))) for k in samples[0]},
            results=results,runtime=processor.runtime,scope='Selected development packets only; decode/exposure excluded; not a capture latency or full throughput benchmark')
        output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print({k:report[k] for k in ('passed','frames','timing_ms')},flush=True)
    finally:processor.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--detections',type=Path,required=True)
    p.add_argument('--candidates',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start',type=int,default=64);p.add_argument('--end',type=int,default=78)
    a=p.parse_args();run(a.input,a.detections,a.candidates,a.output,a.start,a.end)
