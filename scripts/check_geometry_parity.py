"""Compare the CPU solver with the SciPy reference on identical observations."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pose_sim.pipeline import read_jsonl,reconstruct_record
from pose_sim.motion_evaluation import statistics


def run(source,detections,output):
    source,detections,output=map(Path,(source,detections,output))
    if output.exists():raise FileExistsError(output)
    cams=json.loads((source/'cameras.json').read_text())['cameras'];cfg=json.loads((source/'config.json').read_text())
    rows=list(read_jsonl(detections));results={};timings={}
    for solver in ('scipy','newton'):
        results[solver]=[];elapsed=[];cache={}
        for row in rows:
            start=time.perf_counter()
            results[solver].append(reconstruct_record(row,cams,{**cfg,'triangulation_solver':solver},cache))
            elapsed.append((time.perf_counter()-start)*1000)
        timings[solver]=statistics(elapsed)
    differences=[];coverage_changes=inlier_changes=fallbacks=0
    for a,b in zip(results['scipy'],results['newton']):
        for p,q,qa,qb in zip(a['joints_world_m'],b['joints_world_m'],a['quality'],b['quality']):
            coverage_changes+=int((p is None)!=(q is None))
            if p is None or q is None:continue
            differences.append(float(np.linalg.norm(np.array(p)-q)*1000))
            inlier_changes+=int(qa['inlier_cameras']!=qb['inlier_cameras']);fallbacks+=int(qb['solver_fallback'])
    report=dict(position_difference_mm=statistics(differences),coverage_changes=coverage_changes,
        inlier_changes=inlier_changes,fallbacks=fallbacks,processing_ms=timings,
        input_sha256=hashlib.sha256(detections.read_bytes()).hexdigest(),
        passed=bool(differences and not coverage_changes and not inlier_changes and max(differences)<.05))
    output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2));return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--detections',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not run(a.source,a.detections,a.output)['passed']:raise SystemExit(1)
