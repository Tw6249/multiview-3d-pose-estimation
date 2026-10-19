"""Paired foot-state check on cached RGB reconstructions; no contact truth claim."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from pose_sim.human_motion import HumanMotionEstimator
from pose_sim.foot_support import FootSupportEstimator,load_config


class PreviousHumanMotion(HumanMotionEstimator):
    def __init__(self,cfg):super().__init__(cfg);self.previous_contact_memory={}

    def _support(self,p,velocity,states,t):
        # Exact previous support rule for paired diagnostics, no production use.
        contacts={}
        for side in ('left','right'):
            ns=[side+'_'+n for n in ('heel','big_toe','small_toe')]
            attachment_ok=not self.cfg.get('auxiliary_consistency',True) or self._attachment_state[side+'_foot']['status']=='supported'
            if not attachment_ok or self.cfg['ground_height_m'] is None or not all(n in p and n in velocity for n in ns):
                contacts[side]=dict(state='unknown',score=None);self.previous_contact_memory.pop(side,None);continue
            sigma=max(states[n]['max_std_m'] for n in ns)
            scores=[float(np.exp(-.5*((p[n][2]-self.cfg['ground_height_m'])/.035)**2-.5*(np.linalg.norm(velocity[n])/.2)**2)) for n in ns]
            score=max(scores);old=self.previous_contact_memory.get(side,dict(state='unknown',candidate=None,count=0,last=t))
            target='contact' if score>.65 and sigma<.03 else 'swing' if score<.25 else old['state']
            count=old['count']+1 if old['candidate']==target and t-old['last']<=150_000_000 else 1
            state=target if count>=3 else old['state']
            self.previous_contact_memory[side]=dict(state=state,candidate=target,count=count,last=t)
            contacts[side]=dict(state=state,score=score,calibrated_probability=False)
        return dict(feet=contacts)


def run(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve support audit')
    output.mkdir(parents=True)
    sources=sorted(Path('outputs/vision_attachment/accepted/estimates').glob('*/poses3d.jsonl'))
    if len(sources)!=6:raise ValueError('Expected frozen six-scene cache')
    settings=json.loads(Path('config/vision_motion.json').read_text())['state']
    cfg={**settings,'ground_height_m':0.}
    freeze=dict(configuration=cfg,foot_configuration=load_config(),sources_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('pose_sim/foot_support.py'),Path('pose_sim/human_motion.py')]},
        scope='Known rendered RGB reconstruction, supplied rendered floor Z=0 for this diagnostic only. Runtime default floor remains unknown. No independent human contact labels, no control integration or threshold fitting.')
    (output/'freeze.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8');cases=[];timings=[]
    for path in sources:
        before=PreviousHumanMotion(cfg);after=HumanMotionEstimator(cfg);no_floor=FootSupportEstimator()
        confusion=Counter();reasons=Counter();none=Counter();frames=0
        for line in path.read_text().splitlines():
            row=json.loads(line);a=before.update(row);b=after.update(row);frames+=1
            for key in ('joints_world_m','velocities_world_m_s','shape','joint_states','orientations','orientation_quality','joint_rotations'):
                if a[key]!=b[key]:raise AssertionError('Support change modified human kinematics: '+key)
            for side in ('left','right'):
                x=a['support']['feet'][side]['state'];y=b['support']['feet'][side]
                confusion[x+'->'+y['state']]+=1;reasons[y['reason']]+=1
            p={n:v for n,v,s in zip(b['joint_names'],b['joints_world_m'],b['joint_states']) if s['status']=='filtered'}
            v=dict(zip(b['joint_names'],b['velocities_world_m_s']));states=dict(zip(b['joint_names'],b['joint_states']))
            # Time the low-dimensional module with current inputs and known plane,
            # separate from filtering and all GPU work. Fresh module per frame
            # only for speed profiling, never counted in classification scores.
            e=FootSupportEstimator();tick=time.perf_counter();e.update(b['timestamp_ns'],p,v,states,dict(left=True,right=True),0.)
            timings.append((time.perf_counter()-tick)*1000)
            r=no_floor.update(b['timestamp_ns'],p,v,states,dict(left=True,right=True),None)
            none.update(f['state'] for f in r['feet'].values())
        cases.append(dict(id=path.parent.name,frames=frames,paired_label_counts=dict(confusion),current_reasons=dict(reasons),
            unspecified_floor_labels=dict(none),human_kinematics_exactly_unchanged=True))
        print(path.parent.name,dict(confusion),flush=True)
    result=dict(freeze=freeze,cases=cases,module_cpu_ms=dict(mean=float(np.mean(timings)),p95=float(np.percentile(timings,95))),
        latency_scope='CPU support evidence only, module construction excluded; not exposure-to-state latency')
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
