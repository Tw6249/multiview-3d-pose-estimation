"""Plot measured improvements and failed gates without mixing evaluation domains."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(report,output):
    r=json.loads(Path(report).read_text(encoding='utf-8'));fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    actions=['walk','turn','squat','cross_step','fast_arms','occlusion'];x=np.arange(len(actions))
    for i,(method,label,color) in enumerate([('baseline','Baseline','#78879a'),('joint','Joint association + state','#197d9d')]):
        means=[np.mean([s['methods'][method]['position_error_mm']['mean'] for s in r['benchmark']['sequence_results'] if s['action']==a]) for a in actions]
        axes[0,0].bar(x+(i-.5)*.36,means,.36,label=label,color=color)
    axes[0,0].set_xticks(x,[a.replace('_','\n') for a in actions]);axes[0,0].set_ylabel('World position error (mm)')
    axes[0,0].set_title('Held-out synthetic observation stress test');axes[0,0].legend(fontsize=8)
    for i,(method,color) in enumerate([('baseline','#78879a'),('joint','#197d9d')]):
        values=[r['benchmark']['results'][method][key]['mean'] for key in ['landmark_rotation_error_deg','occlusion_recovery_excess_step_mm']]
        axes[0,1].bar(np.arange(2)+(i-.5)*.36,values,.36,color=color)
    axes[0,1].set_xticks([0,1],['Foot/palm rotation (deg)','Recovery excess step (mm)'])
    axes[0,1].set_title('Other metrics: improvement is not uniform')
    old=r['rgb_regression']['previous_state']['proxy_error_mm']['mean'];new=r['rgb_regression']['new_state']['proxy_error_mm']['mean']
    axes[1,0].bar(['Previous state','Experimental state'],[old,new],color=['#78879a','#c76644'])
    axes[1,0].set_ylabel('Proxy error (mm)');axes[1,0].set_title('Existing RGB clip: mean error REGRESSED')
    for i,value in enumerate([old,new]):axes[1,0].text(i,value+1,f'{value:.2f}',ha='center')
    axes[1,0].set_ylim(0,65)
    names=['serial','concurrent_detector'];means=[r['online'][n]['latency_ms']['mean'] for n in names]
    p95=[r['online'][n]['latency_ms']['p95'] for n in names]
    axes[1,1].bar([0,1],means,color=['#78879a','#197d9d'],label='Mean')
    axes[1,1].scatter([0,1],p95,marker='_',s=300,color='#c76644',label='P95')
    axes[1,1].axhline(150,color='#c76644',linestyle='--',label='Output deadline')
    axes[1,1].set_xticks([0,1],['Serial detector','4 concurrent detectors']);axes[1,1].set_ylabel('Replay exposure proxy to output (ms)')
    axes[1,1].set_title('Paced RGB replay: physical exposure unverified');axes[1,1].legend(fontsize=8)
    for ax in axes.flat:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('Vision / motion evaluation | Experimental configuration, not G1 acceptance',fontsize=14)
    Path(output).parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('report',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();plot(a.report,a.output)
