"""Scientific before/after plots and a markerless RGB / skeleton comparison video."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pose_sim.pipeline import read_jsonl
from pose_sim.landmarks import edges


def build(source,output,baseline):
    source,output,baseline=map(Path,(source,output,baseline))
    frames=list(read_jsonl(source/'frames.jsonl'))
    old=list(read_jsonl(baseline/'poses3d.jsonl'))
    raw=list(read_jsonl(output/'poses3d.jsonl'));state=list(read_jsonl(output/'states3d.jsonl'))
    detects=list(read_jsonl(output/'detections.jsonl'))
    def series(rows,a,b):
        values=[]
        for r in rows:
            ps=dict(zip(r['joint_names'],r['joints_world_m']))
            values.append(np.linalg.norm(np.array(ps[a])-ps[b]) if ps.get(a) is not None and ps.get(b) is not None else np.nan)
        return np.array(values)
    ts=np.array([r['timestamp_ns'] for r in state])/1e9
    fig,axs=plt.subplots(2,1,figsize=(11,6),sharex=True)
    for ax,(a,b,title) in zip(axs,[('right_knee','right_ankle','Right shank length'),('left_elbow','left_wrist','Left forearm length')]):
        for rows,label,color in [(old,'Original RTMPose raw','#b74535'),(raw,'Hybrid raw','#c99d35'),(state,'Causal constrained state','#167c82')]:
            ax.plot(ts,series(rows,a,b),label=label,color=color,lw=1.5)
        ax.set(ylabel='Length (m)',title=title);ax.grid(alpha=.2);ax.legend(loc='upper right')
    axs[-1].set_xlabel('Source time (s)')
    fig.suptitle('Same RGB input: visual skeleton stability (not real-human accuracy)')
    fig.tight_layout();fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)
    fps=json.loads((source/'config.json').read_text())['fps']
    writer=cv2.VideoWriter(str(output/'comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1440,760))
    if not writer.isOpened():raise RuntimeError('Cannot create comparison video')
    def draw_skeleton(canvas,r,x0,width,label,color,origin):
        p=np.array([v if v is not None else [np.nan]*3 for v in r['joints_world_m']])
        # Oblique orthographic world projection, translated only for display.
        x=p[:,0]-origin[0];y=p[:,1]-origin[1];z=p[:,2]
        screen=np.column_stack((x0+width/2+(x*.8-y*.6)*180,650-z*280))
        def valid(i):return np.isfinite(screen[i]).all()
        for a,b in edges(r['joint_names']):
            if valid(a) and valid(b):cv2.line(canvas,tuple(screen[a].astype(int)),tuple(screen[b].astype(int)),color,2,cv2.LINE_AA)
        for i,pix in enumerate(screen):
            if not valid(i):continue
            c=(80,110,230) if r.get('joint_states',[{}]*len(p))[i].get('status')=='predicted' else color
            cv2.circle(canvas,tuple(pix.astype(int)),3,c,-1,cv2.LINE_AA)
        cv2.putText(canvas,label,(x0+10,75),cv2.FONT_HERSHEY_SIMPLEX,.48,(45,45,45),1,cv2.LINE_AA)
    try:
        for frame,o,r,s,d in zip(frames,old,raw,state,detects):
            canvas=np.full((760,1440,3),247,np.uint8)
            for n,(cid,v) in enumerate(frame['views'].items()):
                image=cv2.imdecode(np.fromfile(source/v['image'],np.uint8),1)
                xy=d['views'][cid]['keypoints_px'];score=d['views'][cid]['scores']
                if xy is not None:
                    xy=np.array(xy)
                    for a,b in edges(d['joint_names']):
                        if min(score[a],score[b])>=.35:cv2.line(image,tuple(xy[a].astype(int)),tuple(xy[b].astype(int)),(100,220,30),2)
                image=cv2.resize(image,(320,320));y,x=90+(n//2)*320,(n%2)*320
                canvas[y:y+320,x:x+320]=image
                cv2.putText(canvas,cid,(x+8,y+22),cv2.FONT_HERSHEY_SIMPLEX,.6,(20,90,240),2)
            hips=[s['joints_world_m'][s['joint_names'].index(n)] for n in ('left_hip','right_hip')]
            origin=np.mean([p for p in hips if p is not None],axis=0) if any(p is not None for p in hips) else np.zeros(3)
            draw_skeleton(canvas,o,650,250,'Original raw',(65,70,190),origin)
            draw_skeleton(canvas,r,910,250,'Hybrid raw',(35,160,190),origin)
            draw_skeleton(canvas,s,1170,250,'Filtered / bounded prediction',(120,120,15),origin)
            cv2.putText(canvas,f'Markerless multiview vision | source t={frame["timestamp_ns"]/1e9:.3f}s',(15,32),cv2.FONT_HERSHEY_SIMPLEX,.8,(40,40,40),2)
            cv2.putText(canvas,'RGB detection: body + feet + palm landmarks. Red state points = prediction. Offline replay at source FPS.',(15,750),cv2.FONT_HERSHEY_SIMPLEX,.52,(60,60,60),1)
            if frame is frames[0]:cv2.imencode('.png',canvas)[1].tofile(output/'preview.png')
            writer.write(canvas)
    finally:writer.release()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);p.add_argument('--baseline',required=True)
    a=p.parse_args();build(a.source,a.output,a.baseline)
