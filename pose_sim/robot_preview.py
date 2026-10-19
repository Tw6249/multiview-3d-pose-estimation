"""Robot-source observations and G1 request preview; no executed motion implied."""
import argparse
import json
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .pipeline import read_jsonl
from motion_control.unitree import G1Description


def render(session):
    session = Path(session)
    poses = list(read_jsonl(session/'link_poses.jsonl'))
    frames = list(read_jsonl(session/'frames.jsonl'))
    detections = list(read_jsonl(session/'detections.jsonl'))
    requests = list(read_jsonl(session/'g1_transfer/g1_locomotion_requests.jsonl'))
    selected = len(poses)//2
    fig = plt.figure(figsize=(12,9),layout='constrained')
    ax = fig.add_subplot(221)
    cid,view = next(iter(frames[selected]['views'].items()))
    image = cv2.imdecode(np.fromfile(session/view['image'],np.uint8),cv2.IMREAD_COLOR)
    ax.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    points = [p for p in (detections[selected]['views'][cid]['keypoints_px'] or []) if p is not None]
    if points: ax.scatter(*np.array(points).T,c='#ff8f00',s=20)
    ax.set(title='Source robot: observed natural landmarks',xlabel='Image X (px)',ylabel='Image Y (px)')
    ax = fig.add_subplot(222,projection='3d')
    for name,link in poses[selected]['links'].items():
        if not link['valid']: continue
        t = np.array(link['T_world_link']); origin = t[:3,3]
        for i,color in enumerate(('r','g','b')):
            ax.quiver(*origin,*(t[:3,i]*0.1),color=color)
        ax.text(*origin,name)
    ax.set(title='Recovered source link SE(3) poses',xlabel='World X (m)',ylabel='World Y (m)',zlabel='World Z (m)')
    origins = np.array([np.array(link['T_world_link'])[:3,3] for row in poses
                        for link in row['links'].values() if link['valid']])
    if len(origins):
        centre = (origins.min(axis=0)+origins.max(axis=0))/2
        radius = max(float(np.ptp(origins,axis=0).max())/2+0.15,0.25)
        ax.set(xlim=(centre[0]-radius,centre[0]+radius),ylim=(centre[1]-radius,centre[1]+radius),
               zlim=(centre[2]-radius,centre[2]+radius))
    ax.set_box_aspect((1,1,1))
    ax = fig.add_subplot(223,projection='3d')
    target_variant = json.loads((session/'g1_transfer/transfer_report.json').read_text())['target_variant']
    model = G1Description(target_variant); links = model.forward(np.zeros(model.size))
    for node in model.nodes:
        if node['type'] != 'revolute': continue
        p = np.array([links[node['parent']][:3,3],links[node['child']][:3,3]])
        ax.plot(p[:,0],p[:,1],p[:,2],color='#327bb5',lw=2)
    ax.set(title='G1 target URDF (model only, not executing)',xlabel='Base X (m)',ylabel='Base Y (m)',zlabel='Base Z (m)')
    ax.set_box_aspect((0.5,0.7,1.4)); ax.view_init(15,35)
    ax.set_xticks([0,0.2]); ax.set_yticks([-0.2,0,0.2]); ax.tick_params(labelsize=8)
    ax.set_xlabel('Base X (m)',fontsize=9,labelpad=0)
    ax.set_ylabel('Base Y (m)',fontsize=9,labelpad=0)
    ax = fig.add_subplot(224)
    times = (np.array([r['timestamp_ns'] for r in requests])-requests[0]['timestamp_ns'])/1e9
    velocities = np.array([r['request']['parameters']['velocity'] for r in requests])
    ax.plot(times,velocities[:,0],label='Forward vx (m/s)')
    ax.plot(times,velocities[:,1],label='Lateral vy (m/s)')
    ax.plot(times,velocities[:,2],label='Yaw rate (rad/s)',linestyle='--')
    ax.set(title='G1 native-locomotion requests: OFFLINE',xlabel='Replay time (s)',ylabel='Requested value')
    ax.legend(fontsize=9); ax.grid(alpha=0.2)
    fig.suptitle('Source robot perception → explicit task mapping → G1 target\nOffline requests; no physical robot/network execution',fontsize=13)
    fig.savefig(session/'robot_transfer_preview.png',dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('session',type=Path)
    render(p.parse_args().session)
