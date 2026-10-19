"""Visualize commanded and simulated joint motion, without pose ground truth."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from .model import RobotModel
from .replay import read_rows


def render(output):
    output = Path(output)
    model = RobotModel.load(output/'robot_model.json')
    commands = read_rows(output/'commands.jsonl')
    feedback = read_rows(output/'feedback.jsonl')[:len(commands)]
    times = (np.array([c['timestamp_ns'] for c in commands])-commands[0]['timestamp_ns'])/1e9
    q = np.array([c['positions_rad'] for c in commands])
    actual = np.array([f['positions_rad'] for f in feedback])
    fig = plt.figure(figsize=(11,5.5),layout='constrained')
    ax = fig.add_subplot(121,projection='3d')
    graph = fig.add_subplot(122)
    lines = []
    for color,label,values in [('#2764bb','Command',q),('#d65f22','Feedback',actual)]:
        for side in ('left','right'):
            line, = ax.plot([],[],[],color=color,marker='o',lw=2,label=label if side=='left' else None)
            lines.append((line,values,side))
    ax.plot([0,0],[0,0],[0.7,1.2],color='0.7',lw=10)
    ax.plot([0,0],[-0.22,0.22],[1.2,1.2],color='0.7',lw=7)
    ax.set(xlim=(-0.6,0.6),ylim=(-0.7,0.7),zlim=(0.45,1.6),xlabel='Forward (m)',ylabel='Left (m)',zlabel='Up (m)')
    ax.set_box_aspect((1.2,1.4,1.15))
    ax.view_init(20,35)
    ax.legend(loc='upper left')
    for i,color in [(1,'#2764bb'),(3,'#379a66'),(5,'#d65f22'),(7,'#8c50a0')]:
        graph.plot(times,q[:,i],color=color,label=model.names[i].replace('shoulder_',''))
        graph.plot(times,actual[:,i],color=color,linestyle=':',alpha=0.7)
    cursor = graph.axvline(0,color='black',lw=1)
    graph.set(xlabel='Replay time (s)',ylabel='Joint angle (rad)',title='Solid: commands / Dotted: simulated feedback')
    graph.grid(alpha=0.2)
    graph.legend(fontsize=8,loc='upper left')
    def update(index):
        for line,values,side in lines:
            chain = model.forward(values[index])[side]
            p = np.array([chain['shoulder'],chain['elbow'],chain['wrist']])
            line.set_data_3d(p[:,0],p[:,1],p[:,2])
        cursor.set_xdata([times[index]]*2)
        fig.suptitle(f'Visual pose to robot | {times[index]:.2f} s | {commands[index]["state"]}\nKinematic simulation: fixed-base dual arm, 8 joints',fontsize=12)
        return [line for line,_,_ in lines]+[cursor]
    update(len(commands)//2)
    fig.savefig(output/'control_preview.png',dpi=140)
    fps = model.config['control']['rate_hz']
    animation = FuncAnimation(fig,update,frames=len(commands),interval=1000/fps,blit=False)
    animation.save(output/'control_preview.gif',writer=PillowWriter(fps=fps),dpi=85)
    plt.close(fig)
    print('CONTROL_PREVIEW',output/'control_preview.gif')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    render(parser.parse_args().output)
