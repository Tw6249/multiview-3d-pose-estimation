import argparse
import io
import json
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from .geometry import EDGES
from .pipeline import read_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument('session', type=Path)
    a = p.parse_args()
    s = a.session
    frames = list(read_jsonl(s/'frames.jsonl'))
    detects = {r['frame_id']: r for r in read_jsonl(s/'detections.jsonl')}
    poses = {r['frame_id']: r for r in read_jsonl(s/'poses3d.jsonl')}
    gt = {r['frame_id']: r for r in read_jsonl(s/'ground_truth.jsonl')} if (s/'ground_truth.jsonl').exists() else {}
    cfg = json.loads((s/'config.json').read_text())
    video = None
    samples = []
    for packet in frames:
        fid = packet['frame_id']
        fig = plt.figure(figsize=(14, 8), dpi=100, facecolor='#f3f5f7')
        grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25])
        for n, (cid, v) in enumerate(list(packet['views'].items())[:4]):
            ax = fig.add_subplot(grid[n//2, n%2])
            rgb = np.asarray(Image.open(s/v['image']).convert('RGB'))
            ax.imshow(rgb)
            d = detects[fid]['views'][cid]
            if d['keypoints_px'] is not None:
                k, c = np.array(d['keypoints_px']), np.array(d['scores'])
                for u, vj in EDGES:
                    if min(c[u], c[vj]) >= cfg['confidence_threshold']:
                        ax.plot(k[[u, vj], 0], k[[u, vj], 1], color='#00edb0', lw=1.5)
                ok = c >= cfg['confidence_threshold']
                ax.scatter(k[ok, 0], k[ok, 1], s=8, color='#ffb000')
            ax.set_title(cid+' / RGB + detected 2D joints', fontsize=10)
            ax.axis('off')
        ax = fig.add_subplot(grid[:, 2], projection='3d')
        for row, color, label in [(gt.get(fid), '#aab1ba', 'Rig proxy GT (12)'),
                                  (poses[fid], '#007f9f', 'Reconstructed from RGB')]:
            if row is None:
                continue
            points = row['joints_world_m']
            x = np.array([v if v is not None else [np.nan]*3 for v in points])
            ax.scatter(x[:, 0], x[:, 1], x[:, 2], color=color, s=20, label=label)
            for u, v in EDGES:
                ax.plot(x[[u,v], 0], x[[u,v], 1], x[[u,v], 2], color=color, lw=2)
        ax.set(xlim=(-0.9, 0.9), ylim=(-0.9, 0.9), zlim=(0, 2), xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)')
        ax.set_box_aspect((1.8, 1.8, 2))
        ax.view_init(elev=17, azim=-60)
        ax.legend(loc='upper center', fontsize=8)
        valid = sum(v is not None for v in poses[fid]['joints_world_m'])
        ax.set_title(f'3D pose / {valid}/17 joints valid')
        fig.suptitle(f'Multi-view RGB to 3D pose | frame {fid:03d} | {packet["timestamp_ns"]/1e9:.3f} s', fontsize=15)
        fig.text(0.02, 0.015, 'RTMPose inference on rendered RGB. Grey rig centres are evaluation proxies; not exact COCO ground truth.', fontsize=9)
        fig.subplots_adjust(top=0.91, bottom=0.07, left=0.02, right=0.98, wspace=0.12, hspace=0.15)
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        plt.close(fig)
        im = Image.open(buffer).convert('RGB')
        if fid == frames[0]['frame_id']:
            im.save(s/'preview.png')
        samples.append(im.copy())
        if video is None:
            video = cv2.VideoWriter(str(s/'preview.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), cfg['fps'], im.size)
            if not video.isOpened():
                raise RuntimeError('Cannot open MP4 video writer')
        video.write(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR))
    video.release()
    samples[0].save(s/'preview.gif', save_all=True, append_images=samples[1:], duration=round(1000/cfg['fps']), loop=0)
    print('PREVIEW_READY', str(s/'preview.png'))


if __name__ == '__main__':
    main()
