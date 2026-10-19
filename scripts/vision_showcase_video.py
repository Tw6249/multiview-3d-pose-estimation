"""Review-only video compositor. Reads RGB and estimator output, never labels.

Frames join by (frame_id, source timestamp_ns); missing/misaligned rows fail.
Only the 3-D display is root-centred. No interpolation or extra temporal filter.
Colour indicates detector anatomical side, not independently verified identity.
"""
from functools import lru_cache
from fractions import Fraction
import json
import math
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pose_sim.landmarks import edges
from pose_sim.pipeline import read_jsonl

WIDTH, HEIGHT = 1920, 1080
LEFT = (74, 175, 250)   # BGR, orange
RIGHT = (240, 182, 72)  # BGR, blue
PREDICTED = (173, 91, 245)
NEUTRAL = (198, 204, 214)


@lru_cache(maxsize=12)
def font(size, bold=False):
    path = Path('C:/Windows/Fonts/msyhbd.ttc' if bold else 'C:/Windows/Fonts/msyh.ttc')
    if not path.exists():
        raise RuntimeError('Chinese caption font not found')
    return ImageFont.truetype(str(path), size)


def caption(draw, xy, text, size=22, color=(216, 224, 235), bold=False):
    draw.text(xy, text, font=font(size, bold), fill=color)


def side_color(name):
    return LEFT if name.startswith('left_') else RIGHT if name.startswith('right_') else NEUTRAL


def join_rows(frames, *streams):
    """Reject missing/duplicate/reordered time records instead of zip truncation."""
    expected = [(f['frame_id'], f['timestamp_ns']) for f in frames]
    if len(set(expected)) != len(expected):
        raise ValueError('Duplicate source frame identity')
    if any(b[1] <= a[1] for a, b in zip(expected, expected[1:])):
        raise ValueError('Source timestamps must increase within each scene')
    for rows in streams:
        if [(r['frame_id'], r['timestamp_ns']) for r in rows] != expected:
            raise ValueError('RGB and estimate frame identity/timestamps differ')
    return zip(frames, *streams)


def camera_panel(source, frame, detection, cid):
    image = cv2.imdecode(np.fromfile(source/frame['views'][cid]['image'], np.uint8), 1)
    if image is None:
        raise RuntimeError('Cannot read RGB frame')
    view = detection['views'][cid]
    points, scores = view['keypoints_px'], view['scores']
    if points is not None:
        points = np.array(points)
        for a, b in edges(detection['joint_names']):
            if min(scores[a], scores[b]) >= .35 and np.isfinite(points[[a,b]]).all():
                color = side_color(detection['joint_names'][b])
                cv2.line(image, tuple(points[a].astype(int)), tuple(points[b].astype(int)), color, 3, cv2.LINE_AA)
        for j, (xy, score) in enumerate(zip(points, scores)):
            if score >= .35 and np.isfinite(xy).all():
                cv2.circle(image, tuple(xy.astype(int)), 3, side_color(detection['joint_names'][j]), -1, cv2.LINE_AA)
    return cv2.resize(image, (440, 440), interpolation=cv2.INTER_AREA)


def draw_state(canvas, state, origin, x0, angle):
    """Two fixed orthographic world views, Z up; no ground-truth silhouette."""
    p = np.array([xyz if xyz is not None else [np.nan]*3 for xyz in state['joints_world_m']])
    angle = math.radians(angle)
    c, s = math.cos(angle), math.sin(angle)
    center, floor, scale = 222, 607, 252
    def project(xyz):
        x, y, z = xyz
        return (int(round(center+(c*x+s*y)*scale)), int(round(floor-(z+.14*(-s*x+c*y))*scale)))
    panel = np.full((670, 464, 3), (35, 29, 24), np.uint8)
    for v in np.linspace(-.8, .8, 5):
        for a, b in (([v,-.8,0], [v,.8,0]), ([-.8,v,0], [.8,v,0])):
            cv2.line(panel, project(a), project(b), (66, 56, 44), 1, cv2.LINE_AA)
    cv2.line(panel, project([-.8, -.8, 0]), project([-.8, -.8, 2]), (66,56,44), 1)
    p[:, :2] -= origin[:2]
    valid = np.isfinite(p).all(axis=1)
    points = [project(xyz) if ok else None for xyz, ok in zip(p, valid)]
    statuses = state['joint_states']
    # Draw inside this panel, clipping off-panel failures instead of overwriting
    # other views. Such joints remain counted in the displayed state summary.
    overlay = np.zeros_like(panel)
    for a, b in edges(state['joint_names']):
        if not (valid[a] and valid[b]):
            continue
        color = PREDICTED if any(statuses[i]['status']=='predicted' for i in (a,b)) else side_color(state['joint_names'][b])
        cv2.line(overlay, points[a], points[b], color, 4, cv2.LINE_AA)
    for j, point in enumerate(points):
        if point is None:
            continue
        color = PREDICTED if statuses[j]['status']=='predicted' else side_color(state['joint_names'][j])
        cv2.circle(overlay, point, 5 if j < 17 else 3, color, -1, cv2.LINE_AA)
    dst = panel[47:656]
    src = overlay[47:656]
    mask = np.any(src, axis=2)
    dst[mask] = src[mask]
    canvas[108:778, x0:x0+464] = panel


def make_frame(output, manifest, scene, scene_index, frame, detection, state, global_frame, total_frames):
    canvas = np.full((HEIGHT, WIDTH, 3), (26, 21, 17), np.uint8)
    for n, cid in enumerate(frame['views']):
        x, y = 24+(n%2)*456, 108+(n//2)*448
        canvas[y:y+440, x:x+440] = camera_panel(output/scene['source'], frame, detection, cid)
        cv2.rectangle(canvas, (x+8,y+8), (x+177,y+41), (37,30,24), -1)
    hips = [state['joints_world_m'][state['joint_names'].index(n)] for n in ('left_hip','right_hip')]
    origin = np.mean([p for p in hips if p is not None], axis=0) if any(p is not None for p in hips) else np.zeros(3)
    draw_state(canvas, state, origin, 944, 35)
    draw_state(canvas, state, origin, 1424, 125)
    cv2.rectangle(canvas, (944,790), (1888,996), (35,29,24), -1)
    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    caption(draw, (26,10), '无标记 · 多视角姿态估计', 32, bold=True)
    caption(draw, (26,57), f"{scene_index+1:02d} / {len(manifest['scenes']):02d}   {scene['title']}   |   {scene['description']}", 21)
    caption(draw, (1472,16), f'{global_frame/scene["fps"]:04.1f} / {total_frames/scene["fps"]:.0f} 秒', 27, bold=True)
    caption(draw, (1472,57), f'合成 RGB · 离线 {scene["fps"]} fps 播放', 20, (166,181,199))
    for n, cid in enumerate(frame['views']):
        x, y = 24+(n%2)*456, 108+(n//2)*448
        caption(draw, (x+17,y+10), f'相机 {n+1} · 2D 检测', 18)
    caption(draw, (960,119), '三维估计 · 斜视', 22, bold=True)
    caption(draw, (1440,119), '三维估计 · 侧视', 22, bold=True)
    caption(draw, (960,745), '世界 Z 轴向上 · 根部居中显示', 17, (151,168,188))
    caption(draw, (1440,745), '无真值输入 · 无额外补帧', 17, (151,168,188))
    body = state['joint_states'][5:17]
    predicted = sum(r['status']=='predicted' for r in body)
    missing = sum(p is None for p in state['joints_world_m'][5:17])
    current = 12-predicted-missing
    accepted = sum(v['keypoints_px'] is not None for v in detection['views'].values())
    caption(draw, (964,803), f'身体 12 关节    当前观测 {current:02d}    历史预测 {predicted:02d}    缺失 {missing:02d}', 23, bold=True)
    caption(draw, (964,847), '左侧', 22, tuple(reversed(LEFT)))
    caption(draw, (1057,847), '右侧', 22, tuple(reversed(RIGHT)))
    caption(draw, (1150,847), '历史预测', 22, tuple(reversed(PREDICTED)))
    caption(draw, (1320,847), f'接受单人观测：{accepted}/4 相机', 21)
    caption(draw, (964,891), 'GPU 图像检测  →  多视角融合  →  因果时序状态', 22)
    caption(draw, (964,934), '完整手脚旋转与实际支撑未验证 · 未连接实机', 20, (166,181,199))
    elapsed = 0.0
    for i, item in enumerate(manifest['scenes']):
        label = item['title'].split(' · ')[0]
        end = elapsed + item['frames']/item['fps']
        step = 1872/len(manifest['scenes'])
        left = int(24+i*step)
        color = (79,187,190) if i == scene_index else (58,68,83)
        draw.rounded_rectangle((left,1010,int(left+step-14),1043), radius=4, fill=color)
        caption(draw, (left+14,1011), f'{elapsed:02.0f}–{end:02.0f}s   {label}', 18,
                (14,28,38) if i==scene_index else (189,201,216), bold=i==scene_index)
        elapsed = end
    caption(draw, (26,1052), 'CesiumMan © 2017 Cesium · CC BY 4.0 · 同一模型与步行片段，改变路径、速度、尺寸和环境；不代表真实人体评测。', 15, (144,162,181))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def compose(output, manifest):
    ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        raise RuntimeError('ffmpeg and ffprobe are required')
    fps_set = {s['fps'] for s in manifest['scenes']}
    if len(fps_set) != 1:
        raise ValueError('All scenes must have identical source FPS; no resampling')
    fps = fps_set.pop()
    total = sum(s['frames'] for s in manifest['scenes'])
    movie = output/f'pose_estimation_{total/fps:g}s.mp4'
    command = [ffmpeg, '-y', '-hide_banner', '-loglevel', 'warning', '-f', 'rawvideo',
               '-pix_fmt', 'bgr24', '-s', f'{WIDTH}x{HEIGHT}', '-r', str(fps), '-i', '-',
               '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-threads', '8',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(movie)]
    global_frame = 0
    with (output/'encode.log').open('w') as log:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            for i, scene in enumerate(manifest['scenes']):
                result = output/scene['result']
                frames = list(read_jsonl(output/scene['source']/'frames.jsonl'))
                detections = list(read_jsonl(scene.get('detections', result/'detections.jsonl')))
                states = list(read_jsonl(result/'states3d.jsonl'))
                if len(frames) != scene['frames']:
                    raise ValueError('Incomplete source clip')
                for f, d, s in join_rows(frames, detections, states):
                    canvas = make_frame(output, manifest, scene, i, f, d, s, global_frame, total)
                    proc.stdin.write(canvas.tobytes())
                    if f['frame_id'] in (scene['frames']//2,):
                        cv2.imencode('.png', canvas)[1].tofile(output/f"preview_{scene['id']}.png")
                    global_frame += 1
                print(f"COMPOSED {scene['id']}", flush=True)
        finally:
            proc.stdin.close()
            code = proc.wait()
        if code:
            raise RuntimeError('Video encoding failed; inspect encode.log')
    if global_frame != total:
        raise RuntimeError('Composed frame count mismatch')
    validate_video(output, manifest)


def validate_video(output, manifest):
    """Can re-run validation of the existing video without rerendering/encoding."""
    fps = manifest['scenes'][0]['fps']
    total = sum(s['frames'] for s in manifest['scenes'])
    for scene in manifest['scenes']:
        frames = list(read_jsonl(output/scene['source']/'frames.jsonl'))
        if len(frames) != scene['frames'] or scene['fps'] != fps:
            raise ValueError('Incomplete source clip or mixed FPS')
        result = output/scene['result']
        join_rows(frames, list(read_jsonl(scene.get('detections', result/'detections.jsonl'))),
                  list(read_jsonl(result/'states3d.jsonl')))
    movie = output/f'pose_estimation_{total/fps:g}s.mp4'
    info = json.loads(subprocess.check_output([shutil.which('ffprobe'),'-v','error','-count_frames','-show_streams',
                    '-show_format','-of','json',str(movie)], text=True, encoding='utf-8'))
    stream = next(s for s in info['streams'] if s['codec_type']=='video')
    if (int(stream['nb_read_frames']) != total or
            (stream['width'],stream['height']) != (WIDTH,HEIGHT) or
            Fraction(stream['avg_frame_rate']) != fps or
            abs(float(info['format']['duration'])-total/fps) > .05):
        raise RuntimeError('Encoded video duration/frame count mismatch')
    subprocess.run([shutil.which('ffmpeg'),'-v','error','-xerror','-i',str(movie),
                    '-f','null','-'], check=True, capture_output=True)
    contact_frames = []
    for scene in manifest['scenes']:
        preview = cv2.imdecode(np.fromfile(output/f"preview_{scene['id']}.png",np.uint8),1)
        if preview is None:
            raise RuntimeError('Missing scene preview')
        contact_frames.append(cv2.resize(preview,(960,540),interpolation=cv2.INTER_AREA))
    if contact_frames:
        if len(contact_frames) % 2:
            contact_frames.append(np.zeros_like(contact_frames[0]))
        rows = [np.hstack(contact_frames[i:i+2]) for i in range(0,len(contact_frames),2)]
        cv2.imencode('.jpg', np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(output/'contact_sheet.jpg')
    report = dict(video=movie.name, duration_seconds=total/fps, fps=fps, frames=total,
                  width=WIDTH,height=HEIGHT,codec=stream['codec_name'],bytes=movie.stat().st_size,
                  source='rendered RGB; same CesiumMan walk asset', offline=True,
                  ground_truth_used_for_estimation_or_display=False, extra_interpolation=False,
                  source_frame_time_alignment_verified=True, full_decode_passed=True,
                  scenes=len(manifest['scenes']))
    (output/'video_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
