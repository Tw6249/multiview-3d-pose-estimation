"""Authored RGB scene motion only; never an input to the vision estimator.

Input: scene config and frame index. Output: animation frame, root translation
(metres, world Z up) and yaw (radians about world Z). Time is frame / source FPS.
Unknown trajectories fail explicitly. Calibration remains static within a clip.
"""
import math


def sample_motion(config, frame):
    fps = config['fps']
    if fps <= 0 or frame < 0:
        raise ValueError('Positive FPS and nonnegative frame required')
    animation_frame = frame * config.get('animation_speed', 1.0)
    if config.get('animation_loop_seconds'):
        animation_frame %= config['animation_loop_seconds'] * fps
    seconds = frame / fps
    motion = config.get('root_motion', {'kind': 'linear'})
    kind = motion['kind']
    yaw = math.radians(config.get('human_yaw_deg', 0.0))
    if kind == 'linear':
        velocity = config.get('human_translation_m_s', [0, 0, 0])
        centered = seconds - (config['frames'] - 1) / fps / 2
        translation = [v * centered for v in velocity]
    elif kind == 'circle':
        radius = motion['radius_m']
        speed = motion['angular_speed_rad_s']
        if radius <= 0 or speed == 0:
            raise ValueError('Circle radius must be positive and speed nonzero')
        angle = motion.get('phase_rad', 0.0) + speed * seconds
        translation = [radius * math.cos(angle), radius * math.sin(angle), 0.0]
        # CesiumMan faces -Y. Rotate its heading onto the path tangent.
        yaw += angle + (math.pi if speed > 0 else 0.0)
    else:
        raise ValueError(f'Unknown authored root motion: {kind}')
    return animation_frame, translation, yaw
