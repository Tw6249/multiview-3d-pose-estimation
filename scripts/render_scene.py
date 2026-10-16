"""Run with portable Blender --background --python scripts/render_scene.py -- ..."""
import argparse
import json
import math
from pathlib import Path
import sys
import bpy
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

ROOT = Path(__file__).resolve().parents[1]
BONES = {
    5: 'Skeleton_arm_joint_L__4_', 6: 'Skeleton_arm_joint_R',
    7: 'Skeleton_arm_joint_L__3_', 8: 'Skeleton_arm_joint_R__2_',
    9: 'Skeleton_arm_joint_L__2_', 10: 'Skeleton_arm_joint_R__3_',
    11: 'leg_joint_L_1', 12: 'leg_joint_R_1', 13: 'leg_joint_L_2',
    14: 'leg_joint_R_2', 15: 'leg_joint_L_3', 16: 'leg_joint_R_3'
}


def material(name, color):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*color, 1)
    m.use_nodes = True
    m.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (*color, 1)
    m.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 0.8
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--config', default=str(ROOT/'config/demo.json'))
    p.add_argument('--frames', type=int)
    p.add_argument('--asset', default=str(ROOT/'assets/CesiumMan.glb'))
    args = p.parse_args(sys.argv[sys.argv.index('--')+1:])
    cfg = json.loads(Path(args.config).read_text())
    if args.frames:
        cfg['frames'] = args.frames
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'frames.jsonl').exists():
        raise RuntimeError('Session already exists; use a new output folder')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=str(Path(args.asset).resolve()))
    imported = list(bpy.context.scene.objects)
    rigs = [o for o in imported if o.type == 'ARMATURE']
    if not rigs:
        raise RuntimeError('Human asset has no armature')
    rig = rigs[0]
    missing = set(BONES.values()) - set(rig.pose.bones.keys())
    if missing:
        raise RuntimeError(f'Asset needs a custom joint map: {sorted(missing)}')
    scene = bpy.context.scene
    scene.render.fps = cfg['fps']
    scene.frame_set(0)
    bpy.context.view_layer.update()
    # Use evaluated skinned vertices. A bind-pose bounding box can be far from the
    # actual animated character and would put the feet above the floor.
    mesh_points = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for o in imported:
        # glTF also creates a hidden Icosphere custom-bone shape, which is not
        # part of the person. Only skin meshes driven by this armature qualify.
        if o.type == 'MESH' and (o.parent == rig or any(m.type == 'ARMATURE' and m.object == rig for m in o.modifiers)):
            evaluated = o.evaluated_get(depsgraph)
            mesh = evaluated.to_mesh()
            mesh_points.extend(evaluated.matrix_world @ v.co for v in mesh.vertices)
            evaluated.to_mesh_clear()
    minimum = Vector(tuple(min(v[i] for v in mesh_points) for i in range(3)))
    maximum = Vector(tuple(max(v[i] for v in mesh_points) for i in range(3)))
    scale = cfg['human_height_m'] / (maximum.z-minimum.z)
    wrapper = bpy.data.objects.new('Human_world_transform', None)
    scene.collection.objects.link(wrapper)
    for o in imported:
        if o.parent is None:
            o.parent = wrapper
    wrapper.scale = (scale,)*3
    wrapper.location = (-scale*(minimum.x+maximum.x)/2, -scale*(minimum.y+maximum.y)/2, -scale*minimum.z)
    bpy.context.view_layer.update()
    (out/'scene_metadata.json').write_text(json.dumps({'asset': str(Path(args.asset).resolve()),
        'normalization': 'evaluated animated mesh vertices at frame zero',
        'target_initial_height_m': cfg['human_height_m'], 'uniform_scale': scale,
        'source_bounds_min': list(minimum), 'source_bounds_max': list(maximum)}, indent=2))

    bpy.ops.mesh.primitive_plane_add(size=12)
    floor = bpy.context.object
    floor.name = 'Floor_z0'
    floor.data.materials.append(material('Floor', (0.32, 0.36, 0.41)))
    for name, pos, energy, size in [('Key', (2, -3, 5), 900, 4), ('Fill', (-3, 1, 4), 700, 4), ('Rim', (1, 3, 4), 600, 3)]:
        data = bpy.data.lights.new(name, 'AREA')
        data.energy, data.shape, data.size = energy, 'DISK', size
        obj = bpy.data.objects.new(name, data)
        scene.collection.objects.link(obj)
        obj.location = pos
        obj.rotation_euler = (Vector((0, 0, 0.9))-obj.location).to_track_quat('-Z', 'Y').to_euler()
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs[0].default_value = (0.25, 0.25, 0.25, 1)
    scene.world.node_tree.nodes['Background'].inputs[1].default_value = 0.5
    scene.render.engine = cfg['engine']
    if cfg['engine'] == 'CYCLES':
        scene.cycles.device = 'CPU'
        scene.cycles.samples = cfg['render_samples']
        scene.cycles.use_denoising = True
    w, h = cfg['width'], cfg['height']
    scene.render.resolution_x, scene.render.resolution_y = w, h
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1
    scene.render.pixel_aspect_y = cfg['fx']/cfg['fy']
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = False
    scene.view_settings.view_transform = 'AgX'
    cams, camera_objs = {}, {}
    for i, angle in enumerate(cfg['camera_angles_deg']):
        cid = f'cam{i:02d}'
        a = math.radians(angle)
        data = bpy.data.cameras.new(cid)
        data.sensor_fit, data.sensor_width = 'HORIZONTAL', 36
        data.lens = cfg['fx']*36/w
        data.clip_start, data.clip_end = 0.05, 100
        cam = bpy.data.objects.new(cid, data)
        scene.collection.objects.link(cam)
        cam.location = (cfg['camera_radius_m']*math.cos(a), cfg['camera_radius_m']*math.sin(a), cfg['camera_height_m'])
        cam.rotation_euler = (Vector((0, 0, cfg['target_height_m']))-cam.location).to_track_quat('-Z', 'Y').to_euler()
        bpy.context.view_layer.update()
        inv = cam.matrix_world.inverted()
        r = [[inv[row][col]*(1 if row == 0 else -1) for col in range(3)] for row in range(3)]
        t = [inv[row][3]*(1 if row == 0 else -1) for row in range(3)]
        cams[cid] = {'K': [[cfg['fx'], 0, w/2], [0, cfg['fy'], h/2], [0, 0, 1]],
                     'R_world_to_camera': r, 't_world_to_camera_m': t,
                     'distortion': [0]*5, 'width': w, 'height': h}
        camera_objs[cid] = cam
        (out/'images'/cid).mkdir(parents=True, exist_ok=True)
    (out/'cameras.json').write_text(json.dumps({'schema_version': 1, 'world_axes': 'right-handed, Z-up',
        'units': 'metre', 'camera_axes': 'OpenCV: X-right Y-down Z-forward', 'cameras': cams}, indent=2))
    (out/'config.json').write_text(json.dumps(cfg, indent=2))
    max_projection_error = 0.0
    with (out/'frames.jsonl').open('w') as frames, (out/'ground_truth.jsonl').open('w') as truths:
        for fi in range(cfg['frames']):
            scene.frame_set(fi)
            bpy.context.view_layer.update()
            joints = [None]*17
            for j, bone in BONES.items():
                joints[j] = list(rig.matrix_world @ rig.pose.bones[bone].head)
            packet = {'frame_id': fi, 'timestamp_ns': round(fi/cfg['fps']*1e9), 'views': {}}
            pixels = {}
            for cid, cam in camera_objs.items():
                scene.camera = cam
                rel = f'images/{cid}/{fi:06d}.png'
                scene.render.filepath = str(out/rel)
                bpy.ops.render.render(write_still=True)
                packet['views'][cid] = {'image': rel, 'timestamp_ns': packet['timestamp_ns']}
                uv = [None]*17
                c = cams[cid]
                for j, xyz in enumerate(joints):
                    if xyz is None:
                        continue
                    ndc = world_to_camera_view(scene, cam, Vector(xyz))
                    uv[j] = [ndc.x*w, (1-ndc.y)*h]
                    pc = [sum(c['R_world_to_camera'][row][k]*xyz[k] for k in range(3)) + c['t_world_to_camera_m'][row] for row in range(3)]
                    expected = [cfg['fx']*pc[0]/pc[2]+w/2, cfg['fy']*pc[1]/pc[2]+h/2]
                    max_projection_error = max(max_projection_error, math.dist(expected, uv[j]))
                pixels[cid] = uv
            frames.write(json.dumps(packet)+'\n')
            truths.write(json.dumps({'frame_id': fi, 'timestamp_ns': packet['timestamp_ns'],
                'joints_world_m': joints, 'projected_joints_px': pixels,
                'label_definition': '12 rig joint centres mapped to COCO body names; face landmarks unavailable; anatomical proxy, not exact COCO labels'})+'\n')
            frames.flush()
            truths.flush()
            print(f'FRAME_DONE {fi+1}/{cfg["frames"]}', flush=True)
    (out/'render_validation.json').write_text(json.dumps({'max_projection_difference_px': max_projection_error,
        'passed': max_projection_error < 0.01}, indent=2))
    if max_projection_error >= 0.01:
        raise RuntimeError(f'Camera export disagrees with Blender: {max_projection_error}px')
    scene.frame_set(0)
    scene.camera = camera_objs['cam00']
    scene.frame_end = cfg['frames']-1
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces.active.region_3d.view_perspective = 'CAMERA'
                area.spaces.active.shading.color_type = 'MATERIAL'
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'scene.blend'))
    print('SESSION_READY', str(out), flush=True)


if __name__ == '__main__':
    main()
