"""Synthetic clutter scene, step 2 of 3: render four synchronized RGB cameras in Blender (Cycles).

Run: <blender 4.5>/blender.exe --background --python scripts/blender_clutter_scene.py -- --mesh <mesh.npz> --out <dir>
Input : mesh npz from scripts/blender_clutter_mesh.py (SMPL-X vertices per frame, world Z-up, metres, ground z = 0).
Scene : textured floor (z = 0), walls with posters, and clutter: box stack, table + chair, traffic cones, tripod, potted
        plant, shelf, ball, a coat rack with a hanging coat (person-like distractor), and a crate between the walking area
        and camera C02 that hides the lower legs in that view at times. No other person.
        Human: the SMPL-X mesh, coloured per body part (argmax LBS joint): skin, hair, shirt, trousers, shoes.
Output (same layout as data/public_mamma_session): cameras.json (OpenCV pinhole: K, R_world_to_camera,
        t_world_to_camera_m, zero distortion), frames.jsonl (one timestamp per frame, all cameras exposed together),
        images/<cam>/<frame>.jpg, source.json (motion source, render settings).
Truth : exact (rendered, synchronized, calibrated by construction); the SMPL-X joints in the mesh npz are the body truth.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

W, H = 1280, 960
CAMERAS = [  # name, azimuth deg (camera position around the origin), radius m, height m, focal mm
    ("C01", 20.0, 4.6, 1.55, 26.0),
    ("C02", 115.0, 4.3, 1.95, 24.0),
    ("C03", 205.0, 4.8, 1.45, 28.0),
    ("C04", 290.0, 4.4, 2.25, 24.0),
]
LOOK_AT = (0.0, 0.0, 0.9)
SENSOR_W_MM = 36.0
SKIN, HAIR, SHIRT, TROUSERS, SHOES = range(5)
PART_MATERIAL = {12: SKIN, 15: SKIN, 22: SKIN, 23: HAIR, 24: HAIR, 18: SKIN, 19: SKIN, 20: SKIN, 21: SKIN,
                 3: SHIRT, 6: SHIRT, 9: SHIRT, 13: SHIRT, 14: SHIRT, 16: SHIRT, 17: SHIRT,
                 0: TROUSERS, 1: TROUSERS, 2: TROUSERS, 4: TROUSERS, 5: TROUSERS,
                 7: SHOES, 8: SHOES, 10: SHOES, 11: SHOES}          # 25..54 hands -> skin


def mat(name, rgb, rough=0.7, checker=None, noise=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    if checker is not None:
        c = nt.nodes.new("ShaderNodeTexChecker")
        c.inputs["Color1"].default_value = (*checker[0], 1.0)
        c.inputs["Color2"].default_value = (*checker[1], 1.0)
        c.inputs["Scale"].default_value = checker[2]
        nt.links.new(c.outputs["Color"], bsdf.inputs["Base Color"])
    elif noise is not None:
        n = nt.nodes.new("ShaderNodeTexNoise")
        n.inputs["Scale"].default_value = noise[2]
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = (*noise[0], 1.0)
        ramp.color_ramp.elements[1].color = (*noise[1], 1.0)
        nt.links.new(n.outputs["Fac"], ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    return m


def box(name, center, size, material, rot_z=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=center, rotation=(0.0, 0.0, math.radians(rot_z)))
    o = bpy.context.object
    o.name = name
    o.scale = size
    o.data.materials.append(material)
    return o


def cyl(name, center, radius, depth, material, verts=24, r2=None):
    if r2 is None:
        bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=radius, depth=depth, location=center)
    else:
        bpy.ops.mesh.primitive_cone_add(vertices=verts, radius1=radius, radius2=r2, depth=depth, location=center)
    o = bpy.context.object
    o.name = name
    o.data.materials.append(material)
    return o


def build_clutter():
    cardboard = mat("cardboard", (0.55, 0.40, 0.25), noise=((0.45, 0.32, 0.2), (0.62, 0.47, 0.3), 12.0))
    wood = mat("wood", (0.45, 0.30, 0.18), noise=((0.35, 0.22, 0.12), (0.55, 0.38, 0.22), 30.0))
    metal = mat("metal", (0.25, 0.25, 0.27), rough=0.35)
    orange = mat("cone_orange", (0.9, 0.35, 0.05), rough=0.5)
    green = mat("leaves", (0.12, 0.35, 0.10), noise=((0.08, 0.25, 0.06), (0.2, 0.45, 0.15), 25.0))
    blue = mat("ball", (0.10, 0.25, 0.75), rough=0.4)
    coat = mat("coat", (0.15, 0.17, 0.22), rough=0.9)
    white = mat("shelf_white", (0.85, 0.85, 0.82))
    for i, (c, s, rz) in enumerate([((2.1, 1.3, 0.25), (0.6, 0.5, 0.5), 10), ((2.15, 1.3, 0.7), (0.5, 0.45, 0.4), 25),
                                    ((2.6, 0.7, 0.2), (0.4, 0.4, 0.4), -15)]):
        box(f"box_{i}", c, s, cardboard, rz)
    box("crate_occluder", (-0.75, 2.2, 0.45), (0.9, 0.5, 0.9), wood, 15)        # between the walk and camera C02
    box("table_top", (-2.4, 0.9, 0.74), (1.3, 0.75, 0.05), wood, 5)
    for dx in (-0.58, 0.58):
        for dy in (-0.32, 0.32):
            cyl("table_leg", (-2.4 + dx, 0.9 + dy, 0.36), 0.03, 0.72, metal)
    box("chair_seat", (-2.2, 0.2, 0.45), (0.45, 0.45, 0.05), wood)
    box("chair_back", (-2.2, -0.02, 0.72), (0.45, 0.04, 0.5), wood)
    for dx in (-0.2, 0.2):
        for dy in (-0.2, 0.2):
            cyl("chair_leg", (-2.2 + dx, 0.2 + dy, 0.22), 0.02, 0.44, metal)
    for i, c in enumerate([(1.3, -1.9), (-0.9, -2.2), (2.3, -0.4)]):
        cyl(f"cone_{i}", (c[0], c[1], 0.3), 0.18, 0.6, orange, r2=0.03)
    for k in range(3):                                                            # tripod
        a = math.radians(120 * k)
        bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=0.015, depth=1.5,
                                            location=(0.4 + 0.18 * math.cos(a), -2.7 + 0.18 * math.sin(a), 0.72),
                                            rotation=(0.13 * math.sin(a), -0.13 * math.cos(a), 0.0))
        bpy.context.object.data.materials.append(metal)
    box("tripod_head", (0.4, -2.7, 1.5), (0.12, 0.08, 0.08), metal)
    cyl("pot", (-2.2, -1.7, 0.2), 0.22, 0.4, mat("pot", (0.6, 0.25, 0.15)))
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=0.45, location=(-2.2, -1.7, 0.85))
    bpy.context.object.scale = (1.0, 1.0, 1.3)
    bpy.context.object.data.materials.append(green)
    box("shelf", (2.9, -1.6, 0.9), (0.4, 1.0, 1.8), white, 20)
    for z in (0.5, 0.95, 1.4):
        box("shelf_item", (2.8, -1.6, z + 0.12), (0.25, 0.3, 0.22), cardboard, 20)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.15, location=(1.1, 1.8, 0.15))
    bpy.context.object.data.materials.append(blue)
    cyl("coat_rack_pole", (1.9, 2.3, 0.85), 0.025, 1.7, metal)
    cyl("coat_rack_base", (1.9, 2.3, 0.02), 0.25, 0.04, metal)
    box("coat", (1.9, 2.3, 1.2), (0.42, 0.18, 0.75), coat)
    cyl("coat_sleeve_l", (1.66, 2.3, 1.15), 0.06, 0.6, coat)
    cyl("coat_sleeve_r", (2.14, 2.3, 1.15), 0.06, 0.6, coat)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.11, location=(1.9, 2.3, 1.72))      # hat on the rack
    bpy.context.object.data.materials.append(metal)


def build_room():
    bpy.ops.mesh.primitive_plane_add(size=16.0, location=(0.0, 0.0, 0.0))
    bpy.context.object.name = "floor_z0"
    bpy.context.object.data.materials.append(mat("floor", (0.5, 0.5, 0.5), rough=0.6,
                                                 checker=((0.42, 0.40, 0.37), (0.55, 0.53, 0.50), 24.0)))
    wall = mat("wall", (0.8, 0.78, 0.74), noise=((0.72, 0.70, 0.66), (0.86, 0.84, 0.80), 4.0))
    for c, s in [((0.0, 7.0, 1.6), (14.0, 0.1, 3.2)), ((0.0, -7.0, 1.6), (14.0, 0.1, 3.2)),
                 ((7.0, 0.0, 1.6), (0.1, 14.0, 3.2)), ((-7.0, 0.0, 1.6), (0.1, 14.0, 3.2))]:
        box("wall", c, s, wall)
    rng = np.random.default_rng(3)
    for i in range(10):                                                          # posters on the walls
        u, z = rng.uniform(-5.0, 5.0), rng.uniform(1.0, 2.3)
        wd, ht = rng.uniform(0.5, 1.4), rng.uniform(0.5, 1.0)
        col = tuple(float(x) for x in rng.uniform(0.1, 0.9, 3))
        c, size = [((u, 6.93, z), (wd, 0.02, ht)), ((u, -6.93, z), (wd, 0.02, ht)),
                   ((6.93, u, z), (0.02, wd, ht)), ((-6.93, u, z), (0.02, wd, ht))][i % 4]
        box(f"poster_{i}", c, size, mat(f"poster_{i}", col, rough=0.5))
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy, sun.angle = 2.2, math.radians(8)
    o = bpy.data.objects.new("sun", sun)
    bpy.context.scene.collection.objects.link(o)
    o.rotation_euler = (math.radians(50), math.radians(10), math.radians(35))
    area = bpy.data.lights.new("area", "AREA")
    area.energy, area.size = 500.0, 5.0
    o = bpy.data.objects.new("area", area)
    bpy.context.scene.collection.objects.link(o)
    o.location = (-1.0, 1.0, 3.2)
    w = bpy.context.scene.world or bpy.data.worlds.new("world")
    bpy.context.scene.world = w
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.55, 0.6, 0.68, 1.0)
    w.node_tree.nodes["Background"].inputs[1].default_value = 0.35


def build_human(npz):
    verts, faces, part, joints = npz["verts"], npz["faces"], npz["part"], npz["joints"]
    mesh = bpy.data.meshes.new("smplx_body")
    mesh.from_pydata(verts[0].tolist(), [], faces.tolist())
    mesh.update()
    obj = bpy.data.objects.new("human", mesh)
    bpy.context.scene.collection.objects.link(obj)
    for name, rgb, r in [("skin", (0.78, 0.57, 0.45), 0.55), ("hair", (0.08, 0.06, 0.05), 0.8),
                         ("shirt", (0.70, 0.15, 0.15), 0.8), ("trousers", (0.12, 0.18, 0.35), 0.85),
                         ("shoes", (0.9, 0.9, 0.88), 0.6)]:
        mesh.materials.append(mat(name, rgb, rough=r))
    vmat = np.array([PART_MATERIAL.get(int(p), SKIN) for p in part])
    up = joints[0, 15] - joints[0, 12]
    up /= np.linalg.norm(up)
    hair = (part == 15) & ((verts[0] - joints[0, 15]) @ up > 0.045)
    vmat[hair] = HAIR
    pm = np.array([np.bincount(vmat[f], minlength=5).argmax() for f in faces])
    mesh.polygons.foreach_set("material_index", pm.astype(np.int32))
    mesh.polygons.foreach_set("use_smooth", np.ones(len(faces), bool))
    mesh.update()
    return mesh


def build_cameras():
    cams = {}
    for name, az, r, h, f in CAMERAS:
        data = bpy.data.cameras.new(name)
        data.lens, data.sensor_width, data.sensor_fit = f, SENSOR_W_MM, "HORIZONTAL"
        o = bpy.data.objects.new(name, data)
        bpy.context.scene.collection.objects.link(o)
        o.location = (r * math.cos(math.radians(az)), r * math.sin(math.radians(az)), h)
        o.rotation_euler = (Vector(LOOK_AT) - o.location).to_track_quat("-Z", "Y").to_euler()
        cams[name] = o
    bpy.context.view_layer.update()
    out = {}
    for name, o in cams.items():
        Rbc = np.array(o.matrix_world.to_3x3().normalized())          # Blender camera axes in world (columns)
        C = np.array(o.matrix_world.translation)
        Rwc = np.diag([1.0, -1.0, -1.0]) @ Rbc.T                     # world -> OpenCV camera (x right, y down, z fwd)
        fx = o.data.lens / SENSOR_W_MM * W
        out[name] = dict(K=[[fx, 0.0, (W - 1) / 2.0], [0.0, fx, (H - 1) / 2.0], [0.0, 0.0, 1.0]],
                         R_world_to_camera=Rwc.tolist(), t_world_to_camera_m=(-Rwc @ C).tolist(),
                         distortion=[0.0, 0.0, 0.0, 0.0], width=W, height=H)
    return cams, out


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--mesh", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--samples", type=int, default=32)
    a.add_argument("--frames", type=int, default=0, help="0 = all frames of the mesh file")
    a = a.parse_args(sys.argv[sys.argv.index("--") + 1:])
    t0 = time.perf_counter()
    npz = np.load(a.mesh)
    out = Path(a.out).resolve()        # render.filepath must be absolute: Blender does not resolve against the CWD
    if (out / "frames.jsonl").exists():
        raise RuntimeError(f"{out} already holds a session; use a new folder")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    build_room()
    build_clutter()
    mesh = build_human(npz)
    cams, cam_json = build_cameras()
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    used = [d.name for d in prefs.devices if d.type == "OPTIX"]
    for d in prefs.devices:
        d.use = d.type == "OPTIX"
    if not used:
        raise RuntimeError("no OptiX device")
    sc.cycles.device = "GPU"
    sc.cycles.samples = a.samples
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPTIX"
    sc.cycles.max_bounces = 4
    sc.cycles.seed = 0
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = W, H, 100
    sc.render.image_settings.file_format = "JPEG"
    sc.render.image_settings.quality = 92
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "Standard"
    verts = npz["verts"]
    T = len(verts) if a.frames <= 0 else min(a.frames, len(verts))
    fps = float(npz["fps"])
    for c in cams:
        (out / "images" / c).mkdir(parents=True, exist_ok=True)
    lines = []
    for k in range(T):
        mesh.vertices.foreach_set("co", verts[k].astype(np.float64).ravel())
        mesh.update()
        views = {}
        for c, o in cams.items():
            sc.camera = o
            rel = f"images/{c}/{k:06d}.jpg"
            sc.render.filepath = str(out / rel)
            bpy.ops.render.render(write_still=True)
            views[c] = dict(timestamp_ns=int(round(k / fps * 1e9)), image=rel)
        lines.append(json.dumps(dict(frame_id=k, timestamp_ns=int(round(k / fps * 1e9)), views=views)))
        if k % 20 == 0:
            print(f"frame {k}/{T} {time.perf_counter() - t0:.0f}s", flush=True)
    (out / "cameras.json").write_text(json.dumps(dict(schema_version=1, units="metre",
                                                      world_axes="Blender world = pipeline world: Z up, ground z = 0",
                                                      cameras=cam_json), indent=1), encoding="utf-8")
    (out / "frames.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "source.json").write_text(json.dumps(dict(
        source="synthetic Blender clutter scene (scripts/blender_clutter_scene.py)", blender=bpy.app.version_string,
        mesh=str(Path(a.mesh).resolve()), motion=f"SMPL-X fit of AdvancedTake {str(npz['take'])} from slot "
        f"{int(npz['start_slot'])} (30 Hz)", frames=T, fps=fps, render=dict(engine="CYCLES", device=used,
        samples=a.samples, denoiser="OPTIX", width=W, height=H), cameras=[c[0] for c in CAMERAS],
        reference_labels_available=True, truth="mesh npz joints (SMPL-X), same world frame",
        seconds=round(time.perf_counter() - t0, 1)), indent=1), encoding="utf-8")
    print(f"done {T} frames x {len(cams)} cameras in {time.perf_counter() - t0:.0f}s -> {out}")


main()
