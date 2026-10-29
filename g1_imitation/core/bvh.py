"""CHINGMU BVH reader (offline scoring only; BVH is never a pipeline input).

Moved unchanged from the removed pose_sim/chingmu.py (2026-10-29): the parser, its unit/axis conversion and FK.
Input : local BVH with explicit YXZ intrinsic channels; source lengths are ASSUMED cm (template scale, no unit
        metadata). Frame zero must be the rest-pose template and is removed.
Output: FK positions in m and 3x3 orientation matrices in a Z-up template-aligned frame (permutation z,x,y); its
        alignment with the room/robot is unverified without calibration. Source-local seconds.
Failure: malformed dimensions, unsupported channel order, non-finite values or a non-template frame zero raise.
"""
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

BVH_TO_REP103_PERM = np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])


def bvh_vec_to_rep103_m(vec_cm: np.ndarray) -> np.ndarray:
    """Apply the assumed cm scale and (z,x,y) permutation to positions.

    The result is template-aligned and Z-up, not verified room/robot REP-103.
    """
    arr = np.asarray(vec_cm, dtype=float)
    return (arr @ BVH_TO_REP103_PERM.T) * 0.01


def bvh_rot_to_rep103(rot_bvh: np.ndarray) -> np.ndarray:
    """Change basis of 3x3 rotations into the template-aligned Z-up frame."""
    return BVH_TO_REP103_PERM @ rot_bvh @ BVH_TO_REP103_PERM.T


def parse_bvh_and_fk(bvh_path: Path) -> dict[str, Any]:
    """Parse supported CHINGMU BVH and compute hierarchy forward kinematics.

    Source lengths are assumed cm; output is template-aligned Z-up metres.
    """
    lines = [line.strip() for line in bvh_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    nodes: list[dict[str, Any]] = []
    stack: list[int] = []
    i = 0
    n_lines = len(lines)
    ch_cursor = 0

    while i < n_lines:
        line = lines[i]
        if line == "MOTION":
            i += 1
            break
        if line.startswith("ROOT ") or line.startswith("JOINT "):
            name = line.split()[1]
            parent = stack[-1] if stack else -1
            idx = len(nodes)
            nodes.append(
                {
                    "name": name,
                    "parent": parent,
                    "offset_cm": np.zeros(3, dtype=float),
                    "channels": [],
                    "ch_start": 0,
                    "is_end_site": False,
                }
            )
            stack.append(idx)
        elif line.startswith("End Site"):
            parent = stack[-1]
            parent_name = nodes[parent]["name"]
            idx = len(nodes)
            nodes.append(
                {
                    "name": f"{parent_name}_EndSite",
                    "parent": parent,
                    "offset_cm": np.zeros(3, dtype=float),
                    "channels": [],
                    "ch_start": ch_cursor,
                    "is_end_site": True,
                }
            )
            stack.append(idx)
        elif line.startswith("OFFSET "):
            parts = [float(x) for x in line.split()[1:4]]
            nodes[stack[-1]]["offset_cm"] = np.array(parts, dtype=float)
        elif line.startswith("CHANNELS "):
            tokens = line.split()
            n_ch = int(tokens[1])
            ch_names = tokens[2 : 2 + n_ch]
            nodes[stack[-1]]["channels"] = ch_names
            nodes[stack[-1]]["ch_start"] = ch_cursor
            ch_cursor += n_ch
        elif line == "}":
            stack.pop()
        i += 1

    n_total_frames = int(lines[i].split(":")[1].strip())
    frame_time_s = float(lines[i + 1].split(":")[1].strip())
    data_lines = lines[i + 2 : i + 2 + n_total_frames]
    raw_frames = np.array([[float(v) for v in row.split()] for row in data_lines], dtype=float)

    if raw_frames.shape != (n_total_frames, ch_cursor) or n_total_frames < 3:
        raise ValueError("BVH dimensions do not match header")
    if not np.isfinite(raw_frames).all() or frame_time_s <= 0:
        raise ValueError("Invalid BVH numeric data")
    for node in nodes:
        expected = (["Xposition", "Yposition", "Zposition"] if node["parent"] == -1 else []) + ["Yrotation", "Xrotation", "Zrotation"]
        if not node["is_end_site"] and node["channels"] != expected:
            raise ValueError("Unsupported BVH channel order")
    if not np.allclose(raw_frames[0, 3:], 0, atol=1e-8) or not np.allclose(raw_frames[0, [0,2]], 0, atol=1e-8):
        raise ValueError("Frame zero is not the expected rest-pose template")
    n_nodes = len(nodes)
    pos_bvh_cm = np.zeros((n_total_frames, n_nodes, 3), dtype=float)
    rot_bvh = np.zeros((n_total_frames, n_nodes, 3, 3), dtype=float)

    for node_idx, node in enumerate(nodes):
        parent = node["parent"]
        offset = node["offset_cm"]
        chs = node["channels"]
        c0 = node["ch_start"]

        if not chs:
            rot_local = np.broadcast_to(np.eye(3, dtype=float), (n_total_frames, 3, 3)).copy()
            trans_local = np.broadcast_to(offset, (n_total_frames, 3)).copy()
        elif len(chs) == 6:
            trans_local = raw_frames[:, c0 : c0 + 3] + offset[None, :]
            # CHINGMU BVH channel order: Yrotation Xrotation Zrotation (intrinsic YXZ)
            eul = raw_frames[:, c0 + 3 : c0 + 6]
            rot_local = Rotation.from_euler("YXZ", eul, degrees=True).as_matrix()
        else:
            trans_local = np.broadcast_to(offset, (n_total_frames, 3)).copy()
            eul = raw_frames[:, c0 : c0 + 3]
            rot_local = Rotation.from_euler("YXZ", eul, degrees=True).as_matrix()

        if parent == -1:
            pos_bvh_cm[:, node_idx, :] = trans_local
            rot_bvh[:, node_idx, :, :] = rot_local
        else:
            pos_bvh_cm[:, node_idx, :] = pos_bvh_cm[:, parent, :] + np.einsum(
                "fij,fj->fi", rot_bvh[:, parent, :, :], trans_local
            )
            rot_bvh[:, node_idx, :, :] = np.einsum(
                "fij,fjk->fik", rot_bvh[:, parent, :, :], rot_local
            )

    pos_world_m = bvh_vec_to_rep103_m(pos_bvh_cm)
    rot_world = np.einsum("ij,fnjk,lk->fnil", BVH_TO_REP103_PERM, rot_bvh, BVH_TO_REP103_PERM)

    node_names = [n["name"] for n in nodes]
    name_to_idx = {name: idx for idx, name in enumerate(node_names)}

    tpose_world_m = {name: pos_world_m[0, idx].copy() for idx, name in enumerate(node_names)}
    motion_world_m = pos_world_m[1:]
    motion_rot_world = rot_world[1:]
    n_motion = motion_world_m.shape[0]

    hips_traj = motion_world_m[:, name_to_idx["Hips"], :]
    hips_std_m = float(np.linalg.norm(np.std(hips_traj, axis=0)))
    mocap_frozen = bool(np.max(np.ptp(raw_frames[1:], axis=0)) < 1e-8)

    return {
        "bvh_path": str(bvh_path),
        "node_names": node_names,
        "name_to_idx": name_to_idx,
        "nodes": nodes,
        "frame_time_s": frame_time_s,
        "fps": round(1.0 / frame_time_s, 2),
        "n_total_frames": n_total_frames,
        "n_motion_frames": n_motion,
        "duration_s": (n_motion - 1) * frame_time_s,
        "sample_coverage_s": n_motion * frame_time_s,
        "num_bones": sum(not n["is_end_site"] for n in nodes),
        "tpose_world_m": tpose_world_m,
        "motion_world_m": motion_world_m,
        "motion_rot_world": motion_rot_world,
        "raw_motion_channels": raw_frames[1:],
        "hips_std_m": hips_std_m,
        "mocap_frozen": mocap_frozen,
    }

