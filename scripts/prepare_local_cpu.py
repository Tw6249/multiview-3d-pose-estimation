"""Restore pinned G1/SONIC assets and generate a relocated CPU configuration.

Uses Python's standard library. Downloads stay in outputs/local_cpu; the original
configuration and any supplied ZIP are read-only. Every robot file is checked
against the pinned Git blob (including LFS SHA256 pointers); policies must match
the main configuration's SHA256. Failure aborts without publishing a new config.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/local_cpu"
COMMIT = "45c03facbf848e0c743091b7d9c17952096ff908"
REPO = "NVlabs/GR00T-WholeBodyControl"
PREFIX = "gear_sonic/data/robot_model/model_data/g1/"


def fetch(url: str) -> bytes:
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "g1-local-baseline/1"})
            with urllib.request.urlopen(req, timeout=60) as response:
                return response.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)
    raise AssertionError("unreachable")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_policy(url: str) -> bytes:
    """Resume parallel HTTP ranges; the caller verifies the full pinned SHA256."""
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "g1-local-baseline/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        size = int(response.headers["Content-Length"])
    cache = OUT / "download-parts" / hashlib.sha256(url.encode()).hexdigest()[:16]
    cache.mkdir(parents=True, exist_ok=True)
    chunk = 2 * 1024 * 1024

    def part(begin):
        end = min(begin + chunk, size) - 1
        dest = cache / f"{begin:010d}.part"
        if dest.exists() and dest.stat().st_size == end - begin + 1:
            return dest
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={begin}-{end}",
                    "User-Agent": "g1-local-baseline/1"})
                with urllib.request.urlopen(req, timeout=60) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {begin}-{end}/{size}":
                        raise ValueError("Server did not honor exact byte range")
                    data = response.read()
                if len(data) != end - begin + 1:
                    raise ValueError("Incomplete range")
                dest.write_bytes(data)
                print(f"Policy download {begin // chunk + 1}/{(size+chunk-1)//chunk}", flush=True)
                return dest
            except Exception:
                if attempt == 2:
                    raise
        raise AssertionError("unreachable")

    with ThreadPoolExecutor(max_workers=8) as pool:
        parts = list(pool.map(part, range(0, size, chunk)))
    return b"".join(p.read_bytes() for p in parts)


def git_blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def matches_git(data: bytes, expected: str) -> bool:
    pointer = ("version https://git-lfs.github.com/spec/v1\n"
               f"oid sha256:{sha256(data)}\nsize {len(data)}\n").encode()
    return git_blob(data) == expected or git_blob(pointer) == expected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-zip", type=Path, help="Optional existing G1 deploy ZIP, verified before reuse")
    ap.add_argument("--low-latency", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "g1_imitation/config/pipeline_config.json").read_text(encoding="utf-8"))
    tree_file = OUT / "upstream_tree.json"
    if not tree_file.exists():
        tree_file.write_bytes(fetch(f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1"))
    tree = json.loads(tree_file.read_text(encoding="utf-8-sig"))
    if tree["sha"] != COMMIT or tree.get("truncated"):
        raise ValueError("Wrong or incomplete pinned upstream tree")
    local_files = {}
    if args.model_zip:
        with zipfile.ZipFile(args.model_zip) as archive:
            for name in archive.namelist():
                if "/model/" in name and not name.endswith("/"):
                    local_files[name.split("/model/", 1)[1]] = archive.read(name)
    assets = OUT / "robot-assets"
    entries = [e for e in tree["tree"] if e["type"] == "blob" and e["path"].startswith(PREFIX)
               and (e["path"][len(PREFIX):] in ("scene_43dof.xml", "g1_29dof_with_hand.xml")
                    or e["path"][len(PREFIX):].startswith("meshes/"))]

    def restore_robot(entry):
        rel = entry["path"][len(PREFIX):]
        dest = assets / "sonic_official_g1" / rel
        data = dest.read_bytes() if dest.exists() else local_files.get(rel)
        source = "verified_local_copy"
        if data is None or not matches_git(data, entry["sha"]):
            source = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{entry['path']}"
            data = fetch(source)
            if data.startswith(b"version https://git-lfs.github.com/spec/v1"):
                if git_blob(data) != entry["sha"]:
                    raise ValueError(f"LFS pointer mismatch: {rel}")
                source = f"https://media.githubusercontent.com/media/{REPO}/{COMMIT}/{entry['path']}"
                data = fetch(source)
        if not matches_git(data, entry["sha"]):
            raise ValueError(f"Robot asset fingerprint mismatch: {rel}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return dict(path=str(dest.relative_to(OUT)), sha256=sha256(data), git_blob=entry["sha"], source=source)

    with ThreadPoolExecutor(max_workers=4) as pool:
        manifest = list(pool.map(restore_robot, entries))
    print(f"Verified {len(manifest)} pinned robot files", flush=True)
    variants = [("sonic_v11", "sonic_v1_1")]
    if args.low_latency:
        variants.append(("sonic_low_latency", "low_latency"))
    for key, remote in variants:
        for part in ("encoder", "decoder"):
            settings = cfg["s6_closed_loop"][key]
            dest = assets / settings[f"{part}_onnx_relative_path"]
            expected = settings[f"{part}_onnx_sha256"]
            source = f"https://huggingface.co/nvidia/GEAR-SONIC/resolve/main/{remote}/model_{part}.onnx"
            data = dest.read_bytes() if dest.exists() else b""
            if sha256(data) != expected:
                print(f"Downloading {key} {part}", flush=True)
                data = fetch_policy(source)
            if sha256(data) != expected:
                raise ValueError(f"Policy fingerprint mismatch: {key}/{part}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            manifest.append(dict(path=str(dest.relative_to(OUT)), sha256=expected, source=source))
    cfg["runtime"] = dict(python=str(Path(sys.executable).resolve()), robot_assets_root=str(assets),
                          rtmlib_cache_root=str(OUT / "vision-models"),
                          reference_repos_root=str(OUT / "reference-repos"))
    (OUT / "asset_manifest.json").write_text(json.dumps(dict(commit=COMMIT, files=manifest), indent=2), encoding="utf-8")
    (OUT / "pipeline_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"Configuration ready: {OUT / 'pipeline_config.json'}", flush=True)


if __name__ == "__main__":
    main()
