"""Verified single-stream downloader, extractor, and auditor for EgoHumans.

Reads `config/public_egohumans.json` and manages EgoHumans dataset assets in
`D:\\datasets\\egohumans` (linked into the project at `data/egohumans` and
`outputs/public_datasets/egohumans`).

Supports three authentication / download modes:
1. `--oauth-token <BEARER_TOKEN>` (or `GOOGLE_DRIVE_TOKEN` env var):
   Uses Google Drive API v3 (`alt=media&acknowledgeAbuse=true`) with the user's
   authenticated Google account quota, bypassing anonymous `Quota exceeded`.
2. `--cookies <COOKIES_TXT_PATH>`:
   Uses a Netscape-format `cookies.txt` from a logged-in Google browser session.
3. Anonymous single-stream (`Range: bytes=<offset>-`):
   Uses a single continuous HTTP stream per file with dynamic `uuid` virus-scan
   confirmation so that anonymous request quota is not exhausted by chunking.

All downloads are verified against exact byte counts and MD5 checksums before
replacing `.part` files. Partial outputs are never accepted as final files.
"""
from __future__ import annotations

import argparse
import hashlib
import http.cookiejar
import json
import os
from pathlib import Path
import re
import socket
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "public_egohumans.json"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def detect_proxy(explicit_proxy: str | None) -> str | None:
    if explicit_proxy:
        return explicit_proxy
    env_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    if env_proxy:
        return env_proxy
    for port in (7897, 7890, 10809):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return f"http://127.0.0.1:{port}"
    return None


def make_opener(
    proxy_url: str | None,
    cookies_path: Path | None = None,
) -> tuple[urllib.request.OpenerDirector, http.cookiejar.CookieJar]:
    ctx = ssl.create_default_context()
    if cookies_path is not None:
        cj: http.cookiejar.CookieJar = http.cookiejar.MozillaCookieJar(str(cookies_path))
        cj.load(ignore_discard=True, ignore_expires=True)
    else:
        cj = http.cookiejar.CookieJar()
    handlers: list[urllib.request.BaseHandler] = [
        urllib.request.HTTPSHandler(context=ctx),
        urllib.request.HTTPCookieProcessor(cj),
    ]
    if proxy_url:
        handlers.insert(0, urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url}))
    opener = urllib.request.build_opener(*handlers)
    opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")]
    return opener, cj


def compute_hashes(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def resolve_download_request(
    file_id: str,
    opener: urllib.request.OpenerDirector,
    oauth_token: str | None,
) -> tuple[str, dict[str, str]]:
    if oauth_token:
        url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&acknowledgeAbuse=true"
        return url, {"Authorization": f"Bearer {oauth_token}"}

    uc_url = f"https://drive.google.com/uc?id={file_id}&export=download"
    with opener.open(uc_url, timeout=25) as resp:
        ctype = resp.headers.get("Content-Type", "")
        if "text/html" not in ctype:
            return resp.geturl(), {}
        html = resp.read(32768).decode("utf-8", "ignore")
        if "Quota exceeded" in html or "Too many users have viewed or downloaded" in html:
            raise RuntimeError(
                f"Google Drive anonymous quota exceeded for file_id={file_id} (owner: egohumans2023@gmail.com). "
                "Provide --oauth-token <GOOGLE_OAUTH_TOKEN> or --cookies <cookies.txt>, or retry after Google's 24h quota window resets."
            )
        uuid_m = re.search(r'name="uuid"\s+value="([^"]+)"', html)
        if uuid_m:
            uuid_val = uuid_m.group(1)
            dl_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t&uuid={uuid_val}"
            return dl_url, {}
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t", {}


def download_single_stream(
    file_id: str,
    destination: Path,
    expected_bytes: int,
    expected_md5: str | None,
    proxy_url: str | None,
    oauth_token: str | None = None,
    cookies_path: Path | None = None,
) -> dict:
    if destination.exists():
        local_size = destination.stat().st_size
        if local_size == expected_bytes:
            md5_hex, sha256_hex = compute_hashes(destination)
            if expected_md5 and md5_hex != expected_md5:
                raise ValueError(f"MD5 mismatch on existing file {destination}: {md5_hex} != {expected_md5}")
            print(f"[VERIFIED] {destination.name} ({local_size:,} bytes, md5={md5_hex})", flush=True)
            return {
                "file_id": file_id,
                "bytes": local_size,
                "md5": md5_hex,
                "sha256": sha256_hex,
                "status": "verified",
            }
        destination.unlink()

    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_suffix(destination.suffix + ".part")
    offset = part_path.stat().st_size if part_path.exists() else 0
    if offset > expected_bytes:
        part_path.unlink()
        offset = 0

    t0 = time.time()
    initial_offset = offset
    last_log_bytes = offset
    retries = 0

    while offset < expected_bytes:
        try:
            opener, _ = make_opener(proxy_url, cookies_path)
            dl_url, extra_headers = resolve_download_request(file_id, opener, oauth_token)
            headers = dict(extra_headers)
            if offset > 0:
                headers["Range"] = f"bytes={offset}-"
            req = urllib.request.Request(dl_url, headers=headers)
            with opener.open(req, timeout=60) as resp:
                ctype = resp.headers.get("Content-Type", "")
                if "text/html" in ctype:
                    body = resp.read(2048).decode("utf-8", "ignore")
                    if "Quota exceeded" in body or "Too many users" in body:
                        raise RuntimeError(
                            f"Google Drive anonymous quota exceeded for {destination.name} (file_id={file_id}, owner=egohumans2023@gmail.com). "
                            "Pass --oauth-token <GOOGLE_OAUTH_TOKEN> or --cookies <cookies.txt>, or retry after the 24h quota reset."
                        )
                    raise ValueError(f"Unexpected HTML response for {destination.name}: {body[:160]}")
                if offset > 0 and resp.status != 206:
                    raise ValueError(f"Server did not honor byte offset {offset} (status={resp.status})")

                mode = "ab" if offset > 0 else "wb"
                with part_path.open(mode) as f:
                    while offset < expected_bytes:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        offset += len(chunk)
                        retries = 0
                        if offset - last_log_bytes >= 32 * 1024 * 1024 or offset == expected_bytes:
                            f.flush()
                            last_log_bytes = offset
                            elapsed = max(0.001, time.time() - t0)
                            speed_mb = ((offset - initial_offset) / (1024**2)) / elapsed
                            pct = 100.0 * offset / expected_bytes
                            print(
                                f"  [{destination.name}] {pct:5.1f}% ({offset / (1024**2):.1f}/{expected_bytes / (1024**2):.1f} MB) "
                                f"speed={speed_mb:.2f} MB/s",
                                flush=True,
                            )
        except RuntimeError:
            if part_path.exists() and part_path.stat().st_size == 0:
                part_path.unlink()
            raise
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", "ignore")
            if exc.code == 403 and "downloadQuotaExceeded" in err_body:
                if part_path.exists() and part_path.stat().st_size == 0:
                    part_path.unlink()
                raise RuntimeError(
                    f"Google Drive API reported downloadQuotaExceeded for {destination.name} ({file_id}). "
                    "Provide a valid personal OAuth token via --oauth-token or --cookies."
                ) from exc
            retries += 1
            if retries > 8:
                raise
            time.sleep(2.0 * retries)
        except Exception as exc:
            retries += 1
            if retries > 8:
                raise RuntimeError(f"Download failed at {offset}/{expected_bytes}: {exc}") from exc
            time.sleep(2.0 * retries)

    if part_path.stat().st_size != expected_bytes:
        raise RuntimeError(f"Incomplete download: {part_path.stat().st_size} != {expected_bytes}")

    md5_hex, sha256_hex = compute_hashes(part_path)
    if expected_md5 and md5_hex != expected_md5:
        raise ValueError(f"MD5 verification failed for {destination.name}: got {md5_hex}, expected {expected_md5}")

    part_path.replace(destination)
    print(f"[DONE] {destination} ({expected_bytes:,} bytes, md5={md5_hex})", flush=True)
    return {
        "file_id": file_id,
        "bytes": expected_bytes,
        "md5": md5_hex,
        "sha256": sha256_hex,
        "status": "downloaded_and_verified",
    }


def extract_subsequence_tar(tar_path: Path, extract_parent: Path, expected_dir: Path, strip_components: int = 8) -> dict:
    if expected_dir.exists() and (expected_dir / "colmap").exists() and (expected_dir / "exo").exists():
        print(f"[SKIP EXTRACT] {expected_dir} already extracted", flush=True)
    else:
        extract_parent.mkdir(parents=True, exist_ok=True)
        print(f"[EXTRACT] {tar_path.name} -> {extract_parent} (--strip-components={strip_components})...", flush=True)
        cmd = [
            "tar.exe",
            "-xzf",
            str(tar_path),
            f"--strip-components={strip_components}",
            "-C",
            str(extract_parent),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"tar extraction failed ({res.returncode}): {res.stderr}")

    exo_cams = sorted(p.name for p in (expected_dir / "exo").iterdir() if p.is_dir()) if (expected_dir / "exo").exists() else []
    ego_cams = sorted(p.name for p in (expected_dir / "ego").iterdir() if p.is_dir()) if (expected_dir / "ego").exists() else []
    processed = sorted(p.name for p in (expected_dir / "processed_data").iterdir()) if (expected_dir / "processed_data").exists() else []
    colmap_files = sorted(p.name for p in (expected_dir / "colmap" / "workplace").iterdir()) if (expected_dir / "colmap" / "workplace").exists() else []

    sample_cam_frames = 0
    if exo_cams:
        cam01_imgs = expected_dir / "exo" / exo_cams[0] / "images"
        if cam01_imgs.exists():
            sample_cam_frames = sum(1 for _ in cam01_imgs.glob("*.jpg"))

    return {
        "extracted_path": str(expected_dir),
        "exo_cameras": exo_cams,
        "num_exo_cameras": len(exo_cams),
        "ego_cameras": ego_cams,
        "frames_per_exo_camera": sample_cam_frames,
        "colmap_workplace_files": colmap_files,
        "processed_data_entries": processed,
    }


def build_audit_report(output_root: Path, cfg: dict) -> dict:
    repo_dir = output_root / "egohumans_repo"
    catalog_path = output_root / "catalog.json"
    configs_dir = repo_dir / "egohumans" / "configs"
    yaml_configs = sorted(str(p.relative_to(repo_dir)) for p in configs_dir.rglob("*.yaml")) if configs_dir.exists() else []
    blend_scenes = sorted(str(p.relative_to(repo_dir)) for p in (repo_dir / "egohumans" / "assets").rglob("*.blend")) if (repo_dir / "egohumans" / "assets").exists() else []

    target_states = {}
    for key, meta in cfg["targets"].items():
        dest = output_root / meta["rel_path"]
        part = dest.with_suffix(dest.suffix + ".part")
        if dest.exists() and dest.stat().st_size == meta["bytes"]:
            state = "downloaded_verified"
        elif part.exists() and part.stat().st_size > 0:
            state = f"partial ({part.stat().st_size}/{meta['bytes']} bytes)"
        else:
            state = "pending_gdrive_quota"
        target_states[key] = {
            "rel_path": meta["rel_path"],
            "file_id": meta["file_id"],
            "expected_bytes": meta["bytes"],
            "expected_md5": meta["md5"],
            "state": state,
        }

    return {
        "dataset": cfg["dataset"],
        "official_site": cfg["official_site"],
        "official_github": cfg["official_github"],
        "google_drive_root_id": cfg["google_drive_root_id"],
        "owner_account": cfg["owner_account"],
        "storage_root": str(output_root),
        "project_junctions": {
            "data/egohumans": (REPO_ROOT / "data" / "egohumans").exists(),
            "outputs/public_datasets/egohumans": (REPO_ROOT / "outputs" / "public_datasets" / "egohumans").exists(),
        },
        "local_assets_downloaded": {
            "official_repository_cloned": repo_dir.exists(),
            "official_repository_path": str(repo_dir),
            "sequence_yaml_configs_count": len(yaml_configs),
            "blender_scene_files": blend_scenes,
            "full_drive_catalog_indexed": catalog_path.exists(),
            "catalog_path": str(catalog_path),
            "catalog_total_files": 142,
            "catalog_total_bytes": 600067087226,
        },
        "targets": target_states,
        "gdrive_quota_note": (
            "All 142 archive/annotation files in the official EgoHumans Google Drive are owned by a single account "
            "(egohumans2023@gmail.com). When Google Drive's 24-hour anonymous download quota is exceeded on that "
            "owner account (HTTP 403 downloadQuotaExceeded / 'Quota exceeded' HTML page), unauthenticated downloads "
            "are blocked by Google Drive across all IPs. Pass --oauth-token <GOOGLE_OAUTH_TOKEN> or --cookies <cookies.txt> "
            "to download immediately using an authenticated Google account's quota, or re-run after the 24h reset."
        ),
    }


def main() -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Download EgoHumans dataset subset for multi-view pose & retargeting.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(cfg["storage_root"]),
        help="Root directory on disk where EgoHumans data is stored.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=cfg["recommended_subset"]["targets"],
        choices=sorted(cfg["targets"].keys()),
        help="Dataset items to download.",
    )
    parser.add_argument("--proxy", type=str, default=None, help="Explicit HTTP/HTTPS proxy URL.")
    parser.add_argument(
        "--oauth-token",
        type=str,
        default=os.environ.get("GOOGLE_DRIVE_TOKEN"),
        help="Google Drive OAuth2 Bearer token (bypasses anonymous quota limit).",
    )
    parser.add_argument(
        "--cookies",
        type=Path,
        default=None,
        help="Path to Netscape-format cookies.txt from a logged-in Google session.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Write and print current local dataset & audit status without downloading.",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_root / "audit.json"

    if args.status:
        audit = build_audit_report(args.output_root, cfg)
        audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        return 0

    proxy_url = detect_proxy(args.proxy)
    print(f"[INFO] Output root: {args.output_root}", flush=True)
    print(f"[INFO] Using proxy: {proxy_url or 'DIRECT'}", flush=True)
    print(
        f"[INFO] Auth mode: {'OAuth2 Bearer token' if args.oauth_token else ('Cookies (' + str(args.cookies) + ')' if args.cookies else 'Anonymous single-stream')}",
        flush=True,
    )

    exit_code = 0
    for target_key in args.targets:
        meta = cfg["targets"][target_key]
        dest = args.output_root / meta["rel_path"]
        try:
            download_single_stream(
                file_id=meta["file_id"],
                destination=dest,
                expected_bytes=meta["bytes"],
                expected_md5=meta.get("md5"),
                proxy_url=proxy_url,
                oauth_token=args.oauth_token,
                cookies_path=args.cookies,
            )
            if "extract_dir" in meta:
                extract_subsequence_tar(
                    tar_path=dest,
                    extract_parent=args.output_root / meta["extract_dir"],
                    expected_dir=args.output_root / meta["expected_dir"],
                    strip_components=meta["strip_components"],
                )
        except RuntimeError as exc:
            print(f"[QUOTA BLOCKED] {target_key}: {exc}", flush=True)
            exit_code = 2
            break

    audit = build_audit_report(args.output_root, cfg)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[AUDIT] Updated {audit_path}", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
