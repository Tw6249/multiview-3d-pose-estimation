"""Resolve and verify pinned, externally cached robot baseline assets."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cache_root():
    path=ROOT/'.robot-runtime.json'
    if not path.exists():path=ROOT/'.runtime.json'
    runtime = json.loads(path.read_text(encoding='utf-8'))
    return Path(runtime['python']).parents[2]/'robot-assets'


def require_asset(relative):
    manifest = json.loads((ROOT/'assets/humanoid/manifest.json').read_text(encoding='utf-8'))
    entry = next((row for row in manifest['files'] if row['path'] == relative), None)
    if entry is None:
        raise ValueError(f'Unpinned robot asset: {relative}')
    path = cache_root()/relative
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    if digest != entry['sha256']:
        raise ValueError(f'Robot asset checksum mismatch: {relative}')
    return path
