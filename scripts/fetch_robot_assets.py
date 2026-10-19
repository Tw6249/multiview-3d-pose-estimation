"""Download content-pinned research assets into the external runtime cache."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pose_sim.robot_assets import cache_root,require_asset


def fetch():
    manifest=json.loads((ROOT/'assets/humanoid/manifest.json').read_text())
    cache=cache_root();cache.mkdir(parents=True,exist_ok=True)
    entries=manifest['files']
    def one(entry):
        path=(cache/entry['path']).resolve()
        if not path.is_relative_to(cache.resolve()):raise ValueError('Asset path escapes cache')
        if path.exists():require_asset(entry['path']);return
        path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.part')
        try:
            if entry.get('google_drive_id'):
                import gdown
                if not gdown.download(id=entry['google_drive_id'],output=str(temporary),quiet=True):raise RuntimeError('Download failed')
            elif entry.get('url'):
                urllib.request.urlretrieve(entry['url'],temporary)
            else:raise ValueError('Asset has no downloadable source')
            with temporary.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
            if digest!=entry['sha256'] or temporary.stat().st_size!=entry['size']:raise ValueError('Downloaded asset hash/size mismatch: '+entry['path'])
            temporary.replace(path)
            print('VERIFIED',entry['path'],flush=True)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    with ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(one,entries))
    print('HUMANOID ASSETS VERIFIED',len(entries),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    fetch()
