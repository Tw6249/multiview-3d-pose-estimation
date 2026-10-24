"""Fetch only pinned inference assets; never changes Python/CUDA environments."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pose_sim.robot_assets import cache_root


def main():
    rows = json.loads((ROOT/'assets/humanoid/controller_comparison_manifest.json').read_text())['files']
    for row in rows:
        target = cache_root()/row['path']
        if target.exists():
            with target.open('rb') as f:
                if hashlib.file_digest(f,'sha256').hexdigest() == row['sha256']:
                    print('verified',row['path']);continue
            raise ValueError('Existing asset hash mismatch: '+str(target))
        target.parent.mkdir(parents=True,exist_ok=True)
        temp = target.with_suffix(target.suffix+'.part')
        with urllib.request.urlopen(row['url'],timeout=60) as response,temp.open('wb') as f:
            shutil.copyfileobj(response,f)
        with temp.open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest() != row['sha256']:
                raise ValueError('Downloaded asset hash mismatch')
        temp.replace(target)
        print('downloaded',row['path'])


if __name__ == '__main__':main()
