"""Install the reproducible robot environment without modifying CPU/CUDA baselines."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from bootstrap import ROOT,CACHE,PYTHON_VERSION,UV_VERSION


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    args=parser.parse_args()
    baseline=ROOT/'.runtime.json'
    base=Path(json.loads(baseline.read_text())['python']).parents[2] if baseline.exists() else CACHE
    base.mkdir(parents=True,exist_ok=True);uv=base/'bootstrap-tools/bin/uv.exe'
    if not uv.exists():
        subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--target',str(base/'bootstrap-tools'),f'uv=={UV_VERSION}'],check=True)
    if subprocess.check_output([str(uv),'--version'],text=True).strip().split()[1]!=UV_VERSION:raise RuntimeError('Unexpected uv version')
    env=dict(os.environ,UV_CACHE_DIR=str(base/'uv-cache'),UV_PYTHON_INSTALL_DIR=str(base/'python'),PYTHONNOUSERSITE='1',PYTHONUTF8='1')
    python=base/'venv-py311-robot/Scripts/python.exe'
    if not python.exists():
        if baseline.exists():interpreter=json.loads(baseline.read_text())['python']
        else:
            subprocess.run([str(uv),'python','install',PYTHON_VERSION,'--no-bin','--no-registry'],env=env,check=True)
            interpreter=PYTHON_VERSION
        subprocess.run([str(uv),'venv','--python',interpreter,str(python.parents[1])],env=env,check=True)
    if subprocess.check_output([str(python),'-c','import platform; print(platform.python_version())'],text=True).strip()!=PYTHON_VERSION:
        raise RuntimeError('Wrong Python version in robot environment')
    lock=ROOT/'requirements-robot-lock.txt'
    subprocess.run([str(uv),'pip','sync','--python',str(python),'--require-hashes',str(lock)],env=env,check=True)
    runtime={'python':str(python.resolve()),'python_version':PYTHON_VERSION,'device':'cuda','render_device':'MuJoCo OpenGL',
             'dependency_lock_sha256':hashlib.sha256(lock.read_bytes()).hexdigest(),
             'asset_manifest_sha256':hashlib.sha256((ROOT/'assets/humanoid/manifest.json').read_bytes()).hexdigest()}
    (ROOT/'.robot-runtime.json').write_text(json.dumps(runtime,indent=2),encoding='utf-8')
    cmd=[str(python),str(ROOT/'scripts/fetch_robot_assets.py')]
    subprocess.run(cmd,env=env,cwd=ROOT,check=True)
    subprocess.run([str(uv),'pip','check','--python',str(python)],env=env,check=True)
    print(json.dumps(runtime,indent=2),flush=True)

if __name__=='__main__':main()
