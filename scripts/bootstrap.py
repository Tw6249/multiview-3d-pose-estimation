"""Isolated Windows runtime. No system Python packages or file associations changed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get('LOCALAPPDATA', Path.home())) / 'multiview-pose-sim'
VERSION = '4.5.3'
PYTHON_VERSION = (ROOT / '.python-version').read_text(encoding='utf-8').strip()
UV_VERSION = '0.9.7'
ASSET_URL = ('https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/'
             'edc7c9e67c639d230715049ee31f9a96a6babbbe/Models/CesiumMan/glTF-Binary/CesiumMan.glb')
ASSET_SHA256 = 'b7001eaeea8254bd44773bcd247e78696d94169388fbb2a1800fc69434e777d9'


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print('Downloading:', url, flush=True)
        temp = path.with_suffix(path.suffix + '.part')
        subprocess.run(['curl.exe', '--fail', '--location', '--retry', '2',
                        '--output', str(temp), url], check=True)
        temp.replace(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    uv = CACHE / 'bootstrap-tools' / 'bin' / 'uv.exe'
    if not uv.exists():
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                        '--target', str(CACHE / 'bootstrap-tools'), f'uv=={UV_VERSION}'], check=True)
    uv_version = subprocess.check_output([str(uv), '--version'], text=True).strip().split()[1]
    if uv_version != UV_VERSION:
        raise RuntimeError(f'Expected uv {UV_VERSION}; found {uv_version} at {uv}')
    env = dict(os.environ, UV_PYTHON_INSTALL_DIR=str(CACHE / 'python'),
               UV_CACHE_DIR=str(CACHE / 'uv-cache'), PYTHONNOUSERSITE='1', PYTHONUTF8='1')
    subprocess.run([str(uv), 'python', 'install', PYTHON_VERSION, '--no-bin', '--no-registry'],
                   env=env, check=True)
    venv = CACHE / f'venv-py311-{args.device}'
    python = venv / 'Scripts' / 'python.exe'
    if not python.exists():
        subprocess.run([str(uv), 'venv', '--managed-python', '--python', PYTHON_VERSION, str(venv)],
                       env=env, check=True)
    actual_version = subprocess.check_output([str(python), '-c',
        'import platform; print(platform.python_version())'], env=env, text=True).strip()
    if actual_version != PYTHON_VERSION:
        raise RuntimeError(f'Expected Python {PYTHON_VERSION}; found {actual_version} in {venv}')
    lock = ROOT / ('requirements-gpu-lock.txt' if args.device == 'cuda' else 'requirements-lock.txt')
    subprocess.run([str(uv), 'pip', 'sync', '--python', str(python), '--require-hashes', str(lock)],
                   env=env, check=True)
    name = f'blender-{VERSION}-windows-x64'
    blender = CACHE / name / 'blender.exe'
    if not blender.exists():
        base = 'https://download.blender.org/release/Blender4.5/'
        archive = download(base + name + '.zip', CACHE / (name + '.zip'))
        checksum_file = download(base + f'blender-{VERSION}.sha256', CACHE / f'blender-{VERSION}.sha256')
        checksums = checksum_file.read_text()
        expected = next(line.split()[0] for line in checksums.splitlines() if name + '.zip' in line)
        with archive.open('rb') as f:
            actual = hashlib.file_digest(f, 'sha256').hexdigest()
        if actual != expected:
            raise RuntimeError('Blender checksum mismatch; archive was not extracted')
        with zipfile.ZipFile(archive) as z:
            z.extractall(CACHE)
    model = ROOT / 'assets' / 'CesiumMan.glb'
    download(ASSET_URL, model)
    if hashlib.sha256(model.read_bytes()).hexdigest() != ASSET_SHA256:
        raise RuntimeError('CesiumMan checksum mismatch')
    # Resolve MSIX filesystem redirects: Windows side-by-side DLL loading does not
    # reliably resolve the virtual LocalAppData location used by desktop hosts.
    paths = {'python': str(python.resolve()), 'python_version': PYTHON_VERSION,
             'device': args.device, 'render_device': 'OPTIX' if args.device == 'cuda' else 'CPU',
             'dependency_lock_sha256': hashlib.sha256(lock.read_bytes()).hexdigest(),
             'blender': str(blender.resolve()), 'asset': str(model.resolve())}
    (ROOT / '.runtime.json').write_text(json.dumps(paths, indent=2), encoding='utf-8')
    print(json.dumps(paths, indent=2), flush=True)


if __name__ == '__main__':
    main()
