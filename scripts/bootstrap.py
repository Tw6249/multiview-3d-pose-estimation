"""Isolated Windows runtime. No system Python packages or file associations changed."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get('LOCALAPPDATA', Path.home())) / 'multiview-pose-sim'
VERSION = '4.5.3'


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
    CACHE.mkdir(parents=True, exist_ok=True)
    python = CACHE / 'venv' / 'Scripts' / 'python.exe'
    if not python.exists():
        subprocess.run([sys.executable, '-m', 'venv', str(CACHE / 'venv')], check=True)
    subprocess.run([str(python), '-m', 'pip', 'install', '--disable-pip-version-check',
                    '-r', str(ROOT / 'requirements.txt')], check=True)
    name = f'blender-{VERSION}-windows-x64'
    blender = CACHE / name / 'blender.exe'
    if not blender.exists():
        base = 'https://download.blender.org/release/Blender4.5/'
        archive = download(base + name + '.zip', CACHE / (name + '.zip'))
        checksum_file = download(base + f'blender-{VERSION}.sha256', CACHE / f'blender-{VERSION}.sha256')
        checksums = checksum_file.read_text()
        expected = next(line.split()[0] for line in checksums.splitlines() if name + '.zip' in line)
        actual = hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest()
        if actual != expected:
            raise RuntimeError('Blender checksum mismatch; archive was not extracted')
        with zipfile.ZipFile(archive) as z:
            z.extractall(CACHE)
    model = ROOT / 'assets' / 'CesiumMan.glb'
    download('https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/CesiumMan/glTF-Binary/CesiumMan.glb', model)
    # Resolve MSIX filesystem redirects: Windows side-by-side DLL loading does not
    # reliably resolve the virtual LocalAppData location used by desktop hosts.
    paths = {'python': str(python.resolve()), 'blender': str(blender.resolve()), 'asset': str(model.resolve())}
    (ROOT / '.runtime.json').write_text(json.dumps(paths, indent=2), encoding='utf-8')
    print(json.dumps(paths, indent=2), flush=True)


if __name__ == '__main__':
    main()
