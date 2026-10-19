"""Keep verbose render logs on disk while reporting frame-level progress."""
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
runtime = json.loads((root/'.runtime.json').read_text())
args = sys.argv[1:]
out = Path(args[args.index('--output')+1]).resolve()
if (out/'frames.jsonl').exists():
    raise RuntimeError('Session already exists; use a new output folder')
out.mkdir(parents=True, exist_ok=True)
command = [runtime['blender'], '--background', '--factory-startup', '--threads', '8',
           '--python-exit-code', '1', '--python', str(root/'scripts/render_scene.py'), '--', *args]
with (out/'render.log').open('w', encoding='utf-8') as log:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding='utf-8', errors='replace')
    tail = []
    for line in process.stdout:
        log.write(line)
        tail = (tail+[line])[-25:]
        if 'FRAME_DONE' in line or 'SESSION_READY' in line or 'RENDER_DEVICE' in line or 'Error' in line:
            print(line.strip(), flush=True)
    code = process.wait()
if code:
    print(''.join(tail), file=sys.stderr)
    raise SystemExit(code)
