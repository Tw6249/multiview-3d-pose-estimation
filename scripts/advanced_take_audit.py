"""Audit ZIP membership, extracted bytes and independently decoded video PTS.

Inputs : data/archives/AdvancedTake.zip and its extraction data/advanced_take_raw/ (MP4/BVH).
Outputs: outputs/advanced_take/audit/source_audit.json and per-video decoder logs (read by advanced_take_s1_extract.py).
PTS are stream-local seconds, NOT exposure or shared hardware timestamps.
Any ZIP/extraction mismatch aborts; decoder warnings and missing frames remain
in the report. No source files are changed and no inference cache is regenerated.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    crc = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
            crc = zlib.crc32(block, crc)
    return h.hexdigest(), crc & 0xffffffff


def main():
    raw = ROOT / 'data/advanced_take_raw'
    out = ROOT / 'outputs/advanced_take/audit'
    out.mkdir(parents=True, exist_ok=True)
    archive = ROOT / 'data/archives/AdvancedTake.zip'
    entries = []
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            path = (raw / info.filename).resolve()
            if not path.is_relative_to(raw.resolve()):
                raise ValueError('Unsafe ZIP member')
            sha, crc = digest(path)
            if path.stat().st_size != info.file_size or crc != info.CRC:
                raise ValueError(f'Extracted source mismatch: {info.filename}')
            entries.append(dict(member=info.filename, bytes=info.file_size,
                                sha256=sha, crc32=f'{crc:08x}', zip_mtime=list(info.date_time)))

    def probe(path):
        cmd = ['ffprobe', '-v', 'warning', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,codec_name:frame=best_effort_timestamp_time',
               '-show_frames', '-of', 'json', str(path)]
        result = subprocess.run(cmd, capture_output=True, check=False)
        (out / f'{path.stem}_decoder.log').write_bytes(result.stderr)
        if result.returncode:
            raise RuntimeError(f'ffprobe failed: {path.name}')
        payload = json.loads(result.stdout)
        meta = payload['streams'][0]
        pts = [float(f['best_effort_timestamp_time']) for f in payload['frames']
               if 'best_effort_timestamp_time' in f]
        row = dict(stream=meta, decoded_frames=len(payload['frames']), pts_s=pts,
                   timestamped_frames=len(pts), decoder_warning_bytes=len(result.stderr),
                   decoder_log=f'outputs/advanced_take/audit/{path.stem}_decoder.log')
        print(f'{path.name}: {row["decoded_frames"]}/{meta.get("nb_frames")} frames, warnings={len(result.stderr)} bytes', flush=True)
        return path.name, row

    with ThreadPoolExecutor(max_workers=4) as pool:
        videos = dict(pool.map(probe, sorted(raw.glob('*.mp4'))))
    manifest = dict(zip_sha256=digest(archive)[0], zip_members=entries, videos=videos,
                    clock_status='stream-local PTS only; shared exposure clock not supplied',
                    calibration_files_in_zip=[], source_files_untouched=True)
    (out / 'source_audit.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Source audit saved', flush=True)


if __name__ == '__main__':
    main()
