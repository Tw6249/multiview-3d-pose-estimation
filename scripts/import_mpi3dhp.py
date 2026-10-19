"""Download a small official research subset and build isolated RGB/reference inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import cv2
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pose_sim.public_download import download_file, download_zip_members
from pose_sim.mpi3dhp import parse_calibration, reference_rows
from pose_sim.pipeline import write_jsonl


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(1024*1024):
            h.update(chunk)
    return h.hexdigest()


def prepare(cache, output, config, download=False):
    cfg = json.loads(Path(config).read_text())
    cache = Path(cache).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve existing dataset manifest')
    provenance = {}
    for sequence in sorted({c['sequence'] for c in cfg['clips']}):
        folder = cache/sequence
        base = cfg['source_base']+sequence+'/'
        if download:
            for name in ('camera.calibration', 'annot.mat'):
                download_file(base+name, folder/name)
            download_zip_members(base+'imageSequence/vnect_cameras.zip', folder/'imageSequence',
                                 [f'video_{cid}.avi' for cid in cfg['camera_ids']])
        hashes = {n: sha256(folder/n) for n in ('camera.calibration', 'annot.mat')}
        videos = {}
        for cid in cfg['camera_ids']:
            path = folder/'imageSequence'/f'video_{cid}.avi'
            cap = cv2.VideoCapture(str(path))
            meta = dict(fps=cap.get(cv2.CAP_PROP_FPS), frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                        width=int(cap.get(3)), height=int(cap.get(4)))
            cap.release()
            # Some sequences are 50 Hz; they are decimated by an integer stride to cfg['fps'].
            if (meta['fps'] % cfg['fps'] or (meta['width'], meta['height']) != (2048, 2048)
                    or meta['fps'] != next(iter(videos.values()), meta)['fps']):
                raise ValueError('Unexpected source video dimensions/frame rate')
            videos[str(cid)] = dict(path=str(path), sha256=sha256(path), **meta)
        provenance[sequence] = dict(base_url=base, metadata_sha256=hashes, videos=videos,
                                    frame_stride=int(next(iter(videos.values()))['fps']//cfg['fps']))
    output.mkdir(parents=True)
    (output/'references').mkdir()
    entries = []
    for clip in cfg['clips']:
        folder = cache/clip['sequence']
        dest = output/'inputs'/clip['id']
        dest.mkdir(parents=True)
        calibration, original = parse_calibration((folder/'camera.calibration').read_text(), cfg['camera_ids'], cfg['width'], cfg['height'])
        count = cfg['frames_per_clip']
        videos = provenance[clip['sequence']]['videos']
        stride = provenance[clip['sequence']]['frame_stride']
        if any(clip['start_frame']+(count-1)*stride+1 > v['frames'] for v in videos.values()):
            raise ValueError('Clip exceeds video length')
        source = dict(**clip, fps=cfg['fps'], frames=count, videos=videos, frame_stride=stride,
                      time_base='source frame index / (fps*frame_stride); dataset synchronization, not measured exposure latency')
        (dest/'source.json').write_text(json.dumps(source, indent=2), encoding='utf-8')
        (dest/'cameras.json').write_text(json.dumps(calibration, indent=2), encoding='utf-8')
        truth, checks = reference_rows(folder/'annot.mat', original, clip['start_frame'], count, cfg['fps'], stride)
        write_jsonl(output/'references'/(clip['id']+'.jsonl'), truth)
        entries.append(dict(**clip, frames=count, reference_checks=checks))
    manifest = dict(configuration=cfg, configuration_sha256=sha256(config), provenance=provenance, clips=entries,
                    reference_isolation='inputs contain only RGB paths, camera calibration and video timing; references are scoring-only')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(dict(clips=len(entries), states=len(entries)*cfg['frames_per_clip'], output=str(output)), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, default=ROOT/'outputs/public_datasets/mpi_inf_3dhp')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=ROOT/'config/public_mpi3dhp.json')
    p.add_argument('--download', action='store_true')
    a = p.parse_args()
    prepare(a.cache, a.output, a.config, a.download)
