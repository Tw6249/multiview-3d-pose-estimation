"""Remove explicitly selected generated artifacts below outputs; preview by default."""
import argparse
import os
from pathlib import Path
import shutil
import stat

ROOT = Path(__file__).resolve().parents[1]


def checked_target(path, output_root):
    root = Path(output_root).resolve(strict=True)
    original = Path(path).absolute()
    target = Path(path).resolve(strict=True)
    if target == root or not target.is_relative_to(root):
        raise ValueError(f'Refusing cleanup outside generated outputs: {target}')
    def reject_link(entry):
        info = entry.lstat()
        if entry.is_symlink() or getattr(info,'st_file_attributes',0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError(f'Refusing linked artifact tree: {entry}')
    for ancestor in [original,*original.parents]:
        if ancestor == root: break
        reject_link(ancestor)
    reject_link(target)
    if target.is_dir():
        for folder, directories, files in os.walk(target,followlinks=False):
            for name in directories+files:
                reject_link(Path(folder)/name)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths',nargs='+',type=Path)
    parser.add_argument('--apply',action='store_true')
    args = parser.parse_args()
    targets = [checked_target(p,ROOT/'outputs') for p in args.paths]
    # Check ALL targets before any deletion.
    for target in targets:
        size = sum(p.stat().st_size for p in target.rglob('*') if p.is_file()) if target.is_dir() else target.stat().st_size
        print('REMOVE' if args.apply else 'WOULD_REMOVE',target,f'({size} bytes)')
        if args.apply:
            if target.is_dir(): shutil.rmtree(target)
            else: target.unlink()


if __name__ == '__main__':
    main()
