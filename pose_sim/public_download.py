"""Bounded, verified HTTPS range downloads for public dataset archives.

No upstream scripts are executed. ZIP names are selected explicitly; archive
paths are never used as filesystem paths. Partial outputs are not accepted.
"""
from concurrent.futures import ThreadPoolExecutor
import io
import re
import time
import urllib.request
import zlib
import zipfile
from pathlib import Path


class HTTPRanges(io.RawIOBase):
    chunk_size = 65536

    def __init__(self, url):
        if not url.startswith('https://'):
            raise ValueError('HTTPS required')
        self.url = url
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=30) as response:
            self.size = int(response.headers['Content-Length'])
            self.etag = response.headers.get('ETag')
        self.position = 0

    def range(self, start, end):
        if not 0 <= start <= end < self.size:
            raise ValueError('Invalid byte range')
        for attempt in range(4):
            try:
                headers = {'Range': f'bytes={start}-{end}'}
                if self.etag:
                    headers['If-Match'] = self.etag
                with urllib.request.urlopen(urllib.request.Request(self.url, headers=headers), timeout=20) as response:
                    expected = f'bytes {start}-{end}/{self.size}'
                    if response.status != 206 or response.headers.get('Content-Range') != expected:
                        raise ValueError('Server did not honor exact byte range')
                    data = response.read(end-start+2)
                    if len(data) != end-start+1:
                        raise ValueError('Truncated range response')
                    return data
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(.25*(attempt+1))

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position+offset if whence == 1 else self.size+offset
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.size, self.position+size)
        data = b''.join(self.range(i, min(i+self.chunk_size, end)-1)
                        for i in range(self.position, end, self.chunk_size))
        self.position = end
        return data

    def chunks(self, start, size, workers=16):
        # executor.map preserves byte order; submit only one bounded window.
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for window in range(start, start+size, workers*self.chunk_size):
                spans = [(i, min(i+self.chunk_size, start+size)-1)
                         for i in range(window, min(window+workers*self.chunk_size, start+size), self.chunk_size)]
                yield from executor.map(lambda span: self.range(*span), spans)


def download_file(url, destination):
    destination = Path(destination)
    source = HTTPRanges(url)
    if destination.exists():
        if destination.stat().st_size != source.size:
            raise ValueError('Existing download size mismatch')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix+'.part')
    count = 0
    with partial.open('wb') as file:
        for data in source.chunks(0, source.size):
            file.write(data)
            count += len(data)
            if count % (16*1024*1024) == 0:
                print(destination.name, count, '/', source.size, flush=True)
    if count != source.size:
        raise ValueError('Incomplete download')
    partial.replace(destination)
    print('DOWNLOADED', destination, count, flush=True)


def download_zip_members(url, destination, names):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if any(not re.fullmatch(r'video_\d+\.avi', name) for name in names):
        raise ValueError('Only explicit dataset video basenames accepted')
    source = HTTPRanges(url)
    with zipfile.ZipFile(source) as archive:
        infos = [archive.getinfo(name) for name in names]
    for info in infos:
        target = destination / info.filename
        if target.exists():
            crc = 0
            with target.open('rb') as file:
                while data := file.read(1024*1024):
                    crc = zlib.crc32(data, crc)
            if target.stat().st_size != info.file_size or crc != info.CRC:
                raise ValueError('Existing video failed ZIP integrity check')
            continue
        source.seek(info.header_offset)
        header = source.read(30)
        if header[:4] != b'PK\x03\x04' or info.flag_bits & 1 or info.compress_type != zipfile.ZIP_DEFLATED:
            raise ValueError('Unsupported ZIP member')
        name_length = int.from_bytes(header[26:28], 'little')
        extra_length = int.from_bytes(header[28:30], 'little')
        start = info.header_offset+30+name_length+extra_length
        decoder = zlib.decompressobj(-15)
        partial = target.with_suffix('.avi.part')
        crc = count = compressed = 0
        with partial.open('wb') as file:
            for data in source.chunks(start, info.compress_size):
                block = decoder.decompress(data)
                file.write(block)
                crc = zlib.crc32(block, crc)
                count += len(block)
                compressed += len(data)
                if compressed % (16*1024*1024) == 0:
                    print(info.filename, compressed, '/', info.compress_size, flush=True)
            block = decoder.flush()
            file.write(block)
            crc = zlib.crc32(block, crc)
            count += len(block)
        if not decoder.eof or count != info.file_size or crc != info.CRC:
            raise ValueError('Downloaded member failed ZIP CRC/size check')
        partial.replace(target)
        print('DOWNLOADED', target, count, flush=True)
