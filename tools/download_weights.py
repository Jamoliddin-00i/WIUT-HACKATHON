"""Fetch and verify official weights during setup, never during inference."""
import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt'
SHA256 = '9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92'


def valid(path):
    if not path.exists():
        return False
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest() == SHA256


def main():
    target = ROOT / 'weights/yolo26x.pt'
    target.parent.mkdir(exist_ok=True)
    if valid(target):
        print('Official YOLO26x weights already present; checksum verified.')
        return
    partial = target.with_suffix('.download')
    urllib.request.urlretrieve(URL, partial)
    if not valid(partial):
        raise RuntimeError(f'Checksum failed: {partial}; existing weights were not replaced')
    partial.replace(target)
    print(f'Saved verified model: {target}')


if __name__ == '__main__':
    main()
