"""Run the official harness with network connections blocked and check its log."""
import argparse
import json
import os
from pathlib import Path
import runpy
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'debug/offline-config'))
    os.environ.setdefault('SALEN_WORK_DIR', str(ROOT / 'debug/offline-runtime'))
    attempts = []

    def blocked(*unused_args, **unused_kwargs):
        attempts.append(True)
        raise RuntimeError('Network connections are forbidden during inference')

    originals = (socket.socket.connect, socket.socket.connect_ex, socket.create_connection,
                 socket.getaddrinfo)
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = blocked
    socket.getaddrinfo = blocked
    previous_args = sys.argv
    sys.argv = ['run_submission.py', '--videos', str(args.videos.resolve()),
                '--out', str(args.out.resolve()), '--team', 'Salen']
    try:
        try:
            runpy.run_path(str(ROOT / 'run_submission.py'), run_name='__main__')
        except SystemExit as exc:
            if exc.code:
                raise
    finally:
        (socket.socket.connect, socket.socket.connect_ex, socket.create_connection,
         socket.getaddrinfo) = originals
        sys.argv = previous_args
    result = json.loads(args.out.read_text(encoding='utf-8'))
    problems = [f'{video}: {error}' for video, row in result['log'].items() for error in row['errors']]
    if attempts:
        problems.append(f'{len(attempts)} network connection attempts were blocked')
    if problems:
        raise RuntimeError('\n'.join(problems))
    print('PASS: no network connections attempted; all harness runs completed within budget.')


if __name__ == '__main__':
    main()
