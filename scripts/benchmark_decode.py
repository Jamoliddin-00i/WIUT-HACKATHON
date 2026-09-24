"""Measure the OpenCV BGR decode path used by the organizer harness."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import time
from pathlib import Path

import cv2


def limit_windows_affinity(logical_cpus: int) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    kernel32.SetProcessAffinityMask.restype = ctypes.c_int
    process = kernel32.GetCurrentProcess()
    mask = (1 << logical_cpus) - 1
    if not kernel32.SetProcessAffinityMask(process, mask):
        raise ctypes.WinError(ctypes.get_last_error())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--cores", type=int, help="limit to first N logical CPUs on Windows")
    args = parser.parse_args()
    if args.cores:
        if os.name != "nt":
            parser.error("--cores is implemented for Windows only")
        limit_windows_affinity(args.cores)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    start = time.perf_counter()
    count = 0
    while count < args.frames:
        ok, frame = capture.read()
        if not ok:
            break
        count += 1
    elapsed = time.perf_counter() - start
    capture.release()
    print(json.dumps({
        "video": args.video.name,
        "frames": count,
        "source_seconds": count / fps,
        "wall_seconds": elapsed,
        "decode_fps": count / elapsed,
        "wall_per_source": elapsed / (count / fps),
        "logical_cpus": args.cores or os.cpu_count(),
        "opencv": cv2.__version__,
    }, indent=2))


if __name__ == "__main__":
    main()
