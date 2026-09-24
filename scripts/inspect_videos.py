"""Save basic metadata and small frame contact sheets for local sample videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def inspect_video(path: Path, output_dir: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0.0
    metadata = {
        "file": path.name,
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frame_count,
        "duration_sec": duration,
    }

    thumbs = []
    for fraction in (0.05, 0.3, 0.55, 0.8):
        frame_index = min(frame_count - 1, round(fraction * frame_count))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {frame_index} from {path}")
        thumb = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
        cv2.putText(
            thumb,
            f"{path.name}  {frame_index / fps:.1f}s",
            (16, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        thumbs.append(thumb)
    capture.release()
    sheet = np.vstack((np.hstack(thumbs[:2]), np.hstack(thumbs[2:])))
    cv2.imwrite(str(output_dir / f"{path.stem}_contact.jpg"), sheet)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", type=Path)
    parser.add_argument("--out", type=Path, default=Path("debug"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        (path for path in args.videos.iterdir() if path.suffix.lower() == ".mp4"),
        key=lambda path: path.name.lower(),
    )
    if not paths:
        parser.error(f"No MP4 files in {args.videos}")
    metadata = [inspect_video(path, args.out) for path in paths]
    (args.out / "video_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
