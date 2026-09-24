"""Generate conservative event proposals from local YOLO tracks and scene geometry.

The output has ground-truth JSON shape, but it is a set of *proposals*, not
validated ground truth. It never modifies dev_labels.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "debug"))
PERSON_CLASS = 0
VEHICLE_CLASSES = {1, 2, 3, 5, 7}  # COCO bicycle, car, motorcycle, bus, truck
TRACK_CLASSES = [PERSON_CLASS, *sorted(VEHICLE_CLASSES)]


def polygons(scene: dict, key: str) -> list[np.ndarray]:
    value = scene[key]
    entries = value.values() if isinstance(value, dict) else value
    return [np.asarray(points, dtype=np.float32) for points in entries]


def signed_distance(point: tuple[float, float], polygon: np.ndarray) -> float:
    return cv2.pointPolygonTest(polygon, point, True)


def inside(point: tuple[float, float], polygon: np.ndarray) -> bool:
    return signed_distance(point, polygon) >= 0


def segments(flags: list[float], sample_period: float, minimum: float,
             duration: float) -> list[list[float]]:
    """Merge nearby positive samples into valid temporal segments."""
    if not flags:
        return []
    events = []
    start = previous = flags[0]
    for t_sec in flags[1:]:
        if t_sec - previous > max(1.0, 2 * sample_period):
            end = min(duration, previous + sample_period)
            if end - start >= minimum:
                events.append([round(start, 3), round(end, 3)])
            start = t_sec
        previous = t_sec
    end = min(duration, previous + sample_period)
    if end - start >= minimum:
        events.append([round(start, 3), round(end, 3)])
    return events


def merge_intervals(intervals: list[list[float]], gap: float = 0.2) -> list[list[float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def get_detections(result, width: int, height: int) -> list[dict]:
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    ids = boxes.id.int().cpu().tolist()
    classes = boxes.cls.int().cpu().tolist()
    scores = boxes.conf.cpu().tolist()
    items = []
    for bbox, track_id, cls, score in zip(xyxy, ids, classes, scores):
        x1, y1, x2, y2 = map(float, bbox)
        items.append({
            "id": track_id,
            "class": cls,
            "score": score,
            "anchor": ((x1 + x2) / (2 * width), y2 / height),
            "bbox": [x1, y1, x2, y2],
        })
    return items


def events_from_tracks(track_path: Path, scene: dict, width: int, height: int,
                       sample_period: float, duration: float) -> list[list]:
    """Apply scene rules to cached tracks, allowing quick rule-only reruns."""
    crosswalks = polygons(scene, "crosswalks")
    road = polygons(scene, "carriageway")[0]
    pedestrian_ignore = polygons(scene, "pedestrian_ignore_zones")
    jaywalking_by_track: dict[int, list[float]] = defaultdict(list)
    yield_flags: list[float] = []

    def process_frame(t_sec: float, items: list[dict]) -> None:
        people_by_crossing: dict[int, list[dict]] = defaultdict(list)
        vehicles_by_crossing: dict[int, list[dict]] = defaultdict(list)
        for item in items:
            anchor = item["anchor"]
            crossing_ids = [i for i, polygon in enumerate(crosswalks)
                            if inside(anchor, polygon)]
            if item["class"] == PERSON_CLASS:
                for crossing_id in crossing_ids:
                    people_by_crossing[crossing_id].append(item)
                near_crossing = any(signed_distance(anchor, polygon) >= -0.025
                                    for polygon in crosswalks)
                ignored = any(inside(anchor, polygon)
                              for polygon in pedestrian_ignore)
                if (inside(anchor, road) and not near_crossing and not ignored
                        and item["speed"] >= 0.005):
                    jaywalking_by_track[item["id"]].append(t_sec)
            elif item["class"] in VEHICLE_CLASSES and item["speed"] >= 0.025:
                for crossing_id in crossing_ids:
                    vehicles_by_crossing[crossing_id].append(item)
        for crossing_id, pedestrians in people_by_crossing.items():
            vehicles = vehicles_by_crossing[crossing_id]
            if any(person["anchor"][1] < 0.95 and
                   math.dist(person["anchor"], vehicle["anchor"]) < 0.07
                   for person in pedestrians for vehicle in vehicles):
                yield_flags.append(t_sec)
                break

    with track_path.open(newline="", encoding="utf-8") as track_file:
        current_time = None
        items: list[dict] = []
        for row in csv.DictReader(track_file):
            t_sec = float(row["t_sec"])
            if t_sec >= duration:
                break
            if current_time is not None and t_sec != current_time:
                process_frame(current_time, items)
                items = []
            current_time = t_sec
            x1, y1, x2, y2 = (float(row[key]) for key in ("x1", "y1", "x2", "y2"))
            items.append({
                "id": int(row["track_id"]),
                "class": int(row["coco_class"]),
                "anchor": ((x1 + x2) / (2 * width), y2 / height),
                "speed": float(row["speed_norm_per_sec"]),
            })
        if current_time is not None:
            process_frame(current_time, items)

    events = []
    jaywalking_intervals = [interval
                            for times in jaywalking_by_track.values()
                            for interval in segments(times, sample_period, 0.6, duration)]
    for start, end in merge_intervals(jaywalking_intervals):
        events.append([start, end, "jaywalking"])
    for start, end in segments(yield_flags, sample_period, 0.3, duration):
        events.append([start, end, "failure_to_yield"])
    events.sort(key=lambda event: (event[0], event[2]))
    return events


def propose(video: Path, scene: dict, weights: Path, sample_fps: float,
            image_size: int, max_seconds: float | None,
            tracks_dir: Path, reuse_tracks: bool = False) -> dict:
    started = time.perf_counter()
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or frame_count <= 0:
        capture.release()
        raise RuntimeError(f"Invalid video metadata: {video}")
    duration = frame_count / fps
    limit_frames = min(frame_count, round(max_seconds * fps)) if max_seconds else frame_count
    stride = max(1, round(fps / sample_fps))
    sample_period = stride / fps
    tracks_dir.mkdir(parents=True, exist_ok=True)
    track_path = tracks_dir / f"{video.stem}_auto_tracks.csv"
    if reuse_tracks:
        capture.release()
        if not track_path.exists():
            raise FileNotFoundError(f"No cached tracks: {track_path}")
    else:
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            capture.release()
            raise RuntimeError("Install CUDA PyTorch and ultralytics in .venv") from exc
        if not torch.cuda.is_available():
            capture.release()
            raise RuntimeError("CUDA unavailable; refusing heavy detector inference on CPU")
        if not weights.exists():
            capture.release()
            raise FileNotFoundError(f"Local model weights are missing: {weights}")
        print(f"GPU: {torch.cuda.get_device_name(0)}; PyTorch {torch.__version__}; "
              f"CUDA {torch.version.cuda}", flush=True)
        model = YOLO(str(weights))
        track_history: dict[tuple[int, int], tuple[float, tuple[float, float]]] = {}
        try:
            with track_path.open("w", newline="", encoding="utf-8") as track_file:
                writer = csv.writer(track_file)
                writer.writerow(["t_sec", "track_id", "coco_class", "confidence",
                                 "x1", "y1", "x2", "y2", "speed_norm_per_sec"])
                for frame_index in range(limit_frames):
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if frame_index % stride:
                        continue
                    t_sec = frame_index / fps
                    result = model.track(
                        frame, persist=True, tracker="bytetrack.yaml", device=0,
                        imgsz=image_size, conf=0.25, classes=TRACK_CLASSES,
                        verbose=False,
                    )[0]
                    detections = get_detections(result, width, height)
                    for item in detections:
                        track_key = (item["class"], item["id"])
                        previous = track_history.get(track_key)
                        speed = 0.0
                        if previous and 0 < t_sec - previous[0] <= 2.0:
                            speed = math.dist(item["anchor"], previous[1]) / (t_sec - previous[0])
                        track_history[track_key] = (t_sec, item["anchor"])
                        writer.writerow([round(t_sec, 3), item["id"], item["class"],
                                         round(item["score"], 4),
                                         *[round(x, 1) for x in item["bbox"]],
                                         round(speed, 4)])
                    if frame_index % max(stride, round(15 * fps)) == 0:
                        print(f"{video.name}: {t_sec:.0f}/{limit_frames / fps:.0f}s, "
                              f"{len(detections)} tracks", flush=True)
        finally:
            capture.release()

    processed_duration = min(duration, limit_frames / fps)
    events = events_from_tracks(track_path, scene, width, height, sample_period,
                                processed_duration)
    print(f"{video.name}: {len(events)} proposals in "
          f"{time.perf_counter() - started:.1f}s; tracks: {track_path}", flush=True)
    return {"duration": duration, "fps": fps, "events": events}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", type=Path, help="one MP4 or a folder of MP4s")
    parser.add_argument("--out", type=Path, default=ROOT / "auto_proposals.json")
    parser.add_argument("--scene", type=Path, default=ROOT / "config" / "scene.json")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights" / "yolo26n.pt")
    parser.add_argument("--tracks-dir", type=Path, default=ROOT / "debug")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--max-seconds", type=float, help="process only this many seconds")
    parser.add_argument("--reuse-tracks", action="store_true",
                        help="reapply scene rules to the cached track CSV without model inference")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip videos already present in the output JSON")
    args = parser.parse_args()
    if args.sample_fps <= 0:
        parser.error("--sample-fps must be positive")
    if args.max_seconds is not None and args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    if args.videos.is_dir():
        videos = sorted(path for path in args.videos.iterdir()
                        if path.suffix.lower() == ".mp4")
    else:
        videos = [args.videos]
    if not videos:
        parser.error("No MP4 files found")
    scene = json.loads(args.scene.read_text(encoding="utf-8"))
    output = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    for video in videos:
        if args.skip_existing and video.name in output:
            print(f"Skipping {video.name}: already in {args.out}")
            continue
        output[video.name] = propose(video, scene, args.weights, args.sample_fps,
                                     args.image_size, args.max_seconds, args.tracks_dir,
                                     args.reuse_tracks)
        args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved event proposals to {args.out}")


if __name__ == "__main__":
    main()
