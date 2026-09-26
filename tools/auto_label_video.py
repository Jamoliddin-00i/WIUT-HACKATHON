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

if __package__:
    from .crossing_rules import CrossingEvents
    from .pedestrian_rules import (follows_edge, reliable_foot, road_outline,
                                   simplified_polygons, trajectory_intervals)
else:
    from crossing_rules import CrossingEvents
    from pedestrian_rules import (follows_edge, reliable_foot, road_outline,
                                  simplified_polygons, trajectory_intervals)


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "debug"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
PERSON_CLASS = 0
VEHICLE_CLASSES = {1, 2, 3, 5, 7}  # COCO bicycle, car, motorcycle, bus, truck
TRACK_CLASSES = [PERSON_CLASS, *sorted(VEHICLE_CLASSES)]


def polygons(scene: dict, key: str) -> list[np.ndarray]:
    value = scene[key]
    entries = value.values() if isinstance(value, dict) else value
    return [np.asarray(points, dtype=np.float32) for points in entries]


def scene_for_video(scene: dict, stem: str, registration: dict) -> dict:
    """Warp rough C3905 polygons into a clip using the team's 4K homographies."""
    reference = scene.get("reference_video", "C3905")
    if stem == reference:
        return scene
    videos = registration["videos"]
    reference_stem = registration["reference"]["stem"]
    h_to_ref = (np.eye(3) if reference == reference_stem else
                np.asarray(videos[reference]["homography"]["H_video_to_ref"]))
    h_from_ref = (np.eye(3) if stem == reference_stem else
                  np.asarray(videos[stem]["homography"]["H_ref_to_video"]))
    matrix = h_from_ref @ h_to_ref
    width, height = scene["reference_resolution"]
    scale = np.asarray([width, height], dtype=np.float32)
    warped = dict(scene)
    for key in ("crosswalks", "carriageway", "ignore_zones",
                "pedestrian_ignore_zones"):
        value = scene[key]
        def move(points):
            pixels = (np.asarray(points, dtype=np.float32) * scale).reshape(1, -1, 2)
            mapped = cv2.perspectiveTransform(pixels, matrix)[0]
            return (mapped / scale).tolist()
        warped[key] = ({name: move(points) for name, points in value.items()}
                       if isinstance(value, dict) else [move(points) for points in value])
    return warped


def scene_from_zones(zones_doc: dict, stem: str, registration: dict) -> dict:
    """Map hand-drawn C3897 reference zones into one video's normalized pixels."""
    if (zones_doc.get("frame") != "C3897 reference"
            or zones_doc.get("image_size") != [3840, 2160]):
        raise ValueError("zones.json must use the C3897 3840x2160 reference frame")
    matrix = (np.eye(3, dtype=np.float32) if stem == "C3897" else
              np.asarray(registration["videos"][stem]["homography"]["H_ref_to_video"],
                         dtype=np.float32))
    size = np.asarray([3840, 2160], dtype=np.float32)
    grouped: dict[str, list] = defaultdict(list)
    for zone in zones_doc["zones"]:
        if zone["type"] == "stop_line":
            continue
        points = np.asarray(zone["points"], dtype=np.float32)
        if len(points) < 3 or not np.isfinite(points).all():
            raise ValueError(f"Invalid zone {zone.get('name', zone.get('id'))}")
        mapped = cv2.perspectiveTransform(points.reshape(1, -1, 2), matrix)[0]
        grouped[zone["type"]].append((mapped / size).tolist())
    for required in ("zebra_crossing", "live_lane", "non_carriageway", "bus_stop"):
        if not grouped[required]:
            raise ValueError(f"zones.json has no {required} zone")
    return {
        "crosswalks": grouped["zebra_crossing"],
        "carriageway": grouped["live_lane"],
        "pedestrian_ignore_zones": (grouped["non_carriageway"]
                                    + grouped["parking_bay"]),
        "bus_stops": grouped["bus_stop"],
        "crossing_margin": 0.01,
        "bus_stop_margin": 0.005,
    }


def signed_distance(point: tuple[float, float], polygon: np.ndarray) -> float:
    return cv2.pointPolygonTest(polygon, point, True)


def inside(point: tuple[float, float], polygon: np.ndarray) -> bool:
    return signed_distance(point, polygon) >= 0


def person_inside_vehicle(person: dict, items: list[dict]) -> bool:
    """Reject a person box high inside a vehicle, where its bottom is not feet.

    A pedestrian crossing in front of a vehicle usually reaches the vehicle's
    lower edge. The conservative coverage and height gates keep those cases.
    """
    px1, py1, px2, py2 = person["bbox"]
    person_area = (px2 - px1) * (py2 - py1)
    if person_area <= 0:
        return False
    cx, cy = (px1 + px2) / 2, (py1 + py2) / 2
    for vehicle in items:
        if vehicle["class"] not in {2, 5, 7}:  # car, bus, truck
            continue
        vx1, vy1, vx2, vy2 = vehicle["bbox"]
        if not (vx1 <= cx <= vx2 and vy1 <= cy <= vy2 and vy2 > vy1):
            continue
        overlap_w = max(0.0, min(px2, vx2) - max(px1, vx1))
        overlap_h = max(0.0, min(py2, vy2) - max(py1, vy1))
        coverage = overlap_w * overlap_h / person_area
        if coverage >= 0.85 and (py2 - vy1) / (vy2 - vy1) <= 0.75:
            return True
    return False


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


def bus_dwell_intervals(tracks: dict[int, list[tuple[float, tuple[float, float]]]],
                        sample_period: float, duration: float) -> list[list[float]]:
    """Propose a bus dwell when its ground point stays within ~1% for 10 s."""
    intervals = []
    for samples in tracks.values():
        if len(samples) < 10 / sample_period:
            continue
        chunks = []
        chunk = []
        for sample in samples:
            if chunk and sample[0] - chunk[-1][0] > max(1.0, 2 * sample_period):
                chunks.append(chunk)
                chunk = []
            chunk.append(sample)
        if chunk:
            chunks.append(chunk)
        for chunk in chunks:
            still = set()
            for left, (start, _) in enumerate(chunk):
                right = left
                while right < len(chunk) and chunk[right][0] < start + 10 - sample_period:
                    right += 1
                if right >= len(chunk):
                    break
                window = chunk[left:right + 1]
                xs = [point[0] for _, point in window]
                ys = [point[1] for _, point in window]
                if max(xs) - min(xs) <= 0.012 and max(ys) - min(ys) <= 0.012:
                    still.update(range(left, right + 1))
            times = [chunk[i][0] for i in sorted(still)]
            intervals.extend(segments(times, sample_period, 10.0, duration))
    return merge_intervals(intervals, gap=1.0)


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
                       sample_period: float, duration: float,
                       exclude_vehicle_occupants: bool = True,
                       trajectory_pedestrians: bool = True,
                       crossing_episodes: bool = True) -> list[list]:
    """Apply scene rules to cached tracks, allowing quick rule-only reruns."""
    crosswalks = polygons(scene, "crosswalks")
    road = polygons(scene, "carriageway")
    if "bus_stops" not in scene:
        road = road[:1]  # preserve the old rough-scene replay for comparison
    pedestrian_ignore = polygons(scene, "pedestrian_ignore_zones")
    bus_stops = (polygons(scene, "bus_stops") if "bus_stops" in scene else
                 [np.asarray(scene["ignore_zones"]["far_bus_stop"], dtype=np.float32)])
    jaywalking_by_track: dict[int, list[float]] = defaultdict(list)
    pedestrian_samples = defaultdict(list)
    crossing_edges = simplified_polygons(crosswalks, width, height)
    road_edges = road_outline(road, width, height)
    buses_at_stop: dict[int, list[tuple[float, tuple[float, float]]]] = defaultdict(list)
    yield_flags: list[float] = []
    crossing_events = CrossingEvents(crosswalks, road, width, height, sample_period, duration)

    def process_frame(t_sec: float, items: list[dict]) -> None:
        excluded_people = set()
        people_by_crossing: dict[int, list[dict]] = defaultdict(list)
        vehicles_by_crossing: dict[int, list[dict]] = defaultdict(list)
        for item in items:
            anchor = item["anchor"]
            crossing_ids = [i for i, polygon in enumerate(crosswalks)
                            if inside(anchor, polygon)]
            if item["class"] == PERSON_CLASS:
                occupant = exclude_vehicle_occupants and person_inside_vehicle(item, items)
                x1, y1, x2, y2 = item["bbox"]
                foot = ((x1 + x2) / 2, y2)
                person_height = y2 - y1
                samples = pedestrian_samples[item["id"]]
                if occupant:
                    excluded_people.add(item['id'])
                    samples.append((t_sec, foot, person_height, False))
                    continue
                for crossing_id in crossing_ids:
                    people_by_crossing[crossing_id].append(item)
                near_crossing = any(signed_distance(anchor, polygon) >=
                                    -scene.get("crossing_margin", 0.025)
                                    for polygon in crosswalks)
                ignored = any(inside(anchor, polygon)
                              for polygon in pedestrian_ignore)
                eligible = (any(inside(anchor, polygon) for polygon in road)
                            and not near_crossing and not ignored)
                if eligible and item["speed"] >= 0.005:
                    jaywalking_by_track[item["id"]].append(t_sec)
                if trajectory_pedestrians:
                    recent = [s for s in samples[-4:] if 0 < t_sec - s[0] <= 1.6]
                    if recent:
                        motion = np.asarray(foot) - recent[0][1]
                        if 0.1 * person_height <= np.linalg.norm(motion) <= 2 * person_height:
                            beside_zebra = follows_edge(foot, motion, crossing_edges,
                                                        0.15 * person_height)
                            near_kerb = follows_edge(foot, motion, road_edges,
                                                    0.3 * person_height)
                            eligible = eligible and not beside_zebra and not near_kerb
                    eligible = eligible and reliable_foot(item["bbox"], width, height)
                samples.append((t_sec, foot, person_height, eligible))
            else:
                if item["class"] == 5 and any(
                        signed_distance(point, polygon) >=
                        -scene.get("bus_stop_margin", 0.0)
                        for polygon in bus_stops
                        for point in [anchor,
                                      (item['bbox'][0] / width, item['bbox'][3] / height),
                                      (item['bbox'][2] / width, item['bbox'][3] / height)]):
                    buses_at_stop[item["id"]].append((t_sec, anchor))
                if item["class"] in VEHICLE_CLASSES and item["speed"] >= 0.025:
                    for crossing_id in crossing_ids:
                        vehicles_by_crossing[crossing_id].append(item)
        if crossing_episodes:
            crossing_events.observe(t_sec, items, excluded_people)
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
                "bbox": (x1, y1, x2, y2),
                "speed": float(row["speed_norm_per_sec"]),
            })
        if current_time is not None:
            process_frame(current_time, items)

    events = []
    if trajectory_pedestrians:
        jaywalking_intervals = [interval for samples in pedestrian_samples.values()
                               for interval in trajectory_intervals(samples, sample_period, duration)]
    else:
        jaywalking_intervals = [interval for times in jaywalking_by_track.values()
                               for interval in segments(times, sample_period, 0.6, duration)]
    # The team guide merges same-class events separated by at most one second.
    # Allow a little room for the 2 fps sampling grid around that boundary.
    for start, end in merge_intervals(jaywalking_intervals, gap=1.1):
        events.append([start, end, "jaywalking"])
    yield_intervals = (merge_intervals(crossing_events.events(), gap=1.0) if crossing_episodes else
                       segments(yield_flags, sample_period, 0.3, duration))
    for start, end in yield_intervals:
        events.append([start, end, "failure_to_yield"])
    for start, end in bus_dwell_intervals(buses_at_stop, sample_period, duration):
        events.append([start, end, "stopped_vehicle"])
    events.sort(key=lambda event: (event[0], event[2]))
    return events


def propose(video: Path, scene: dict, weights: Path, sample_fps: float,
            image_size: int, max_seconds: float | None,
            tracks_dir: Path, reuse_tracks: bool = False,
            exclude_vehicle_occupants: bool = True,
            trajectory_pedestrians: bool = True,
            frame_observer=None, crossing_episodes: bool = True) -> dict:
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
                    if frame_index % stride:
                        if not capture.grab():
                            break
                        continue
                    ok, frame = capture.read()
                    if not ok:
                        break
                    t_sec = frame_index / fps
                    if frame_observer is not None:
                        frame_observer(frame, t_sec)
                    result = model.track(
                        frame, persist=True, tracker="bytetrack.yaml", device=0,
                        imgsz=image_size, quantize=16, conf=0.25,
                        classes=TRACK_CLASSES,
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
        print(f"Peak CUDA memory reserved: "
              f"{torch.cuda.max_memory_reserved(0) / (1024 ** 3):.2f} GiB",
              flush=True)

    processed_duration = min(duration, limit_frames / fps)
    if callable(scene):
        scene = scene()
    events = events_from_tracks(track_path, scene, width, height, sample_period,
                                processed_duration, exclude_vehicle_occupants,
                                trajectory_pedestrians, crossing_episodes)
    print(f"{video.name}: {len(events)} proposals in "
          f"{time.perf_counter() - started:.1f}s; tracks: {track_path}", flush=True)
    return {"duration": duration, "fps": fps, "events": events}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", type=Path, help="one MP4 or a folder of MP4s")
    parser.add_argument("--out", type=Path, default=ROOT / "auto_proposals.json")
    parser.add_argument("--zones", type=Path, default=ROOT / "reports" / "eda" / "zones.json",
                        help="hand-drawn reference-frame zones (default geometry)")
    parser.add_argument("--scene", type=Path,
                        help="use the old rough C3905 scene JSON instead of --zones")
    parser.add_argument("--registration", type=Path,
                        default=ROOT / "reports" / "eda" / "registration.json")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights" / "yolo26x.pt")
    parser.add_argument("--tracks-dir", type=Path, default=ROOT / "debug")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--image-size", type=int, default=2560)
    parser.add_argument("--max-seconds", type=float, help="process only this many seconds")
    parser.add_argument("--reuse-tracks", action="store_true",
                        help="reapply scene rules to the cached track CSV without model inference")
    parser.add_argument("--no-occupant-filter", action="store_true",
                        help="disable conservative person-in-vehicle suppression for comparison")
    parser.add_argument("--legacy-pedestrians", action="store_true",
                        help="compare with the original per-frame jaywalking rule")
    parser.add_argument("--legacy-crossings", action="store_true",
                        help="compare with the original per-frame yielding rule")
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
    zone_doc = (None if args.scene else
                json.loads(args.zones.read_text(encoding="utf-8")))
    scene = (json.loads(args.scene.read_text(encoding="utf-8"))
             if args.scene else None)
    registration = json.loads(args.registration.read_text(encoding="utf-8"))
    output = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    for video in videos:
        if args.skip_existing and video.name in output:
            print(f"Skipping {video.name}: already in {args.out}")
            continue
        video_scene = (scene_for_video(scene, video.stem, registration) if scene else
                       scene_from_zones(zone_doc, video.stem, registration))
        output[video.name] = propose(video, video_scene, args.weights, args.sample_fps,
                                     args.image_size, args.max_seconds, args.tracks_dir,
                                     args.reuse_tracks, not args.no_occupant_filter,
                                     not args.legacy_pedestrians,
                                     crossing_episodes=not args.legacy_crossings)
        args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved event proposals to {args.out}")


if __name__ == "__main__":
    main()
