"""Check whether local YOLO26x-pose sees ankles on representative tracks."""

import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "debug"))
from tools.auto_label_video import scene_from_zones  # noqa: E402
from ultralytics import YOLO  # noqa: E402

CASES = [
    ("C3905", 16.016, 789, "true"),
    ("C3905", 75.576, 4104, "true"),
    ("C3905", 106.606, 4687, "true"),
    ("C3905", 34.535, 1004, "rider"),
    ("C3905", 38.038, 1857, "occupant"),
    ("C3905", 85.085, 4063, "zebra-edge"),
    ("C3896", 169.669, 6944, "true"),
    ("C3896", 42.543, 1432, "occupant"),
    ("C3896", 328.328, 12682, "occupant"),
]
registration = json.loads((ROOT / "reports/eda/registration.json").read_text())
zones = json.loads((ROOT / "reports/eda/zones.json").read_text())
model = YOLO(str(ROOT / "weights/yolo26x-pose.pt"))
captures = {}
rows = {}
scenes = {}
tiles = []
for stem, t, tid, label in CASES:
    if stem not in captures:
        captures[stem] = cv2.VideoCapture(str(ROOT.parents[1] / "videos" / f"{stem}.MP4"))
        with (ROOT / "debug" / f"{stem}_auto_tracks.csv").open(newline="") as source:
            rows[stem] = list(csv.DictReader(source))
        scenes[stem] = scene_from_zones(zones, stem, registration)
    row = next(r for r in rows[stem] if int(r["track_id"]) == tid
               and abs(float(r["t_sec"]) - t) < 0.001 and int(r["coco_class"]) == 0)
    box = np.asarray([float(row[k]) for k in ("x1", "y1", "x2", "y2")])
    cap = captures[stem]
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    assert ok
    cx, cy = (box[:2] + box[2:]) / 2
    side = int(max(512, 3 * max(box[2] - box[0], box[3] - box[1])))
    x0, y0 = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
    x1, y1 = min(frame.shape[1], x0 + side), min(frame.shape[0], y0 + side)
    crop = frame[y0:y1, x0:x1].copy()
    result = model.predict(crop, imgsz=1024, device=0, conf=0.15, verbose=False)[0]
    best = None
    if result.boxes is not None and len(result.boxes):
        pose_boxes = result.boxes.xyxy.cpu().numpy()
        target = box - [x0, y0, x0, y0]
        for i, p in enumerate(pose_boxes):
            iw = max(0, min(p[2], target[2]) - max(p[0], target[0]))
            ih = max(0, min(p[3], target[3]) - max(p[1], target[1]))
            overlap = iw * ih / ((target[2] - target[0]) * (target[3] - target[1]))
            if best is None or overlap > best[0]:
                best = (overlap, i)
    anchor = ((box[0] + box[2]) / 7680, box[3] / 2160)
    crossings = [np.asarray(p, dtype=np.float32) for p in scenes[stem]["crosswalks"]]
    box_d = max(cv2.pointPolygonTest(p, anchor, True) for p in crossings)
    cv2.rectangle(crop, tuple((box[:2] - [x0, y0]).astype(int)),
                  tuple((box[2:] - [x0, y0]).astype(int)), (0, 0, 255), 5)
    foot_text = "no matched pose"
    if best and best[0] >= 0.25:
        kpts = result.keypoints.data[best[1]].cpu().numpy()
        visible = [(kpts[k][0], kpts[k][1], kpts[k][2]) for k in (15, 16)
                   if kpts[k][2] >= 0.35]
        for x, y, c in visible:
            cv2.circle(crop, (round(x), round(y)), 12, (0, 255, 0), -1)
        if visible:
            ax = float(np.mean([x + x0 for x, _, _ in visible])) / 3840
            ay = float(np.mean([y + y0 for _, y, _ in visible])) / 2160
            foot_d = max(cv2.pointPolygonTest(p, (ax, ay), True) for p in crossings)
            foot_text = f"ankles={len(visible)} cross_d={foot_d:.3f}"
        else:
            foot_text = "ankles=0"
    print(stem, t, tid, label, f"box_cross_d={box_d:.3f}", foot_text,
          f"pose_overlap={best[0]:.2f}" if best else "no pose", flush=True)
    tile = cv2.resize(crop, (640, 640), interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (640, 68), (0, 0, 0), -1)
    cv2.putText(tile, f"{stem} {t:.1f}s {label} #{tid}", (9, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(tile, foot_text, (9, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    tiles.append(tile)
for cap in captures.values():
    cap.release()
sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
out = ROOT.parents[1] / "videos" / "pose_ankle_reliability_check.jpg"
assert cv2.imwrite(str(out), sheet)
print(out)
