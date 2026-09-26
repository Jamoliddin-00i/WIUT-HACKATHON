"""Detect and track road users in one video for EDA.

Runs Ultralytics YOLO + ByteTrack on every Nth frame of a (proxy) video and
writes tracks.csv with boxes in the ORIGINAL video pixel space, so results
are comparable across proxy resolutions.

Examples
  # quick throughput check on 60 processed frames, no CSV
  python track.py --video proxies/C3905_1080p.mp4 --out eda/C3905 --bench 60

  # full run
  python track.py --video proxies/C3905_1080p.mp4 --out eda/C3905 \
      --model models/yolo11s.pt --imgsz 1280 --stride 3 --orig-width 3840
"""
import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

# COCO ids: person, bicycle, car, motorcycle, bus, truck
DEFAULT_CLASSES = [0, 1, 2, 3, 5, 7]
COLORS = {0: (60, 200, 255), 1: (255, 160, 0), 2: (80, 220, 80),
          3: (255, 80, 200), 5: (0, 140, 255), 7: (220, 220, 60)}


def log(fh, msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def draw(frame, boxes, names):
    img = frame.copy()
    for x1, y1, x2, y2, tid, c, conf in boxes:
        col = COLORS.get(c, (255, 255, 255))
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, col, 2)
        label = f"{names[c]} {tid} {conf:.2f}"
        cv2.putText(img, label, (p1[0], max(12, p1[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="proxy video path")
    ap.add_argument("--out", required=True, help="output dir, e.g. eda/C3905")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--stride", type=int, default=3, help="process every Nth frame")
    ap.add_argument("--conf", type=float, default=0.1,
                    help="detector floor; ByteTrack uses low boxes in its 2nd pass")
    ap.add_argument("--orig-width", type=int, default=3840,
                    help="width of the original video; boxes are scaled to it")
    ap.add_argument("--classes", default=",".join(map(str, DEFAULT_CLASSES)))
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--bench", type=int, default=0,
                    help="only time this many processed frames and exit")
    ap.add_argument("--samples", default="10,40,70,100",
                    help="seconds at which to save annotated JPEGs")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    from ultralytics import YOLO

    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = args.orig_width / w
    classes = [int(c) for c in args.classes.split(",")]
    sample_secs = [float(s) for s in args.samples.split(",") if s]

    model = YOLO(args.model)
    names = model.names

    logf = None if args.bench else open(os.path.join(args.out, "track.log"), "a")
    log(logf, f"video={args.video} {w}x{h} fps={fps:.3f} frames={n_frames} "
              f"model={args.model} imgsz={args.imgsz} stride={args.stride} scale={scale}")

    writer = None
    if not args.bench:
        fcsv = open(os.path.join(args.out, "tracks.csv"), "w", newline="")
        writer = csv.writer(fcsv)
        writer.writerow(["frame", "t_sec", "track_id", "cls", "conf", "x1", "y1", "x2", "y2"])

    idx = -1
    done = 0
    rows = 0
    t0 = time.time()
    t_warm = None
    infer_time = 0.0
    saved = set()
    while True:
        ok = cap.grab()
        if not ok:
            break
        idx += 1
        if idx % args.stride:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        t_sec = idx / fps
        ti = time.time()
        res = model.track(frame, persist=True, tracker=args.tracker, imgsz=args.imgsz,
                          conf=args.conf, classes=classes, verbose=False)[0]
        infer_time += time.time() - ti
        done += 1
        if done == 5:
            t_warm = time.time()
        boxes = []
        if res.boxes is not None and res.boxes.id is not None:
            xyxy = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.cpu().numpy().astype(int)
            cls = res.boxes.cls.cpu().numpy().astype(int)
            conf = res.boxes.conf.cpu().numpy()
            for b, tid, c, cf in zip(xyxy, ids, cls, conf):
                boxes.append((*b, tid, c, cf))
                if writer:
                    x1, y1, x2, y2 = (b * scale).round(1)
                    writer.writerow([idx, f"{t_sec:.3f}", tid, names[c], f"{cf:.3f}",
                                     x1, y1, x2, y2])
                    rows += 1
        for s in sample_secs:
            if s not in saved and t_sec >= s:
                saved.add(s)
                if not args.bench:
                    cv2.imwrite(os.path.join(args.out, f"sample_t{int(s):03d}.jpg"),
                                draw(frame, boxes, names), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if args.bench and done >= args.bench:
            break
        if not args.bench and done % 50 == 0:
            el = time.time() - t0
            total = n_frames // args.stride
            log(logf, f"{done}/{total} processed, t={t_sec:.1f}s, {done / el:.2f} f/s, "
                      f"eta {(total - done) / max(done / el, 1e-6) / 60:.1f} min, rows={rows}")

    el = time.time() - t0
    stats = {"model": args.model, "imgsz": args.imgsz, "stride": args.stride,
             "processed_frames": done, "wall_s": round(el, 1),
             "fps_wall": round(done / el, 3), "fps_infer_only": round(done / infer_time, 3),
             "rows": rows, "video": args.video, "video_fps": fps, "video_frames": n_frames,
             "proxy_size": [w, h], "scale_to_orig": scale}
    if t_warm and done > 5:
        # steady state, excluding the first 5 frames (model warm-up)
        stats["fps_steady"] = round((done - 5) / (time.time() - t_warm), 3)
    if args.bench:
        log(logf, f"BENCH {json.dumps(stats)}")
    else:
        fcsv.close()
        with open(os.path.join(args.out, "track_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        log(logf, f"DONE {json.dumps(stats)}")


if __name__ == "__main__":
    main()
