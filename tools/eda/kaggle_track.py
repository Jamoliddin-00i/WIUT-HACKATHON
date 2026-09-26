"""Detect and track road users in the 4K sample videos on a Kaggle GPU.

Downloads each original from VIDEO_BASE_URL, decodes it with ffmpeg (every 3rd
frame, scaled to 1080p), runs YOLO11 + ByteTrack, and writes one CSV per video:
frame, t_sec, track_id, cls, conf, x1, y1, x2, y2 (boxes in original 4K pixels).
Same schema as tools/eda/track.py, so tools/eda/analyze.py works on either.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

VIDEO_BASE_URL = os.environ.get("VIDEO_BASE_URL", "")
VIDEOS = os.environ.get("VIDEOS", "C3905.MP4").split(",")
MODEL = os.environ.get("MODEL", "yolo11m.pt")
IMGSZ = int(os.environ.get("IMGSZ", "1280"))
STRIDE = 3
W, H, SCALE = 1920, 1080, 2.0  # decode size and factor back to 4K
CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
OUT = Path("/kaggle/working/tracks")
TMP = Path("/tmp/videos")


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def download(name):
    TMP.mkdir(parents=True, exist_ok=True)
    dst = TMP / name
    sh(f"curl -sS --fail --retry 20 --retry-delay 5 -C - -o {dst} {VIDEO_BASE_URL}/{name}")
    return dst


def fps_of(path):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                   "-show_entries", "stream=r_frame_rate,nb_frames", "-of", "json", str(path)])
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return float(num) / float(den), int(s.get("nb_frames", 0))


def _sync_args():
    """Older ffmpeg builds (like Kaggle's) only know -vsync; newer ones prefer -fps_mode."""
    opts = subprocess.run(["ffmpeg", "-hide_banner", "-h", "long"], capture_output=True, text=True).stdout
    return ["-fps_mode", "passthrough"] if "-fps_mode" in opts else ["-vsync", "0"]


SYNC_ARGS = _sync_args()


def frames(path):
    """Yield (frame_index, BGR 1080p frame) for every STRIDE-th frame."""
    cmd = ["ffmpeg", "-v", "error", "-threads", "0", "-i", str(path),
           "-vf", f"select=not(mod(n\\,{STRIDE})),scale={W}:{H}:flags=area",
           *SYNC_ARGS, "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=W * H * 3 * 4)
    size, k = W * H * 3, 0
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        yield k * STRIDE, np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        k += 1
    if proc.wait() != 0 or k == 0:
        raise RuntimeError(f"ffmpeg decode failed for {path} after {k} frames")


def track(path, model):
    fps, n = fps_of(path)
    stem = path.stem
    rows, t0 = [], time.time()
    model.predictor = None  # fresh tracker state per video
    for i, (fi, img) in enumerate(frames(path)):
        r = model.track(img, persist=True, tracker="bytetrack.yaml", imgsz=IMGSZ,
                        classes=list(CLASSES), half=True, verbose=False)[0]
        b = r.boxes
        if b is not None and b.id is not None:
            xyxy = (b.xyxy.cpu().numpy() * SCALE).tolist()
            for (x1, y1, x2, y2), tid, c, cf in zip(xyxy, b.id.int().tolist(), b.cls.int().tolist(), b.conf.tolist()):
                rows.append(f"{fi},{fi / fps:.3f},{tid},{CLASSES[c]},{cf:.3f},{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}")
        if i % 300 == 0:
            done = fi / max(n, 1)
            print(f"[{stem}] frame {fi}/{n} ({done:.0%}) {i / max(time.time() - t0, 1e-6):.1f} proc fps", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{stem}.csv").write_text("frame,t_sec,track_id,cls,conf,x1,y1,x2,y2\n" + "\n".join(rows) + "\n")
    secs = time.time() - t0
    return {"video": path.name, "fps": fps, "n_frames": n, "processed": n // STRIDE,
            "rows": len(rows), "seconds": round(secs, 1), "model": MODEL, "imgsz": IMGSZ}


def main():
    if not VIDEO_BASE_URL:
        sys.exit("set VIDEO_BASE_URL")
    from ultralytics import YOLO
    os.chdir("/tmp")  # weights auto-download here, not into /kaggle/working outputs
    model = YOLO(MODEL)
    report = []
    for name in VIDEOS:
        try:
            path = download(name)
            report.append(track(path, model))
            path.unlink()
        except Exception as e:  # keep going with the other videos
            report.append({"video": name, "error": repr(e)})
        print(json.dumps(report[-1]), flush=True)
        (Path("/kaggle/working") / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
