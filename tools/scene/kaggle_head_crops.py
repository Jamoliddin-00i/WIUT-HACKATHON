"""Crop every signal head from the 4K originals in one decode pass per video (Kaggle).

Input (environment):
  VIDEO_BASE_URL  base URL that serves <STEM>.MP4 (secret, never commit it)
  HEAD_BOXES      JSON {stem: {head: [x1, y1, x2, y2]}} in that video's 4K pixels,
                  e.g. reference boxes mapped with tools/scene/register.py
  DENSE_HEADS     heads cropped every STRIDE frames (default "A,D"); the others
                  every SPARSE_STRIDE frames
Output in /kaggle/working/crops:
  <STEM>_<HEAD>.npz  frames (int32 frame index), crops (uint8 N x h x w x 3, BGR)
  <STEM>_t<SEC>.jpg  two full 4K frames per video for inspection
  report.json        decode counts and timings

ffmpeg decodes the 10-bit 4:2:2 H.264 in software, selects every STRIDE-th frame,
crops the horizontal band that contains all heads, and pipes it as bgr24. The next
video downloads while the current one decodes.
"""
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

BASE = os.environ["VIDEO_BASE_URL"]
BOXES = json.loads(os.environ["HEAD_BOXES"])
DENSE = os.environ.get("DENSE_HEADS", "A,D").split(",")
STRIDE, SPARSE_STRIDE = int(os.environ.get("STRIDE", 3)), int(os.environ.get("SPARSE_STRIDE", 15))
OUT, TMP = Path("/kaggle/working/crops"), Path("/tmp/videos")
W = 3840


def sync_args():
    opts = subprocess.run(["ffmpeg", "-hide_banner", "-h", "long"], capture_output=True, text=True).stdout
    return ["-fps_mode", "passthrough"] if "-fps_mode" in opts else ["-vsync", "0"]


def start_download(stem):
    TMP.mkdir(parents=True, exist_ok=True)
    dst = TMP / f"{stem}.MP4"
    p = subprocess.Popen(["curl", "-sS", "--fail", "--retry", "20", "--retry-delay", "5", "-C", "-",
                          "-o", str(dst), f"{BASE}/{stem}.MP4"])
    return dst, p


def process(stem, path):
    boxes = BOXES[stem]
    y0 = max(0, min(b[1] for b in boxes.values()) // 2 * 2)
    y1 = min(2160, (max(b[3] for b in boxes.values()) + 1) // 2 * 2)
    bh = y1 - y0
    cmd = ["ffmpeg", "-v", "error", "-threads", "0", "-i", str(path), "-an", "-sn", "-dn",
           "-vf", f"select=not(mod(n\\,{STRIDE})),crop={W}:{bh}:0:{y0}", *sync_args(),
           "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=W * bh * 3 * 2)
    crops = {k: [] for k in boxes}
    idx = {k: [] for k in boxes}
    size, k, t0 = W * bh * 3, 0, time.time()
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        fi = k * STRIDE
        band = np.frombuffer(buf, np.uint8).reshape(bh, W, 3)
        for h, (x1, yy1, x2, yy2) in boxes.items():
            if h in DENSE or fi % SPARSE_STRIDE == 0:
                crops[h].append(band[yy1 - y0:yy2 - y0, x1:x2].copy())
                idx[h].append(fi)
        k += 1
        if k % 500 == 0:
            print(f"[{stem}] {fi} frames, {k / (time.time() - t0):.1f} decoded samples/s", flush=True)
    rc = proc.wait()
    OUT.mkdir(parents=True, exist_ok=True)
    for h in boxes:
        np.savez_compressed(OUT / f"{stem}_{h}.npz", frames=np.array(idx[h], np.int32),
                            crops=np.stack(crops[h]), box=np.array(boxes[h]))
    for sec in (10, 60):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(sec), "-i", str(path), "-frames:v", "1",
                        "-q:v", "3", str(OUT / f"{stem}_t{sec}.jpg")])
    return {"stem": stem, "ffmpeg_rc": rc, "samples": k, "last_frame": (k - 1) * STRIDE,
            "band_y": [y0, y1], "decode_s": round(time.time() - t0, 1)}


def main():
    stems = list(BOXES)
    report, t_all = [], time.time()
    nxt = start_download(stems[0])
    for i, stem in enumerate(stems):
        path, p = nxt
        t0 = time.time()
        rc = p.wait()
        wait_s = round(time.time() - t0, 1)
        if i + 1 < len(stems):
            nxt = start_download(stems[i + 1])
        try:
            if rc != 0:
                raise RuntimeError(f"curl rc {rc}")
            r = process(stem, path)
            r["download_wait_s"] = wait_s
        except Exception as e:
            r = {"stem": stem, "error": repr(e)}
        path.unlink(missing_ok=True)
        report.append(r)
        print(json.dumps(r), flush=True)
        (Path("/kaggle/working") / "report.json").write_text(json.dumps(report, indent=1))
    print("total s", round(time.time() - t_all, 1))


if __name__ == "__main__":
    main()
