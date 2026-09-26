"""Background images for the zone editor (runs on the server, OpenCV + numpy).

- ref_4k.jpg: 4K median of 15 frames of the C3897 original (8 s to 308 s), the
  reference frame at full resolution for placing zone points. It is checked
  against tools/scene/reference_C3897.jpg (the 1080p median the homographies were
  estimated on) by phase correlation; the shift is printed and should be ~0.
- bg_<STEM>.jpg: the 60-frame 1080p median of each proxy (background_pass.py
  output), as JPEG, for the verification view.

Decoding runs one ffmpeg thread at nice 19 (the server also hosts production
sites); ~55 s for the 4K median. Existing outputs are kept unless --force.

  ~/wiut/.venv/bin/python make_backgrounds.py --original ~/wiut/originals/C3897.MP4 \
      --ref-1080 reference_C3897.jpg --medians ~/wiut/verify/bg --out ~/wiut/scene/assets
"""
import argparse
import os
import subprocess
import time

import cv2
import numpy as np

W, H = 3840, 2160
STEMS = ["C3905", "C3896", "C3897", "C3902"]


def median_4k(src, out, ref_1080):
    times = np.linspace(8, 308, 15)
    stack = np.empty((len(times), H, W, 3), np.uint8)
    t0 = time.time()
    for k, t in enumerate(times):
        raw = subprocess.run(["nice", "-n", "19", "ffmpeg", "-v", "error", "-threads", "1", "-ss", f"{t:.2f}",
                              "-i", src, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
                             capture_output=True, check=True).stdout
        stack[k] = np.frombuffer(raw, np.uint8).reshape(H, W, 3)
        print("frame", k, round(float(t), 1), "s", round(time.time() - t0, 1), flush=True)
    med = np.empty((H, W, 3), np.uint8)
    for r0 in range(0, H, 120):   # row strips keep the float temporaries small
        med[r0:r0 + 120] = np.median(stack[:, r0:r0 + 120], axis=0).astype(np.uint8)
    del stack
    cv2.imwrite(out, med, [cv2.IMWRITE_JPEG_QUALITY, 90])
    ref = cv2.imread(ref_1080, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    small = cv2.resize(cv2.cvtColor(med, cv2.COLOR_BGR2GRAY), (1920, 1080),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
    shift, resp = cv2.phaseCorrelate(ref, small, cv2.createHanningWindow((1920, 1080), cv2.CV_32F))
    print(f"ref_4k vs reference_C3897.jpg: shift ({shift[0]:.3f}, {shift[1]:.3f}) px at 1080p, response {resp:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--original", required=True, help="C3897 4K original")
    ap.add_argument("--ref-1080", required=True, help="tools/scene/reference_C3897.jpg")
    ap.add_argument("--medians", required=True, help="dir with <STEM>_median.png (background_pass.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "ref_4k.jpg")
    if a.force or not os.path.exists(p):
        median_4k(a.original, p, a.ref_1080)
    for stem in STEMS:
        p = os.path.join(a.out, f"bg_{stem}.jpg")
        if a.force or not os.path.exists(p):
            img = cv2.imread(os.path.join(a.medians, f"{stem}_median.png"))
            cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            print(p, img.shape)


if __name__ == "__main__":
    main()
