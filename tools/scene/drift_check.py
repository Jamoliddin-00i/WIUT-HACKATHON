"""Camera drift inside one video: register single frames against the video's own
median background and report the largest displacement of 5 test points (4K px).

  python tools/scene/drift_check.py --video proxies/C3896_1080p.mp4 --median work/bg/C3896_median.png \
      --frames 0:90:6,90:2000:60 --stem C3896 --out work/drift_C3896.json

--frames takes start:stop:step ranges (frame indices). The JSON is
{stem: [[frame, max_disp_px], ...]}, the format verify_registration.py --drift reads
(merge several stems into one file). This is the check behind "the camera settles
for up to ~30 s after recording starts".
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.scene.register import _apply, estimate  # noqa: E402

PTS = np.array([[200, 200], [3640, 200], [1920, 1080], [200, 1960], [3640, 1960]], float)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--median", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    want = sorted({f for part in a.frames.split(",") for f in range(*map(int, part.split(":")))})
    bg = cv2.imread(a.median)
    cap = cv2.VideoCapture(a.video)
    res = []
    for f in want:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read()
        if not ok:
            continue
        r = estimate(bg, cv2.resize(im, (bg.shape[1], bg.shape[0])) if im.shape[1] != bg.shape[1] else im)
        res.append([f, round(float(np.linalg.norm(_apply(r["H_ref_to_video"], PTS) - PTS, axis=1).max()), 2)])
        print(f, res[-1][1], flush=True)
    with open(a.out, "w") as fh:
        json.dump({a.stem: res}, fh)


if __name__ == "__main__":
    main()
