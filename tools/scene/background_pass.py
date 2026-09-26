"""One sequential pass over a (proxy) video: median background, static-pixel mask and
per-frame brightness. Light enough for the 2-core server (about 75 s per 5-min proxy).

  python tools/scene/background_pass.py --video proxies/C3905_1080p.mp4 --stem C3905 --out work/bg

Writes <STEM>_median.png (median of --n frames spread over the clip),
<STEM>_static_mask.png (255 = static pixel, see register.static_mask) and
<STEM>_luma.csv: frame, t, mean, top_third, bottom_two_thirds, static_pixels
(mean luma of every --stride-th frame at 480x270; static_pixels = mean over the
static mask only, so moving traffic does not change it).
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.scene.register import median_frame, static_mask  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--stride", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    cap = cv2.VideoCapture(a.video)
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    want = set(np.linspace(0, n - 2, a.n).astype(int).tolist())
    frames, small, rows, idx = [], [], [], -1
    while True:
        if not cap.grab():
            break
        idx += 1
        if idx % a.stride and idx not in want:
            continue
        ok, fr = cap.retrieve()
        if not ok:
            break
        if idx in want:
            frames.append(cv2.resize(fr, (1920, 1080), interpolation=cv2.INTER_AREA) if fr.shape[1] != 1920 else fr)
        if idx % a.stride == 0:
            y = cv2.cvtColor(cv2.resize(fr, (480, 270), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2YCrCb)[:, :, 0]
            small.append(cv2.resize(y, (240, 135), interpolation=cv2.INTER_AREA))
            rows.append({"frame": idx, "t": round(idx / fps, 3), "mean": float(y.mean()),
                         "top_third": float(y[:90].mean()), "bottom_two_thirds": float(y[90:].mean())})
    cap.release()
    bg = median_frame(frames)
    mask = static_mask(frames, bg)
    cv2.imwrite(os.path.join(a.out, f"{a.stem}_median.png"), bg)
    cv2.imwrite(os.path.join(a.out, f"{a.stem}_static_mask.png"), mask)
    m = cv2.resize(mask, (240, 135), interpolation=cv2.INTER_NEAREST) > 0
    for r, y in zip(rows, small):
        r["static_pixels"] = float(y[m].mean())
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"{a.stem}_luma.csv"), index=False)
    print(a.stem, n, "frames,", len(frames), "for the median,", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
