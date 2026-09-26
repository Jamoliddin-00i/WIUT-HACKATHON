"""Spot-check pack for human QA of the automatic lamp labels (no UI).

For each video: N random sample times (uniform over the clip, fixed seed, so the
share of wrong labels is an unbiased estimate), the native 4K crop of each head at
that time (JPEG q95, no resizing), and one contact sheet per video (crops shown at
2x). One CSV lists everything with an empty human_label column to fill in.

  python tools/scene/spotcheck_pack.py --crops work/crops --signals reports/eda/signals \
      --out reports/eda/spotcheck --heads D,A --n 30
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd

FPS = 30000 / 1001


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--crops", required=True)
    ap.add_argument("--signals", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stems", default="C3905,C3896,C3897,C3902")
    ap.add_argument("--heads", default="D,A")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    rows = []
    for stem in a.stems.split(","):
        os.makedirs(os.path.join(a.out, stem), exist_ok=True)
        heads = a.heads.split(",")
        base = pd.read_csv(os.path.join(a.signals, f"{stem}_{heads[0]}_samples.csv"))
        pick = np.sort(rng.choice(len(base), a.n, replace=False))
        frames = base.frame.to_numpy()[pick]
        sheet_rows = []
        for h in heads:
            z = np.load(os.path.join(a.crops, f"{stem}_{h}.npz"))
            lab = pd.read_csv(os.path.join(a.signals, f"{stem}_{h}_samples.csv")).set_index("frame").state
            tiles = []
            for k, fr in enumerate(frames):
                i = int(np.argmin(np.abs(z["frames"] - fr)))
                crop = z["crops"][i]
                name = f"{stem}/{stem}_{h}_{k:02d}_f{fr:05d}.jpg"
                cv2.imwrite(os.path.join(a.out, name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                auto = lab.get(int(z["frames"][i]), "n/a")
                rows.append({"id": f"{stem}_{h}_{k:02d}", "stem": stem, "head": h, "frame": int(fr),
                             "t_s": round(fr / FPS, 2), "auto_label": auto, "image": name, "human_label": ""})
                t = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                t = cv2.copyMakeBorder(t, 34, 4, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                cv2.putText(t, f"{k:02d} {h}", (4, 14), 0, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(t, str(auto), (4, 29), 0, 0.45, (0, 0, 160), 1, cv2.LINE_AA)
                tiles.append(t)
            hh = max(x.shape[0] for x in tiles)
            tiles = [cv2.copyMakeBorder(x, 0, hh - x.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for x in tiles]
            for r0 in range(0, len(tiles), 15):
                sheet_rows.append(np.hstack(tiles[r0:r0 + 15]))
        w = max(r.shape[1] for r in sheet_rows)
        sheet_rows = [cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
                      for r in sheet_rows]
        cv2.imwrite(os.path.join(a.out, f"{stem}_sheet.jpg"), np.vstack(sheet_rows), [cv2.IMWRITE_JPEG_QUALITY, 55])
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "spotcheck.csv"), index=False)
    print(pd.DataFrame(rows).groupby(["head", "auto_label"]).size())


if __name__ == "__main__":
    main()
