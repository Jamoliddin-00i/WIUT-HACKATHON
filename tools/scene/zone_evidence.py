"""Detector evidence for drawing zones, mapped into the reference frame.

Reads the Kaggle track files (YOLO11m + ByteTrack, 4K px of each video) and
writes one small JSON the zone editor shows as optional overlays:

- stop_points: every vehicle stop of 2 s or more (ground point = bottom centre of
  the box, rolling median of 5 samples, speed <= 25 px/s over about +-0.5 s, the
  same rule as the EDA stop_runs), as [x_ref, y_ref, seconds, class, video].
- ped_density: distinct person tracks per 40 px cell of the reference frame,
  summed over the videos, cells with 2 or more people, as [x0, y0, count].

Everything is moved into the reference frame with H_video_to_ref from
registration.json (C3897 is the reference, identity). This is detector output:
it helps to find stop lines, crossings and parking, it is not ground truth.

  python tools/scene/zone_evidence.py --tracks-dir work/tracks \
      --registration reports/eda/registration.json --out reports/eda/zone_evidence.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

STEMS = ["C3905", "C3896", "C3897", "C3902"]
VEH = {"car", "bus", "truck", "motorcycle"}
STOP_THR = 25.0
MIN_STOP_S = 2.0
CELL = 40


def to_ref(H, x, y):
    p = np.stack([x, y, np.ones_like(x)], 1) @ np.asarray(H, float).T
    return p[:, 0] / p[:, 2], p[:, 1] / p[:, 2]


def smooth_speed(g):
    sx = pd.Series(g.gx.to_numpy()).rolling(5, center=True, min_periods=1).median().to_numpy()
    sy = pd.Series(g.gy.to_numpy()).rolling(5, center=True, min_periods=1).median().to_numpy()
    t = g.t_sec.to_numpy()
    n = len(g)
    i0 = np.clip(np.arange(n) - 5, 0, n - 1)
    i1 = np.clip(np.arange(n) + 5, 0, n - 1)
    dt = np.maximum(t[i1] - t[i0], 1e-6)
    sp = np.hypot(sx[i1] - sx[i0], sy[i1] - sy[i0]) / dt
    if n == 1:
        sp[:] = 0
    return sx, sy, t, sp


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tracks-dir", required=True, help="<dir>/<STEM>.csv or <dir>/<STEM>/tracks_kaggle.csv")
    ap.add_argument("--registration", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    reg = json.load(open(a.registration))
    stops, cells, per_video = [], {}, {}
    for stem in STEMS:
        p = os.path.join(a.tracks_dir, f"{stem}.csv")
        if not os.path.exists(p):
            p = os.path.join(a.tracks_dir, stem, "tracks_kaggle.csv")
        df = pd.read_csv(p)
        H = np.eye(3) if stem == reg["reference"]["stem"] else reg["videos"][stem]["homography"]["H_video_to_ref"]
        df["gx"] = (df.x1 + df.x2) / 2
        df["gy"] = df.y2
        n_stops = 0
        for tid, g in df[df.cls.isin(VEH)].sort_values("frame").groupby("track_id"):
            sx, sy, t, sp = smooth_speed(g)
            st = sp <= STOP_THR
            i = 0
            while i < len(st):
                if not st[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < len(st) and st[j + 1]:
                    j += 1
                if t[j] - t[i] >= MIN_STOP_S:
                    x, y = to_ref(H, np.array([np.median(sx[i:j + 1])]), np.array([np.median(sy[i:j + 1])]))
                    stops.append([round(float(x[0]), 1), round(float(y[0]), 1), round(float(t[j] - t[i]), 1),
                                  g.cls.iloc[0], stem])
                    n_stops += 1
                i = j + 1
        ped = df[df.cls == "person"]
        x, y = to_ref(H, ped.gx.to_numpy(float), ped.gy.to_numpy(float))
        key = pd.DataFrame({"tid": ped.track_id.to_numpy(),
                            "cx": (x // CELL).astype(int), "cy": (y // CELL).astype(int)}).drop_duplicates()
        for (cx, cy), n in key.groupby(["cx", "cy"]).size().items():
            if 0 <= cx * CELL < 3840 and 0 <= cy * CELL < 2160:
                cells[(cx, cy)] = cells.get((cx, cy), 0) + int(n)
        per_video[stem] = {"stops": n_stops, "person_tracks": int(ped.track_id.nunique())}
        print(stem, per_video[stem], flush=True)
    dens = [[cx * CELL, cy * CELL, n] for (cx, cy), n in sorted(cells.items()) if n >= 2]
    out = {
        "note": "detector evidence (YOLO11m + ByteTrack), mapped into the reference frame with H_video_to_ref; "
                "verify by eye, not ground truth",
        "coords": "reference 4K px (3840x2160)",
        "stop_rule": f"vehicle ground point (bottom centre), speed <= {STOP_THR} px/s for >= {MIN_STOP_S} s",
        "stop_points_fields": ["x", "y", "seconds", "class", "video"],
        "stop_points": stops,
        "ped_density_cell_px": CELL,
        "ped_density_fields": ["x0", "y0", "distinct_person_tracks"],
        "ped_density": dens,
        "per_video": per_video,
    }
    with open(a.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print("stops", len(stops), "density cells", len(dens), "->", a.out)


if __name__ == "__main__":
    main()
