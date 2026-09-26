"""Compare two tracks CSVs of the same video (detector / tracker ablation).

Example
  python compare.py --a eda/C3905_yolo11n/tracks.csv --a-label "yolo11n 1280, CPU" \
      --b eda/C3905/tracks_kaggle.csv --b-label "yolo11m 1280, T4" --out eda/C3905
Writes detector_compare.json and detector_compare.png.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CLASSES = ["person", "car", "bus", "truck", "motorcycle", "bicycle"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL_A, COL_B = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False,
})


def stats(path, min_det):
    df = pd.read_csv(path)
    df["bh"] = df.y2 - df.y1
    maj = df.groupby("track_id").cls.agg(lambda s: s.value_counts().index[0])
    n = df.groupby("track_id").size()
    keep = n[n >= min_det].index
    tdur = df.groupby("track_id").t_sec.agg(lambda s: s.max() - s.min())
    frames = df.frame.nunique()
    out = {"rows": int(len(df)), "processed_frames_with_boxes": int(frames),
           "unique_tracks": {c: int((maj == c).sum()) for c in CLASSES},
           "tracks_min_det": {c: int((maj[keep] == c).sum()) for c in CLASSES},
           "detections_per_frame": {c: round(float((df.cls == c).sum() / frames), 2) for c in CLASSES},
           "median_track_s": {c: round(float(tdur[maj == c].median()), 1) if (maj == c).any() else None
                              for c in CLASSES}}
    ph = df[df.cls == "person"].bh
    out["person_box_height_4k"] = {"median": round(float(ph.median()), 1), "p05": round(float(ph.quantile(0.05)), 1),
                                   "share_below_96px": round(float((ph < 96).mean()), 3),
                                   "share_below_80px": round(float((ph < 80).mean()), 3),
                                   "count_below_96px_per_frame": round(float((ph < 96).sum() / frames), 2)}
    return df, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--a-label", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--b-label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-det", type=int, default=3)
    args = ap.parse_args()
    da, sa = stats(args.a, args.min_det)
    db, sb = stats(args.b, args.min_det)
    res = {"a": {"label": args.a_label, "file": os.path.basename(args.a), **sa},
           "b": {"label": args.b_label, "file": os.path.basename(args.b), **sb}}
    with open(os.path.join(args.out, "detector_compare.json"), "w") as f:
        json.dump(res, f, indent=1)

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    x = np.arange(len(CLASSES))
    w = 0.38
    for ax_, key, ttl in [(a1, "detections_per_frame", "detections per processed frame"),
                          (a2, "tracks_min_det", f"unique tracks (>= {args.min_det} detections)")]:
        va = [sa[key][c] for c in CLASSES]
        vb = [sb[key][c] for c in CLASSES]
        ax_.bar(x - w / 2 - 0.01, va, w, color=COL_A, label=args.a_label)
        ax_.bar(x + w / 2 + 0.01, vb, w, color=COL_B, label=args.b_label)
        ax_.set_xticks(x, CLASSES, rotation=20)
        ax_.set_title(ttl)
        ax_.grid(axis="x", visible=False)
    a1.legend(loc="upper right")
    bins = np.logspace(np.log10(30), np.log10(600), 30)
    for d, col, lab in [(da, COL_A, args.a_label), (db, COL_B, args.b_label)]:
        ph = d[d.cls == "person"].bh
        a3.hist(ph, bins=bins, histtype="step", lw=2, color=col, label=lab)
    a3.set_xscale("log")
    a3.set_xticks([40, 60, 100, 200, 400], ["40", "60", "100", "200", "400"])
    a3.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a3.set_xlabel("person box height (px, 4K, log)")
    a3.set_ylabel("detections")
    a3.set_title("person sizes found")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "detector_compare.png"), dpi=100)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
