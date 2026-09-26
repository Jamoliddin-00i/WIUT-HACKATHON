"""Lane-direction map in the REFERENCE frame, merged over all videos.

Each video's vehicle ground points and velocities (same smoothing as
tools/eda/analyze.py) are moved into the reference frame with H_video_to_ref from
registration.json; the velocity is mapped through the local Jacobian of the
homography. Then the reference frame is cut into the same 48 x 27 grid (80 px
cells in 4K) and, per cell, every vehicle track contributes one mean unit heading.

Per cell:
  heading_deg      image-coordinate heading (0 = right, 90 = down the image)
  coherence        length of the mean of the per-track unit vectors (1 = all agree)
  tracks, samples  support over all videos
  per_video        {stem: {tracks, heading_deg, coherence}}
  videos_agree     videos with >= 3 tracks whose own heading is within 30 deg
  status           "ok" if coherence >= --min-coh, tracks >= --min-tracks and at
                   least --min-videos videos agree; else the reason
  flow             geometric name of the flow (only for status ok)

Cross-check: the per-video direction_field.json cells (made in each video's own
pixels) are warped into the reference and compared with the merged heading of the
cell they land in; the angular differences are reported. This checks the warp and
binning, not the data (both come from the same tracks). The data check is
per-video agreement: "all_videos_agree_share" is the share of ok cells in which
every video with >= 3 tracks has its own heading within 30 deg of the merged one.

  python tools/scene/scene_reference.py --tracks-dir work/tracks --registration reports/eda/registration.json \
      --eda-dir reports/eda --out reports/eda
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "eda"))
from analyze import VEH, load_tracks  # noqa: E402
from tools.scene.register import REF_PATH  # noqa: E402

W, H, GX, GY = 3840, 2160, 48, 27
CW, CH = W / GX, H / GY
REF_STEM = "C3897"

# Flow names by geometry. Headings are image-coordinate degrees (0 = right, 90 = down).
# Decided from the merged map (see SUMMARY.md): the far carriageway runs right to left
# up the image (away from the camera), the near carriageway runs from top-left
# towards the bottom-right (towards the camera). The line between them is the
# median kerb, approximated by the image line through two reference points.
MEDIAN_LINE = ((220, 240), (2360, 1020))  # 4K ref px: far-left end and nose of the median kerb


def side_of_median(x, y):
    (x1, y1), (x2, y2) = MEDIAN_LINE
    return np.sign((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1))  # > 0 below / right of the line


def flow_name(heading, x, y):
    h = heading % 360
    below = side_of_median(x, y) > 0
    if 150 <= h <= 240:
        return ("far carriageway, right to left (away from camera)" if not below
                else "right to left below the median line (turns near the median nose, lower-left exit)")
    if h <= 60 or h >= 330:
        return "near carriageway, top-left to bottom-right (towards camera)" if below else "left to right, above median line (turning)"
    if 60 < h < 150:
        return "down and left (turn into the lower-left road)"
    return "up the image (turning away from camera)"


def jac_map(Hm, x, y, vx, vy, eps=0.1):
    P = np.stack([x, y], 1).reshape(-1, 1, 2).astype(np.float64)
    Q = np.stack([x + vx * eps, y + vy * eps], 1).reshape(-1, 1, 2).astype(np.float64)
    p = cv2.perspectiveTransform(P, Hm).reshape(-1, 2)
    q = cv2.perspectiveTransform(Q, Hm).reshape(-1, 2)
    v = (q - p) / eps
    return p[:, 0], p[:, 1], v[:, 0], v[:, 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tracks-dir", required=True, help="folder with <STEM>.csv (tracks_kaggle.csv schema)")
    ap.add_argument("--registration", required=True)
    ap.add_argument("--eda-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stems", default="C3896,C3897,C3902,C3905")
    ap.add_argument("--move-thr", type=float, default=60.0)
    ap.add_argument("--min-coh", type=float, default=0.7)
    ap.add_argument("--min-tracks", type=int, default=8)
    ap.add_argument("--min-videos", type=int, default=2)
    a = ap.parse_args()
    reg = json.load(open(a.registration))["videos"]
    parts = []
    for stem in a.stems.split(","):
        Hm = np.eye(3) if stem == REF_STEM else np.array(reg[stem]["homography"]["H_video_to_ref"])
        df, _ = load_tracks(os.path.join(a.tracks_dir, f"{stem}.csv"), 3, 29.97, 0.5)
        v = df[df.tcls.isin(VEH)].copy()
        v["rx"], v["ry"], v["rvx"], v["rvy"] = jac_map(Hm, v.sx.to_numpy(), v.sy.to_numpy(), v.vx.to_numpy(), v.vy.to_numpy())
        v["rspeed"] = np.hypot(v.rvx, v.rvy)
        v = v[v.rspeed > a.move_thr]
        v["stem"] = stem
        v["tid"] = stem + "_" + v.track_id.astype(str)
        parts.append(v[["stem", "tid", "rx", "ry", "rvx", "rvy", "rspeed"]])
        print(stem, len(v), "moving vehicle samples", flush=True)
    mv = pd.concat(parts)
    mv = mv[(mv.rx >= 0) & (mv.rx < W) & (mv.ry >= 0) & (mv.ry < H)]
    mv["ci"] = (mv.rx // CW).astype(int)
    mv["cj"] = (mv.ry // CH).astype(int)
    mv["ux"], mv["uy"] = mv.rvx / mv.rspeed, mv.rvy / mv.rspeed
    cells = []
    for (i, j), g in mv.groupby(["ci", "cj"]):
        pt = g.groupby(["stem", "tid"])[["ux", "uy"]].mean()
        nrm = np.hypot(pt.ux, pt.uy).replace(0, 1)
        pt["ux"], pt["uy"] = pt.ux / nrm, pt.uy / nrm  # per-track unit heading
        if len(pt) < 3:
            continue
        ux, uy = pt.ux.mean(), pt.uy.mean()
        coh = math.hypot(ux, uy)
        hd = math.degrees(math.atan2(uy, ux)) % 360
        per, agree = {}, []
        for stem, q in pt.groupby(level=0):
            qx, qy = q.ux.mean(), q.uy.mean()
            h2 = math.degrees(math.atan2(qy, qx)) % 360
            per[stem] = {"tracks": int(len(q)), "heading_deg": round(h2, 1), "coherence": round(math.hypot(qx, qy), 3)}
            if len(q) >= 3 and abs((h2 - hd + 180) % 360 - 180) <= 30:
                agree.append(stem)
        cx, cy = (i + 0.5) * CW, (j + 0.5) * CH
        reasons = []
        if coh < a.min_coh:
            reasons.append("low_coherence")
        if len(pt) < a.min_tracks:
            reasons.append("low_support")
        if len(agree) < a.min_videos:
            reasons.append("few_videos_agree")
        c = {"i": int(i), "j": int(j), "cx": round(cx), "cy": round(cy), "heading_deg": round(hd, 1),
             "dx": round(ux / max(coh, 1e-6), 3), "dy": round(uy / max(coh, 1e-6), 3), "coherence": round(coh, 3),
             "tracks": int(len(pt)), "samples": int(len(g)), "per_video": per, "videos_agree": agree,
             "status": "ok" if not reasons else "+".join(reasons)}
        if not reasons:
            c["flow"] = flow_name(hd, cx, cy)
        cells.append(c)
    ok = [c for c in cells if c["status"] == "ok"]
    # cross-check with the per-video direction_field.json, warped into the reference
    lut = {(c["i"], c["j"]): c for c in cells}
    xcheck = {}
    for stem in a.stems.split(","):
        Hm = np.eye(3) if stem == REF_STEM else np.array(reg[stem]["homography"]["H_video_to_ref"])
        f = json.load(open(os.path.join(a.eda_dir, stem, "direction_field.json")))["cells"]
        f = [c for c in f if c["coherence"] >= a.min_coh]
        x, y, dx, dy = jac_map(Hm, np.array([c["cx"] for c in f], float), np.array([c["cy"] for c in f], float),
                               np.array([c["dx"] for c in f]), np.array([c["dy"] for c in f]))
        diffs = []
        for k in range(len(f)):
            m = lut.get((int(x[k] // CW), int(y[k] // CH)))
            if m and m["status"] == "ok":
                h1 = math.degrees(math.atan2(dy[k], dx[k])) % 360
                diffs.append(abs((h1 - m["heading_deg"] + 180) % 360 - 180))
        if diffs:
            xcheck[stem] = {"cells_compared": len(diffs), "median_abs_diff_deg": round(float(np.median(diffs)), 1),
                            "p90_abs_diff_deg": round(float(np.percentile(diffs, 90)), 1),
                            "share_over_30deg": round(float(np.mean(np.array(diffs) > 30)), 3)}
    flows = {}
    for c in ok:
        fl = flows.setdefault(c["flow"], {"cells": 0, "headings": []})
        fl["cells"] += 1
        fl["headings"].append(c["heading_deg"])
    for k, fl in flows.items():
        hs = np.radians(fl.pop("headings"))
        fl["mean_heading_deg"] = round(float(np.degrees(np.arctan2(np.sin(hs).mean(), np.cos(hs).mean())) % 360), 1)
    full = [c for c in ok if len(c["videos_agree"]) == sum(1 for v in c["per_video"].values() if v["tracks"] >= 3)]
    status_counts = pd.Series([c["status"] for c in cells]).value_counts().to_dict()
    out = {"frame": "reference C3897, 4K px (3840x2160); map into a video with H_ref_to_video from registration.json",
           "grid": [GX, GY], "cell_px": [CW, CH], "moving_thr_px_s": a.move_thr,
           "gates": {"min_coherence": a.min_coh, "min_tracks": a.min_tracks, "min_videos_agree": a.min_videos,
                     "agree_within_deg": 30},
           "heading_convention": "image degrees: 0 = +x (right), 90 = +y (down the image)",
           "median_line_ref_px": MEDIAN_LINE, "flows": flows, "status_counts": status_counts,
           "crosscheck_warped_direction_field": xcheck,
           "all_videos_agree_share": round(len(full) / max(len(ok), 1), 3), "cells": cells}
    with open(os.path.join(a.out, "scene_reference.json"), "w") as fh:
        json.dump(out, fh)
    print(json.dumps({"all_agree": round(len(full) / max(len(ok), 1), 3), "status": status_counts, "flows": flows, "xcheck": xcheck}, indent=1))
    figure(cells, os.path.join(a.out, "scene_reference.jpg"), a)


def figure(cells, path, a):
    bg = cv2.imread(REF_PATH)
    img = (cv2.resize(bg, (1920, 1080)) * 0.55).astype(np.uint8)
    cols = {}
    palette = [(214, 120, 42), (52, 104, 235), (122, 175, 27), (0, 161, 237), (164, 123, 232), (72, 73, 227)]
    s = 0.5
    for c in cells:
        x, y = c["cx"] * s, c["cy"] * s
        if c["status"] != "ok":
            cv2.drawMarker(img, (int(x), int(y)), (150, 150, 150), cv2.MARKER_TILTED_CROSS, 8, 1)
            continue
        col = cols.setdefault(c["flow"], palette[len(cols) % len(palette)])
        L = 34 * (0.4 + 0.6 * c["coherence"])
        p0 = (int(x - c["dx"] * L / 2), int(y - c["dy"] * L / 2))
        p1 = (int(x + c["dx"] * L / 2), int(y + c["dy"] * L / 2))
        cv2.arrowedLine(img, p0, p1, col, 2, cv2.LINE_AA, tipLength=0.35)
    y0 = 1080 - 24 * (len(cols) + 2)
    cv2.rectangle(img, (0, y0 - 10), (1100, 1080), (20, 20, 20), -1)
    cv2.putText(img, f"reference frame C3897; arrow = merged heading of moving vehicles, all 4 videos; grey x = rejected "
                f"(coherence < {a.min_coh}, < {a.min_tracks} tracks or < {a.min_videos} videos agree)",
                (10, y0 + 10), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for k, (name, col) in enumerate(cols.items()):
        cv2.putText(img, name, (10, y0 + 36 + 24 * k), 0, 0.6, col, 2, cv2.LINE_AA)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == "__main__":
    main()
