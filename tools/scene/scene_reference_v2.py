"""Lane-direction map v2 in the REFERENCE frame: several heading modes per cell,
per-video support, and an explicit wrong_way_safe flag.

Why v2: in v1 (scene_reference.json, now SUPERSEDED) an accepted ("ok") cell
means only that the merged heading is coherent and that 2 videos agree. It does
not mean one permissible direction: 5 accepted cells have a video with >= 3
tracks whose own heading is more than 30 deg off (cell (29,21) at (2360,1720):
merged 32.4 deg, C3905 124.6 deg), and 483 of 52,760 track-cell contributions
in accepted cells are more than 90 deg off the merged heading, in 182 of the
635 accepted cells. v1 is kept unchanged for traceability.

Contributions. Same ground points and velocities as v1 (tools/eda/analyze.py
smoothing; moved into the reference with H_video_to_ref, velocity through the
local Jacobian), same 48 x 27 grid of 80 px cells, same moving threshold. One
contribution = one (video, track, cell): the mean unit heading of that track's
moving samples in the cell. Masked before anything is computed:
  * samples whose box touches the video frame edge (within EDGE_PX): the ground
    point of a clipped box is not the vehicle's ground point;
  * motorcycle tracks: on this footage they include bicycles and scooters that
    cross on the zebras (reported separately, not used for modes).

Modes per cell (greedy circular clustering). Kernel density of the
contribution headings on a 1-deg circle (von Mises kernel, kappa = KAPPA, about
20 deg); the highest peak is a mode, every contribution within MODE_HALF_WIDTH
deg of it belongs to it; remove them and repeat while at least MIN_MODE_TRACKS
contributions and MIN_MODE_SHARE of the cell remain. Leftovers are "other".
Each mode: circular-mean heading, share of the cell's contributions, tracks,
per-video tracks.

Per-cell flags
  accepted              the v1 gates on the masked data (coherence >= 0.7,
                        >= 8 tracks, >= 2 videos with >= 3 tracks within 30 deg)
  dominant_heading      heading of the largest mode
  any_video_dissents    a video with >= 3 tracks in the cell has its own
                        circular mean > 30 deg from the dominant heading, or
                        fewer than 2/3 of its tracks in the dominant mode
  frame_edge            the cell centre is within one cell of the edge of the
                        reference frame or of any video's frame (after mapping)
  turning_or_conflict   a second mode with >= MIN_MODE_SHARE, or the dominant
                        heading differs by > CURVE_DEG from an accepted
                        8-neighbour, or the geometric flow name is a turn
  wrong_way_safe        accepted, dominant share >= SAFE_SHARE, >= SAFE_TRACKS
                        tracks, >= 2 supported videos, no video dissents, not
                        frame_edge, not turning_or_conflict. Only in these
                        cells may a rule call a vehicle "wrong way" from its
                        heading alone.
  wrong_way_blockers    codes of the conditions that failed (empty when safe):
                        not_accepted, dominant_share_below_safe, too_few_tracks,
                        fewer_than_2_supported_videos, video_dissents,
                        frame_edge, turning_or_conflict

  python tools/scene/scene_reference_v2.py --tracks-dir work/tracks \
      --registration reports/eda/registration.json --out reports/eda
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
sys.path.insert(0, os.path.join(ROOT, "tools", "scene"))
from analyze import load_tracks  # noqa: E402
from scene_reference import CH, CW, GX, GY, H, REF_STEM, W, flow_name, jac_map  # noqa: E402
from tools.scene.register import REF_PATH  # noqa: E402

VERSION = "v2"
MODE_CLASSES = ["car", "bus", "truck"]
EDGE_PX = 3
KAPPA = 8.0
MODE_HALF_WIDTH = 45.0
MIN_MODE_TRACKS = 2
MIN_MODE_SHARE = 0.05
AGREE_DEG = 30.0
CURVE_DEG = 30.0
SAFE_SHARE = 0.95
SAFE_TRACKS = 12
TURN_FLOWS = ("down and left", "left to right, above median line", "up the image")


def adiff(a, b):
    return np.abs((np.asarray(a) - b + 180.0) % 360.0 - 180.0)


def circ_mean(deg):
    r = np.radians(np.asarray(deg, float))
    return float(np.degrees(math.atan2(np.sin(r).mean(), np.cos(r).mean())) % 360)


def modes_of(h):
    """Greedy circular clustering of headings (deg). Returns (modes, labels);
    labels = mode index per heading, -1 = other."""
    grid = np.arange(360.0)
    lab = np.full(len(h), -1)
    modes = []
    left = np.ones(len(h), bool)
    while left.sum() >= MIN_MODE_TRACKS and left.sum() >= MIN_MODE_SHARE * len(h):
        r = np.radians(grid[:, None] - h[left][None, :])
        dens = np.exp(KAPPA * (np.cos(r) - 1)).sum(1)
        peak = grid[int(np.argmax(dens))]
        mem = left & (adiff(h, peak) <= MODE_HALF_WIDTH)
        if mem.sum() < MIN_MODE_TRACKS or mem.sum() < MIN_MODE_SHARE * len(h):
            break
        centre = circ_mean(h[mem])
        mem = left & (adiff(h, centre) <= MODE_HALF_WIDTH)  # re-centre once
        lab[mem] = len(modes)
        modes.append(centre)
        left &= ~mem
    return modes, lab


def contributions(tracks_dir, reg, stems, move_thr):
    parts, masked = [], {}
    for stem in stems:
        Hm = np.eye(3) if stem == REF_STEM else np.array(reg[stem]["homography"]["H_video_to_ref"])
        df, _ = load_tracks(os.path.join(tracks_dir, f"{stem}.csv"), 3, 29.97, 0.5)
        v = df[df.tcls.isin(MODE_CLASSES + ["motorcycle"])].copy()
        v["rx"], v["ry"], v["rvx"], v["rvy"] = jac_map(Hm, v.sx.to_numpy(), v.sy.to_numpy(),
                                                       v.vx.to_numpy(), v.vy.to_numpy())
        v["rspeed"] = np.hypot(v.rvx, v.rvy)
        v = v[v.rspeed > move_thr]
        clip = (v.x1 <= EDGE_PX) | (v.y1 <= EDGE_PX) | (v.x2 >= W - EDGE_PX) | (v.y2 >= H - EDGE_PX)
        moto = v.tcls == "motorcycle"
        masked[stem] = {"moving_samples": int(len(v)), "clipped_box_samples": int(clip.sum()),
                        "motorcycle_samples": int((moto & ~clip).sum())}
        v = v[~clip & ~moto]
        v["stem"] = stem
        v["tid"] = stem + "_" + v.track_id.astype(str)
        parts.append(v[["stem", "tid", "rx", "ry", "rvx", "rvy", "rspeed"]])
    mv = pd.concat(parts)
    mv = mv[(mv.rx >= 0) & (mv.rx < W) & (mv.ry >= 0) & (mv.ry < H)].copy()
    mv["ci"] = (mv.rx // CW).astype(int)
    mv["cj"] = (mv.ry // CH).astype(int)
    mv["ux"], mv["uy"] = mv.rvx / mv.rspeed, mv.rvy / mv.rspeed
    pt = mv.groupby(["ci", "cj", "stem", "tid"])[["ux", "uy"]].mean().reset_index()
    pt["h"] = np.degrees(np.arctan2(pt.uy, pt.ux)) % 360
    return pt, masked


def edge_cell(cx, cy, reg, stems):
    if min(cx, cy, W - cx, H - cy) < CW:
        return True
    for stem in stems:
        if stem == REF_STEM:
            continue
        Hm = np.array(reg[stem]["homography"]["H_ref_to_video"])
        p = cv2.perspectiveTransform(np.array([[[cx, cy]]], np.float64), Hm)[0, 0]
        if min(p[0], p[1], W - p[0], H - p[1]) < CW:
            return True
    return False


def build(pt, reg, stems, min_coh=0.7, min_tracks=8, min_videos=2):
    cells = {}
    for (i, j), g in pt.groupby(["ci", "cj"]):
        if len(g) < 3:
            continue
        h = g.h.to_numpy()
        coh = float(np.hypot(np.cos(np.radians(h)).mean(), np.sin(np.radians(h)).mean()))
        mean_h = circ_mean(h)
        modes, lab = modes_of(h)
        dom = modes[0] if modes else mean_h
        per_video, agree, dissent = {}, [], []
        for stem, q in g.groupby("stem"):
            hv = q.h.to_numpy()
            in_dom = float((lab[g.stem.to_numpy() == stem] == 0).mean()) if modes else 0.0
            mh = circ_mean(hv)
            per_video[stem] = {"tracks": int(len(q)), "heading_deg": round(mh, 1), "share_in_dominant": round(in_dom, 3)}
            if len(q) >= 3:
                if adiff(mh, mean_h) <= AGREE_DEG:
                    agree.append(stem)
                if adiff(mh, dom) > AGREE_DEG or in_dom < 2 / 3:
                    dissent.append(stem)
        cx, cy = (i + 0.5) * CW, (j + 0.5) * CH
        reasons = []
        if coh < min_coh:
            reasons.append("low_coherence")
        if len(g) < min_tracks:
            reasons.append("low_support")
        if len(agree) < min_videos:
            reasons.append("few_videos_agree")
        md = []
        for k, m in enumerate(modes):
            sel = lab == k
            md.append({"heading_deg": round(m, 1), "share": round(float(sel.mean()), 3), "tracks": int(sel.sum()),
                       "per_video_tracks": {s: int(c) for s, c in g[sel].stem.value_counts().items()}})
        cells[(int(i), int(j))] = {
            "i": int(i), "j": int(j), "cx": round(cx), "cy": round(cy), "tracks": int(len(g)),
            "coherence": round(coh, 3), "mean_heading_deg": round(mean_h, 1),
            "dominant_heading": round(dom, 1), "modes": md,
            "other_share": round(float((lab < 0).mean()), 3),
            "per_video": per_video, "videos_agree": agree, "dissenting_videos": dissent,
            "any_video_dissents": bool(dissent), "accepted": not reasons,
            "status": "ok" if not reasons else "+".join(reasons),
            "flow": flow_name(dom, cx, cy), "frame_edge": edge_cell(cx, cy, reg, stems)}
    # neighbourhood curvature and the final flag
    for (i, j), c in cells.items():
        nb = [cells[(i + di, j + dj)] for di in (-1, 0, 1) for dj in (-1, 0, 1)
              if (di or dj) and (i + di, j + dj) in cells and cells[(i + di, j + dj)]["accepted"]]
        curve = max([float(adiff(n["dominant_heading"], c["dominant_heading"])) for n in nb], default=0.0)
        second = len(c["modes"]) > 1 and c["modes"][1]["share"] >= MIN_MODE_SHARE
        turn_flow = c["flow"].startswith(TURN_FLOWS)
        c["max_neighbour_heading_diff_deg"] = round(curve, 1)
        c["turning_or_conflict"] = bool(second or curve > CURVE_DEG or turn_flow)
        dom_share = c["modes"][0]["share"] if c["modes"] else 0.0
        supported = sum(1 for v in c["per_video"].values() if v["tracks"] >= 3)
        block = []
        if not c["accepted"]:
            block.append("not_accepted")
        if dom_share < SAFE_SHARE:
            block.append("dominant_share_below_safe")
        if c["tracks"] < SAFE_TRACKS:
            block.append("too_few_tracks")
        if supported < 2:
            block.append("fewer_than_2_supported_videos")
        if c["any_video_dissents"]:
            block.append("video_dissents")
        if c["frame_edge"]:
            block.append("frame_edge")
        if c["turning_or_conflict"]:
            block.append("turning_or_conflict")
        c["dominant_share"] = round(dom_share, 3)
        c["supported_videos"] = supported
        c["wrong_way_safe"] = not block
        c["wrong_way_blockers"] = block
    return [cells[k] for k in sorted(cells)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tracks-dir", required=True, help="folder with <STEM>.csv (tracks_kaggle.csv schema)")
    ap.add_argument("--registration", default="reports/eda/registration.json")
    ap.add_argument("--out", default="reports/eda")
    ap.add_argument("--stems", default="C3896,C3897,C3902,C3905")
    ap.add_argument("--move-thr", type=float, default=60.0)
    a = ap.parse_args()
    stems = a.stems.split(",")
    reg = json.load(open(a.registration))["videos"]
    pt, masked = contributions(a.tracks_dir, reg, stems, a.move_thr)
    cells = build(pt, reg, stems)
    acc = [c for c in cells if c["accepted"]]
    safe = [c for c in cells if c["wrong_way_safe"]]
    blockers = pd.Series([b for c in acc for b in c["wrong_way_blockers"]]).value_counts().to_dict()
    lut = {(c["i"], c["j"]): c for c in cells}
    key = list(zip(pt.ci, pt.cj))
    dom = np.array([lut[k]["dominant_heading"] if k in lut and lut[k]["accepted"] else np.nan for k in key])
    off = adiff(pt.h.to_numpy(), np.nan_to_num(dom))
    in_acc = ~np.isnan(dom)
    out = {"version": VERSION, "status": "PROVISIONAL (tracker output only, no labelled wrong_way event)",
           "supersedes": "scene_reference.json (v1)",
           "frame": "reference C3897, 4K px (3840x2160); map into a video with H_ref_to_video from registration.json",
           "grid": [GX, GY], "cell_px": [CW, CH], "moving_thr_px_s": a.move_thr,
           "heading_convention": "image degrees: 0 = +x (right), 90 = +y (down the image)",
           "params": {"mode_classes": MODE_CLASSES, "edge_px": EDGE_PX, "kappa": KAPPA,
                      "mode_half_width_deg": MODE_HALF_WIDTH, "min_mode_tracks": MIN_MODE_TRACKS,
                      "min_mode_share": MIN_MODE_SHARE, "agree_deg": AGREE_DEG, "curve_deg": CURVE_DEG,
                      "safe_share": SAFE_SHARE, "safe_tracks": SAFE_TRACKS},
           "masked_samples": masked,
           "summary": {"cells": len(cells), "accepted": len(acc), "wrong_way_safe": len(safe),
                       "accepted_with_2plus_modes": sum(1 for c in acc if len(c["modes"]) > 1),
                       "accepted_with_dissenting_video": sum(1 for c in acc if c["any_video_dissents"]),
                       "contributions_in_accepted": int(in_acc.sum()),
                       "contributions_over_90deg_from_dominant": int((off[in_acc] > 90).sum()),
                       "accepted_cells_with_such_contributions": int(len({k for k, o, m in zip(key, off, in_acc)
                                                                          if m and o > 90})),
                       "accepted_blocker_counts": blockers},
           "cells": cells}
    with open(os.path.join(a.out, "scene_reference_v2.json"), "w") as fh:
        json.dump(out, fh)
    print(json.dumps({"masked": masked, "summary": out["summary"]}, indent=1))
    figure(cells, os.path.join(a.out, "scene_reference_v2.jpg"))


def figure(cells, path):
    bg = cv2.imread(REF_PATH)
    img = (cv2.resize(bg, (1920, 1080)) * 0.5).astype(np.uint8)
    s = 0.5

    def arrow(c, hd, col, L, w):
        x, y = c["cx"] * s, c["cy"] * s
        dx, dy = math.cos(math.radians(hd)), math.sin(math.radians(hd))
        cv2.arrowedLine(img, (int(x - dx * L / 2), int(y - dy * L / 2)), (int(x + dx * L / 2), int(y + dy * L / 2)),
                        col, w, cv2.LINE_AA, tipLength=0.35)

    for c in cells:
        if not c["accepted"]:
            cv2.drawMarker(img, (int(c["cx"] * s), int(c["cy"] * s)), (120, 120, 120), cv2.MARKER_TILTED_CROSS, 7, 1)
            continue
        if c["wrong_way_safe"]:
            arrow(c, c["dominant_heading"], (90, 220, 60), 32, 2)
        else:
            arrow(c, c["dominant_heading"], (0, 165, 255), 26, 2)
            for m in c["modes"][1:]:
                arrow(c, m["heading_deg"], (230, 60, 230), 18, 1)
            if c["any_video_dissents"]:
                cv2.circle(img, (int(c["cx"] * s), int(c["cy"] * s)), 17, (60, 60, 255), 1, cv2.LINE_AA)
    n_safe = sum(c["wrong_way_safe"] for c in cells)
    n_acc = sum(c["accepted"] for c in cells)
    cv2.rectangle(img, (0, 1080 - 120), (1250, 1080), (20, 20, 20), -1)
    lines = [(f"scene_reference v2, reference frame C3897; {n_acc} accepted cells, {n_safe} wrong_way_safe", (255, 255, 255)),
             ("green arrow = wrong_way_safe cell (one mode >= 95%, >= 12 tracks, all videos agree, not edge, not turning)",
              (90, 220, 60)),
             ("orange arrow = accepted but NOT safe (dominant mode); magenta = further modes", (0, 165, 255)),
             ("red ring = a video with >= 3 tracks dissents; grey x = not accepted", (60, 60, 255))]
    for k, (t, col) in enumerate(lines):
        cv2.putText(img, t, (10, 1080 - 95 + 26 * k), 0, 0.6, col, 1, cv2.LINE_AA)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 78])


if __name__ == "__main__":
    main()
