"""Data files for zones.html and spotcheck.html (stdlib only).

Reads the EDA outputs in the repo and writes the static JSON the two pages load,
plus a copy of the spot-check crops:

  <out>/scene/scene_data.json   reference size, per-video H_ref_to_video and
                                background image, signal heads, registration
                                landmarks, lane-direction cells, detector evidence
  <out>/spotcheck/items.json    spot-check items in answering order (rare
                                automatic labels first), crop and context strip paths
  <out>/spotcheck/crops/<STEM>/*.jpg   the pack's 4K crops (copied)

  python3 tools/scene/web/build_data.py --eda reports/eda --out work/web \
      [--strips work/strips/strips.json]

The background images (scene/ref_4k.jpg, scene/bg_<STEM>.jpg) come from
make_backgrounds.py and the context strips from context_strips.py, both on the
server; see tools/scene/web/README.md.
"""
import argparse
import csv
import json
import math
import os
import shutil

STEMS = ["C3905", "C3896", "C3897", "C3902"]
RARE = ["red_amber", "amber", "green_flash", "occluded"]   # answered first, in this order
IDENTITY = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def scene_data(eda):
    reg = json.load(open(os.path.join(eda, "registration.json")))
    heads = json.load(open(os.path.join(eda, "signals", "signal_heads.json")))["heads"]
    # v2 map (v1 is superseded); headings in image degrees, 0 = +x, 90 = +y
    sref = json.load(open(os.path.join(eda, "scene_reference_v2.json")))
    ev_path = os.path.join(eda, "zone_evidence.json")
    ev = json.load(open(ev_path)) if os.path.exists(ev_path) else None
    ref_stem = reg["reference"]["stem"]
    videos = []
    for stem in STEMS:
        v = reg["videos"].get(stem)
        videos.append({
            "stem": stem,
            "H_ref_to_video": IDENTITY if stem == ref_stem else v["homography"]["H_ref_to_video"],
            "bg": f"scene/bg_{stem}.jpg",
            "bg_size": [1920, 1080],
            "is_reference": stem == ref_stem,
            "landmark_error": None if stem == ref_stem else v.get("landmark_error_summary"),
            "displacement_px": None if stem == ref_stem else v.get("displacement_ref_to_video_px"),
        })
    flows = sorted({c.get("flow") for c in sref["cells"] if c.get("flow")})
    cells = []
    for c in sref["cells"]:
        h = math.radians(c["dominant_heading"]) if c.get("dominant_heading") is not None else None
        if h is None:
            continue
        cells.append([c["cx"], c["cy"], round(math.cos(h), 3), round(math.sin(h), 3),
                      flows.index(c["flow"]) if c.get("flow") in flows else -1,
                      1 if c["status"] == "ok" else 0])
    return {
        "image_size": [3840, 2160],
        "coords": "reference C3897, 4K px",
        "reference": {"stem": ref_stem, "img_1080": "scene/ref_1080.jpg", "img_4k": "scene/ref_4k.jpg"},
        "videos": videos,
        "landmarks": reg["landmarks_ref_4k"],
        "heads": {h: {"box": d["box_ref"], "type": d["type"], "lamps": d["lamps"], "faces": d["faces"],
                      "controls": d["controls"], "confidence": d["controls_confidence"],
                      "readable": d["readable_from_camera"]} for h, d in heads.items()},
        "lanes": {"cell_px": sref["cell_px"], "flows": flows,
                  "fields": ["cx", "cy", "dx", "dy", "flow_index", "ok"], "cells": cells,
                  "gates": sref.get("gates", sref.get("params")),
                  # the median line is only in the v1 file; it is display-only
                  "median_line": sref.get("median_line_ref_px") or json.load(open(os.path.join(eda, "scene_reference.json"))).get("median_line_ref_px")},
        "evidence": ev,
    }


def spot_items(eda, strips):
    rows = list(csv.DictReader(open(os.path.join(eda, "spotcheck", "spotcheck.csv"))))
    by_head = {}
    for r in rows:
        by_head.setdefault((r["stem"], r["head"]), []).append(r)
    nb = {}
    for rs in by_head.values():
        rs.sort(key=lambda r: int(r["frame"]))
        for i, r in enumerate(rs):
            nb[r["id"]] = (rs[i - 1]["id"] if i else None, rs[i + 1]["id"] if i + 1 < len(rs) else None)
    items = []
    for r in rows:
        s = (strips or {}).get(r["id"])
        items.append({
            "id": r["id"], "stem": r["stem"], "head": r["head"], "frame": int(r["frame"]),
            "t_s": float(r["t_s"]), "auto": r["auto_label"],
            "img": "spotcheck/crops/" + r["image"], "pack_image": r["image"],
            "prev": nb[r["id"]][0], "next": nb[r["id"]][1],
            "strip": None if not s else {"src": "spotcheck/strips/" + s["strip"], "tiles": s["tiles"],
                                         "center": s["center"], "step_frames": s["step_frames"],
                                         "source": s["source"]},
        })
    rank = {lab: i for i, lab in enumerate(RARE)}
    stem_rank = {s: i for i, s in enumerate(STEMS)}
    items.sort(key=lambda it: (rank.get(it["auto"], len(RARE)), stem_rank[it["stem"]], it["head"], it["frame"]))
    return {"items": items, "rare_labels": RARE,
            "note": "30 random moments per video (fixed seed), heads D (vehicle, 3 lamps) and A "
                    "(pedestrian, 2 lamps); crops are native 4K; strips are 13 crops 0.2 s apart"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eda", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strips", default="")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "scene"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "spotcheck"), exist_ok=True)
    with open(os.path.join(a.out, "scene", "scene_data.json"), "w") as f:
        json.dump(scene_data(a.eda), f, separators=(",", ":"))
    strips = json.load(open(a.strips)) if a.strips and os.path.exists(a.strips) else None
    sp = spot_items(a.eda, strips)
    with open(os.path.join(a.out, "spotcheck", "items.json"), "w") as f:
        json.dump(sp, f, indent=0)
    for stem in STEMS:
        src = os.path.join(a.eda, "spotcheck", stem)
        dst = os.path.join(a.out, "spotcheck", "crops", stem)
        os.makedirs(dst, exist_ok=True)
        for fn in os.listdir(src):
            if fn.endswith(".jpg"):
                shutil.copy2(os.path.join(src, fn), dst)
    print("items", len(sp["items"]), "with strips", sum(1 for it in sp["items"] if it["strip"]))


if __name__ == "__main__":
    main()
