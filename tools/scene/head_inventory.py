"""Signal-head inventory: appearance table + automatic cycle check + figure.

Appearance (lamp count, figure icons, which side of the head faces the camera) was
read by eye from 4K crops of C3897 (the reference) at six moments over one cycle;
it is written in HEADS below, not inferred from traffic. The automatic part
tests whether any pixel of each head (its own box, minus pixels inside another
head's box) follows the signal cycle: the max over pixels of |correlation| between
its red / green colour score and the vehicle head D's green indicator, against a
null of circularly shifted indicators. A head whose lit face points away from the
camera has no pixel clearly above the null. (For D itself the test is circular.)

  python tools/scene/head_inventory.py --crops work/crops --signals reports/eda/signals \
      --out reports/eda/signals --stem C3897
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lamp_states import FPS, colour_scores  # noqa: E402

# reference (C3897) 4K boxes: YOLO traffic-light boxes on a 4K frame, tiled scan
HEADS = {
    "A": {"box_ref": [517, 964, 554, 1038], "type": "pedestrian", "lamps": 2,
          "appearance": "front visible: red standing figure (top), green walking figure (bottom)",
          "mount": "left gantry pole, lower head", "faces": "towards the camera (down-right in the image)",
          "controls": "pedestrians on the lower-left zebra (across the side road at the lower left)",
          "controls_confidence": "MEDIUM: from facing direction only; its green is synchronous with D"},
    "B": {"box_ref": [549, 957, 594, 1034], "type": "pedestrian (likely)", "lamps": 2,
          "appearance": "side / back view, two visors, no lit face visible",
          "mount": "left gantry pole, next to A", "faces": "right, along the zebra across the near carriageway",
          "controls": "pedestrians crossing the near carriageway from the median nose",
          "controls_confidence": "LOW"},
    "C": {"box_ref": [2244, 719, 2281, 779], "type": "pedestrian (likely)", "lamps": 2,
          "appearance": "side / back view, two visors pointing left, no lit face visible",
          "mount": "median-nose pole, left of D", "faces": "left, along the zebra across the near carriageway",
          "controls": "pedestrians crossing the near carriageway from the left corner",
          "controls_confidence": "LOW"},
    "D": {"box_ref": [2291, 719, 2340, 827], "type": "vehicle", "lamps": 3,
          "appearance": "front visible: three round lamps, red top, amber middle, green bottom",
          "mount": "median-nose pole", "faces": "towards the camera",
          # EDA v4: text corrected after the full-cycle validation (signal_validate.py v2); the far
          # carriageway is cycle-locked too, so "controls the near carriageway only" was not supported
          "controls": ("UNKNOWN which movements D physically controls. D is a strong timing proxy for the near-"
                       "carriageway flow: held-out, near-carriageway crossings rise by +1.10 (daylight) and +1.27 "
                       "(evening) per s after D green onset (p <= 0.011 under the most conservative periodicity-"
                       "preserving null), and only 2.5 to 6.3% of them happen while D is red (red is 45 to 53% of the "
                       "time). The far carriageway is also cycle-locked, with a different phase: its flow drops to a "
                       "low-rate stretch about 47 to 61-65 s after D green onset, inside D red "
                       "(signal_validation_v2.json). Coordinated signals or upstream platoons could explain either. See"
                       " reports/eda/SUMMARY.md, finding 3."),
          "controls_confidence": ("HIGH as a timing proxy for the near-carriageway flow; UNKNOWN for the movements it controls; v4"
                                  " timeline coverage only, no human-verified lamp labels yet")},
    "E": {"box_ref": [1442, 493, 1502, 615], "type": "vehicle (likely)", "lamps": None,
          "appearance": "back of a tall housing hanging from the gantry (similar height to D), lamps not visible",
          "mount": "gantry over the near carriageway, right end", "faces": "away from the camera, up-left",
          "controls": "near carriageway (towards camera): drivers approaching the gantry face it",
          "controls_confidence": "MEDIUM: position over the near lanes and facing direction; not readable"},
    "F": {"box_ref": [799, 530, 835, 640], "type": "vehicle (likely)", "lamps": None,
          "appearance": "back of a tall housing hanging from the gantry, lamps not visible",
          "mount": "gantry over the near carriageway, left part", "faces": "away from the camera, up-left",
          "controls": "near carriageway (towards camera)",
          "controls_confidence": "MEDIUM: as E"},
    "G": {"box_ref": [3782, 708, 3816, 753], "type": "pedestrian (likely)", "lamps": 2,
          "appearance": "side view at the right frame edge, two visors pointing left",
          "mount": "right kerb pole", "faces": "left, along the right-hand zebra",
          "controls": "pedestrians on the right-hand zebra",
          "controls_confidence": "LOW"},
}


def cycle_corr(z, own_box, other_boxes, green, rng, n_null=200):
    """Max over pixels of the head's own box (not inside another head's box) of
    |corr(colour score, vehicle-head green indicator)|, for red and green scores,
    with a null from circular shifts of the indicator (95th percentile)."""
    x0, y0 = z["box"][0], z["box"][1]
    h, w = z["crops"].shape[1:3]
    yy, xx = np.mgrid[0:h, 0:w]
    X, Y = xx + x0, yy + y0
    keep = (X >= own_box[0]) & (X <= own_box[2]) & (Y >= own_box[1]) & (Y <= own_box[3])
    for b in other_boxes:
        keep &= ~((X >= b[0]) & (X <= b[2]) & (Y >= b[1]) & (Y <= b[3]))
    if keep.sum() < 20:
        return {"pixels": int(keep.sum())}
    g = (green - green.mean()) / (green.std() + 1e-9)
    out = {"pixels": int(keep.sum())}
    for nm in ("red", "green"):
        S = np.stack([colour_scores(c)[nm][keep] for c in z["crops"]]).astype(np.float32)
        S = (S - S.mean(0)) / (S.std(0) + 1e-6)
        obs = float(np.abs(g @ S / len(g)).max())
        null = [float(np.abs(np.roll(g, int(rng.integers(len(g) // 8, len(g) - len(g) // 8))) @ S / len(g)).max())
                for _ in range(n_null)]
        out[nm] = {"max_abs_corr": round(obs, 3), "null_p95": round(float(np.percentile(null, 95)), 3)}
    return out


def main():
    rng = np.random.default_rng(0)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--crops", required=True)
    ap.add_argument("--signals", required=True, help="folder with <STEM>_D_samples.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stems", default="C3897,C3896,C3902,C3905")
    ap.add_argument("--stem", default="C3897", help="video used for the figure")
    a = ap.parse_args()
    inv = {"note": ("appearance, facing and controlled approach are read by eye from 4K crops of the reference "
                    "video; cycle_corr is automatic (see module docstring). Crop boxes carry a 20 px margin, "
                    "so a head's own box is its crop box shrunk by 20 px."), "heads": {}}
    boxes = {}
    for stem in a.stems.split(","):
        for h in HEADS:
            p = os.path.join(a.crops, f"{stem}_{h}.npz")
            if os.path.exists(p):
                b = np.load(p)["box"].tolist()
                boxes.setdefault(stem, {})[h] = [b[0] + 20, b[1] + 20, b[2] - 20, b[3] - 20]
    for h, d in HEADS.items():
        e = dict(d)
        e["cycle_corr"] = {}
        for stem in a.stems.split(","):
            p = os.path.join(a.crops, f"{stem}_{h}.npz")
            if not os.path.exists(p):
                continue
            z = np.load(p)
            sv = pd.read_csv(os.path.join(a.signals, f"{stem}_D_samples.csv")).set_index("frame").state
            st = sv.reindex(z["frames"]).fillna("unknown").to_numpy()
            ok = st != "occluded"
            green = np.isin(st, ["green", "green_flash"]).astype(np.float32)
            zz = {"box": z["box"], "crops": z["crops"][ok]}
            e["cycle_corr"][stem] = cycle_corr(zz, boxes[stem][h], [b for k, b in boxes[stem].items() if k != h],
                                               green[ok], rng)
            e.setdefault("crop_box_4k", {})[stem] = z["box"].tolist()
        e["readable_from_camera"] = h in ("A", "D")
        inv["heads"][h] = e
        print(h, e["cycle_corr"], flush=True)
    with open(os.path.join(a.out, "signal_heads.json"), "w") as f:
        json.dump(inv, f, indent=1)
    # figure: every head at a vehicle-red and a vehicle-green moment of the reference video
    sv = pd.read_csv(os.path.join(a.signals, f"{a.stem}_D_samples.csv"))
    picks = {}
    for st in ("red", "green"):
        c = sv[sv.state == st]
        picks[st] = int(c.frame.iloc[len(c) // 2])
    tiles = []
    for h in HEADS:
        p = os.path.join(a.crops, f"{a.stem}_{h}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        row = []
        for st, fr in picks.items():
            i = int(np.argmin(np.abs(z["frames"] - fr)))
            im = cv2.resize(z["crops"][i], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            im = cv2.copyMakeBorder(im, 22, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(im, f"{h} D={st} t={z['frames'][i] / FPS:.0f}s", (4, 16), 0, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            row.append(im)
        hh = max(r.shape[0] for r in row)
        row = [cv2.copyMakeBorder(r, 0, hh - r.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in row]
        tiles.append(np.hstack(row))
    hh = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, hh - t.shape[0], 0, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    cv2.imwrite(os.path.join(a.out, f"signal_heads_{a.stem}.jpg"), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    main()
