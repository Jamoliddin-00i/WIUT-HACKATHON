"""Read signal-head lamps from 4K crops with a per-head, per-video calibration.

Input: <crops>/<STEM>_<HEAD>.npz from tools/scene/kaggle_head_crops.py (frames,
crops as N x h x w x 3 BGR, box = crop box in the video's 4K px).

Each crop is first re-aligned to the median crop by gradient NCC (the camera
drifts up to ~30 px while it settles after recording starts).

Calibration, per head and per video, no fixed colour thresholds:
  1. Lamp positions. Per pixel, a colour score (red: R - max(G, B); green:
     G - R, lit green here is cyan; amber: min(R, G) - B). The temporal 99.5th
     percentile minus the 5th percentile is high only where a lamp turns on and
     off (a lamp can be on for more than half the clip, so not the median). Restricted to the dark housing (temporal median gray in the darkest part
     of the crop) so tail lights of cars behind the head are ignored. The centroid
     of the strongest blob is the lamp centre. Amber, when its own blob is weak,
     sits at the midpoint of red and green (3 lamps in a vertical column).
  2. Per sample, per lamp: mean colour score in a disc at the lamp centre
     (radius = 0.28 x lamp pitch). Scores subtract the other channels, so a
     global brightness change (daylight, exposure) mostly cancels.
  3. Per lamp, a 1-D two-means split of that feature over the whole video gives
     the on / off threshold; separation d' = (on mean - off mean) / pooled sd says
     whether the lamp is readable at all (d' >= 3).
State per sample: the set of lamps above threshold. vehicle: {red} red, {amber}
amber, {green} green, {red, amber} red_amber, {} dark, anything else unclear.
pedestrian: {red} red, {green} green, {} dark, both unclear.
Samples where a vehicle passes in front of the head (many housing pixels far
brighter than usual, or a lamp's surroundings covered) are labelled occluded,
unless exactly one lamp is clearly on (halfway into its on-cluster) and its own
surroundings are clear. Occluded samples are left out of the thresholds.
"dark" runs of at most 1.5 s that follow green and are followed by green, amber or
red are relabelled green_flash (blinking green); other dark samples count as unclear.

BEFORE (for comparison): the v2 rule from tools/eda/analyze.py (HSV V > 150,
S > 90, lit share > 0.003 and 1.5 x runner-up) applied to the same crops.

  python tools/scene/lamp_states.py --crops work/crops --out reports/eda/signals \
      --heads A:pedestrian,D:vehicle --stems C3905,C3896,C3897,C3902
"""
import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd

FPS = 30000 / 1001
COLS = {"red": "#e34948", "amber": "#eda100", "green": "#008300", "red_amber": "#f08a24",
        "green_flash": "#7cc47c", "dark": "#52514e", "unclear": "#d8d6d0", "occluded": "#9fc5e8"}


def colour_scores(c):
    c = c.astype(np.int16)
    b, g, r = c[..., 0], c[..., 1], c[..., 2]
    return {"red": r - np.maximum(g, b), "green": g - r, "amber": np.minimum(r, g) - b}


def locate_lamps(crops, kind):
    """Lamp centres (x, y) in crop px, from where each colour switches on and off."""
    sub = crops[:: max(1, len(crops) // 1500)]
    gray = np.median(sub.mean(-1), axis=0)
    housing = gray < np.percentile(gray, 35)
    housing = cv2.dilate(housing.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    names = ["red", "green"] + (["amber"] if kind == "vehicle" else [])
    act, cen = {}, {}
    for nm in names:
        # clipped at 0: a lit lamp of another colour drives this score negative, which
        # must not count as activity
        s = np.stack([np.maximum(colour_scores(x)[nm], 0) for x in sub]).astype(np.float32)
        a = np.percentile(s, 99.5, axis=0) - np.percentile(s, 5, axis=0)
        a = cv2.GaussianBlur(a, (0, 0), 1.5) * housing
        act[nm] = a
        if a.max() <= 0:
            continue
        m = (a > 0.5 * a.max()).astype(np.uint8)
        n, lab, stats, cents = cv2.connectedComponentsWithStats(m)
        k = 1 + int(np.argmax([a[lab == i].sum() for i in range(1, n)]))
        ys, xs = np.nonzero(lab == k)
        w = a[ys, xs]
        cen[nm] = (float((xs * w).sum() / w.sum()), float((ys * w).sum() / w.sum()), float(a.max()))
    if kind == "vehicle" and "red" in cen and "green" in cen:
        mid = ((cen["red"][0] + cen["green"][0]) / 2, (cen["red"][1] + cen["green"][1]) / 2)
        am = cen.get("amber")
        # trust the amber blob only if it is strong and close to the red-green midpoint
        spacing = np.hypot(cen["red"][0] - cen["green"][0], cen["red"][1] - cen["green"][1])
        if am is None or am[2] < 0.3 * min(cen["red"][2], cen["green"][2]) or \
                np.hypot(am[0] - mid[0], am[1] - mid[1]) > 0.25 * spacing:
            cen["amber"] = (mid[0], mid[1], -1.0)
    return cen, housing


def two_means(v, iters=50):
    lo, hi = np.percentile(v, 30), np.percentile(v, 99.7)
    for _ in range(iters):
        thr = (lo + hi) / 2
        a, b = v[v <= thr], v[v > thr]
        if len(a) == 0 or len(b) == 0:
            break
        lo2, hi2 = a.mean(), b.mean()
        if abs(lo2 - lo) < 1e-3 and abs(hi2 - hi) < 1e-3:
            break
        lo, hi = lo2, hi2
    thr = (lo + hi) / 2
    a, b = v[v <= thr], v[v > thr]
    sd = np.sqrt(((a.var() if len(a) > 1 else 0) + (b.var() if len(b) > 1 else 0)) / 2) + 1e-6
    return float(thr), float((hi - lo) / sd), float(len(b) / len(v)), float(hi)


def old_rule(crops):
    """The v2 analyze.py classifier on each crop (per sample, before smoothing)."""
    out = []
    for c in crops:
        hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
        hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        lit = (vv > 150) & (ss > 90)
        f = np.array([(lit & ((hh < 10) | (hh > 165))).mean(), (lit & (hh >= 12) & (hh <= 32)).mean(),
                      (lit & (hh >= 45) & (hh <= 100)).mean()])
        s = np.sort(f)
        out.append(["red", "amber", "green"][int(f.argmax())] if s[-1] > 0.003 and s[-1] > 1.5 * s[-2] else "none")
    return np.array(out)


def mode_filter(lab, win=9):
    out = lab.copy()
    h = win // 2
    for i in range(len(lab)):
        v, c = np.unique(lab[max(0, i - h):i + h + 1], return_counts=True)
        out[i] = v[c.argmax()]
    return out


def runs(t, lab):
    res, i = [], 0
    while i < len(lab):
        j = i
        while j + 1 < len(lab) and lab[j + 1] == lab[i]:
            j += 1
        res.append([lab[i], i, j])
        i = j + 1
    return res


def classify(feat, thr, kind):
    on = {k: feat[k] > thr[k] for k in feat}
    n = len(next(iter(feat.values())))
    lab = np.full(n, "unclear", dtype=object)
    r, g = on["red"], on["green"]
    if kind == "vehicle":
        a = on["amber"]
        lab[r & ~a & ~g] = "red"
        lab[a & ~r & ~g] = "amber"
        lab[g & ~r & ~a] = "green"
        lab[r & a & ~g] = "red_amber"
        lab[~r & ~a & ~g] = "dark"
    else:
        lab[r & ~g] = "red"
        lab[g & ~r] = "green"
        lab[~r & ~g] = "dark"
    return lab.astype("<U12")


def occlusion(crops, housing, discs, jump=40, frac=0.15):
    """True where something passes in front of the head: a large share of the housing
    pixels (outside the lamp discs) is much brighter than their temporal median."""
    keep = housing.copy()
    for d in discs.values():
        keep &= ~cv2.dilate(d.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    g = crops.mean(-1)[:, keep] if keep.any() else crops.mean(-1).reshape(len(crops), -1)
    med = np.median(g, axis=0)
    return ((g - med) > jump).mean(axis=1) > frac


def lamp_ring_occlusion(crops, discs, jump=40, frac=0.3):
    """True where any lamp's surroundings (a ring just outside the lamp disc, on the
    housing) is covered by something much brighter, e.g. a truck roof passing
    in front of the lower lamp."""
    g = crops.mean(-1)
    med = np.median(g[:: max(1, len(g) // 400)], axis=0)
    hit = {}
    for k, d in discs.items():
        ring = cv2.dilate(d.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~d
        hit[k] = ((g[:, ring] - med[ring]) > jump).mean(axis=1) > frac
    return hit


def label_flash(t, lab, max_dark_s=1.5):
    """Occluded samples are skipped (the pattern is read on the visible samples)."""
    lab = lab.astype("<U12")
    vis = np.nonzero(lab != "occluded")[0]
    sub, ts = lab[vis], t[vis]
    rr = runs(ts, sub)
    for k, (s, i, j) in enumerate(rr):
        # a short dark gap after green that is followed by green again or by the next
        # phase (amber on a vehicle head, red on a pedestrian head) = blinking green
        if s == "dark" and 0 < k < len(rr) - 1 and rr[k - 1][0] in ("green", "green_flash") \
                and rr[k + 1][0] in ("green", "amber", "red") and ts[j] - ts[i] <= max_dark_s:
            sub[i:j + 1] = "green_flash"
    lab[vis] = sub
    return lab


def phases(t, lab, min_s=1.0):
    """Collapse per-sample labels into phases; runs shorter than min_s are absorbed
    into the previous phase (flicker). green_flash merges into green; occluded
    samples keep the previous state."""
    lab = np.where(lab == "green_flash", "green", lab)
    lab = pd.Series(np.where(lab == "occluded", None, lab)).ffill().fillna("occluded").to_numpy()
    out = []
    for s, i, j in runs(t, lab):
        dur = t[j] - t[i] + (t[1] - t[0] if len(t) > 1 else 0)
        if out and (dur < min_s or out[-1]["state"] == s):
            out[-1]["end"] = round(float(t[j]), 2)
            continue
        out.append({"state": s, "start": round(float(t[i]), 2), "end": round(float(t[j]), 2)})
    for p in out:
        p["duration"] = round(p["end"] - p["start"], 2)
    return out


def _grad(g):
    g = g.astype(np.float32)
    return cv2.GaussianBlur(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)), (0, 0), 1.0)


def stabilise(crops, max_shift=20, min_ncc=0.6):
    """Undo small camera drift (the camera settles for up to ~30 s after recording
    starts): the central part of the median crop (head housing and pole) is found in
    each crop by NCC of gradient images within +-max_shift px, and the crop is
    shifted back. Weak matches (e.g. a bus in front of the head) are not applied."""
    med = np.median(crops[:: max(1, len(crops) // 400)].mean(-1), axis=0)
    m = max_shift
    tmpl = _grad(med)[m:-m, m:-m]
    out = np.empty_like(crops)
    shifts = np.zeros((len(crops), 2), np.float32)
    for i, c in enumerate(crops):
        res = cv2.matchTemplate(_grad(c.mean(-1)), tmpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, (x, y) = cv2.minMaxLoc(res)
        dx, dy = x - m, y - m
        if mx >= min_ncc and (dx or dy):
            shifts[i] = (dx, dy)
            M = np.float32([[1, 0, -dx], [0, 1, -dy]])
            out[i] = cv2.warpAffine(c, M, (c.shape[1], c.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        else:
            out[i] = c
    return out, shifts


def analyse(npz, kind):
    d = np.load(npz)
    frames = d["frames"]
    crops, shifts = stabilise(d["crops"])
    t = frames / FPS
    cen, housing = locate_lamps(crops, kind)
    if "red" not in cen or "green" not in cen:
        return None
    # lamp pitch: red to green spans two pitches on a 3-lamp head, one on a 2-lamp head
    spacing = np.hypot(cen["red"][0] - cen["green"][0], cen["red"][1] - cen["green"][1]) / (2 if kind == "vehicle" else 1)
    rad = max(3.0, 0.28 * spacing)
    hh, ww = crops.shape[1:3]
    yy, xx = np.mgrid[0:hh, 0:ww]
    lamps = ["red", "amber", "green"] if kind == "vehicle" else ["red", "green"]
    disc = {k: (xx - cen[k][0]) ** 2 + (yy - cen[k][1]) ** 2 <= rad ** 2 for k in lamps}
    feat = {k: np.empty(len(crops), np.float32) for k in lamps}
    for n, c in enumerate(crops):
        sc = colour_scores(c)
        for k in lamps:
            feat[k][n] = sc[k][disc[k]].mean()
    occ = occlusion(crops, housing, disc)
    thr, sep, share, on_mean = {}, {}, {}, {}
    for k in lamps:
        thr[k], sep[k], share[k], on_mean[k] = two_means(feat[k][~occ])
    lab = classify(feat, thr, kind)
    ring = lamp_ring_occlusion(crops, disc)
    any_ring = np.any(np.stack(list(ring.values())), axis=0)
    # keep a reading despite something in front when exactly one lamp is clearly on
    # (at least halfway into its on-cluster) and that lamp's own surroundings are clear
    single = {"red": "red", "green": "green", "amber": "amber"}
    keep = np.zeros(len(lab), bool)
    for k, st in single.items():
        if k in feat:
            keep |= (lab == st) & (feat[k] >= (thr[k] + on_mean[k]) / 2) & ~ring[k]
    lab[((occ | any_ring) & ~keep) | (any_ring & np.isin(lab, ["dark", "unclear"]))] = "occluded"
    occ = lab == "occluded"
    lab = label_flash(t, lab)
    before = old_rule(crops)
    return {"frames": frames, "t": t, "feat": feat, "thr": thr, "sep": sep, "on_share": share, "lab": lab,
            "before": before, "before_smoothed": mode_filter(before), "centres": cen, "radius": rad,
            "crops": crops, "box": d["box"].tolist(), "spacing_px": spacing, "shifts": shifts, "occluded": occ}


def fig_timeline(stem, res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    heads = [h for h in res if res[h] is not None]
    fig, axs = plt.subplots(len(heads) * 2, 1, figsize=(14, 2.6 * len(heads) * 2), sharex=True,
                            gridspec_kw={"height_ratios": [1, 2] * len(heads)})
    for k, h in enumerate(heads):
        r = res[h]
        a0, a1 = axs[2 * k], axs[2 * k + 1]
        t, lab = r["t"], r["lab"]
        for s, i, j in runs(t, lab):
            a0.axvspan(t[i], t[j] + (t[1] - t[0]), color=COLS.get(s, "#999"), lw=0)
        a0.set_yticks([])
        a0.set_title(f"{stem} head {h} ({r['kind']}): per-sample state, every 3rd frame", loc="left", fontsize=12)
        for nm in r["feat"]:
            a1.plot(t, r["feat"][nm], color=COLS[nm], lw=0.6, label=f"{nm} lamp (thr {r['thr'][nm]:.0f}, d' {r['sep'][nm]:.1f})")
            a1.axhline(r["thr"][nm], color=COLS[nm], lw=0.8, ls="--")
        a1.set_ylabel("lamp colour score")
        a1.legend(loc="upper right", fontsize=8, ncol=3, frameon=False)
    axs[-1].set_xlabel("time (s)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLS.values()]
    fig.legend(handles, list(COLS), loc="lower center", ncol=len(COLS), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--crops", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--heads", default="A:pedestrian,D:vehicle")
    ap.add_argument("--stems", default="C3905,C3896,C3897,C3902")
    ap.add_argument("--eda-dir", default="reports/eda", help="for the v2 published share_unclear")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    heads = dict(x.split(":") for x in a.heads.split(","))
    summary = {}
    for stem in a.stems.split(","):
        res = {}
        for h, kind in heads.items():
            p = os.path.join(a.crops, f"{stem}_{h}.npz")
            r = analyse(p, kind) if os.path.exists(p) else None
            if r is None:
                res[h] = None
                continue
            r["kind"] = kind
            res[h] = r
            lab = r["lab"]
            n = len(lab)
            classified = np.isin(lab, ["red", "amber", "green", "red_amber", "green_flash"])
            visible = lab != "occluded"
            ph = phases(r["t"], lab)
            pd.DataFrame({"frame": r["frames"], "t": np.round(r["t"], 3), "state": lab,
                          **{f"{k}_score": np.round(v, 1) for k, v in r["feat"].items()},
                          "before_v2_rule": r["before"]}).to_csv(os.path.join(a.out, f"{stem}_{h}_samples.csv"), index=False)
            pd.DataFrame(ph).to_csv(os.path.join(a.out, f"{stem}_{h}_phases.csv"), index=False)
            vals, cnt = np.unique(lab, return_counts=True)
            summary.setdefault(stem, {})[h] = {
                "kind": kind, "samples": n, "crop_box_4k": r["box"],
                "lamp_centres_crop_px": {k: [round(v[0], 1), round(v[1], 1)] for k, v in r["centres"].items()},
                "amber_from_midpoint": bool(kind == "vehicle" and r["centres"]["amber"][2] < 0),
                "disc_radius_px": round(float(r["radius"]), 1),
                "thresholds": {k: round(v, 1) for k, v in r["thr"].items()},
                "separation_dprime": {k: round(v, 1) for k, v in r["sep"].items()},
                "drift_shift_px": {"max_abs": round(float(np.abs(r["shifts"]).max()), 1),
                                   "p99_abs": round(float(np.percentile(np.abs(r["shifts"]), 99)), 1)},
                "after_classified_share": round(float(classified.mean()), 4),
                "after_classified_share_of_unoccluded": round(float(classified[visible].mean()), 4),
                "after_state_share": {str(v): round(int(c) / n, 4) for v, c in zip(vals, cnt)},
                "before_v2_rule_classified_share": round(float((r["before"] != "none").mean()), 4),
                "before_v2_rule_smoothed_classified_share": round(float((r["before_smoothed"] != "none").mean()), 4),
                "phases": ph}
            print(stem, h, kind, "classified", summary[stem][h]["after_classified_share"],
                  "before", summary[stem][h]["before_v2_rule_classified_share"], "d'", summary[stem][h]["separation_dprime"],
                  flush=True)
        fig_timeline(stem, res, os.path.join(a.out, f"{stem}_signal_timeline.png"))
    with open(os.path.join(a.out, "lamp_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
