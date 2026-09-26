"""Validate the vehicle head (D) timeline against traffic, over the FULL signal
cycle, for every gate, with a periodicity-preserving null. Version 2 of this
script (writes signal_validation_v2.json; the v3-era signal_validation.json is
kept and marked superseded).

Inputs: v4 intervals (<signals_v4>/<STEM>_<HEAD>_intervals_v4.csv, from
tools/scene/signal_timeline.py), signal_heads.json (the vehicle head is taken
from there: type vehicle, 3 lamps; it must be D), tracks (<STEM>.csv,
tracks_kaggle.csv schema) and registration.json.

What v2 changes, and why (second adversarial review):
  * v1 only looked at [-20, +40) s around green onset and at one window per
    gate, and concluded "far carriageway: no effect". Over the full cycle the
    far carriageway is strongly cycle-locked too, with a different phase
    offset. v2 folds every crossing onto the cycle phase (time since the last
    D green onset, cycle fitted per video) and reports the whole curve.
  * v1 reported the number of onsets available, not the number used (an onset
    is used only if its windows fit inside the video). v2 reports both.
  * v1's null drew random onset times, which destroys the periodic structure of
    traffic and makes any periodic flow look significant. v2 adds a
    periodicity-preserving null: each test video's crossing sequence is shifted
    circularly by a random offset (mod the video duration), onsets fixed; once
    with an independent shift per video and once with one shared shift per
    lighting group (more conservative). All three p-values are reported.

Checks
  1. Grammar on the v4 intervals: transitions (unknown intervals shown as
     such), complete-phase durations (both neighbours known), cycle length.
  2. Pedestrian head A vs D: onset lags from the v4 intervals.
  3. Gate crossings (same four gates, reference 4K px) of vehicles moved into
     the reference frame.
  4. Test A, held out (as v1): effect = rate in [o+a, o+b) - rate in [o-15, o),
     window chosen on one lighting group (evening C3905, C3902 / daylight
     C3896, C3897), tested on the other.
  5. Test B, held out, full cycle: on the training group pick the 10 s window
     (start 0 .. 64 s after green onset, 1 s steps) with the highest crossing
     rate and the one with the lowest; on the test group, contrast = rate(high)
     - rate(low). This is the test that shows the far carriageway is locked to
     the cycle.
  6. Phase: per gate and group, crossing rate against cycle phase (1 s bins,
     exposure-corrected), the low-rate stretch of each gate (longest run of the
     5 s smoothed rate below 25% of its peak), and the far-minus-near offsets of
     that stretch's start and end with 95% intervals from a cycle-block
     bootstrap.
  7. Share of crossings per D state (v4, unknown shown separately).

  python tools/scene/signal_validate.py --signals-v4 reports/eda/signals/v4 --tracks-dir work/tracks \
      --registration reports/eda/registration.json --out reports/eda/signals
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools", "eda"))
sys.path.insert(0, os.path.join(ROOT, "tools", "scene"))
from analyze import VEH, load_tracks  # noqa: E402
from scene_reference import jac_map  # noqa: E402

REF_STEM = "C3897"
GROUPS = {"evening": ["C3905", "C3902"], "daylight": ["C3896", "C3897"]}
# Gates in reference 4K px: segment end points and the side the vehicles move to.
GATES = {
    "far_carriageway_at_median_nose": {"seg": [[2560, 720], [2560, 1140]], "dir": [-1, 0], "carriageway": "far",
                                       "desc": "far carriageway (right to left, away from camera) passing the "
                                               "median nose, just downstream of the right-hand zebra"},
    "near_carriageway_below_gantry": {"seg": [[1400, 660], [1120, 1200]], "dir": [1, 0], "carriageway": "near",
                                      "desc": "near carriageway (towards camera) between the gantry and the "
                                              "zebra below it"},
    "far_approach_from_right": {"seg": [[3400, 1000], [3400, 1500]], "dir": [-1, 0], "carriageway": "far",
                                "desc": "vehicles entering from the right edge, heading up-left towards the far "
                                        "carriageway, upstream of the right-hand zebra"},
    "lower_left_exit": {"seg": [[300, 1600], [1000, 1600]], "dir": [0, 1], "carriageway": "turn",
                        "desc": "vehicles turning down-left into the lower-left road"},
}
LEGAL = {("green", "green_blink"), ("green_blink", "amber"), ("green", "amber"), ("amber", "red"),
         ("red", "red_amber"), ("red_amber", "green")}
N_NULL = 2000
WIN_B = 10          # test B window length, s
PRE = 15            # test A baseline window, s


def crossings(df, seg, direction):
    (x1, y1), (x2, y2) = seg
    ex, ey = x2 - x1, y2 - y1
    out = []
    for tid, g in df.groupby("tid"):
        x, y, t = g.rx.to_numpy(), g.ry.to_numpy(), g.t_sec.to_numpy()
        side = ex * (y - y1) - ey * (x - x1)
        u = ((x - x1) * ex + (y - y1) * ey) / (ex * ex + ey * ey)
        for k in range(len(x) - 1):
            if side[k] == 0 or np.sign(side[k]) == np.sign(side[k + 1]):
                continue
            if not (0 <= u[k] <= 1 or 0 <= u[k + 1] <= 1):
                continue
            if (x[k + 1] - x[k]) * direction[0] + (y[k + 1] - y[k]) * direction[1] > 0:
                out.append(t[k + 1])
                break
    return np.sort(np.array(out))


def vehicle_head(heads_file):
    if not os.path.exists(heads_file):
        raise SystemExit(f"missing {heads_file}: the vehicle head identity must come from signal_heads.json")
    heads = json.load(open(heads_file))["heads"]
    veh = [h for h, d in heads.items() if d.get("type") == "vehicle" and d.get("lamps") == 3
           and d.get("readable_from_camera")]
    if veh != ["D"]:
        raise SystemExit(f"expected exactly one readable 3-lamp vehicle head, D; found {veh}")
    return "D"


def green_onsets(iv, smp, max_unc=1.1):
    """Green onsets from the v4 files. A green interval counts as an onset when
    the last known state before it is red or red_amber and, in any unknown
    stretch in between, every sample that is not occluded has its GREEN lamp
    clearly off (green_z <= -1; the per-lamp evidence, even where another lamp
    is ambiguous). The onset time is the start of the green interval; its
    uncertainty is one sample plus the occluded run right before it (the green
    may have come on while the head was covered). Onsets with more than max_unc
    s of uncertainty are dropped. Returns [(t, uncertainty_s)]."""
    out = []
    rows = iv.to_dict("records")
    fr = smp.frame.to_numpy()
    for k in range(1, len(rows)):
        if rows[k]["state"] != "green":
            continue
        j = k - 1
        while j >= 0 and rows[j]["state"] == "unknown":
            j -= 1
        if j < 0 or rows[j]["state"] not in ("red_amber", "red"):
            continue
        a_, b_ = rows[j]["end_frame"], rows[k]["start_frame"]
        gap = smp[(fr >= a_) & (fr < b_)]
        vis = gap[gap.observation != "occluded"]
        if (vis.green_z > -1).any():
            continue
        occ_tail = 0
        for o in gap.observation.to_numpy()[::-1]:
            if o != "occluded":
                break
            occ_tail += 1
        unc = round((occ_tail + 1) * 3 / (30000 / 1001), 3)
        if unc <= max_unc:
            out.append((float(rows[k]["start_t"]), unc))
    return out


def state_onsets(iv, state, prev):
    rows = iv.to_dict("records")
    return [float(rows[k]["start_t"]) for k in range(1, len(rows))
            if rows[k]["state"] == state and rows[k - 1]["state"] in prev]


def fit_cycle(onsets):
    t = np.array([o for o, _ in onsets])
    if len(t) < 2:
        return None
    d = np.diff(t)
    # cycles between consecutive onsets; the smallest gap is one cycle as long as
    # at least two consecutive onsets were observed (checked by the residuals)
    k = np.concatenate([[0], np.cumsum(np.round(d / d.min()))])
    A = np.vstack([np.ones_like(k), k]).T
    (a, T), *_ = np.linalg.lstsq(A, t, rcond=None)
    res = t - (a + T * k)
    if np.abs(res).max() > 0.5:
        raise SystemExit(f"cycle fit residual {np.abs(res).max():.2f} s: onsets {t.tolist()} do not fit a fixed-time plan")
    return {"t0": round(float(a), 3), "cycle_s": round(float(T), 3), "max_abs_residual_s": round(float(np.abs(res).max()), 3),
            "onsets": len(t), "cycles_spanned": int(k[-1])}


def phase_of(t, fit):
    return np.mod(np.asarray(t) - fit["t0"], fit["cycle_s"])


def window_rate(ev, onsets, a, b, dur, need_pre=False):
    vals = []
    for o in onsets:
        if o + a < 0 or o + b > dur or (need_pre and o - PRE < 0):
            continue
        vals.append(((ev >= o + a) & (ev < o + b)).sum() / (b - a))
    return (float(np.mean(vals)) if vals else np.nan), len(vals)


def effect_a(ev, onsets, a, b, dur):
    vals = []
    for o in onsets:
        if o - PRE < 0 or o + b > dur:
            continue
        vals.append(((ev >= o + a) & (ev < o + b)).sum() / (b - a) - ((ev >= o - PRE) & (ev < o)).sum() / PRE)
    return (float(np.mean(vals)) if vals else np.nan), len(vals)


def pooled(fn, stems, data, *args):
    num = den = 0.0
    for s in stems:
        v, n = fn(data[s]["ev"], data[s]["on"], *args, data[s]["dur"])
        if n:
            num += v * n
            den += n
    return (num / den if den else np.nan), int(den)


def contrast_b(ev, onsets, hi, lo, dur):
    rh, nh = window_rate(ev, onsets, hi, hi + WIN_B, dur)
    rl, nl = window_rate(ev, onsets, lo, lo + WIN_B, dur)
    if not nh or not nl:
        return np.nan, 0
    return rh - rl, min(nh, nl)


def nulls(stat, stems, data, rng, args):
    """Two nulls for a pooled statistic on the test group.
    random_onsets: onset times drawn uniformly (v1 null; breaks traffic periodicity).
    circular_shift: each video's crossing sequence shifted by U(0, duration) mod duration,
    real onsets kept (keeps the periodic structure, breaks the phase relation with D).
    circular_shift_joint: as circular_shift, but one shared shift fraction for all videos of
    the group (fewer effective draws, wider null; the most conservative of the three)."""
    out = {"random_onsets": [], "circular_shift": [], "circular_shift_joint": []}
    for _ in range(N_NULL):
        fake = {}
        for s in stems:
            d = data[s]
            k = max(1, len(d["on"]))
            fake[s] = {"ev": d["ev"], "dur": d["dur"], "on": list(rng.uniform(0, d["dur"], k))}
        out["random_onsets"].append(pooled(stat, stems, fake, *args)[0])
        sh = {}
        for s in stems:
            d = data[s]
            sh[s] = {"ev": np.sort(np.mod(d["ev"] + rng.uniform(0, d["dur"]), d["dur"])), "on": d["on"], "dur": d["dur"]}
        out["circular_shift"].append(pooled(stat, stems, sh, *args)[0])
        u = rng.uniform(0, 1)
        sj = {s: {"ev": np.sort(np.mod(data[s]["ev"] + u * data[s]["dur"], data[s]["dur"])), "on": data[s]["on"],
                  "dur": data[s]["dur"]} for s in stems}
        out["circular_shift_joint"].append(pooled(stat, stems, sj, *args)[0])
    return {k: np.array([x for x in v if np.isfinite(x)]) for k, v in out.items()}


def circ_phase(ph, T):
    ang = 2 * np.pi * np.asarray(ph) / T
    c, s = np.cos(ang).mean(), np.sin(ang).mean()
    return float(np.mod(np.arctan2(s, c), 2 * np.pi) * T / (2 * np.pi)), float(np.hypot(c, s))


def smooth(rate, k=5):
    h = k // 2
    return np.convolve(np.concatenate([rate[-h:], rate, rate[:h]]), np.ones(k) / k, mode="valid")


def low_stretch(sm, T, frac=0.25):
    """Longest circular run of bins below frac * max -> (start_s, end_s) of cycle phase."""
    nb = len(sm)
    low = sm < frac * sm.max()
    if low.all() or not low.any():
        return (np.nan, np.nan)
    k0 = int(np.argmin(low))  # a high bin: start scanning there so runs do not wrap
    best, cur, bs = (0, 0), 0, 0
    for n in range(1, nb + 1):
        k = (k0 + n) % nb
        if low[k]:
            if cur == 0:
                bs = k
            cur += 1
            if cur > best[1]:
                best = (bs, cur)
        else:
            cur = 0
    return (best[0] * T / nb, np.mod((best[0] + best[1]) * T / nb, T))


def pct(boot, est, T):
    dev = np.mod(boot[np.isfinite(boot)] - est + T / 2, T) - T / 2
    return [round(float(np.mod(est + np.percentile(dev, 2.5), T)), 1), round(float(np.mod(est + np.percentile(dev, 97.5), T)), 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--signals-v4", default="reports/eda/signals/v4")
    ap.add_argument("--heads-file", default="reports/eda/signals/signal_heads.json")
    ap.add_argument("--tracks-dir", required=True)
    ap.add_argument("--registration", default="reports/eda/registration.json")
    ap.add_argument("--out", default="reports/eda/signals")
    ap.add_argument("--ped-head", default="A")
    ap.add_argument("--stems", default="C3905,C3896,C3897,C3902")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    vh = vehicle_head(a.heads_file)
    reg = json.load(open(a.registration))["videos"]
    res = {"version": "v2", "vehicle_head": vh, "supersedes": "signal_validation.json (v1 of this script)",
           "gates_ref_px": GATES, "groups": GROUPS, "n_null": N_NULL, "seed": a.seed, "videos": {}}
    data = {}
    for stem in a.stems.split(","):
        iv = pd.read_csv(os.path.join(a.signals_v4, f"{stem}_{vh}_intervals_v4.csv"))
        ivp = pd.read_csv(os.path.join(a.signals_v4, f"{stem}_{a.ped_head}_intervals_v4.csv"))
        smp = pd.read_csv(os.path.join(a.signals_v4, f"{stem}_{vh}_samples_v4.csv"))
        dur = float(iv.end_t.iloc[-1])
        r = {}
        seq = iv.state.tolist()
        tr = pd.Series([f"{x}->{y}" for x, y in zip(seq[:-1], seq[1:])]).value_counts().to_dict()
        known_pairs = [(x, y) for x, y in zip(seq[:-1], seq[1:]) if "unknown" not in (x, y)]
        rows = iv.to_dict("records")
        durs = {}
        for k in range(1, len(rows) - 1):
            if rows[k - 1]["state"] != "unknown" and rows[k + 1]["state"] != "unknown" and rows[k]["state"] != "unknown":
                durs.setdefault(rows[k]["state"], []).append(round(rows[k]["duration_s"], 2))
        on = green_onsets(iv, smp)
        fit = fit_cycle(on)
        r["grammar"] = {"transitions": tr,
                        "illegal_between_known_states": sorted({f"{x}->{y}" for x, y in known_pairs if (x, y) not in LEGAL}),
                        "durations_s_complete_phases_with_known_neighbours": durs,
                        "green_onsets": [{"t": t, "uncertainty_s": u} for t, u in on], "cycle_fit": fit}
        pg = state_onsets(ivp, "green", ("red",))
        pr = state_onsets(ivp, "red", ("green", "green_blink"))
        d_amb = state_onsets(iv, "amber", ("green", "green_blink"))
        d_blk = state_onsets(iv, "green_blink", ("green",))
        def near(xs, ys, tol=5.0):
            """lag to the nearest reference onset; onsets with none within tol s are counted, not listed"""
            lags = [min((x - y for y in ys), key=abs) for x in xs] if ys else []
            return {"lags_s": [round(v, 2) for v in lags if abs(v) <= tol],
                    "unmatched": int(sum(abs(v) > tol for v in lags)) + (len(xs) if not ys else 0)}
        r["ped_vs_vehicle"] = {"A_green_onset_minus_D_green_onset_s": near(pg, [t for t, _ in on]),
                               "A_red_onset_minus_D_amber_onset_s": near(pr, d_amb),
                               "A_red_onset_minus_D_green_blink_onset_s": near(pr, d_blk)}
        Hm = np.eye(3) if stem == REF_STEM else np.array(reg[stem]["homography"]["H_video_to_ref"])
        df, _ = load_tracks(os.path.join(a.tracks_dir, f"{stem}.csv"), 3, 29.97, 0.5)
        v = df[df.tcls.isin(VEH)].copy()
        v["rx"], v["ry"], _, _ = jac_map(Hm, v.sx.to_numpy(), v.sy.to_numpy(), v.vx.to_numpy(), v.vy.to_numpy())
        v["tid"] = v.track_id
        ev = {g: crossings(v.sort_values("frame"), gd["seg"], gd["dir"]) for g, gd in GATES.items()}
        r["gate_crossings"] = {g: int(len(x)) for g, x in ev.items()}
        st_share = {}
        for g, x in ev.items():
            idx = np.clip(np.searchsorted(iv.start_t.to_numpy(), x, side="right") - 1, 0, len(iv) - 1)
            evs = pd.Series(iv.state.to_numpy()[idx]).value_counts(normalize=True)
            tm = iv.groupby("state").duration_s.sum() / dur
            st_share[g] = {k: {"crossings": round(float(evs.get(k, 0)), 3), "time": round(float(tm.get(k, 0)), 3)}
                           for k in ["green", "green_blink", "amber", "red", "red_amber", "unknown"]}
        r["crossings_by_D_state"] = st_share
        res["videos"][stem] = r
        data[stem] = {"ev": ev, "on": [t for t, _ in on], "dur": dur, "fit": fit}
        print(stem, "onsets", len(on), "cycle", fit and round(fit["cycle_s"], 2), r["gate_crossings"], flush=True)

    # ---- held-out tests A and B, with both nulls
    windows_a = [(x, x + w) for x in range(0, 11) for w in (5, 10, 15)]
    starts_b = list(range(0, 65))
    res["tests"] = {}
    for g in GATES:
        out = {}
        for train, test in (("evening", "daylight"), ("daylight", "evening")):
            dtr = {s: {"ev": data[s]["ev"][g], "on": data[s]["on"], "dur": data[s]["dur"]} for s in GROUPS[train]}
            dte = {s: {"ev": data[s]["ev"][g], "on": data[s]["on"], "dur": data[s]["dur"]} for s in GROUPS[test]}
            # test A
            best = max(windows_a, key=lambda w: np.nan_to_num(pooled(effect_a, GROUPS[train], dtr, *w)[0], nan=-1e9))
            obs, n_used = pooled(effect_a, GROUPS[test], dte, *best)
            nl = nulls(effect_a, GROUPS[test], dte, rng, best)
            ta = {"window_s_after_green_onset": list(best),
                  "train_effect_per_s": round(pooled(effect_a, GROUPS[train], dtr, *best)[0], 3),
                  "test_effect_per_s": round(obs, 3),
                  "test_onsets_available": int(sum(len(dte[s]["on"]) for s in dte)), "test_onsets_used": n_used}
            for k, arr in nl.items():
                ta[f"p_{k}_null"] = round(float((arr >= obs).mean()), 4)
                ta[f"{k}_null_p95"] = round(float(np.percentile(arr, 95)), 3)
            # test B
            rates = {x: pooled(window_rate, GROUPS[train], dtr, x, x + WIN_B)[0] for x in starts_b}
            hi = max(starts_b, key=lambda x: np.nan_to_num(rates[x], nan=-1e9))
            lo = min(starts_b, key=lambda x: np.nan_to_num(rates[x], nan=1e9))
            rh, nh = pooled(window_rate, GROUPS[test], dte, hi, hi + WIN_B)
            rl, nlo = pooled(window_rate, GROUPS[test], dte, lo, lo + WIN_B)
            obs_b = rh - rl
            nb = nulls(contrast_b, GROUPS[test], dte, rng, (hi, lo))
            tb = {"high_window_s": [hi, hi + WIN_B], "low_window_s": [lo, lo + WIN_B],
                  "train_rates_per_s": [round(rates[hi], 3), round(rates[lo], 3)],
                  "test_rate_high_per_s": round(rh, 3), "test_rate_low_per_s": round(rl, 3),
                  "test_contrast_per_s": round(obs_b, 3), "test_onsets_used_high_low": [nh, nlo]}
            for k, arr in nb.items():
                tb[f"p_{k}_null"] = round(float((arr >= obs_b).mean()), 4)
                tb[f"{k}_null_p95"] = round(float(np.percentile(arr, 95)), 3)
            out[f"train_{train}_test_{test}"] = {"A_green_onset_window": ta, "B_full_cycle_contrast": tb}
        res["tests"][g] = out
        print(g, json.dumps(out), flush=True)

    # ---- full-cycle phase curves, circular mean phase with cycle-block bootstrap
    res["phase"] = {}
    for grp, stems in GROUPS.items():
        T = float(np.mean([data[s]["fit"]["cycle_s"] for s in stems]))
        nb = int(np.floor(T))
        # cycle blocks: (video, cycle index) with their own exposure and crossing histograms
        blocks = []
        for s in stems:
            f = data[s]["fit"]
            tt = np.arange(0, data[s]["dur"], 0.05)
            cyc_t = np.floor((tt - f["t0"]) / f["cycle_s"])
            for c in np.unique(cyc_t):
                m = cyc_t == c
                blk = {"id": f"{s}_{int(c)}", "expo": np.histogram(phase_of(tt[m], f), bins=nb, range=(0, T))[0] * 0.05,
                       "cnt": {}}
                for g in GATES:
                    e = data[s]["ev"][g]
                    e = e[np.floor((e - f["t0"]) / f["cycle_s"]) == c]
                    blk["cnt"][g] = np.histogram(phase_of(e, f), bins=nb, range=(0, T))[0]
                    blk.setdefault("ph", {})[g] = phase_of(e, f)
                blocks.append(blk)
        expo = np.sum([b_["expo"] for b_ in blocks], axis=0)
        gp = {"cycle_s": round(T, 2), "bin_s": round(T / nb, 3), "exposure_s_per_bin": np.round(expo, 1).tolist(),
              "cycle_blocks": len(blocks), "gates": {}}
        boot_idx = [rng.integers(0, len(blocks), len(blocks)) for _ in range(1000)]
        stretch_boot = {}
        for g in GATES:
            cnt = np.sum([b_["cnt"][g] for b_ in blocks], axis=0)
            ph = np.concatenate([b_["ph"][g] for b_ in blocks])
            rate = cnt / np.maximum(expo, 1e-9)
            sm = smooth(rate)
            m, R = circ_phase(ph, T) if len(ph) else (np.nan, 0.0)
            st = low_stretch(sm, T)
            bs = []
            for idx in boot_idx:
                e_ = np.sum([blocks[i]["expo"] for i in idx], axis=0)
                c_ = np.sum([blocks[i]["cnt"][g] for i in idx], axis=0)
                bs.append(low_stretch(smooth(c_ / np.maximum(e_, 1e-9)), T))
            bs = np.array(bs)
            stretch_boot[g] = bs
            gp["gates"][g] = {"crossings": int(len(ph)), "rate_per_s": np.round(rate, 3).tolist(),
                              "circular_mean_phase_s": round(m, 2), "resultant_length_R": round(R, 3),
                              "peak_phase_s_5s_smoothed": round(float((np.argmax(sm) + 0.5) * T / nb), 1),
                              "peak_and_min_rate_per_s_5s_smoothed": [round(float(sm.max()), 3), round(float(sm.min()), 3)],
                              "low_rate_stretch_s": {"start": round(st[0], 1), "end": round(st[1], 1),
                                                     "length": round(np.mod(st[1] - st[0], T), 1),
                                                     "start_95ci": pct(bs[:, 0], st[0], T),
                                                     "end_95ci": pct(bs[:, 1], st[1], T)}}
        fn = "far_carriageway_at_median_nose"
        nn = "near_carriageway_below_gantry"
        d0 = np.mod(stretch_boot[fn][:, 0] - stretch_boot[nn][:, 0] + T / 2, T) - T / 2
        d1 = np.mod(stretch_boot[fn][:, 1] - stretch_boot[nn][:, 1] + T / 2, T) - T / 2
        f_, n_ = gp["gates"][fn]["low_rate_stretch_s"], gp["gates"][nn]["low_rate_stretch_s"]
        e0 = float(np.mod(f_["start"] - n_["start"] + T / 2, T) - T / 2)
        e1 = float(np.mod(f_["end"] - n_["end"] + T / 2, T) - T / 2)
        gp["far_minus_near"] = {
            "low_stretch_start_offset_s": round(e0, 1),
            "low_stretch_start_offset_95ci_s": [round(float(np.percentile(d0, 2.5)), 1), round(float(np.percentile(d0, 97.5)), 1)],
            "low_stretch_end_offset_s": round(e1, 1),
            "low_stretch_end_offset_95ci_s": [round(float(np.percentile(d1, 2.5)), 1), round(float(np.percentile(d1, 97.5)), 1)],
            "note": "low-rate stretch = longest run of the 5 s smoothed, exposure-corrected rate below 25% of its "
                    "peak; offsets = far minus near, in seconds of cycle phase; 95% intervals from 1000 "
                    "cycle-block bootstrap resamples"}
        res["phase"][grp] = gp
    res["phase_folding"] = ("phase = (t - t0) mod cycle_s, with t0 and cycle_s fitted per video to the D green "
                            "onsets (fixed-time plan; max residuals in videos[*].grammar.cycle_fit)")
    with open(os.path.join(a.out, "signal_validation_v2.json"), "w") as f:
        json.dump(res, f, indent=1)
    figure(res, data, a.signals_v4, vh, os.path.join(a.out, "cycle_phase_v2.png"))


def figure(res, data, signals_v4, vh, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"near_carriageway_below_gantry": "#2a78d6", "far_carriageway_at_median_nose": "#eb6834",
            "far_approach_from_right": "#9085e9", "lower_left_exit": "#1baf7a"}
    scol = {"red": "#e34948", "red_amber": "#f08a24", "amber": "#eda100", "green": "#008300",
            "green_blink": "#7cc47c", "dark": "#52514e", "unknown": "#d8d6d0"}
    fig, axs = plt.subplots(2, 2, figsize=(13, 6.4), sharex="col", gridspec_kw={"height_ratios": [1, 5]})
    for c, (grp, gp) in enumerate(res["phase"].items()):
        T, nb = gp["cycle_s"], len(gp["exposure_s_per_bin"])
        s0 = GROUPS[grp][1]
        iv = pd.read_csv(os.path.join(signals_v4, f"{s0}_{vh}_intervals_v4.csv"))
        fit = data[s0]["fit"]
        x0 = fit["t0"] + fit["cycle_s"] * 1  # the second full cycle of that video
        for r in iv.itertuples():
            a_, b_ = r.start_t - x0, r.end_t - x0
            if b_ > 0 and a_ < T:
                axs[0, c].axvspan(max(a_, 0), min(b_, T), color=scol[r.state], lw=0)
        axs[0, c].set_yticks([])
        axs[0, c].set_title(f"{grp}: head {vh} state over one cycle ({s0}), cycle {T:.1f} s", loc="left", fontsize=11)
        x = (np.arange(nb) + 0.5) * T / nb
        for g, gd in gp["gates"].items():
            y = np.convolve(np.concatenate([gd["rate_per_s"][-2:], gd["rate_per_s"], gd["rate_per_s"][:2]]),
                            np.ones(5) / 5, mode="valid")
            axs[1, c].plot(x, y, color=cols[g], lw=1.8, label=g.replace("_", " "))
            ls_ = gd["low_rate_stretch_s"]
            if g in ("near_carriageway_below_gantry", "far_carriageway_at_median_nose") and np.isfinite(ls_["start"]):
                yb = -0.04 if g.startswith("near") else -0.08
                a_, b_ = ls_["start"], ls_["end"]
                segs = [(a_, b_)] if b_ >= a_ else [(a_, T), (0, b_)]
                for u_, v_ in segs:
                    axs[1, c].plot([u_, v_], [yb, yb], color=cols[g], lw=4, solid_capstyle="butt")
        axs[1, c].set_xlim(0, T)
        axs[1, c].set_xlabel(f"seconds since {vh} green onset (phase-folded, 5 s smoothing); "
                             "bars below 0 = low-rate stretch")
        axs[1, c].grid(alpha=0.3)
    axs[1, 0].set_ylabel("gate crossings per second")
    axs[1, 0].legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    main()
