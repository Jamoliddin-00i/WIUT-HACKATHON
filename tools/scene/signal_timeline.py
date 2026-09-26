"""Signal timeline v4: per-sample states with explicit observation status, and
gap-free, frame-indexed intervals that never hide imputation.

Input: the v3 per-sample files from tools/scene/lamp_states.py
(<signals>/<STEM>_<HEAD>_samples.csv: frame, t, state, <lamp>_score, ...) and the
published per-lamp thresholds in <signals>/lamp_summary.json. Nothing is
re-read from video; v4 only re-interprets the v3 readings honestly.

Why v4 (the v3 problems it fixes):
  * v3 phases.csv forward-filled occluded samples with the previous state, so
    29 occluded + 2 dark samples in C3896 (frames 5676 to 5766) became a 3 s
    "dark" phase that nobody observed.
  * v3 phases.csv used duration = end - start with end = time of the last
    sample, so every phase missed one sampling interval and consecutive phases
    left gaps (62 gaps, 6.21 s in total over the four D files).

Per-sample file <STEM>_<HEAD>_samples_v4.csv
  frame        video frame index (0-based, the video's own frames)
  t_sec        frame / (30000/1001)
  state        red | red_amber | amber | green | green_blink | dark | unknown
               (pedestrian heads use red | green | green_blink | dark | unknown)
  observation  observed | occluded | low_confidence
  confidence   0..1 = min(1, margin_sd / 2); 0 for occluded samples
  margin_sd    min over lamps of |lamp score - lamp threshold| / pooled within-
               cluster SD of that lamp (how far the weakest lamp decision is
               from its threshold)
  reason       why a sample is not "observed": occluded | unclear_lamp_combination
               | lamp_near_threshold | short_unsupported_run | dark_inside_green
  v3_label     the v3 per-sample label, unchanged
  <lamp>_on    per-lamp decision (1 = above the published v3 threshold)
  <lamp>_z     signed distance of the lamp score from its threshold in pooled SD
               (lamp decisions under occlusion are not trustworthy)
Rules, in order:
  1. v3 "occluded" -> state unknown, observation occluded.
     v3 "unclear" (an impossible lamp combination) -> unknown, low_confidence.
  2. margin_sd < 1 (a lamp within one pooled SD of its threshold) -> keep the
     state guess, observation low_confidence.
  3. A run of one state family (green and green_flash count as one family)
     shorter than MIN_RUN_S that is not next to a run of at least MIN_RUN_S of
     the same family (looking across at most MAX_BRIDGE_FRAMES of non-observed
     samples) and does not directly (no sample in between) follow or precede a
     long run through a legal state change (e.g. the last 0.8 s of a clip that
     ends in red_amber right after red) -> low_confidence. This removes flicker
     and misreads next to an occlusion (e.g. two "amber" runs of 0.3 and 0.6 s inside the C3896 green
     at 125 to 127 s while a green bus covers the head).
  4. Blinking green: inside a green-family run, v3 "green_flash" samples in the
     last BLINK_TAIL_S of the run mark the blink; from the first of them to the
     end of the run every sample is green_blink (lit and dark halves alike).
     A v3 "green_flash" sample outside such a tail -> dark, low_confidence.

Interval file <STEM>_<HEAD>_intervals_v4.csv (half-open [start_frame, end_frame))
  start_frame, end_frame, start_t, end_t, duration_s, state, n_samples,
  n_observed, n_imputed, n_unobserved, imputed, start_uncertainty_frames
Sample k stands for frames [frame_k, frame_{k+1}); the last sample stands for
[frame_last, n_frames). The intervals therefore tile [0, n_frames) exactly, no
gaps, no overlaps. Construction:
  * consecutive observed samples with the same state form an interval;
  * a run of non-observed samples (occluded or low_confidence) of at most
    MAX_BRIDGE_FRAMES frames (1 s) whose observed neighbours on BOTH sides have
    the same state is bridged: it takes that state, n_imputed counts it and
    imputed = true;
  * any other non-observed run (longer than 1 s, at a state change, or at the
    start / end of the video) is its own interval with state "unknown".
    Nothing is forward-filled through it.
Boundary uncertainty (column start_uncertainty_frames): the change INTO an
interval's state happened in [start_frame - start_uncertainty_frames,
start_frame]. Between two known states this is one sample (3 frames, 0.1 s):
the change happened between the last sample of one interval and the first
sample of the next. After an "unknown" interval it is that interval's length
plus one sample (the state may have begun anywhere inside the unknown
stretch). 0 for the interval starting at frame 0. The end of an interval is the
start of the next one, with the next one's uncertainty. The green_blink start
is the first dark sample seen in the tail of green; if that dark half-period
was occluded the blink began up to ~0.5 s earlier.

"coverage" in timeline_v4.json is the share of the video with a known state.
It is NOT accuracy: no human-verified label exists yet (reports/eda/spotcheck).

  python tools/scene/signal_timeline.py --signals reports/eda/signals \
      --out reports/eda/signals/v4 --eda-dir reports/eda
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

FPS = 30000 / 1001
VERSION = "v4"
MAX_BRIDGE_FRAMES = 30        # 1 s at 29.97 fps: longer occlusions stay unknown
MIN_RUN_S = 1.0               # shorter runs need support from a long run of the same family
BLINK_TAIL_S = 5.0            # green blink sits in the last ~3 s of green
LOW_MARGIN_SD = 1.0
VEHICLE_STATES = ["red", "red_amber", "amber", "green", "green_blink", "dark", "unknown"]
PED_STATES = ["red", "green", "green_blink", "dark", "unknown"]
FAMILY = {"green": "G", "green_flash": "G"}
# legal state changes between families (G = green incl. its blinking end)
LEGAL_NEXT = {"vehicle": {("G", "amber"), ("amber", "red"), ("red", "red_amber"), ("red_amber", "G"), ("red", "G")},
              "pedestrian": {("G", "red"), ("red", "G")}}


def frame_counts(eda_dir, stems):
    """Frames per video from eda.json (proxy frame count = original frame count,
    checked against the Sony rtmd per-frame records in camera_settings.json)."""
    out = {}
    cam = {}
    p = os.path.join(eda_dir, "camera", "camera_settings.json")
    if os.path.exists(p):
        cam = json.load(open(p)).get("videos", {})
    for s in stems:
        n = json.load(open(os.path.join(eda_dir, s, "eda.json")))["extra"]["video"]["frames"]
        if s in cam and cam[s].get("frames") not in (None, n):
            raise SystemExit(f"{s}: frame count mismatch eda.json {n} vs camera metadata {cam[s]['frames']}")
        out[s] = int(n)
    return out


def pooled_sd(v, thr):
    a, b = v[v <= thr], v[v > thr]
    return float(np.sqrt(((a.var() if len(a) > 1 else 0) + (b.var() if len(b) > 1 else 0)) / 2) + 1e-6)


def runs_of(keys):
    """[(key, i, j)] for maximal runs of equal keys, j inclusive."""
    out, i = [], 0
    while i < len(keys):
        j = i
        while j + 1 < len(keys) and keys[j + 1] == keys[i]:
            j += 1
        out.append((keys[i], i, j))
        i = j + 1
    return out


def per_sample(df, thresholds, step, kind="vehicle"):
    """v3 samples -> v4 per-sample table (state, observation, confidence, reason)."""
    lamps = [c[:-6] for c in df.columns if c.endswith("_score")]
    v3 = df.state.astype(str).to_numpy()
    vis = v3 != "occluded"
    margins, lamp_cols = [], {}
    for k in lamps:
        s = df[f"{k}_score"].to_numpy(float)
        sd = pooled_sd(s[vis], thresholds[k])
        z = (s - thresholds[k]) / sd
        margins.append(np.abs(z))
        lamp_cols[f"{k}_on"] = (z > 0).astype(int)
        lamp_cols[f"{k}_z"] = np.round(z, 2)
    margin = np.min(margins, axis=0)
    n = len(df)
    obs = np.full(n, "observed", dtype=object)
    reason = np.full(n, "", dtype=object)
    obs[v3 == "occluded"], reason[v3 == "occluded"] = "occluded", "occluded"
    m = v3 == "unclear"
    obs[m], reason[m] = "low_confidence", "unclear_lamp_combination"
    m = (obs == "observed") & (margin < LOW_MARGIN_SD)
    obs[m], reason[m] = "low_confidence", "lamp_near_threshold"

    # rule 3: short runs need a long run of the same family next to them
    fam = np.array([FAMILY.get(x, x) for x in v3], dtype=object)
    min_run = int(round(MIN_RUN_S * FPS / step))
    changed = True
    while changed:  # a demoted run can change its neighbours' support, iterate
        changed = False
        key = np.where(obs == "observed", fam, None)
        rr = [r for r in runs_of(list(key)) if r[0] is not None]
        long_ = [(r[2] - r[1] + 1) >= min_run for r in rr]
        for k, (f, i, j) in enumerate(rr):
            if long_[k]:
                continue
            ok = False
            for nb in (k - 1, k + 1):
                if not (0 <= nb < len(rr) and long_[nb]):
                    continue
                gap = i - rr[nb][2] - 1 if nb < k else rr[nb][1] - j - 1
                if rr[nb][0] == f:
                    ok |= gap * step <= MAX_BRIDGE_FRAMES
                else:  # directly after / before a long run through a legal change
                    pair = (rr[nb][0], f) if nb < k else (f, rr[nb][0])
                    ok |= gap == 0 and pair in LEGAL_NEXT[kind]
            if not ok:
                obs[i:j + 1], reason[i:j + 1] = "low_confidence", "short_unsupported_run"
                changed = True
    state = np.where(v3 == "green_flash", "green", v3).astype(object)
    state[np.isin(v3, ["occluded", "unclear"])] = "unknown"
    # rule 4: blinking green in the tail of each observed green-family stretch
    key = np.where(obs == "observed", fam, None)
    tail = int(round(BLINK_TAIL_S * FPS / step))
    fams = [r for r in runs_of(list(key)) if r[0] == "G"]
    # join green-family runs separated by bridgeable gaps into one stretch
    stretches = []
    for f, i, j in fams:
        if stretches and (i - stretches[-1][1] - 1) * step <= MAX_BRIDGE_FRAMES:
            stretches[-1][1] = j
        else:
            stretches.append([i, j])
    for i, j in stretches:
        fl = [k for k in range(max(i, j - tail + 1), j + 1) if v3[k] == "green_flash" and obs[k] == "observed"]
        if fl:
            seg = np.arange(fl[0], j + 1)
            seg = seg[state[seg] == "green"]
            state[seg] = "green_blink"
    # a dark ("green_flash") reading outside a blink tail is unexplained: dark, low confidence
    stray = (v3 == "green_flash") & (state == "green")
    state[stray] = "dark"
    obs[stray & (obs == "observed")] = "low_confidence"
    reason[stray & (reason == "")] = "dark_inside_green"
    conf = np.where(obs == "occluded", 0.0, np.minimum(1.0, margin / 2.0))
    return pd.DataFrame({"frame": df.frame.astype(int), "t_sec": np.round(df.frame / FPS, 3),
                         "state": state.astype(str), "observation": obs.astype(str),
                         "confidence": np.round(conf, 3), "margin_sd": np.round(margin, 2),
                         "reason": reason.astype(str), "v3_label": v3, **lamp_cols})


def intervals(samples, n_frames):
    """Gap-free half-open intervals over [0, n_frames); see module docstring."""
    fr = samples.frame.to_numpy()
    if fr[0] != 0 or np.any(np.diff(fr) <= 0) or fr[-1] >= n_frames:
        raise ValueError("sample frames must start at 0, increase, and stay below n_frames")
    ends = np.append(fr[1:], n_frames)
    st = samples.state.to_numpy().astype(object)
    ob = samples.observation.to_numpy()
    known = (ob == "observed") & (st != "unknown")
    lab = np.where(known, st, None).astype(object)
    imputed = np.zeros(len(st), bool)
    # bridge short non-observed runs inside one steady state
    for k, i, j in runs_of([x is None for x in lab]):
        if not k:
            continue
        span = ends[j] - fr[i]
        if 0 < i and j < len(lab) - 1 and span <= MAX_BRIDGE_FRAMES and lab[i - 1] == lab[j + 1]:
            lab[i:j + 1] = lab[i - 1]
            imputed[i:j + 1] = True
    lab = np.where(pd.isna(lab), "unknown", lab)
    rows = []
    for s, i, j in runs_of(list(lab)):
        o = known[i:j + 1]
        rows.append({"start_frame": int(fr[i]), "end_frame": int(ends[j]),
                     "start_t": round(fr[i] / FPS, 3), "end_t": round(ends[j] / FPS, 3),
                     "duration_s": round((ends[j] - fr[i]) / FPS, 3), "state": s,
                     "n_samples": int(j - i + 1), "n_observed": int(o.sum()),
                     "n_imputed": int(imputed[i:j + 1].sum()),
                     "n_unobserved": int((~o).sum() - imputed[i:j + 1].sum()) if s == "unknown" else 0,
                     "imputed": bool(imputed[i:j + 1].any()),
                     "start_uncertainty_frames": 0 if fr[i] == 0 else int(fr[i] - fr[i - 1])})
    # a state that follows an unknown interval may have begun anywhere inside it
    for k in range(1, len(rows)):
        if rows[k - 1]["state"] == "unknown" and rows[k]["state"] != "unknown":
            rows[k]["start_uncertainty_frames"] += rows[k - 1]["end_frame"] - rows[k - 1]["start_frame"]
    return pd.DataFrame(rows)


def v3_phase_gaps(signals, stems, head):
    """Reproduce the v3 phases.csv gap defect: sum of (next.start - prev.end) > 0."""
    n, tot = 0, 0.0
    for s in stems:
        ph = pd.read_csv(os.path.join(signals, f"{s}_{head}_phases.csv"))
        g = ph.start.to_numpy()[1:] - ph.end.to_numpy()[:-1]
        n += int((g > 1e-6).sum())
        tot += float(g[g > 1e-6].sum())
    return {"gaps": n, "total_s": round(tot, 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--signals", default="reports/eda/signals")
    ap.add_argument("--out", default="reports/eda/signals/v4")
    ap.add_argument("--eda-dir", default="reports/eda")
    ap.add_argument("--heads-file", default="reports/eda/signals/signal_heads.json")
    ap.add_argument("--heads", default="D,A")
    ap.add_argument("--stems", default="C3905,C3896,C3897,C3902")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    stems = a.stems.split(",")
    heads_meta = json.load(open(a.heads_file))["heads"]
    summ = json.load(open(os.path.join(a.signals, "lamp_summary.json")))
    nfr = frame_counts(a.eda_dir, stems)
    meta = {"version": VERSION, "schema": __doc__.split("Per-sample file")[1].split("  python ")[0].strip(),
            "time_base": "t = frame / (30000/1001); frames are the video's own 0-based frame indices",
            "sampling": "every 3rd frame (0.1 s)",
            "params": {"max_bridge_frames": MAX_BRIDGE_FRAMES, "min_run_s": MIN_RUN_S,
                       "blink_tail_s": BLINK_TAIL_S, "low_margin_sd": LOW_MARGIN_SD},
            "status": "PROVISIONAL: coverage only, no human-verified labels yet (reports/eda/spotcheck is unfilled)",
            "v3_phase_gap_defect": {h: v3_phase_gaps(a.signals, stems, h) for h in a.heads.split(",")},
            "videos": {}}
    for s in stems:
        for h in a.heads.split(","):
            kind = "vehicle" if heads_meta[h]["type"] == "vehicle" else "pedestrian"
            df = pd.read_csv(os.path.join(a.signals, f"{s}_{h}_samples.csv"))
            step = int(np.median(np.diff(df.frame)))
            smp = per_sample(df, summ[s][h]["thresholds"], step, kind)
            iv = intervals(smp, nfr[s])
            allowed = VEHICLE_STATES if kind == "vehicle" else PED_STATES
            bad = set(smp.state) - set(allowed)
            if bad:
                raise SystemExit(f"{s} {h}: states outside the {kind} vocabulary: {bad}")
            smp.to_csv(os.path.join(a.out, f"{s}_{h}_samples_v4.csv"), index=False)
            iv.to_csv(os.path.join(a.out, f"{s}_{h}_intervals_v4.csv"), index=False)
            unk = iv[iv.state == "unknown"]
            dur = nfr[s] / FPS
            meta["videos"].setdefault(s, {"n_frames": nfr[s], "duration_s": round(dur, 3), "heads": {}})
            meta["videos"][s]["heads"][h] = {
                "kind": kind, "lamps": heads_meta[h].get("lamps"), "sample_step_frames": step,
                "samples": int(len(smp)),
                "observation_counts": smp.observation.value_counts().to_dict(),
                "low_confidence_reasons": smp[smp.observation == "low_confidence"].reason.value_counts().to_dict(),
                "intervals": int(len(iv)),
                "coverage_known_state": round(1 - float(unk.duration_s.sum()) / dur, 4),
                "unknown_total_s": round(float(unk.duration_s.sum()), 2),
                "unknown_intervals_over_1s": [[int(r.start_frame), int(r.end_frame)] for r in unk.itertuples()
                                              if r.end_frame - r.start_frame > MAX_BRIDGE_FRAMES],
                "imputed_intervals": int(iv.imputed.sum()), "imputed_samples": int(iv.n_imputed.sum()),
                "time_share_by_state": {k: round(float(v) / dur, 4) for k, v in
                                        iv.groupby("state").duration_s.sum().items()}}
            print(s, h, "intervals", len(iv), "coverage", meta["videos"][s]["heads"][h]["coverage_known_state"],
                  "unknown", meta["videos"][s]["heads"][h]["unknown_total_s"], "s", flush=True)
    with open(os.path.join(a.out, "timeline_v4.json"), "w") as f:
        json.dump(meta, f, indent=1)
    fig_timelines(a.out, stems, a.heads.split(","))


COLS = {"red": "#e34948", "red_amber": "#f08a24", "amber": "#eda100", "green": "#008300",
        "green_blink": "#7cc47c", "dark": "#52514e", "unknown": "#d8d6d0"}


def fig_timelines(out, stems, heads):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(stems) * len(heads), 1, figsize=(13, 0.62 * len(stems) * len(heads) + 1.2))
    k = 0
    for s in stems:
        for h in heads:
            iv = pd.read_csv(os.path.join(out, f"{s}_{h}_intervals_v4.csv"))
            ax = axs[k]
            for r in iv.itertuples():
                ax.axvspan(r.start_t, r.end_t, fc=COLS[r.state], ec="white", lw=0, hatch="///" if r.imputed else None)
            ax.set_xlim(0, 341)
            ax.set_yticks([])
            ax.set_ylabel(f"{s} {h}", rotation=0, ha="right", va="center", fontsize=9)
            if k < len(axs) - 1:
                ax.set_xticks([])
            k += 1
    axs[-1].set_xlabel("time (s); hatched = contains imputed samples (short occlusion bridged inside one state)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLS.values()]
    fig.legend(handles, list(COLS), loc="upper center", ncol=len(COLS), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out, "timelines_v4.png"), dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    main()
