"""Scene EDA from tracks.csv (made by track.py) plus the proxy video.

Writes PNG figures, eda.json (consumed by the website) and direction_field.json
into --out. These are descriptive; the rule inputs are listed in
tools/scene/CONTRACT.md (direction_field.json is DEPRECATED for rules, use
reports/eda/scene_reference_v2.json).

Signal (EDA v4): the vehicle head is taken from reports/eda/signals/signal_heads.json
(the one readable 3-lamp head, D), its box is mapped into the video with
registration.json, and its states come from the v4 intervals
(reports/eda/signals/v4, tools/scene/signal_timeline.py). Any of these missing is
an error; --no-signal runs without a signal head on purpose. The v1/v2 practice
of picking "the vehicle light" by correlation with traffic is gone (it picked the
pedestrian head A in C3896 and C3902).

Example
  python tools/eda/analyze.py --stem C3905 --tracks work/tracks/C3905/tracks_kaggle.csv \
      --video proxies/C3905_1080p.mp4 --out reports/eda/C3905 \
      --detector "yolo11m imgsz 1280 + ByteTrack (Kaggle T4)" \
      --benchmark reports/eda/C3905/benchmark.json --lane-names reports/eda/C3905/lane_names.json

All coordinates are in ORIGINAL video pixels (3840x2160 for the samples).
Ground point of a box = bottom centre. Speeds are px/s in that space.
"""
import argparse
import json
import math
import os
import time

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VEH = ["car", "bus", "truck", "motorcycle"]
CLASSES = ["car", "bus", "truck", "motorcycle", "bicycle", "person"]
# categorical slots, fixed order per class so colour always means the same class
PALETTE = {"car": "#2a78d6", "person": "#eb6834", "motorcycle": "#1baf7a",
           "bus": "#eda100", "truck": "#e87ba4", "bicycle": "#008300"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
FIG_W = 16  # inches at dpi 100 -> 1600 px

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 13, "axes.titlesize": 16, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False,
})


# ----------------------------------------------------------------- helpers
def rolling_median(a, k):
    if len(a) < k:
        return a.copy()
    pad = k // 2
    ap = np.pad(a, pad, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(ap, k)
    return np.median(win, axis=1)


def box_blur(img, sigma):
    k = int(sigma * 6) | 1
    return cv2.GaussianBlur(img.astype(np.float32), (k, k), sigma)


def kmeans(X, k, iters=100, seeds=8):
    best = None
    for s in range(seeds):
        rng = np.random.default_rng(s)
        # k-means++ init
        c = [X[rng.integers(len(X))]]
        for _ in range(1, k):
            d = np.min(((X[:, None, :] - np.array(c)[None]) ** 2).sum(-1), axis=1)
            c.append(X[rng.choice(len(X), p=d / d.sum())])
        c = np.array(c)
        for _ in range(iters):
            lab = np.argmin(((X[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
            nc = np.array([X[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
            if np.allclose(nc, c):
                break
            c = nc
        inertia = ((X - c[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, lab, c)
    return best[1], best[2]


def silhouette(X, lab):
    D = np.sqrt(((X[:, None, :] - X[None]) ** 2).sum(-1))
    s = []
    for i in range(len(X)):
        same = lab == lab[i]
        if same.sum() <= 1:
            s.append(0.0)
            continue
        a = D[i, same].sum() / (same.sum() - 1)
        b = min(D[i, lab == j].mean() for j in set(lab) if j != lab[i])
        s.append((b - a) / max(a, b))
    return float(np.mean(s))


def merge_flows(flows, max_ang=20.0, max_off=250.0):
    """Merge k-means clusters that are pieces of the same lane (same heading,
    same line), which happens when tracks are cut by occlusion."""
    merged = 0
    while True:
        ids = flows.flow.value_counts().index.tolist()
        med = {f: flows.loc[flows.flow == f, ["x0", "y0", "x1", "y1"]].median().to_numpy() for f in ids}
        done = False
        for ai, a in enumerate(ids):
            ax0, ay0, ax1, ay1 = med[a]
            da = np.array([ax1 - ax0, ay1 - ay0])
            da = da / max(np.linalg.norm(da), 1e-6)
            nrm = np.array([-da[1], da[0]])
            for b in ids[ai + 1:]:
                bx0, by0, bx1, by1 = med[b]
                db = np.array([bx1 - bx0, by1 - by0])
                db = db / max(np.linalg.norm(db), 1e-6)
                ang = math.degrees(math.acos(float(np.clip(da @ db, -1, 1))))
                off = max(abs((np.array([bx0, by0]) - [ax0, ay0]) @ nrm),
                          abs((np.array([bx1, by1]) - [ax0, ay0]) @ nrm))
                if ang < max_ang and off < max_off:
                    flows.loc[flows.flow == b, "flow"] = a
                    merged += 1
                    done = True
                    break
            if done:
                break
        if not done:
            return merged


def region_name(x, y, W, H):
    col = ["left", "centre", "right"][min(2, int(3 * x / W))]
    row = ["top", "middle", "bottom"][min(2, int(3 * y / H))]
    if row == "middle" and col == "centre":
        return "centre"
    return f"{row}-{col}" if col != "centre" else f"{row}-centre"


def direction_words(dx, dy):
    """Image-space heading to words. Camera is elevated, so moving down the
    image is roughly towards the camera and moving up is away from it."""
    ang = math.degrees(math.atan2(dy, dx)) % 360
    parts = []
    if abs(dy) > 0.35 * math.hypot(dx, dy):
        parts.append("towards camera" if dy > 0 else "away from camera")
    if abs(dx) > 0.35 * math.hypot(dx, dy):
        parts.append("left to right" if dx > 0 else "right to left")
    return ", ".join(parts), ang


def heading_color(ang_rad):
    return plt.cm.hsv((np.asarray(ang_rad) % (2 * np.pi)) / (2 * np.pi))


def add_heading_wheel(fig):
    fig.subplots_adjust(right=0.83)
    ax = fig.add_axes([0.855, 0.58, 0.13, 0.26], projection="polar")
    th = np.linspace(0, 2 * np.pi, 361)
    r = np.linspace(0.55, 1, 2)
    T, R = np.meshgrid(th, r)
    # image y points down, so a heading of +90 deg (down the image) is drawn at the bottom
    ax.pcolormesh(-T, R, T, cmap="hsv", shading="auto", vmin=0, vmax=2 * np.pi)
    ax.set_yticks([])
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax.set_xticklabels(["right", "up the image\n(away)", "left", "down the image\n(towards camera)"],
                       fontsize=10, color=INK)
    ax.set_title("heading", fontsize=12, pad=28, loc="center")
    ax.grid(False)
    ax.set_facecolor("none")
    ax.spines["polar"].set_visible(False)
    return ax


# ----------------------------------------------------------------- video pass
def video_pass(video, n_bg=25):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, int(round(fps)))
    bg_idx = set(np.linspace(0, n - 2, n_bg).astype(int).tolist())
    luma, frames, raw_samples = [], [], []
    idx = -1
    while True:
        if not cap.grab():
            break
        idx += 1
        want_l = idx % step == 0
        if not (want_l or idx in bg_idx):
            continue
        ok, fr = cap.retrieve()
        if not ok:
            break
        if want_l:
            y = cv2.cvtColor(fr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
            luma.append({"t": round(idx / fps, 2), "mean": float(y.mean()),
                         "top_third": float(y[: h // 3].mean()),
                         "bottom_two_thirds": float(y[h // 3:].mean()),
                         "p95": float(np.percentile(y[::4, ::4], 95))})
        if idx in bg_idx:
            frames.append(fr)
    cap.release()
    stack = np.stack(frames)
    bg = np.empty_like(frames[0])
    for r0 in range(0, h, 120):  # chunked median keeps memory low
        bg[r0:r0 + 120] = np.median(stack[:, r0:r0 + 120], axis=0).astype(np.uint8)
    raw_samples = [frames[len(frames) // 4], frames[len(frames) // 2], frames[3 * len(frames) // 4]]
    return dict(fps=fps, n=n, w=w, h=h, luma=luma, bg=bg, raw=raw_samples)


# The v1/v2 signal code (COCO traffic-light search on the 1080p proxy, HSV lamp
# reading, and picking "the vehicle light" by correlation with traffic) was
# removed in EDA v4: it picked the pedestrian head A as the vehicle light in
# C3896 and C3902. Head identity now comes from reports/eda/signals/signal_heads.json
# and the timeline from the v4 intervals (tools/scene/signal_timeline.py).
SIGNAL_COLS = {"red": "#e34948", "red_amber": "#f08a24", "amber": "#eda100", "green": "#008300",
               "green_blink": "#7cc47c", "dark": "#52514e", "unknown": "#bdbdbd"}


def load_signal(stem, heads_file, registration_file, signals_v4):
    """Vehicle head (and the readable pedestrian head) for this video, from the
    head inventory, mapped into the video with the registration, with its v4
    intervals. Fails loudly: a missing file or an ambiguous head is an error,
    never a silent fallback to guessing."""
    if not os.path.exists(heads_file):
        raise SystemExit(f"signal heads file missing: {heads_file}. Head identity must come from the inventory "
                         "(tools/scene/head_inventory.py); pass --no-signal to run without any signal head.")
    heads = json.load(open(heads_file))["heads"]
    veh = [h for h, d in heads.items() if d.get("type") == "vehicle" and d.get("lamps") == 3
           and d.get("readable_from_camera")]
    ped = [h for h, d in heads.items() if d.get("type") == "pedestrian" and d.get("readable_from_camera")]
    if len(veh) != 1:
        raise SystemExit(f"{heads_file}: expected exactly one readable 3-lamp vehicle head, found {veh}")
    if not os.path.exists(registration_file):
        raise SystemExit(f"registration file missing: {registration_file}")
    reg = json.load(open(registration_file))
    ref_stem = reg["reference"]["stem"]
    if stem == ref_stem:
        Hm = np.eye(3)
    elif stem in reg["videos"]:
        Hm = np.array(reg["videos"][stem]["homography"]["H_ref_to_video"])
    else:
        raise SystemExit(f"{stem} is not in {registration_file}: register it first (tools/scene/register.py)")
    out = {}
    for h in veh + ped[:1]:
        x1, y1, x2, y2 = heads[h]["box_ref"]
        pts = cv2.perspectiveTransform(np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], np.float64), Hm)
        box = [round(float(pts[:, 0, 0].min())), round(float(pts[:, 0, 1].min())),
               round(float(pts[:, 0, 0].max())), round(float(pts[:, 0, 1].max()))]
        f = os.path.join(signals_v4, f"{stem}_{h}_intervals_v4.csv")
        if not os.path.exists(f):
            raise SystemExit(f"v4 signal intervals missing: {f} (tools/scene/signal_timeline.py)")
        iv = pd.read_csv(f)
        out[h] = {"head": h, "type": heads[h]["type"], "lamps": heads[h]["lamps"], "box_ref": heads[h]["box_ref"],
                  "box_video": box, "intervals": iv, "file": os.path.relpath(f, ROOT)}
    return veh[0], (ped[0] if ped else None), out


def state_per_second(iv, secs):
    """State that covers most of each whole second (unknown where unknown dominates)."""
    out = []
    for s_ in secs:
        a_, b_ = s_, s_ + 1
        ov = (np.minimum(iv.end_t, b_) - np.maximum(iv.start_t, a_)).clip(lower=0)
        out.append(iv.state.iloc[int(ov.to_numpy().argmax())] if ov.sum() > 0 else "unknown")
    return np.array(out)


def shade_states(ax, iv, alpha=0.14):
    for r in iv.itertuples():
        ax.axvspan(r.start_t, r.end_t, fc=SIGNAL_COLS.get(r.state, "#999999"), ec="#8a8a8a", alpha=alpha, lw=0,
                   hatch="//" if r.state == "unknown" else None)


def draw_samples(video, raw, secs, scale, fps, out):
    """Annotated JPEGs straight from the tracks CSV, so any tracker output can be eyeballed."""
    cols = {"person": (60, 200, 255), "bicycle": (0, 160, 0), "car": (230, 140, 60),
            "motorcycle": (140, 200, 30), "bus": (0, 170, 240), "truck": (170, 120, 240)}
    cap = cv2.VideoCapture(video)
    frames = np.sort(raw.frame.unique())
    for sec in secs:
        f = int(frames[np.argmin(np.abs(frames - sec * fps))])
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        for r in raw[raw.frame == f].itertuples():
            c = cols.get(r.cls, (255, 255, 255))
            p1 = (int(r.x1 / scale), int(r.y1 / scale))
            p2 = (int(r.x2 / scale), int(r.y2 / scale))
            cv2.rectangle(img, p1, p2, c, 2)
            cv2.putText(img, f"{r.cls} {r.track_id}", (p1[0], max(12, p1[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out, f"sample_t{int(sec):03d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    cap.release()


# ----------------------------------------------------------------- tracks
def load_tracks(path, min_det, fps, rider_iou):
    df = pd.read_csv(path)
    df["gx"] = (df.x1 + df.x2) / 2
    df["gy"] = df.y2
    df["bh"] = df.y2 - df.y1
    # one class per track (majority vote), tracker class can flicker
    maj = df.groupby("track_id").cls.agg(lambda s: s.value_counts().index[0])
    df["tcls"] = df.track_id.map(maj)
    ndet = df.groupby("track_id").size()
    df = df[df.track_id.map(ndet) >= min_det].copy()

    # riders: person boxes that sit on a motorcycle / bicycle box most of the time
    riders = set()
    two = df[df.tcls.isin(["motorcycle", "bicycle"])]
    ppl = df[df.tcls == "person"]
    if len(two) and len(ppl):
        tw = {f: g[["x1", "y1", "x2", "y2"]].to_numpy() for f, g in two.groupby("frame")}
        hits = {}
        for f, g in ppl.groupby("frame"):
            if f not in tw:
                continue
            B = tw[f]
            for tid, x1, y1, x2, y2 in g[["track_id", "x1", "y1", "x2", "y2"]].to_numpy():
                cx = (x1 + x2) / 2
                inside = (cx > B[:, 0]) & (cx < B[:, 2]) & (y2 > B[:, 1]) & (y2 < B[:, 3] + 0.25 * (B[:, 3] - B[:, 1]))
                hits[tid] = hits.get(tid, 0) + int(inside.any())
        npp = ppl.groupby("track_id").size()
        riders = {t for t, hcount in hits.items() if hcount >= rider_iou * npp[t]}
    df.loc[df.track_id.isin(riders), "tcls"] = "rider"

    # smoothed ground point and velocity per track
    out = []
    for tid, g in df.sort_values("frame").groupby("track_id"):
        g = g.copy()
        g["sx"] = rolling_median(g.gx.to_numpy(), 5)
        g["sy"] = rolling_median(g.gy.to_numpy(), 5)
        t = g.t_sec.to_numpy()
        k = 5  # about 0.5 s either side at 10 processed fps
        n = len(g)
        i0 = np.clip(np.arange(n) - k, 0, n - 1)
        i1 = np.clip(np.arange(n) + k, 0, n - 1)
        dt = np.maximum(t[i1] - t[i0], 1e-6)
        g["vx"] = (g.sx.to_numpy()[i1] - g.sx.to_numpy()[i0]) / dt
        g["vy"] = (g.sy.to_numpy()[i1] - g.sy.to_numpy()[i0]) / dt
        if n == 1:
            g["vx"] = g["vy"] = 0.0
        out.append(g)
    df = pd.concat(out)
    df["speed"] = np.hypot(df.vx, df.vy)
    df["heading"] = np.arctan2(df.vy, df.vx)
    return df, riders


def track_table(df, stop_thr):
    rows = []
    for tid, g in df.groupby("track_id"):
        sx, sy = g.sx.to_numpy(), g.sy.to_numpy()
        m = min(3, len(g))
        rows.append({
            "track_id": tid, "cls": g.tcls.iloc[0], "n": len(g),
            "t0": g.t_sec.min(), "t1": g.t_sec.max(),
            "dur": g.t_sec.max() - g.t_sec.min(),
            "x0": sx[:m].mean(), "y0": sy[:m].mean(), "x1": sx[-m:].mean(), "y1": sy[-m:].mean(),
            "med_speed_moving": float(g.speed[g.speed > stop_thr].median()) if (g.speed > stop_thr).any() else 0.0,
            "p85_speed": float(g.speed.quantile(0.85)),
            "frac_stopped": float((g.speed <= stop_thr).mean()),
            "med_h": float(g.bh.median()),
        })
    tt = pd.DataFrame(rows).set_index("track_id")
    tt["disp"] = np.hypot(tt.x1 - tt.x0, tt.y1 - tt.y0)
    return tt


def stop_runs(df, stop_thr, min_s):
    runs = []
    for tid, g in df[df.tcls.isin(VEH)].groupby("track_id"):
        st = (g.speed <= stop_thr).to_numpy()
        t = g.t_sec.to_numpy()
        i = 0
        while i < len(st):
            if not st[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(st) and st[j + 1]:
                j += 1
            if t[j] - t[i] >= min_s:
                runs.append({"track_id": tid, "cls": g.tcls.iloc[0], "t0": t[i], "t1": t[j],
                             "dur": t[j] - t[i], "x": float(np.median(g.sx.to_numpy()[i:j + 1])),
                             "y": float(np.median(g.sy.to_numpy()[i:j + 1]))})
            i = j + 1
    return pd.DataFrame(runs, columns=["track_id", "cls", "t0", "t1", "dur", "x", "y"])


# ----------------------------------------------------------------- figures
def bg_axes(fig_h, bg, W, H, dim=0.45, gray=True, title=None):
    fig, ax = plt.subplots(figsize=(FIG_W, fig_h))
    img = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    if gray:
        img = np.repeat(img.mean(2, keepdims=True), 3, 2)
    ax.imshow(img * dim + (1 - dim) * 0.0, extent=(0, W, H, 0))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.grid(False)
    ax.set_xlabel("x (px, 4K frame)")
    ax.set_ylabel("y (px, 4K frame)")
    if title:
        ax.set_title(title)
    return fig, ax


def save(fig, out, name):
    p = os.path.join(out, name)
    fig.savefig(p, dpi=100, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--video", required=True, help="1080p proxy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None, help="DEPRECATED, ignored (was the COCO traffic light search)")
    ap.add_argument("--signal-heads", default=os.path.join(ROOT, "reports/eda/signals/signal_heads.json"),
                    help="head inventory; the vehicle head is the one readable 3-lamp head (D). Required.")
    ap.add_argument("--registration", default=os.path.join(ROOT, "reports/eda/registration.json"))
    ap.add_argument("--signals-v4", default=os.path.join(ROOT, "reports/eda/signals/v4"),
                    help="folder with <STEM>_<HEAD>_intervals_v4.csv")
    ap.add_argument("--no-signal", action="store_true",
                    help="explicitly run without a signal head (new video not yet read); no signal fields written")
    ap.add_argument("--orig-width", type=int, default=3840)
    ap.add_argument("--orig-height", type=int, default=2160)
    ap.add_argument("--stop-thr", type=float, default=25.0, help="px/s, below = stationary")
    ap.add_argument("--move-thr", type=float, default=60.0, help="px/s, above = moving")
    ap.add_argument("--min-det", type=int, default=3)
    ap.add_argument("--k", default="auto", help="number of lane flows or auto")
    ap.add_argument("--url-prefix", default="/media/eda")
    ap.add_argument("--lane-names", default=None,
                    help='optional JSON {"F1": "descriptive name", ...} after looking at lane_flows.png')
    ap.add_argument("--detector", default="", help="label of the detector that made the tracks, for eda.json")
    ap.add_argument("--run-info", default=None, help="optional track_stats.json of the tracking run")
    ap.add_argument("--benchmark", default=None, help="optional benchmark.json (CPU throughput tests)")
    ap.add_argument("--samples", default="10,40,70,100", help="seconds for annotated frames drawn from the CSV")
    ap.add_argument("--notes-file", default=None,
                    help="optional JSON list of reviewed notes; replaces the automatic ones, "
                         "which stay in extra.auto_notes")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    W, H = args.orig_width, args.orig_height
    T0 = time.time()

    print("video pass ...", flush=True)
    vp = video_pass(args.video)
    scale = W / vp["w"]
    duration = vp["n"] / vp["fps"]
    cv2.imwrite(os.path.join(args.out, "background.jpg"), vp["bg"], [cv2.IMWRITE_JPEG_QUALITY, 88])
    bg = vp["bg"]

    print("tracks ...", flush=True)
    df, riders = load_tracks(args.tracks, args.min_det, vp["fps"], 0.5)
    df["sec"] = df.t_sec.astype(int)
    tt = track_table(df, args.stop_thr)
    raw = pd.read_csv(args.tracks)
    n_raw_tracks = raw.track_id.nunique()

    cls_tracks = tt.cls.value_counts().to_dict()
    veh = df[df.tcls.isin(VEH)]
    ped = df[df.tcls == "person"]
    notes = []
    extra = {"video": {"stem": args.stem, "orig_resolution": [W, H], "fps": round(vp["fps"], 3),
                       "frames": vp["n"], "duration_s": round(duration, 2),
                       "proxy_resolution": [vp["w"], vp["h"]]}}

    # ---- fig 1: counts over time ------------------------------------------
    secs = np.arange(int(math.ceil(duration)))
    per_sec = df.groupby(["sec", "tcls"]).track_id.nunique().unstack(fill_value=0)
    per_sec = per_sec.reindex(secs, fill_value=0)
    counts_per_sec = []
    for s in secs:
        row = {"t": int(s)}
        for c in CLASSES:
            row[c] = int(per_sec.at[s, c]) if c in per_sec.columns else 0
        counts_per_sec.append(row)
    # mean objects per processed frame, for the concurrency statement
    per_frame = df.groupby(["frame", "tcls"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(FIG_W, 6.5))
    order = ["person", "car", "motorcycle", "bus", "truck", "bicycle"]
    ys = [per_sec[c].to_numpy() if c in per_sec else np.zeros(len(secs)) for c in order]
    ax.stackplot(secs, ys, labels=order, colors=[PALETTE[c] for c in order],
                 edgecolor=SURFACE, linewidth=0.6, alpha=0.95)
    ax.set_xlim(0, secs[-1])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("tracked objects visible in that second")
    ax.set_title(f"{args.stem}: objects on screen per second, by class")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, args.out, "counts_over_time.png")

    # ---- density per minute ----------------------------------------------
    density = []
    for m in range(int(math.ceil(duration / 60))):
        sub = df[(df.t_sec >= 60 * m) & (df.t_sec < 60 * (m + 1))]
        v = sub[sub.tcls.isin(VEH)]
        density.append({"minute": m, "vehicles": int(v.track_id.nunique()),
                        "persons": int(sub[sub.tcls == "person"].track_id.nunique()),
                        "avg_speed_px_s": round(float(v.speed.mean()), 1) if len(v) else 0.0})
    extra["density_last_minute_seconds"] = round(duration - 60 * (len(density) - 1), 1)

    # ---- fig 2: heatmaps ---------------------------------------------------
    cell = 20
    gw, gh = W // cell, H // cell

    def occupancy(sub):
        hm, _, _ = np.histogram2d(sub.sy, sub.sx, bins=[gh, gw], range=[[0, H], [0, W]])
        return hm

    def heat_fig(sub, title, name, cmap):
        hm = box_blur(occupancy(sub), 1.5)
        fig, ax = bg_axes(9.6, bg, W, H, dim=0.55, title=title)
        v = np.log1p(hm)
        a = np.clip(v / (np.percentile(v[v > 0], 99) if (v > 0).any() else 1), 0, 1)
        rgba = plt.get_cmap(cmap)(a)
        rgba[..., 3] = np.clip(a * 1.4, 0, 0.9)
        ax.imshow(rgba, extent=(0, W, H, 0), interpolation="bilinear")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label("relative time spent (log scale)")
        return save(fig, args.out, name), hm

    _, veh_hm = heat_fig(veh, f"{args.stem}: where vehicles are (ground point density, all {veh.track_id.nunique()} vehicle tracks)",
                         "heatmap.png", "inferno")
    _, ped_hm = heat_fig(ped, f"{args.stem}: where pedestrians are (ground point density, {ped.track_id.nunique()} person tracks, riders removed)",
                         "heatmap_persons.png", "viridis")

    # ---- fig 3: trajectories ----------------------------------------------
    long_v = tt[(tt.cls.isin(VEH)) & (tt.dur >= 2)].index
    fig, ax = bg_axes(9.6, bg, W, H, dim=0.5,
                      title=f"{args.stem}: {len(long_v)} vehicle trajectories longer than 2 s, coloured by heading")
    segs, cols = [], []
    for tid in long_v:
        g = df[df.track_id == tid]
        p = g[["sx", "sy"]].to_numpy()
        if len(p) < 2:
            continue
        segs.extend(np.stack([p[:-1], p[1:]], 1))
        mv = g.speed.to_numpy()[1:] > args.stop_thr
        c = heading_color(g.heading.to_numpy()[1:])
        c[~mv] = (1, 1, 1, 1)
        cols.extend(c)
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=1.1, alpha=0.75))
    add_heading_wheel(fig)
    fig.text(0.125, 0.015, "white = stationary part of a track. Ground point = bottom centre of the box, "
             "smoothed with a 5-sample median.", color=INK2, fontsize=11)
    save(fig, args.out, "trajectories.png")

    # ---- fig 4: direction field ------------------------------------------
    GX, GY = 48, 27
    cw, ch = W / GX, H / GY
    mv = veh[veh.speed > args.move_thr].copy()
    mv["ci"] = np.clip((mv.sx // cw).astype(int), 0, GX - 1)
    mv["cj"] = np.clip((mv.sy // ch).astype(int), 0, GY - 1)
    mv["ux"] = mv.vx / mv.speed
    mv["uy"] = mv.vy / mv.speed
    field = []
    for (i, j), g in mv.groupby(["ci", "cj"]):
        ntr = g.track_id.nunique()
        if len(g) < 8 or ntr < 3:
            continue
        # average per track first so one slow car does not dominate a cell
        pt = g.groupby("track_id")[["ux", "uy", "speed"]].mean()
        ux, uy = pt.ux.mean(), pt.uy.mean()
        coh = math.hypot(ux, uy)
        field.append({"i": int(i), "j": int(j), "cx": round((i + 0.5) * cw), "cy": round((j + 0.5) * ch),
                      "dx": round(ux / max(coh, 1e-6), 3), "dy": round(uy / max(coh, 1e-6), 3),
                      "coherence": round(coh, 3), "n": int(len(g)), "tracks": int(ntr),
                      "mean_speed": round(float(pt.speed.mean()), 1)})
    with open(os.path.join(args.out, "direction_field.json"), "w") as f:
        json.dump({"grid": [GX, GY], "cell_px": [cw, ch], "frame": [W, H],
                   "min_samples": 8, "min_tracks": 3, "moving_thr_px_s": args.move_thr,
                   "note": "dx,dy = unit mean heading of moving vehicles; coherence 1 = all agree, "
                           "near 0 = opposing or turning traffic share the cell",
                   "cells": field}, f)
    fig, ax = bg_axes(9.6, bg, W, H, dim=0.5,
                      title=f"{args.stem}: lane direction map ({len(field)} cells of {GX}x{GY} with enough moving vehicles)")
    for c in field:
        L = 0.9 * cw * (0.35 + 0.65 * c["coherence"])
        ang = math.atan2(c["dy"], c["dx"])
        col = heading_color(ang)
        col = (*col[:3], 0.35 + 0.65 * c["coherence"])
        ax.add_patch(FancyArrowPatch((c["cx"] - c["dx"] * L / 2, c["cy"] - c["dy"] * L / 2),
                                     (c["cx"] + c["dx"] * L / 2, c["cy"] + c["dy"] * L / 2),
                                     arrowstyle="-|>", mutation_scale=11, lw=1.6, color=col))
    add_heading_wheel(fig)
    fig.text(0.125, 0.015, f"Cell = {cw:.0f} x {ch:.0f} px. Arrow = mean heading of moving vehicles (> {args.move_thr:.0f} px/s), "
             "per track first. Faded, short arrow = mixed directions (turns, opposing flows).", color=INK2, fontsize=11)
    save(fig, args.out, "direction_field.png")

    # ---- lane flows (clustering) -------------------------------------------
    flows = tt[(tt.cls.isin(VEH)) & (tt.dur >= 2) & (tt.disp >= 300)].copy()
    lane_flows, flow_info = [], []
    lane_names = json.load(open(args.lane_names)) if args.lane_names and os.path.exists(args.lane_names) else {}
    if len(flows) >= 6:
        dx, dy = (flows.x1 - flows.x0), (flows.y1 - flows.y0)
        nrm = np.hypot(dx, dy)
        X = np.column_stack([flows.x0 / W, flows.y0 / H, flows.x1 / W, flows.y1 / H,
                             0.5 * dx / nrm, 0.5 * dy / nrm])
        if args.k == "auto":
            best = None
            for k in range(3, min(10, len(flows) // 4) + 1):
                lab, _ = kmeans(X, k)
                s = silhouette(X, lab)
                if best is None or s > best[0] + 0.01:
                    best = (s, k, lab)
            sil, k, lab = best
        else:
            k = int(args.k)
            lab, _ = kmeans(X, k)
            sil = silhouette(X, lab)
        flows["flow"] = lab
        n_merged = merge_flows(flows)
        extra["lane_flow_clustering"] = {"method": "k-means on start xy, end xy, net heading, then merge "
                                                   "clusters that share heading (< 20 deg) and line (< 250 px)",
                                         "k": int(k), "silhouette": round(sil, 3), "merged": n_merged,
                                         "eligible_tracks": int(len(flows)),
                                         "rule": "vehicle tracks >= 2 s and >= 300 px net travel"}
        order_f = flows.flow.value_counts().index.tolist()
        names = {}
        for rank, f_id in enumerate(order_f):
            g = flows[flows.flow == f_id]
            sx, sy, ex, ey = g.x0.median(), g.y0.median(), g.x1.median(), g.y1.median()
            words, ang = direction_words(ex - sx, ey - sy)
            r0, r1 = region_name(sx, sy, W, H), region_name(ex, ey, W, H)
            nm = f"F{rank + 1} {r0} to {r1}" if r0 != r1 else f"F{rank + 1} within {r0}"
            nm = lane_names.get(f"F{rank + 1}", nm)
            names[f_id] = nm
            lane_flows.append({"lane": nm, "direction": words, "count": int(len(g))})
            flow_info.append({"lane": nm, "flow_id": int(f_id), "count": int(len(g)),
                              "start_median": [round(sx), round(sy)], "end_median": [round(ex), round(ey)],
                              "heading_deg_image": round(ang, 1),
                              "classes": g.cls.value_counts().to_dict(),
                              "median_speed_px_s": round(float(g.med_speed_moving.median()), 1),
                              "share_that_stopped_2s": None})
        tt.loc[flows.index, "flow"] = flows.flow
        # flow figure
        fig, ax = bg_axes(9.6, bg, W, H, dim=0.5,
                          title=f"{args.stem}: {len(order_f)} dominant vehicle flows (k-means on entry, exit, heading)")
        cols8 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#9085e9", "#e34948",
                 "#ffffff", "#8a8a8a"]
        for rank, f_id in enumerate(order_f):
            col = cols8[rank % len(cols8)]
            g = flows[flows.flow == f_id]
            for tid in g.index:
                p = df.loc[df.track_id == tid, ["sx", "sy"]].to_numpy()
                ax.plot(p[:, 0], p[:, 1], color=col, lw=0.9, alpha=0.55)
            fi = flow_info[rank]
            ax.add_patch(FancyArrowPatch(fi["start_median"], fi["end_median"], arrowstyle="-|>",
                                         mutation_scale=28, lw=4, color=col, zorder=5))
            ax.plot([], [], color=col, lw=4, label=f"{names[f_id]} ({fi['count']})")
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=11, frameon=False)
        save(fig, args.out, "lane_flows.png")

    # ---- turning movements -------------------------------------------------
    turns = []
    for tid in tt[(tt.cls.isin(VEH)) & (tt.dur >= 2)].index:
        g = df[(df.track_id == tid) & (df.speed > args.move_thr)]
        if len(g) < 10:
            continue
        a0 = math.atan2(g.vy.iloc[:5].mean(), g.vx.iloc[:5].mean())
        a1_ = math.atan2(g.vy.iloc[-5:].mean(), g.vx.iloc[-5:].mean())
        dlt = math.degrees((a1_ - a0 + math.pi) % (2 * math.pi) - math.pi)
        kind = ("straight" if abs(dlt) < 35 else "u_turn_like" if abs(dlt) > 140
                else "turn_clockwise_on_screen" if dlt > 0 else "turn_anticlockwise_on_screen")
        turns.append({"track_id": int(tid), "kind": kind, "delta_deg": round(dlt, 1),
                      "xy_mid": [round(g.sx.iloc[len(g) // 2]), round(g.sy.iloc[len(g) // 2])],
                      "t": [round(g.t_sec.iloc[0], 1), round(g.t_sec.iloc[-1], 1)]})
    tk = pd.Series([t_["kind"] for t_ in turns]).value_counts().to_dict() if turns else {}
    extra["turning_movements"] = {"counts": {k: int(v) for k, v in tk.items()},
                                  "rule": "heading change between first and last 5 moving samples; "
                                          "|d| < 35 straight, > 140 U-turn like; image-plane, so perspective bends it",
                                  "non_straight": [t_ for t_ in turns if t_["kind"] != "straight"][:40]}

    # ---- signal head (v4: identity from the inventory, timeline from v4 intervals)
    sig_iv, veh_head, sig = None, None, {}
    if args.model:
        print("note: --model is ignored since EDA v4 (no COCO traffic light search)", flush=True)
    if args.no_signal:
        extra["signal"] = {"status": "not_used (--no-signal)"}
    else:
        veh_head, ped_head, sig = load_signal(args.stem, args.signal_heads, args.registration, args.signals_v4)
        sig_iv = sig[veh_head]["intervals"]
        end_frame = int(sig_iv.end_frame.iloc[-1])
        if end_frame != vp["n"]:
            raise SystemExit(f"{args.stem}: v4 intervals end at frame {end_frame}, the video has {vp['n']} frames")
        dur_iv = sig_iv.groupby("state").duration_s.sum()
        g_on = [float(sig_iv.start_t.iloc[i]) for i in range(1, len(sig_iv))
                if sig_iv.state.iloc[i] == "green" and sig_iv.state.iloc[i - 1] in ("red", "red_amber")]
        extra["signal"] = {
            "version": "v4", "status": "PROVISIONAL (coverage only; no human-verified lamp labels yet)",
            "vehicle_head": {k: v for k, v in sig[veh_head].items() if k != "intervals"},
            "pedestrian_head": ({k: v for k, v in sig[ped_head].items() if k != "intervals"} if ped_head else None),
            "head_source": os.path.relpath(args.signal_heads, ROOT),
            "box_mapping": "box_ref (reference C3897 4K px) mapped with H_ref_to_video from registration.json",
            "time_share_by_state": {k: round(float(v) / duration, 4) for k, v in dur_iv.items()},
            "coverage_known_state": round(1 - float(dur_iv.get("unknown", 0.0)) / duration, 4),
            "green_onsets_after_known_red_s": [round(x, 2) for x in g_on],
            "meaning": "head D is a strong timing proxy for the near-carriageway flow; the far carriageway is also "
                       "cycle-locked with a different phase offset; which movements D physically controls is "
                       "UNKNOWN (reports/eda/SUMMARY.md, finding 3)"}
        fig, axes = plt.subplots(3, 1, figsize=(FIG_W, 6.2), sharex=True, gridspec_kw={"height_ratios": [1, 1, 3]})
        for ax_, h in zip(axes[:2], [veh_head, ped_head]):
            if h is None:
                ax_.axis("off")
                continue
            for r in sig[h]["intervals"].itertuples():
                ax_.axvspan(r.start_t, r.end_t, fc=SIGNAL_COLS.get(r.state, "#999999"), ec="white", lw=0,
                            hatch="//" if r.imputed else None)
            ax_.set_yticks([])
            ax_.set_ylabel(f"head {h}\n({sig[h]['type']})", rotation=0, ha="right", va="center", fontsize=11)
        axes[0].set_title(f"{args.stem}: signal heads from the inventory, v4 intervals (grey = unknown, "
                          "hatched = short occlusion bridged)")
        mv_ = veh[veh.speed > args.stop_thr].groupby("sec").track_id.nunique().reindex(secs, fill_value=0)
        st_ = veh[veh.speed <= args.stop_thr].groupby("sec").track_id.nunique().reindex(secs, fill_value=0)
        shade_states(axes[2], sig_iv)
        axes[2].plot(secs, st_, color=PALETTE["truck"], lw=2, label="stationary vehicles")
        axes[2].plot(secs, mv_, color=PALETTE["car"], lw=2, label="moving vehicles")
        axes[2].set_xlim(0, duration)
        axes[2].set_xlabel("time (s)")
        axes[2].set_ylabel("vehicles")
        axes[2].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SIGNAL_COLS.values()]
        fig.legend(handles, list(SIGNAL_COLS), loc="lower center", ncol=len(SIGNAL_COLS), fontsize=10, frameon=False)
        fig.subplots_adjust(bottom=0.16)
        fig.savefig(os.path.join(args.out, "signal.png"), dpi=70, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)

    # ---- fig 5: stop map ---------------------------------------------------
    runs = stop_runs(df, args.stop_thr, 2.0)
    runs["flow"] = runs.track_id.map(tt.get("flow", pd.Series(dtype=float)))
    fig = plt.figure(figsize=(FIG_W, 13.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.2], hspace=0.18)
    ax = fig.add_subplot(gs[0])
    img = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    ax.imshow(np.repeat(img.mean(2, keepdims=True), 3, 2) * 0.5, extent=(0, W, H, 0))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.grid(False)
    ax.set_xlabel("x (px, 4K frame)")
    ax.set_ylabel("y (px, 4K frame)")
    ax.set_title(f"{args.stem}: where vehicles stand still for 2 s or more ({len(runs)} stops, "
                 f"{runs.track_id.nunique()} vehicles)")
    FLOW_COLS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#9085e9", "#e34948"]
    if len(runs):
        groups_ = [(fi["lane"], runs[runs.flow == fi["flow_id"]], FLOW_COLS[i % 8]) for i, fi in enumerate(flow_info)]
        groups_.append(("not in a flow (short, parked or turning)", runs[runs.flow.isna()], "#bdbdbd"))
        for name_, r_, col_ in groups_:
            if len(r_):
                ax.scatter(r_.x, r_.y, s=np.clip(r_.dur * 8, 15, 450), color=col_, alpha=0.8,
                           edgecolors="white", linewidths=0.6, label=f"{name_} ({len(r_)})")
        for d_ in (5, 20, 40):
            ax.scatter([], [], s=d_ * 8, color="#bdbdbd", edgecolors="white", label=f"{d_} s stop")
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=11, labelspacing=1.1)
    ax2 = fig.add_subplot(gs[1])
    stopped_ts = veh[veh.speed <= args.stop_thr].groupby("sec").track_id.nunique().reindex(secs, fill_value=0)
    moving_ts = veh[veh.speed > args.stop_thr].groupby("sec").track_id.nunique().reindex(secs, fill_value=0)
    title2 = "stationary vs moving vehicles per second"
    d_sec = None
    if sig_iv is not None:
        shade_states(ax2, sig_iv)
        d_sec = state_per_second(sig_iv, secs)
        known = d_sec != "unknown"
        gsec = np.isin(d_sec, ["green", "green_blink"]).astype(float)
        diff_ = (moving_ts - stopped_ts).to_numpy()
        if known.sum() > 10 and gsec[known].std() > 0:
            extra["signal"]["corr_D_green_vs_moving_minus_stationary"] = round(
                float(np.corrcoef(gsec[known], diff_[known])[0, 1]), 3)
        title2 += f"; background = head {veh_head} state (v4, from the head inventory; hatched = unknown)"
    ax2.plot(secs, stopped_ts, color=PALETTE["truck"], lw=2, label="stationary vehicles")
    ax2.plot(secs, moving_ts, color=PALETTE["car"], lw=2, label="moving vehicles")
    ax2.set_xlim(0, secs[-1])
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("vehicles")
    ax2.set_title(title2, fontsize=14)
    ax2.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))

    # stop line estimate per flow: stops of >= 4 s that begin while head D shows
    # red / red_amber / amber (observed or bridged, never unknown), and the furthest
    # point along the flow that still has >= 3 stops within 400 px behind it
    red_spans = []
    if sig_iv is not None:
        red_spans = [(r_.start_t, r_.end_t) for r_ in sig_iv.itertuples() if r_.state in ("red", "red_amber", "amber")]
    for fi in flow_info:
        r = runs[runs.flow == fi["flow_id"]]
        n_tr = int((flows.flow == fi["flow_id"]).sum())
        fi["share_that_stopped_2s"] = round(r.track_id.nunique() / max(n_tr, 1), 3)
        fi["stops"] = int(len(r))
        fi["median_stop_s"] = round(float(r.dur.median()), 1) if len(r) else None
        q = r[r.dur >= 4]
        if red_spans:
            q = q[[any(a_ <= t0_ < b_ for a_, b_ in red_spans) for t0_ in q.t0]]
        if len(q) >= 3:
            d = np.array(fi["end_median"]) - np.array(fi["start_median"])
            d = d / max(np.linalg.norm(d), 1e-6)
            proj = (q[["x", "y"]].to_numpy() - np.array(fi["start_median"])) @ d
            order_p = np.argsort(proj)
            front = None
            for i_ in order_p[::-1]:
                if ((proj <= proj[i_]) & (proj >= proj[i_] - 400)).sum() >= 3:
                    front = q.iloc[i_]
                    break
            if front is not None:
                fi["furthest_queued_stop_in_D_red"] = [round(front.x), round(front.y)]
                fi["queue_stops_used"] = int(len(q))
        if d_sec is not None:
            known = d_sec != "unknown"
            red = np.isin(d_sec, ["red", "red_amber"]).astype(float)
            members = flows.index[flows.flow == fi["flow_id"]]
            q = veh[veh.track_id.isin(members) & (veh.speed <= args.stop_thr)].groupby("sec").track_id.nunique()
            q = q.reindex(secs, fill_value=0).to_numpy()
            if q[known].std() > 0 and red[known].std() > 0:
                fi["corr_queue_vs_D_red"] = round(float(np.corrcoef(q[known], red[known])[0, 1]), 3)
                fi["mean_queue_D_red_vs_not_red"] = [round(float(q[known & (red > 0.5)].mean()), 1),
                                                     round(float(q[known & (red <= 0.5)].mean()), 1)]

    fig.savefig(os.path.join(args.out, "stop_map.png"), dpi=60, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)

    # parked / long standing vehicles
    # longer than any red phase seen so far (about 40 s), so queued cars do not count
    parked = tt[(tt.cls.isin(VEH)) & (tt.frac_stopped > 0.9) & (tt.dur > 60)]
    extra["long_standing_vehicles"] = [{"track_id": int(i), "cls": r.cls, "duration_s": round(r.dur, 1),
                                        "xy": [round(r.x0), round(r.y0)]} for i, r in parked.iterrows()]

    # ---- fig 6: speeds and durations --------------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIG_W, 6.5))
    order6 = [c for c in ["car", "bus", "truck", "motorcycle", "bicycle", "person"] if cls_tracks.get(c, 0) >= 3]
    data = [tt[(tt.cls == c) & (tt.dur >= 1)].p85_speed.to_numpy() for c in order6]
    bp = a1.boxplot(data, tick_labels=order6, showfliers=False, patch_artist=True, widths=0.55,
                    medianprops=dict(color=INK, lw=2))
    for patch, c in zip(bp["boxes"], order6):
        patch.set_facecolor(PALETTE[c])
        patch.set_alpha(0.8)
    a1.set_ylabel("85th percentile speed of a track (px/s, 4K)")
    a1.set_title("speed per track, by class (tracks >= 1 s)")
    bins = np.logspace(np.log10(0.3), np.log10(max(3, duration)), 30)
    for c in order6:
        d = tt[tt.cls == c].dur.to_numpy()
        a2.hist(np.maximum(d, 0.3), bins=bins, histtype="step", lw=2, color=PALETTE[c], label=f"{c} ({len(d)})")
    a2.set_xscale("log")
    a2.set_xlabel("track duration (s, log scale)")
    a2.set_ylabel("tracks")
    a2.set_title("how long each track lives")
    a2.legend()
    save(fig, args.out, "speed_duration.png")

    # ---- fig 7: lighting ---------------------------------------------------
    L = pd.DataFrame(vp["luma"])
    fig, ax = plt.subplots(figsize=(FIG_W, 5.5))
    ax.plot(L.t, L["mean"], color=INK, lw=2, label="whole frame")
    ax.plot(L.t, L.top_third, color=PALETTE["car"], lw=2, label="top third (far road, sky)")
    ax.plot(L.t, L.bottom_two_thirds, color=PALETTE["person"], lw=2, label="bottom two thirds (near road)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mean luma (0 to 255)")
    ax.set_xlim(0, L.t.max())
    ax.set_title(f"{args.stem}: brightness over time, 1 sample per second")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, args.out, "lighting.png")
    first, last = L["mean"].iloc[:10].mean(), L["mean"].iloc[-10:].mean()
    extra["lighting"] = {"mean_luma_first10s": round(first, 1), "mean_luma_last10s": round(last, 1),
                         "min": round(L["mean"].min(), 1), "max": round(L["mean"].max(), 1),
                         "change_pct": round(100 * (last - first) / first, 1),
                         "p95_luma_median": round(L.p95.median(), 1)}
    m = L["mean"].to_numpy()
    w15 = min(15, len(m) - 1)
    if w15 > 0:
        jumps = (m[w15:] - m[:-w15]) / m[:-w15] * 100
        i = int(np.argmax(np.abs(jumps)))
        extra["lighting"]["largest_15s_change_pct"] = round(float(jumps[i]), 1)
        extra["lighting"]["largest_15s_change_at_s"] = [round(float(L.t.iloc[i]), 1), round(float(L.t.iloc[i + w15]), 1)]

    # ---- fig 8: object size ----------------------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIG_W, 6.5), gridspec_kw={"width_ratios": [1.3, 1]})
    hb = np.logspace(np.log10(30), np.log10(1500), 36)
    for c in order6:
        d = df[df.tcls == c].bh.to_numpy()
        a1.hist(d, bins=hb, histtype="step", lw=2, color=PALETTE[c],
                weights=np.full(len(d), 100.0 / max(len(d), 1)), label=c)
    top = a1.get_ylim()[1]
    for n_, (isz, ls) in enumerate([(640, ":"), (960, "--"), (1280, "-.")]):
        x = 16 * W / isz
        a1.axvline(x, color=INK2, ls=ls, lw=1.2)
        a1.text(x * 1.02, top * (0.97 - 0.1 * n_), f"16 px at imgsz {isz}", fontsize=10, color=INK2, va="top")
    a1.set_xscale("log")
    a1.set_xticks([50, 100, 200, 500, 1000], ["50", "100", "200", "500", "1000"])
    a1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a1.set_xlabel("box height (px, 4K, log scale)")
    a1.set_ylabel("% of that class's detections")
    a1.set_title("detected box height by class")
    a1.legend(loc="upper right")
    p = df[(df.tcls == "person") & (df.y2 < H - 10) & (df.y1 > 10)]  # drop boxes cut by the frame edge
    size_fit = None
    if len(p) > 50:
        a2.scatter(p.gy[::3], p.bh[::3], s=4, alpha=0.25, color=PALETTE["person"])
        yb = np.linspace(p.gy.quantile(0.02), p.gy.quantile(0.98), 12)
        med = [p[(p.gy >= a) & (p.gy < b)].bh.median() for a, b in zip(yb[:-1], yb[1:])]
        mid = (yb[:-1] + yb[1:]) / 2
        ok = ~np.isnan(med)
        coef = np.polyfit(mid[ok], np.array(med)[ok], 1)
        a2.plot(mid[ok], np.array(med)[ok], "o", color=INK, ms=6)
        xx = np.linspace(mid[ok].min(), mid[ok].max(), 10)
        a2.plot(xx, np.polyval(coef, xx), color=INK, lw=2, label=f"h = {coef[0]:.3f} * y + {coef[1]:.0f}")
        a2.legend(loc="upper left")
        size_fit = {"person_height_px_vs_ground_y": {"slope": round(coef[0], 4), "intercept": round(coef[1], 1)},
                    "rough_m_per_px_at_y": {str(int(y)): round(1.7 / max(np.polyval(coef, y), 1), 4)
                                            for y in np.linspace(mid[ok].min(), mid[ok].max(), 5)},
                    "assumption": "person height about 1.7 m; valid only near the fitted y range"}
    a2.set_xlabel("ground y of the person (px, 4K)")
    a2.set_ylabel("person box height (px, 4K)")
    a2.set_title("perspective: person size vs position")
    save(fig, args.out, "object_size.png")
    extra["perspective"] = size_fit

    # ---- pedestrians on the carriageway -----------------------------------
    # road = direction-field cells used by moving vehicles; corridor = road cells
    # crossed by many different people (the zebras); the rest is crossing elsewhere
    road = np.zeros((GY, GX), bool)
    for c in field:
        road[c["j"], c["i"]] = True
    pg = ped.copy()
    pg["ci"] = np.clip((pg.sx // cw).astype(int), 0, GX - 1)
    pg["cj"] = np.clip((pg.sy // ch).astype(int), 0, GY - 1)
    pg["on_road"] = road[pg.cj, pg.ci]
    onr = pg[pg.on_road]
    per_cell = np.zeros((GY, GX))
    for (i, j), g in onr.groupby(["ci", "cj"]):
        per_cell[j, i] = g.track_id.nunique()
    thr = max(4, 0.25 * per_cell.max()) if per_cell.max() > 0 else 1
    corridor = cv2.dilate((per_cell >= thr).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & road
    pg["in_corridor"] = corridor[pg.cj, pg.ci]
    n_on = onr.groupby("track_id").size()
    on_road_tracks = n_on[n_on >= 5]  # at least ~0.5 s in a vehicle lane cell
    outside = pg[pg.on_road & ~pg.in_corridor].groupby("track_id").size()
    outside = outside[outside >= 5]
    spots = []
    if len(outside):
        op = pg[pg.track_id.isin(outside.index) & pg.on_road & ~pg.in_corridor].groupby("track_id")[["sx", "sy"]].median()
        op["bx"] = (op.sx // (W / 6)).astype(int)
        op["by"] = (op.sy // (H / 4)).astype(int)
        for (bx, by), g in op.groupby(["bx", "by"]):
            spots.append({"people": int(len(g)), "median_xy": [round(g.sx.median()), round(g.sy.median())]})
        spots.sort(key=lambda d: -d["people"])
    cj_, ci_ = np.nonzero(corridor)
    corridors = []
    if len(ci_):
        n_lab, lab_img = cv2.connectedComponents(corridor.astype(np.uint8))
        for L_ in range(1, n_lab):
            jj, ii = np.nonzero(lab_img == L_)
            corridors.append({"cells": int(len(ii)),
                              "bbox_px": [round(ii.min() * cw), round(jj.min() * ch),
                                          round((ii.max() + 1) * cw), round((jj.max() + 1) * ch)],
                              "people": int(pg[pg.in_corridor & (lab_img[pg.cj, pg.ci] == L_)].track_id.nunique())})
        corridors.sort(key=lambda d: -d["people"])
    extra["pedestrians_on_carriageway"] = {
        "definition": "person ground point in a direction-field cell used by moving vehicles, >= 5 samples (~0.5 s)",
        "tracks_on_road": int(len(on_road_tracks)), "of_person_tracks": int(cls_tracks.get("person", 0)),
        "crossing_corridors": corridors[:5],
        "corridor_rule": f"road cells with >= {thr:.0f} distinct people, dilated by one cell",
        "tracks_on_road_outside_corridors": int(len(outside)), "outside_spots": spots[:6]}
    fig, ax = bg_axes(9.6, bg, W, H, dim=0.55,
                      title=f"{args.stem}: pedestrians on the carriageway ({len(on_road_tracks)} people), "
                            f"crossing corridors vs elsewhere ({len(outside)} people)")
    ov = np.zeros((GY, GX, 4))
    ov[road] = (0.35, 0.55, 1.0, 0.18)
    ov[corridor] = (0.2, 0.85, 0.35, 0.35)
    ax.imshow(ov, extent=(0, W, H, 0), interpolation="nearest")
    ins = pg[pg.on_road & pg.in_corridor]
    ous = pg[pg.track_id.isin(outside.index) & pg.on_road & ~pg.in_corridor]
    ax.scatter(ins.sx, ins.sy, s=3, color="#7ee08a", alpha=0.5, label="person on road, inside a corridor")
    ax.scatter(ous.sx, ous.sy, s=5, color="#ff5a4e", alpha=0.8, label="person on road, outside corridors")
    ax.plot([], [], "s", color=(0.35, 0.55, 1.0), alpha=0.5, ms=10, label="cells used by moving vehicles")
    ax.plot([], [], "s", color=(0.2, 0.85, 0.35), alpha=0.7, ms=10, label="crossing corridor cells")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=11, markerscale=2)
    save(fig, args.out, "pedestrian_crossings.png")

    # ---- tracker stats ---------------------------------------------------
    margin = 150
    interior_birth = tt[(tt.x0 > margin) & (tt.x0 < W - margin) & (tt.y0 > margin) & (tt.y0 < H - margin)]
    extra["tracker"] = {
        "raw_track_ids": int(n_raw_tracks), "tracks_after_min_det_filter": int(len(tt)),
        "min_detections": args.min_det, "riders_relabelled": int(len(riders)),
        "tracks_by_class": {k: int(v) for k, v in cls_tracks.items()},
        "detections_by_class": {k: int(v) for k, v in df.tcls.value_counts().items()},
        "median_track_duration_s": {c: round(float(tt[tt.cls == c].dur.median()), 1) for c in cls_tracks},
        "short_tracks_under_1s_pct": round(100 * float((tt.dur < 1).mean()), 1),
        "vehicle_tracks_born_inside_frame_pct": round(
            100 * float(interior_birth.cls.isin(VEH).sum() / max(tt.cls.isin(VEH).sum(), 1)), 1),
        "max_concurrent": {c: int(per_frame[c].max()) for c in per_frame.columns},
        "mean_concurrent": {c: round(float(per_frame[c].mean()), 1) for c in per_frame.columns},
    }
    extra["tracks_source"] = {"file": os.path.basename(args.tracks), "detector": args.detector}
    if args.run_info and os.path.exists(args.run_info):
        extra["tracking_run"] = json.load(open(args.run_info))
    if args.benchmark and os.path.exists(args.benchmark):
        extra["cpu_benchmark"] = json.load(open(args.benchmark))
    extra["lane_flow_details"] = flow_info
    extra["stop_runs"] = {"count": int(len(runs)), "vehicles": int(runs.track_id.nunique()),
                          "median_s": round(float(runs.dur.median()), 1) if len(runs) else 0,
                          "stop_thr_px_s": args.stop_thr}
    extra["size_px_4k"] = {c: {"median": round(float(df[df.tcls == c].bh.median()), 1),
                               "p10": round(float(df[df.tcls == c].bh.quantile(0.1)), 1)}
                           for c in order6}
    ph = df[df.tcls == "person"].bh
    extra["person_height_share_below"] = {f"{int(16 * W / isz)}px (16px at imgsz {isz})":
                                          round(float((ph < 16 * W / isz).mean()), 3)
                                          for isz in (640, 960, 1280)}
    extra["speed_px_s"] = {c: {"median_of_track_p85": round(float(tt[(tt.cls == c) & (tt.dur >= 1)].p85_speed.median()), 1)}
                           for c in order6}
    draw_samples(args.video, raw, [float(x) for x in args.samples.split(",") if x], scale, vp["fps"], args.out)
    extra["figures"] = sorted(f for f in os.listdir(args.out) if f.endswith(".png"))

    # ---- automatic notes -----------------------------------------------
    tot = sum(cls_tracks.get(c, 0) for c in CLASSES)
    npers = cls_tracks.get("person", 0)
    notes.append(f"{tot} tracks in {duration:.0f} s: {npers} pedestrians ({100 * npers / max(tot, 1):.0f}%), "
                 + ", ".join(f"{cls_tracks.get(c, 0)} {c}" for c in ["car", "bus", "truck", "motorcycle", "bicycle"])
                 + f"; {len(riders)} person boxes riding two-wheelers were relabelled as riders.")
    mc = extra["tracker"]["mean_concurrent"]
    notes.append(f"On average {mc.get('person', 0)} pedestrians and "
                 f"{sum(mc.get(c, 0) for c in VEH):.0f} vehicles are on screen at once "
                 f"(peak {extra['tracker']['max_concurrent'].get('person', 0)} pedestrians).")
    if lane_flows:
        top = lane_flows[:3]
        notes.append(f"{len(lane_flows)} vehicle flows found; the largest are "
                     + "; ".join(f"{f['lane']} ({f['direction']}, {f['count']} tracks)" for f in top) + ".")
    fr = [fi for fi in flow_info if fi.get("furthest_queued_stop_in_D_red")]
    if fr:
        notes.append("Furthest queued stop that began while head D showed red, red_amber or amber, per flow "
                     "(rough upper bound for a stop line; D is a timing proxy for the near carriageway only): "
                     + "; ".join(f"{fi['lane'].split()[0]} near {fi['furthest_queued_stop_in_D_red']} "
                                 f"({100 * fi['share_that_stopped_2s']:.0f}% of its vehicles stopped >= 2 s"
                                 + (f", queue {fi['mean_queue_D_red_vs_not_red'][0]} vs "
                                    f"{fi['mean_queue_D_red_vs_not_red'][1]} vehicles while D red vs not red"
                                    if fi.get("mean_queue_D_red_vs_not_red") else "") + ")"
                                 for fi in fr) + ".")
    pm = extra["size_px_4k"].get("person", {}).get("median")
    if pm:
        shares = extra["person_height_share_below"]
        notes.append(f"Median pedestrian box height is {pm:.0f} px at 4K; "
                     + ", ".join(f"{100 * v:.0f}% are under {k}" for k, v in shares.items())
                     + ". These are detected boxes, so the smallest people are already missing.")
    lg = extra["lighting"]
    txt = (f"Mean luma goes from {lg['mean_luma_first10s']} to {lg['mean_luma_last10s']} "
           f"({lg['change_pct']:+.1f}%) over the clip, range {lg['min']} to {lg['max']} of 255")
    if "largest_15s_change_pct" in lg:
        txt += (f"; the largest swing is {lg['largest_15s_change_pct']:+.1f}% within 15 s "
                f"({lg['largest_15s_change_at_s'][0]:.0f} to {lg['largest_15s_change_at_s'][1]:.0f} s). "
                "Iris, shutter and gain are constant in the camera metadata (reports/eda/camera), so this is not "
                "exposure control; brightness rules (fire_smoke) must still normalise per frame")
    notes.append(txt + ".")
    pv = extra["pedestrians_on_carriageway"]
    if pv["tracks_on_road"]:
        cor = pv["crossing_corridors"]
        txt = (f"{pv['tracks_on_road']} of {npers} pedestrian tracks spend >= 0.5 s on cells used by moving vehicles; "
               f"{len(cor)} crossing corridor(s) found")
        if cor:
            txt += f", the busiest spans x {cor[0]['bbox_px'][0]} to {cor[0]['bbox_px'][2]}, y {cor[0]['bbox_px'][1]} to {cor[0]['bbox_px'][3]} ({cor[0]['people']} people)"
        txt += f"; {pv['tracks_on_road_outside_corridors']} people are on the road outside corridors"
        if pv["outside_spots"]:
            txt += f", most around {pv['outside_spots'][0]['median_xy']}"
        notes.append(txt + " (includes kerbside waiting and bus boarding, so an upper bound for jaywalking).")
    if "vehicle_head" in extra.get("signal", {}):
        vs = extra["signal"]
        vb = vs["vehicle_head"]["box_video"]
        notes.append(f"Signal: vehicle head {vs['vehicle_head']['head']} (3 lamps, from the head inventory) at "
                     f"({(vb[0] + vb[2]) // 2}, {(vb[1] + vb[3]) // 2}) in this video; its v4 timeline has a known "
                     f"state for {100 * vs['coverage_known_state']:.1f}% of the clip (coverage, not accuracy: no "
                     "human-verified labels yet). D is a timing proxy for the near-carriageway flow; which movements "
                     "it controls is unknown.")
    tm = extra["turning_movements"]["counts"]
    if tm:
        ut = [t_ for t_ in extra["turning_movements"]["non_straight"] if t_["kind"] == "u_turn_like"]
        notes.append(f"Of {sum(tm.values())} vehicle tracks with enough motion, {tm.get('straight', 0)} go straight, "
                     f"{tm.get('turn_clockwise_on_screen', 0) + tm.get('turn_anticlockwise_on_screen', 0)} turn and "
                     f"{len(ut)} reverse heading (U-turn like"
                     + (f", e.g. track {ut[0]['track_id']} at {ut[0]['xy_mid']} around {ut[0]['t'][0]} s" if ut else "")
                     + "); check these by eye before treating them as illegal_u_turn examples.")
    if len(parked):
        notes.append(f"{len(parked)} vehicle(s) stand still for over 90% of a track longer than 60 s "
                     f"at " + ", ".join(str(v["xy"]) for v in extra["long_standing_vehicles"][:5]) + " (parked), which a stopped_vehicle rule must whitelist or time-gate.")
    notes.append(f"Tracker: {extra['tracker']['short_tracks_under_1s_pct']}% of tracks live under 1 s and "
                 f"{extra['tracker']['vehicle_tracks_born_inside_frame_pct']}% of vehicle tracks start away from "
                 f"the frame edge, a sign of occlusion and ID switches.")
    pointer = ("EDA v4: reviewed findings, confidence and rule-input status are in reports/eda/SUMMARY.md and "
               "tools/scene/CONTRACT.md. The notes below are automatic descriptive statistics of this clip only.")
    notes.insert(0, pointer)
    extra["auto_notes"] = notes
    if args.notes_file and os.path.exists(args.notes_file):
        notes = [pointer] + json.load(open(args.notes_file))  # reviewed notes replace the automatic ones
    extra["deprecated"] = {
        "vehicle_light_guess": "REMOVED in v4. v2 picked the light whose green correlated best with traffic; that "
                               "was the pedestrian head A in C3896 and C3902. Use extra.signal (head D from "
                               "signals/signal_heads.json, v4 intervals).",
        "signal_colour_summary": "REMOVED in v4 (HSV reading of COCO-detected boxes on the 1080p proxy, phases "
                                 "forward-filled). Use reports/eda/signals/v4/<STEM>_D_intervals_v4.csv.",
        "traffic_lights_detected": "REMOVED in v4 (COCO detector boxes). The head inventory is "
                                   "reports/eda/signals/signal_heads.json.",
        "signal_timeline.csv": "no longer written (legacy HSV per-sample timeline)."}

    result = {
        "counts_per_sec": counts_per_sec,
        "density_per_min": density,
        "lane_flows": lane_flows,
        "heatmap_url": f"{args.url_prefix}/{args.stem}/heatmap.png",
        "trajectories_url": f"{args.url_prefix}/{args.stem}/trajectories.png",
        "notes": notes,
        "extra": extra,
    }
    with open(os.path.join(args.out, "eda.json"), "w") as f:
        json.dump(result, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"done in {time.time() - T0:.0f} s", flush=True)
    for n_ in notes:
        print(" -", n_)


if __name__ == "__main__":
    main()
