"""Per-frame camera settings from the Sony rtmd data track of an original MP4.

Runs exiftool (needs the -ee "extract embedded" option and LargeFileSupport for
files over 4 GB) and writes one CSV row per video frame:

  frame, t, f_number, exposure_s, iso, master_gain_db, white_balance_raw,
  ext_magnification_pct, gyro_abs_mean, accel_mean_x, accel_mean_y, accel_mean_z

plus a JSON summary (camera model, recording clock time, codec, the distinct
values of every setting and the frames where any setting changes).

  python tools/eda/camera_meta.py --video originals/C3905.MP4 --out eda/C3905
  python tools/eda/camera_meta.py --json C3905_ee.json --out eda/C3905   # reuse an exiftool -j dump

exiftool command used:
  exiftool -api LargeFileSupport=1 -ee -G3 -n -j VIDEO
Without LargeFileSupport exiftool stops at the first 64-bit atom and reports only
the container tags ("End of processing at large atom").

With --luma (a CSV from tools/scene/background_pass.py) it also plots measured
brightness against the per-frame exposure settings (<STEM>_brightness_vs_exposure.png).

Units (exiftool Sony rtmd decoder, values with -n): ExposureTime in seconds, FNumber
is the f-stop, ISO, MasterGainAdjustment in dB. WhiteBalance is kept as exiftool
prints it (column white_balance_raw; the unit is not documented). PitchRollYaw and
Accelerometer hold several raw sensor samples per frame; we keep the per-frame mean
absolute gyro value and the mean accelerometer vector (raw units) as a shake check.
"""
import argparse
import json
import os
import subprocess

import numpy as np
import pandas as pd

FIELDS = {"FNumber": "f_number", "ExposureTime": "exposure_s", "ISO": "iso",
          "MasterGainAdjustment": "master_gain_db", "WhiteBalance": "white_balance_raw",
          "ElectricalExtenderMagnification": "ext_magnification_pct"}
MAIN = ["DeviceModelName", "DeviceManufacturer", "CreationDateValue", "CreateDate", "TimeZone",
        "VideoFormatVideoFrameVideoCodec", "VideoFormatVideoFrameCaptureFps", "ImageWidth",
        "ImageHeight", "DurationValue", "AcquisitionRecordGroupItemValue", "Duration"]


def run_exiftool(video):
    out = subprocess.check_output(["exiftool", "-api", "LargeFileSupport=1", "-ee", "-G3", "-n", "-j", video])
    return json.loads(out)[0]


def parse(d):
    docs = {}
    for k, v in d.items():
        g, _, tag = k.partition(":")
        if g.startswith("Doc") and g[3:].isdigit():
            docs.setdefault(int(g[3:]), {})[tag] = v
    rows = []
    for i in sorted(docs):
        x = docs[i]
        r = {"frame": i - 1, "t": round(float(x.get("SampleTime", np.nan)), 4)}
        for tag, col in FIELDS.items():
            r[col] = x.get(tag, np.nan)
        pry = np.array(str(x.get("PitchRollYaw", "")).split(), float)
        acc = np.array(str(x.get("Accelerometer", "")).split(), float)
        r["gyro_abs_mean"] = float(np.abs(pry).mean()) if pry.size else np.nan
        if acc.size >= 3:
            a = acc[: acc.size // 3 * 3].reshape(-1, 3).mean(0)
            r["accel_mean_x"], r["accel_mean_y"], r["accel_mean_z"] = [round(float(v), 1) for v in a]
        rows.append(r)
    df = pd.DataFrame(rows)
    main = {k.split(":", 1)[-1]: v for k, v in d.items() if k.split(":", 1)[-1] in MAIN and not k.startswith("Doc")}
    changes = []
    for col in FIELDS.values():
        if col in df and df[col].notna().any():
            s = df[col]
            ch = df.frame[(s != s.shift()) & (df.index > 0)].tolist()
            changes.append({"setting": col, "distinct_values": sorted(pd.unique(s.dropna()).tolist()),
                            "change_frames": ch[:50], "n_changes": len(ch)})
    summary = {"camera": main, "n_frame_samples": int(len(df)), "settings": changes,
               "gyro_abs_mean": {"median": round(float(df.gyro_abs_mean.median()), 1),
                                 "p99": round(float(df.gyro_abs_mean.quantile(0.99)), 1)},
               "exiftool_cmd": "exiftool -api LargeFileSupport=1 -ee -G3 -n -j VIDEO"}
    return df, summary


def plot(df, luma, stem, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.4, 1]})
    L = luma.copy()
    L["s"] = (L.t // 1).astype(int)
    L = L.groupby("s").mean(numeric_only=True)
    cols = {"mean": ("whole frame", "#2a78d6"), "bottom_two_thirds": ("bottom 2/3 (road, traffic)", "#eb6834"),
            "top_third": ("top 1/3", "#1baf7a"), "static_pixels": ("static pixels only", "#0b0b0b")}
    for c, (lab, col) in cols.items():
        axs[0].plot(L.t, L[c], label=lab, color=col, lw=2)
    axs[0].set_ylabel("mean luma (1 s bins)")
    axs[0].legend(loc="upper left", fontsize=9, frameon=False)
    axs[0].set_title(f"{stem}: measured brightness vs camera exposure settings from the rtmd track", loc="left")
    ev = np.log2(df.f_number ** 2 / df.exposure_s) - np.log2(df.iso / 100)
    axs[1].plot(df.t, ev, color="#2a78d6", lw=2, label="exposure value EV100 from f-number, shutter, ISO")
    axs[1].set_ylabel("EV (lower = brighter)")
    axs[1].set_ylim(ev.min() - 1.5, ev.max() + 1.5)
    txt = (f"f/{df.f_number.iloc[0]:.1f}, 1/{1 / df.exposure_s.iloc[0]:.0f} s, ISO {df.iso.iloc[0]:.0f}, "
           f"gain {df.master_gain_db.iloc[0]:.0f} dB in frame 0; distinct values over the clip: "
           f"f {df.f_number.nunique()}, shutter {df.exposure_s.nunique()}, ISO {df.iso.nunique()}, gain {df.master_gain_db.nunique()}")
    axs[1].text(0.01, 0.08, txt, transform=axs[1].transAxes, fontsize=9)
    axs[1].legend(loc="upper left", fontsize=9, frameon=False)
    axs[2].plot(df.t, df.gyro_abs_mean, color="#52514e", lw=1)
    axs[2].set_ylabel("gyro |raw| mean")
    axs[2].set_xlabel("time (s)")
    for ax in axs:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=90)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video")
    ap.add_argument("--json", help="existing `exiftool -ee -G3 -n -j` output instead of running exiftool")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stem")
    ap.add_argument("--luma", help="luma CSV from tools/scene/background_pass.py: adds a brightness vs exposure plot")
    a = ap.parse_args()
    d = json.load(open(a.json))[0] if a.json else run_exiftool(a.video)
    df, summary = parse(d)
    os.makedirs(a.out, exist_ok=True)
    stem = a.stem or os.path.splitext(os.path.basename(a.video or a.json))[0].split("_")[0]
    df.to_csv(os.path.join(a.out, f"{stem}_camera_frames.csv"), index=False)
    with open(os.path.join(a.out, f"{stem}_camera.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    if a.luma:
        plot(df, pd.read_csv(a.luma), stem, os.path.join(a.out, f"{stem}_brightness_vs_exposure.png"))
    print(json.dumps({s["setting"]: s["distinct_values"][:8] for s in summary["settings"]}))


if __name__ == "__main__":
    main()
