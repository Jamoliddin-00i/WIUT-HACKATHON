"""Context strips for the signal spot-check page (runs on the server, stdlib + ffmpeg).

For every moment in reports/eda/spotcheck/spotcheck.csv: 13 crops of the same
head, every 6th frame (0.2 s apart) from 1.2 s before to 1.2 s after the
sampled frame, side by side in one JPEG. The sampled frame is tile number
`center` (6 unless the moment is in the first 1.2 s). One decode per moment
gives the strips of all heads (A and D share the moment).

Source: the 4K original when present (same crop box as the pack), else the
1080p proxy with the box halved and the crop upscaled 2x (nearest neighbour),
so tiles have the pack's size either way. The JSON written next to the strips
says which source each strip came from.

  python3 context_strips.py --csv spotcheck.csv --heads signal_heads.json \
      --originals ~/wiut/originals --proxies ~/wiut/proxies --out ~/wiut/scene/strips
"""
import argparse
import csv
import json
import os
import subprocess
import time

FPS = 30000 / 1001
STEP, HALF = 6, 6          # every 6th frame, 6 tiles either side


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--heads", required=True)
    ap.add_argument("--originals", required=True)
    ap.add_argument("--proxies", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="comma list of ids (test)")
    a = ap.parse_args()
    heads = json.load(open(a.heads))["heads"]
    rows = list(csv.DictReader(open(a.csv)))
    only = set(filter(None, a.only.split(",")))
    moments = {}
    for r in rows:
        if only and r["id"] not in only:
            continue
        moments.setdefault((r["stem"], int(r["frame"])), []).append(r)
    meta_path = os.path.join(a.out, "strips.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    t0 = time.time()
    for k, ((stem, fr), rs) in enumerate(sorted(moments.items())):
        os.makedirs(os.path.join(a.out, stem), exist_ok=True)
        orig = os.path.join(a.originals, f"{stem}.MP4")
        use4k = os.path.exists(orig)
        src = orig if use4k else os.path.join(a.proxies, f"{stem}_1080p.mp4")
        back = min(HALF, fr // STEP)
        f0 = fr - back * STEP
        n_tiles = back + 1 + HALF
        ss = max(0.0, (f0 - 0.5) / FPS)
        dur = (n_tiles - 1) * STEP / FPS + 1.0 / FPS
        chains, maps = [], []
        labels = [f"[s{i}]" for i in range(len(rs))]
        split = f"[0:v]select='not(mod(n\\,{STEP}))',split={len(rs)}{''.join(labels)}" if len(rs) > 1 \
            else f"[0:v]select='not(mod(n\\,{STEP}))'[s0]"
        chains.append(split)
        outs = []
        for i, r in enumerate(rs):
            x1, y1, x2, y2 = heads[r["head"]]["crop_box_4k"][stem]
            if use4k:
                crop = f"crop={x2 - x1}:{y2 - y1}:{x1}:{y1}"
            else:
                crop = (f"crop={(x2 - x1) // 2}:{(y2 - y1) // 2}:{x1 // 2}:{y1 // 2},"
                        f"scale=iw*2:ih*2:flags=neighbor")
            chains.append(f"[s{i}]{crop},tile={n_tiles}x1[o{i}]")
            name = f"{stem}/{r['id']}_strip.jpg"
            outs.append((r, name))
            maps += ["-map", f"[o{i}]", "-frames:v", "1", "-q:v", "3", "-update", "1",
                     os.path.join(a.out, name)]
        cmd = ["nice", "-n", "19", "ffmpeg", "-y", "-v", "error", "-threads", "1",
               "-ss", f"{ss:.4f}", "-t", f"{dur:.4f}", "-i", src,
               "-filter_complex", ";".join(chains)] + maps
        subprocess.run(cmd, check=True)
        for r, name in outs:
            meta[r["id"]] = {"strip": name, "tiles": n_tiles, "center": back, "step_frames": STEP,
                             "first_frame": f0, "source": "4K original" if use4k else "1080p proxy (2x)"}
        print(k + 1, "/", len(moments), stem, fr, round(time.time() - t0, 1), "s", flush=True)
        if (k + 1) % 10 == 0:
            json.dump(meta, open(meta_path, "w"), indent=0)
    json.dump(meta, open(meta_path, "w"), indent=0)


if __name__ == "__main__":
    main()
