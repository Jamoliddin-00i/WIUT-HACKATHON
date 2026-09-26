# EDA tools

Scene analysis for one camera video: who is there, where they move, where they
stop, what the signal shows. The camera description was removed by the
organizers, so these scripts are how we get the scene layout (lane directions,
stop lines, crossings, signal head) for the event rules.

All coordinates are in ORIGINAL video pixels (3840x2160 for the sample videos),
so results from different proxies or detectors line up. The ground point of a
box is its bottom centre.

## 1. Track (`track.py`)

YOLO11 (COCO) + ByteTrack on every 3rd frame of a proxy video. Writes
`tracks.csv` with `frame, t_sec, track_id, cls, conf, x1, y1, x2, y2`,
`t_sec = frame / fps`.

```bash
# throughput check on 60 processed frames
python tools/eda/track.py --video proxies/C3905_1080p.mp4 --out eda/C3905 \
    --model models/yolo11s.pt --imgsz 1280 --bench 60

# full run (use setsid/nohup on a remote box)
python tools/eda/track.py --video proxies/C3905_1080p.mp4 --out eda/C3905 \
    --model models/yolo11n.pt --imgsz 1280 --stride 3 --orig-width 3840
```

Throughput on the 2-core server (processed frames/s, 1080p proxy, stride 3):
yolo11s@1280 0.79, yolo11s@960 1.42, yolo11n@1280 1.93. The final tracks for
the website come from yolo11m@1280 on a Kaggle T4 (same CSV schema).

## 2. Analyse (`analyze.py`)

```bash
python tools/eda/analyze.py --stem C3905 --tracks eda/C3905/tracks_kaggle.csv \
    --video proxies/C3905_1080p.mp4 --out eda/C3905 --model models/yolo11s.pt \
    --detector "yolo11m imgsz 1280 + ByteTrack (Kaggle T4)" \
    --lane-names reports/eda/C3905/lane_names.json \
    --notes-file reports/eda/C3905/notes.json
```

`--model` is only used to find traffic lights (COCO class 9) on a few frames.
`--lane-names` and `--notes-file` are optional: run once without them, look at
the figures, then write reviewed lane names and notes and run again. The
automatic notes are always kept in `extra.auto_notes`.

Outputs in `--out`:

| File | What it shows |
|---|---|
| `counts_over_time.png` | tracked objects per second by class |
| `heatmap.png`, `heatmap_persons.png` | where vehicles / pedestrians spend time |
| `trajectories.png` | vehicle tracks over 2 s, coloured by heading |
| `direction_field.png`, `direction_field.json` | mean vehicle heading per 80 px cell (lane direction map) |
| `lane_flows.png` | dominant flows (k-means on entry, exit, heading) |
| `stop_map.png` | where vehicles stand still for 2 s or more, plus moving vs stationary over time with the signal colour behind it |
| `signal.png`, `signal_timeline.csv` | detected traffic lights and their lit colour over time |
| `pedestrian_crossings.png` | people on vehicle lanes, crossing corridors vs elsewhere |
| `speed_duration.png` | speed per track and track lifetime by class |
| `lighting.png` | mean luma per second |
| `object_size.png` | box height by class and person size vs image row |
| `sample_tXXX.jpg` | frames with the CSV boxes drawn, for a sanity check |
| `eda.json` | everything the website needs, plus `extra` with the numbers behind the notes |

The signal reading in `analyze.py` (`signal.png`, `signal_timeline.csv`, fixed HSV
thresholds on 1080p boxes, head chosen by correlation with traffic) is kept for the
website but superseded: use `tools/scene/lamp_states.py` (per-head calibration on
4K crops, heads chosen by appearance). See `reports/eda/SUMMARY.md` finding 3.

## 3. Camera metadata (`camera_meta.py`)

Per-frame exposure settings and gyro / accelerometer from the Sony rtmd track of an
original, via exiftool (needs `-api LargeFileSupport=1` for files over 4 GB).

```bash
python tools/eda/camera_meta.py --video originals/C3905.MP4 --out eda/C3905 \
    --luma work/bg/C3905_luma.csv      # optional: brightness vs exposure plot
```

Writes `<STEM>_camera_frames.csv` (one row per frame), `<STEM>_camera.json`
(camera, clock, codec, distinct values of each setting) and, with `--luma`,
`<STEM>_brightness_vs_exposure.png`. About 10 s per file on the server.

## 4. Compare two detectors (`compare.py`)

```bash
python tools/eda/compare.py --a eda/C3905_yolo11n/tracks.csv --a-label "yolo11n 1280 (CPU)" \
    --b eda/C3905/tracks_kaggle.csv --b-label "yolo11m 1280 (T4)" --out eda/C3905
```

Writes `detector_compare.json` and `detector_compare.png`.

## Requirements

`ultralytics`, `lap`, `pandas`, `matplotlib`, `opencv-python`, `numpy`. No scipy
or sklearn needed.
