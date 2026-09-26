# Scene tools: registration, signal heads, reference scene map

The camera is not perfectly fixed: its framing differs between recordings and it
settles for up to ~30 s after recording starts. Everything scene-related is
therefore defined once on a REFERENCE frame (C3897, `reference_C3897.jpg`, a
60-frame median of the 1080p proxy) in 4K pixels, and moved into each video with
a homography at run time.

| Script | What it does | Runs on |
|---|---|---|
| `register.py` | `register_video(path)` / `register_frames(frames)` -> `H_ref_to_video`, `H_video_to_ref` (4K px), similarity, inliers, residuals; `to_video()`, `to_ref()` helpers; CLI | anywhere (OpenCV with SIFT) |
| `background_pass.py` | one pass over a proxy: 60-frame median, static-pixel mask, per-frame luma CSV | server, ~75 s per 5-min proxy |
| `drift_check.py` | single frames vs the video's own median: camera drift inside a clip | server |
| `verify_registration.py` | independent checks of the registration (template-matched landmarks, YOLO head boxes, displacement field, drift) -> `reports/eda/registration.json` + overlays | Mac / server |
| `kaggle_head_crops.py` | one decode pass per 4K original on Kaggle, crops every signal head (dense heads every 3rd frame, others every 15th) | Kaggle CPU, ~33 min for all four videos |
| `lamp_states.py` | per-head, per-video calibrated lamp reader -> per-sample CSV, phase CSV, timeline figure, `lamp_summary.json` (with the v2 rule re-applied for BEFORE numbers) | Mac, ~30 s |
| `signal_validate.py` | v2: grammar on the v4 intervals, pedestrian vs vehicle head, held-out tests per gate over the FULL cycle with random-onset and periodicity-preserving (circular shift) nulls, phase-folded crossing rates and far-minus-near offsets -> `signal_validation_v2.json`, `cycle_phase_v2.png` (the v1 outputs `signal_validation.json`, `discharge_vs_green_onset.png` are superseded) | Mac, ~20 s |
| `head_inventory.py` | head table (appearance by eye) + automatic cycle-correlation check -> `signal_heads.json`, `signal_heads_C3897.jpg` | Mac |
| `spotcheck_pack.py` | 30 random moments per video, 4K crop of each readable head + automatic label -> `reports/eda/spotcheck/` | Mac |
| `scene_reference.py` | v1 lane-direction map (SUPERSEDED by v2, kept for traceability) -> `reports/eda/scene_reference.json` | Mac, ~5 s |
| `signal_timeline.py` | v4 signal timelines from the v3 per-sample readings: per-sample state + observation status + confidence, gap-free half-open frame intervals, occlusions over 1 s kept `unknown` -> `reports/eda/signals/v4/` | Mac, ~2 s |
| `scene_reference_v2.py` | direction map v2: heading modes per cell, per-video support, clipped boxes and motorcycles masked, `wrong_way_safe` flag -> `reports/eda/scene_reference_v2.json` + `.jpg` | Mac, ~5 s |
| `signal.py` | rule-side accessor for the v4 timelines: `load(stem)`, `state_at(t)`, `intervals()` | model |
| `scene.py` | rule-side accessor for the v2 direction map: `heading_modes(x, y)`, `wrong_way_safe(x, y)`, `against_flow(...)` | model |
| `CONTRACT.md` | which files the rules may consume (and which not), with frames, time base, versions, status | |
| `tests/` | lock-in tests for the rule inputs (pytest) | Mac |
| `zone_evidence.py` | vehicle stop points and pedestrian density from the track files, mapped into the reference frame -> `reports/eda/zone_evidence.json` (overlay in the zone editor) | server / Mac, ~5 s |
| `web/` | zone editor and signal spot-check pages next to the labeling tool, see `web/README.md` | browser + server |

## Registration at run time

```python
from tools.scene.register import register_frames, to_video
r = register_frames(frames_you_already_decoded)      # >= 5 frames, spread over time
zone_video_px = to_video(zone_ref_px, r)             # hand-drawn reference zone -> this video
```

CLI: `python tools/scene/register.py --video proxies/C3902_1080p.mp4 --frames 15`.
Use the homography, not the similarity (the similarity is off by up to ~11 px at
the frame edges). Measured cost on the 2-core server: 15 proxy frames 10 to 15 s
decode + 3 to 4.5 s matching; 10 frames from the first 10 s of a 4K original 20
to 24 s decode + 3 to 4 s matching. Frames from the first ~30 s after recording
starts can be up to 30 px off the settled position (C3896).

## Reproduce

```bash
# server (light): medians, luma, camera metadata
python tools/scene/background_pass.py --video proxies/C3897_1080p.mp4 --stem C3897 --out work/bg   # x4
python tools/eda/camera_meta.py --video originals/C3905.MP4 --out work/camera --luma work/bg/C3905_luma.csv
# checks
python tools/scene/drift_check.py --video proxies/C3896_1080p.mp4 --median work/bg/C3896_median.png \
    --frames 0:90:6,90:2000:60 --stem C3896 --out work/drift_C3896.json   # merge per-stem files into work/drift.json
python tools/scene/verify_registration.py --bg-dir work/bg --eda-dir reports/eda --out reports/eda \
    --drift work/drift.json   # optional --stability: {stem: {frames_vs_own_median: [{frame, max_disp_px}], runtime_s}}
# Kaggle: set VIDEO_BASE_URL (secret) and HEAD_BOXES (reference head boxes mapped with register.to_video)
python tools/scene/kaggle_head_crops.py
python tools/scene/lamp_states.py --crops work/crops --out reports/eda/signals
python tools/scene/signal_timeline.py --signals reports/eda/signals --out reports/eda/signals/v4 --eda-dir reports/eda
python tools/scene/signal_validate.py --signals-v4 reports/eda/signals/v4 --tracks-dir work/tracks \
    --registration reports/eda/registration.json --out reports/eda/signals
python tools/scene/head_inventory.py --crops work/crops --signals reports/eda/signals --out reports/eda/signals
python tools/scene/spotcheck_pack.py --crops work/crops --signals reports/eda/signals --out reports/eda/spotcheck
python tools/scene/scene_reference.py --tracks-dir work/tracks --registration reports/eda/registration.json \
    --eda-dir reports/eda --out reports/eda          # v1, superseded
python tools/scene/scene_reference_v2.py --tracks-dir work/tracks --registration reports/eda/registration.json \
    --out reports/eda
# per-video EDA (needs the v4 intervals and signal_heads.json; fails loudly without them)
python tools/eda/analyze.py --stem C3896 --tracks work/tracks/C3896/tracks_kaggle.csv \
    --video proxies/C3896_1080p.mp4 --out reports/eda/C3896 --detector "yolo11m imgsz 1280 + ByteTrack (Kaggle T4)"
# tests
uvx --with pandas --with numpy --with opencv-python-headless --with matplotlib pytest -q tools/scene/tests
```

Import the accessors as `from tools.scene import signal, scene` from the repo root.
`signal.py` shares its name with the standard-library module; when a script in
this folder is run directly (the folder is then first on `sys.path`) and
something imports `signal`, the file hands over the real standard-library module
(see its docstring). Rule inputs and their status: `CONTRACT.md`.

`work/tracks/<STEM>.csv` are the Kaggle track files (`tracks_kaggle.csv`). The
crops (~1 GB) and the medians are not in git.
