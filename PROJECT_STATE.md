# WIUT Hackathon 2026 — Project State

_Last updated: 2026-09-24, Tashkent time_

This is the living handoff file for Codex and other development sessions. Keep it current. Do not duplicate facts that are already documented and still correct.

## 1. Ownership and working model

- **Jamoliddin:** ML Engineer / Model Lead.
- **Codex is the primary coding agent.** Repository Markdown is the canonical project memory. The user should not need to re-explain context in chat.
- After a meaningful discovery, benchmark, decision, implementation change, or teammate handoff, update this file or the relevant Markdown doc in the same session.
- Commits in the final team repository must use a human team member's configured Git identity. Do not add AI `Co-authored-by:` trailers, bot authors, or AI-owned PR/commit identities.

### Repository status

- `Jamoliddin-00i/WIUT-HACKATHON` is currently a **handoff / staging repository**.
- The real canonical team repository is elsewhere and is managed by the team/Hamid.
- Jamoliddin supplied `https://github.com/abdulhamid-n/salen-traffic-events.git`
  on 2026-09-24. A clone attempt returned GitHub "Repository not found" and
  `gh auth status` showed no GitHub login on this machine. Access or a local
  checkout is still needed; do not push to that repository.
- Jamoliddin placed an extracted snapshot at
  `D:\wiut hackathon\code\salen-traffic-events-main`. It contains Hamid's EDA
  but no `.git` directory, so it is not a synced Git checkout.
- When Jamoliddin has the real team repo locally, Codex should continue there, not maintain a parallel final implementation here.
- Before working in the final repo, verify `git remote -v` so `origin` points to the real team repository.
- AI assistants may edit/generate code locally, but commits pushed to the canonical team repo should be authored by the human contributor, so AI accounts do not appear as contributors.

## 2. Confirmed organizer / video facts

Organizer/team confirmation received on 2026-09-24:

- `camera.md` has been removed. We infer lanes, stop lines, crossings, and directions ourselves from the sample footage/EDA.
- Hidden test footage uses the **same raw camera format/view** as the sample videos.
- Raw video format:
  - `3840x2160`
  - H.264 High 4:2:2
  - 10-bit
  - about `140 Mbps`
  - `29.97 fps`
- The judge GPU is T4-class, but this 4:2:2 H.264 stream should be treated as a **CPU-decode workload** rather than relying on GPU hardware decode.
- The official Part B harness decodes every frame with `cv2.VideoCapture`, converts to BGR, then calls `RiskEstimator.step()` for every frame. We cannot remove that decode cost simply by sampling detector inference less often.
- Total time budget remains `<= 3 x video duration` for Part A + Part B. Because decode itself consumes a meaningful fraction, Part A should target comfortably under ~1x video duration rather than spending the entire budget.

## 3. Current decode benchmark

### Local Windows FFmpeg benchmark

Machine:
- CPU: **Intel Core i5-12500H**
- 12 physical cores
- 16 logical processors

Input tested:
- `C3905.MP4`
- 3840x2160
- H.264 High 4:2:2 10-bit
- ~140 Mbps
- 29.97 fps

Command:

```powershell
ffmpeg -benchmark -i "D:\wiut hackathon\videos\C3905.MP4" -frames:v 600 -f null -
```

Observed:
- 600 frames = ~20.02 s source video
- wall time: **5.713 s**
- ~**105 fps**
- ~**3.5x realtime**
- raw FFmpeg-null decode cost ~**0.285x video duration**

Important: this is **not yet the number that matters most** because `ffmpeg -f null` is cheaper than the actual `cv2.VideoCapture -> BGR ndarray` path used by the harness.

### Local Windows OpenCV BGR decode benchmark

Measured 2026-09-24 with `scripts/benchmark_decode.py` on the first 600 frames of
`C3905.MP4`. This uses `cv2.VideoCapture.read()` and retains the BGR ndarray, as
the organizer harness does. A CUDA PyTorch wheel download was also active during
these measurements, so repeat if a precise final runtime margin is needed.

| CPU affinity | Wall time | Decode rate | Wall / source duration |
| --- | ---: | ---: | ---: |
| 16 logical CPUs (default), first run | 23.78 s | 25.23 fps | 1.19x |
| first 8 logical CPUs | 27.17 s | 22.09 fps | 1.36x |
| 16 logical CPUs, user's exact `time.time()` script, later run | 19.82 s | 30.28 fps | 0.99x |

The OpenCV BGR path is much slower than the FFmpeg-null benchmark above. It
costs roughly one video-duration of wall time on this machine, with meaningful
run-to-run variance; the restricted eight-logical-CPU result is slower. The
combined 3x budget therefore leaves limited margin for Part A detection and
Part B processing; sample model inference sparingly and benchmark end to end.

Optional/secondary benchmark:
- Kaggle/Linux environment for a closer judge-like CPU/T4 setup.
- Record CPU model/core allocation, FFmpeg result, and OpenCV result.

## 4. Hamid's C3905 EDA handoff

Availability as of 2026-09-24:
- Hamid's `reports/eda/C3896/` and `reports/eda/C3905/` are available in the
  extracted snapshot `D:\wiut hackathon\code\salen-traffic-events-main`.
  They include notes, signal timelines/figures, pedestrian crossing maps,
  direction fields, detector comparisons, and scene samples.
- The underlying YOLO11m track CSVs are not in that snapshot, nor are EDA
  directories for C3897 and C3902. Keep using locally cached track CSVs for
  those clips until the teammate tracks arrive.
- Hamid's C3905 detector comparison found 31.55 persons per frame for
  YOLO11m at 1280 px versus 23.03 for YOLO11n at 1280 px. The current local
  YOLO26n 960 px pass therefore misses some small pedestrians; absence of a
  proposal is weak evidence of absence of an event.

### Traffic light

- Readable signal location in original 4K coordinates: approximately **`(2328, 780)`**.
- `signal.png` and `signal_timeline.csv` are now locally available in Hamid's
  C3905 EDA directory. Its notes report red at 0-34.5 s and 75.5-114.4 s,
  green at 34.6-70.4 s and from 114.5 s, with amber in between.
- Reported correlation between signal state and moving cars: **0.88**.
- Practical `red_light` rule direction: **signal is red + vehicle crosses the relevant stop line**.

### Stop line / zebra region

For the near carriageway, cars reportedly wait around:
- **`(1500, 870)`**
- **`(1764, 985)`**

These are just before the zebra crossing and should inform the initial stop-line geometry. Verify exact line/polygon placement against local frames before hard-coding.

### Wrong-way direction field

- `direction_field.json` is now available for C3896 and C3905 in the extracted
  snapshot. It stores vehicle heading and coherence per **80 px cell**.
- A wrong-way rule should require sustained opposing motion in high-coherence
  road cells and reject frame-edge ID switches noted by Hamid.

### Stopped-vehicle ignore zones

Do not treat the following as `stopped_vehicle` anomalies:
- three cars parked for the whole clip at the **left edge**;
- far-kerb bus-stop region approximately **x = 1350..1700, y = 380..520**.

These zones should become explicit ignore masks/regions in scene configuration, not scattered conditionals.

### Exposure jump

- Camera auto-exposure reportedly changes brightness by roughly **35% around 52-67 s**.
- **Do not use raw/global brightness alone for `fire_smoke`.** Any smoke/fire detector should use spatial/temporal/local cues robust to exposure changes.

### YOLO tracks

- Hamid generated **YOLO11m tracks** for EDA on the server/Kaggle; the
  extracted snapshot has reports but no track CSVs.
- The EDA README describes its CSV schema as
  `frame,t_sec,track_id,cls,conf,x1,y1,x2,y2` in original 4K coordinates.
- When shared, inspect the schema once and integrate/reuse the tracks rather than recomputing identical EDA unnecessarily.

## 5. Model / runtime decisions already settled

- Final inference must be offline and reproducible.
- No OpenAI/Gemini/Anthropic/hosted inference APIs in the traffic pipeline.
- Download/install open weights locally and run on the user's GPU with CUDA when available.
- Final implementation must also run on a T4-class 16 GB GPU and stay within the total weight/runtime limits.
- Kaggle may be used for T4-like testing or fine-tuning.
- A local Linux VM is **not currently required**.
- GitHub Actions are not required and should not be a dependency.
- Never modify organizer `run_submission.py` or `evaluate.py`.
- Do not commit the raw multi-GB sample videos.

## 6. Annotation/dev-label state

A local annotation script exists at `scripts/label_video.py` / is being used for sample-video labeling.

Local GUI note:
- `opencv-python-headless` cannot use `cv2.imshow()`.
- The labeler now uses Tkinter and Pillow for its GUI while OpenCV only decodes
  frames; `requirements-labeler.txt` adds Pillow without changing judge deps.
- A desktop smoke test opened and closed successfully on 2026-09-24.

Ground-truth dev annotations should ultimately be saved in the official evaluator-compatible shape, e.g. `dev_labels.json`, then tested with `run_submission.py` and `evaluate.py`.

Manual labeling principle already established:
- be conservative;
- use official event start/end conventions;
- do not label a traffic-law violation unless the relevant signal/lane/crossing rule is actually supported by the footage/scene geometry.

Jamoliddin has asked for automatic event labeling because manually identifying
traffic violations is impractical. Build an automatic first pass from local
detector/tracker tracks and scene rules. Keep its proposals separate from
`dev_labels.json`; pseudo-labels should not be treated as ground truth when
evaluating the model. For `failure_to_yield`, the event is a vehicle **driving
through** the crossing while a pedestrian is on or entering it; stopping to
yield is not that event.

Initial local automatic pass now exists in `scripts/auto_label_video.py` with
normalized scene geometry in `config/scene.json`. It uses locally stored
`weights/yolo26n.pt` and CUDA YOLO tracking to propose `jaywalking` and
`failure_to_yield` segments, writing `auto_proposals.json` separately from
reviewed labels and `debug/*_auto_tracks.csv` for diagnostics. The scene masks
and thresholds are preliminary and must be checked on footage. The first 20 s
run at 3 fps / 1280 px produced false positives; a 2 fps / 960 px pass was
faster. The full 2 fps / 960 px pass completed on all four sample videos:
1 proposal on C3905, 3 on C3896, 10 on C3897, and 9 on C3902. A midpoint
contact-sheet review found clear false positives around sidewalks and islands;
these 23 events remain unverified candidates, not ground truth. C3902 around
83-97 s includes people walking diagonally through the junction and is a
promising jaywalking candidate for focused review. The script now uses
`VideoCapture.grab()` for unsampled frames; this change needs its own runtime
measurement before claiming a speed improvement.

Local ML environment: isolated `.venv`, PyTorch `2.14.0+cu130`, CUDA 13.0,
RTX 3050 Laptop GPU with 4 GB VRAM. Ultralytics `8.4.161` and local YOLO26n
weights (~5.5 MB) are present. The final judge dependency recipe is not yet
settled; do not assume this minimal dev environment is the final package.
Ultralytics states that its code and models use AGPL-3.0 or an Enterprise
license; check the canonical team's license/attribution plan before committing
the weights or packaging Ultralytics in the final submission.

## 7. Immediate next actions for Codex

1. Use the newly available C3896/C3905 EDA to refine scene geometry and
   verify proposed events; the extracted snapshot is not a Git checkout.
2. Use the OpenCV decode numbers above when setting the detector sampling rate.
3. Extend and validate the automatic event-proposal pass; keep it distinct
   from reviewed dev labels and reject sidewalk/island false positives.
4. Use the signal timeline and direction fields now available in EDA for
   candidate `red_light` and `wrong_way` rules, with video verification.
5. Build/verify any missing scene geometry directly from local video frames rather than pretending unavailable EDA files exist.
6. When Hamid shares the YOLO11m track CSVs, inspect and integrate them;
   the direction fields are already available in the extracted snapshot.
7. Use manual review only where it adds value to automatic proposals, with official event boundary conventions.
8. When the canonical team repository is available locally, move/sync the current useful code/docs there and continue in that repo using Jamoliddin's Git identity.
9. Only after runtime/geometry are understood, proceed with detector/tracker/event implementation and Part B TTC/conflict risk logic.

## 8. Performance principle

The raw-video decode path is now a first-order constraint. Optimize model work **around** unavoidable decode cost:

- do not run YOLO on every 29.97-fps frame unless benchmarks justify it;
- sample detector inference and reuse/interpolate tracks;
- avoid independently decoding the same video multiple times inside Part A;
- keep Part B causal and light per frame;
- benchmark the real `run_submission.py` path early, not just isolated model inference.

## 9. Things not to rediscover

Unless new evidence changes them, do not spend time re-deriving these:
- `camera.md` is gone;
- C3905 signal ROI around `(2328, 780)` has already been identified by teammate EDA;
- Hamid has a direction field for C3905, but it is not yet shared locally;
- known parked/bus-stop zones need to be ignored for stopped-vehicle logic;
- exposure changes around 52-67s make global brightness unreliable for fire/smoke;
- raw-video CPU decoding is a material part of the 3x runtime budget;
- Hamid's C3896/C3905 EDA is in the extracted snapshot under `code`, but the
  YOLO11m track CSVs and an authenticated team Git checkout are still absent.
