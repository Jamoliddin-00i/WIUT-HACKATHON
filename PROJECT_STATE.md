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
  on 2026-09-24. GitHub authentication is now available as `Jamoliddin-00i`.
- Jamoliddin placed an extracted snapshot at
  `D:\wiut hackathon\code\salen-traffic-events-main`. It is now a Git checkout
  of the real team repository, clean at `main` commit `46a4c8f` (2026-09-24).
  `origin` points to Hamid's repository. A pull reports "Already up to date."
  The old extracted `tools/eda/README.md` is backed up inside that checkout's
  `.git` directory. No changes were pushed.
- Continue final implementation in the team checkout rather than maintaining
  a parallel final implementation here. Review the team repo's current EDA,
  labeling guide, and tools before carrying over prototype code.
- AI assistants may edit/generate code locally, but commits pushed to the canonical team repo should be authored by the human contributor, so AI accounts do not appear as contributors.

## 2. Confirmed organizer / video facts

Organizer/team confirmation received on 2026-09-24:

- `camera.md` has been removed. We infer lanes, stop lines, crossings, and directions ourselves from the sample footage/EDA.
- Hidden test footage uses the **same raw camera format/view** as the sample videos.
  Hamid's latest EDA shows that even these sample clips have different framing:
  C3902 landmarks shift 64-144 px against C3897, while C3905 shifts up to
  70 px. The camera also settles after recording starts. A single unregistered
  polygon map is not accurate enough for final inference.
- Raw video format:
  - `3840x2160`
  - H.264 High 4:2:2
  - 10-bit
  - about `140 Mbps`
  - `29.97 fps`
- The judge GPU is T4-class, but this 4:2:2 H.264 stream should be treated as a **CPU-decode workload** rather than relying on GPU hardware decode.
- The official Part B harness decodes every frame with `cv2.VideoCapture`, converts to BGR, then calls `RiskEstimator.step()` for every frame. We cannot remove that decode cost simply by sampling detector inference less often.
- Total time budget remains `<= 3 x video duration` for Part A + Part B. Because decode itself consumes a meaningful fraction, Part A should target comfortably under ~1x video duration rather than spending the entire budget.

Organizer labeling clarifications received 2026-09-24 and incorporated in the
team's `docs/LABELING_GUIDE.md` v2:
- A bus stopped for >=10 s in a live lane counts as `stopped_vehicle`, even at
  a bus stop; kerb-parked vehicles do not.
- A pedestrian on a zebra on red is not `jaywalking`; a vehicle driving
  through the pedestrian's carriageway half of that crossing is
  `failure_to_yield` regardless of the signal.
- An ordinary red-light queue is not `congestion`; it must fail to clear on
  green.
- Scoring uses temporal IoU only, with no separate frame tolerance. Event
  boundaries should follow the official rules closely.

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

Kaggle 4-vCPU Linux FFmpeg-null decode of the first 600 C3905 frames took
27.072 s (~22 fps, 1.35x source duration). OpenCV 4.13.0 with FFMPEG can open
and BGR-decode a frame there. A full 600-frame OpenCV loop remains pending.
Do not linearly extrapolate that 4-vCPU VM to the judge's 8-core CPU.

## 4. Hamid's EDA handoff

Availability as of 2026-09-24:
- The current team checkout has EDA directories for all four sample clips,
  `reports/eda/SUMMARY.md` v3, `scene_reference.json`, `registration.json`,
  calibrated signal reports, and `docs/LABELING_GUIDE.md` v2. Read those
  before implementing camera geometry or signal rules.
- The underlying YOLO11m track CSVs are still absent from the checkout. Keep
  using locally cached track CSVs until the teammate tracks arrive.
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

Do not treat cars parked at the **left kerb** as `stopped_vehicle` anomalies.
The team labeling guide incorporates the organizer's answer that a bus at a
bus stop **does count** if it stands in a live carriageway lane for >=10 s.
Do not blanket-exclude the far-kerb bus-stop region x=1350..1700,
y=380..520; first distinguish bay/kerb parking from a live lane.

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
This YOLO26n 960 px pipeline is a local GPU prototype. Hamid's EDA instead
used YOLO11m at 1280 px, which found more small pedestrians. The prototype
does not call a hosted model/API and is not yet wired into `solution.py` or
the final team repository. The rough `failure_to_yield` proximity rule must
be replaced with same-carriageway occupancy and front/rear crossing boundaries
from the team labeling guide before it can be treated as a reliable label.

Local ML environment: isolated `.venv`, PyTorch `2.14.0+cu130`, CUDA 13.0,
RTX 3050 Laptop GPU with 4 GB VRAM. Ultralytics `8.4.161` and local YOLO26n
weights (~5.5 MB) are present. The final judge dependency recipe is not yet
settled; do not assume this minimal dev environment is the final package.
Ultralytics states that its code and models use AGPL-3.0 or an Enterprise
license; check the canonical team's license/attribution plan before committing
the weights or packaging Ultralytics in the final submission.

## 7. Immediate next actions for Codex

1. Continue in the clean team checkout. Use `reports/eda/SUMMARY.md` v3,
   `scene_reference.json`, `registration.json`, and `docs/LABELING_GUIDE.md`
   to replace the staging prototype's unregistered scene geometry.
2. Use the OpenCV decode numbers above when setting the detector sampling rate.
3. Replace the weaker YOLO26n prototype with the strongest feasible local
   detector, benchmark it, and validate proposals separately from ground truth.
4. Extend and validate the automatic event-proposal pass; keep it distinct
   from reviewed dev labels and reject sidewalk/island false positives.
5. Use the calibrated signal timeline and merged direction field in team EDA
   for candidate `red_light` and `wrong_way` rules, with video verification.
6. When Hamid shares the YOLO11m track CSVs, inspect and integrate them;
   reports and direction fields are already available in the team checkout.
7. Use manual review only where it adds value to automatic proposals, with
   official event boundary conventions and no assumed frame tolerance.
8. Carry over only useful staging code into the canonical repo, using
   Jamoliddin's Git identity. Do not push to the canonical repo.
9. Implement Part B TTC/conflict risk logic after detector/runtime and
   registration geometry are verified.

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
- Hamid's merged direction field, registration, and four per-video EDA reports
  are present in the team checkout;
- kerb-parked vehicles are ignored, but a bus stopped in a live lane for >=10 s
  counts even at the bus stop;
- a pedestrian on the zebra is not jaywalking because their signal is red;
- ordinary red-light queues are not congestion;
- temporal IoU scoring has no frame tolerance;
- exposure changes around 52-67s make global brightness unreliable for fire/smoke;
- raw-video CPU decoding is a material part of the 3x runtime budget;
- Kaggle 4-vCPU FFmpeg-null decode measured about 22 fps; Kaggle OpenCV can
  decode the frame to BGR;
- Hamid's EDA for all four clips and an authenticated team Git checkout are
  available under `code`; YOLO11m track CSVs remain absent.
