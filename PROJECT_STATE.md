# WIUT Hackathon 2026 — Project State

_Last updated: 2026-09-24, Tashkent time_

This is the living handoff file for Codex and other development sessions. Keep it current. Do not duplicate facts that are already documented and still correct.

## 1. Ownership and working model

- **Jamoliddin:** ML Engineer / Model Lead.
- **Codex is the primary coding agent.** Repository Markdown is the canonical project memory. The user should not need to re-explain context in chat.
- After a meaningful discovery, benchmark, decision, implementation change, or teammate handoff, update this file or the relevant Markdown doc in the same session.
- Commits in the final team repository must use a human team member's configured Git identity. Do not add AI `Co-authored-by:` trailers, bot authors, or AI-owned PR/commit identities.

### Repository status

- `Jamoliddin-00i/WIUT-HACKATHON` is a **handoff / staging repository**.
- The real canonical team repository is `abdulhamid-n/saleh-traffic-events` and should be the active working repo locally.
- Do not merge the two repositories wholesale. Carry over only useful docs/scripts/code deliberately.
- Before working in the canonical repo, verify `git remote -v` and make sure `origin` points to the real team repository.
- AI assistants may edit/generate code locally, but commits pushed to the canonical team repo should be authored by the human contributor.

## 2. Confirmed organizer / video facts

Organizer/team confirmation received on 2026-09-24:

- `camera.md` has been removed. Infer lanes, stop lines, crossings, and directions from sample footage/EDA.
- Hidden test footage uses the same raw camera format/view as the sample videos.
- Raw video format:
  - `3840x2160`
  - H.264 High 4:2:2
  - 10-bit
  - about `140 Mbps`
  - `29.97 fps`
- Judge GPU is T4-class, but this 4:2:2 H.264 stream should be treated as a CPU-decode workload rather than relying on GPU hardware decode.
- Official Part B harness decodes every frame with `cv2.VideoCapture`, converts to BGR, then calls `RiskEstimator.step()` for every frame.
- Total time budget remains `<= 3 x video duration` for Part A + Part B.

### Organizer labeling clarifications received 2026-09-24

These override earlier assumptions and must be reflected in both manual labels and rule logic:

1. **Bus at bus stop:** a bus stopped at the bus stop **in a live lane for 10 seconds or more is `stopped_vehicle`**. Do not blanket-ignore the bus-stop region.
2. **Kerb parking:** cars parked at the kerb are **ignored completely**, even if they remain for the whole clip.
3. **Pedestrian on zebra on red:** this is **not `jaywalking`**. If a car drives through while the pedestrian is on the zebra, classify the vehicle event as **`failure_to_yield`**.
4. **Normal red-light queue:** this is **not `congestion`**. Only label congestion if the queue **fails to clear when the signal turns green** / remains abnormally stopped or crawling.
5. **Temporal scoring:** scoring uses **temporal IoU only, with no frame tolerance**. Exact start/end boundaries matter; do not pad labels expecting a tolerance window.

The labeling tool has an updated guide link at the top. Treat the latest organizer guide as authoritative when any older note conflicts.

## 3. Current decode benchmarks

### Local Windows FFmpeg benchmark

Machine:
- CPU: Intel Core i5-12500H
- 12 physical cores
- 16 logical processors

Input tested: `C3905.MP4`, 3840x2160, H.264 High 4:2:2 10-bit, ~140 Mbps, 29.97 fps.

Observed for first 600 frames (~20.02 s source):
- wall time: **5.713 s**
- ~**105 fps**
- ~**3.5x realtime**
- raw FFmpeg-null decode cost ~**0.285x video duration**

This is not representative of the actual harness path because `ffmpeg -f null` avoids OpenCV's BGR ndarray conversion/copy path.

### Local Windows OpenCV BGR decode benchmark

Measured with `scripts/benchmark_decode.py` on the first 600 frames of `C3905.MP4`, using `cv2.VideoCapture.read()` and retaining the BGR ndarray.

A CUDA PyTorch wheel download was active during these measurements, so repeat later for a precise final margin.

| CPU affinity / run | Wall time | Decode rate | Wall / source duration |
|---|---:|---:|---:|
| 16 logical CPUs, first run | 23.78 s | 25.23 fps | 1.19x |
| first 8 logical CPUs | 27.17 s | 22.09 fps | 1.36x |
| 16 logical CPUs, later run | 19.82 s | 30.28 fps | 0.99x |

Interpretation:
- Practical local OpenCV/BGR decode is roughly **22-30 fps**, about **1.0-1.36x video duration**.
- The 8-logical-CPU result is not equivalent to an 8-physical-core judge machine because the i5-12500H is hybrid and affinity may map P/E cores unevenly.
- Part B must decode every frame, so model inference and Part A must be aggressively sampled/cached.

### Kaggle Linux CPU benchmark

Dataset path:

```text
/kaggle/input/datasets/jamoliddintoirov/firstvid/C3905.MP4
```

Kaggle CPU allocation:
- x86_64 KVM VM
- Intel Xeon @ 2.20GHz
- 4 logical CPUs total
- 2 cores / socket, 2 threads / core

FFmpeg-null benchmark on first 600 frames:
- wall time: **27.072 s**
- ~**22 fps**
- reported speed: **0.74x realtime**
- decode cost: about **1.35x video duration**
- max RSS: ~619 MB

Do not linearly extrapolate this 4-vCPU VM to the judge's 8-core CPU.

### Kaggle OpenCV status

Kaggle OpenCV 4.13.0 with FFMPEG can open and BGR-decode the sample:

```text
exists: True
opencv: 4.13.0
opened: True
backend: FFMPEG
first read: True
shape: (2160, 3840, 3)
```

The earlier 0-frame result was a script/path/transient failure, not a codec limitation. The next useful Kaggle benchmark is a full 600-frame `cv2.VideoCapture(..., cv2.CAP_FFMPEG)` loop.

## 4. Hamid's C3905 EDA handoff

### Traffic light

- Readable signal location in original 4K coordinates: approximately `(2328, 780)`.
- Hamid reported signal state over time in `signal.png` and a moving-car correlation of 0.88.
- Practical red-light rule direction: signal is red + vehicle crosses the relevant stop line.

### Stop line / zebra region

For the near carriageway, cars reportedly wait around:
- `(1500, 870)`
- `(1764, 985)`

These points are just before the zebra crossing and should inform initial stop-line geometry. Verify exact line/polygon placement against frames before hard-coding.

### Wrong-way direction field

- Hamid has `direction_field.json` with lane/traffic directions per roughly 80 px cell.
- In the canonical repo, inspect teammate-generated EDA artifacts before recreating them.

### Stopped-vehicle handling

Current organizer-backed rule:
- **Kerb-parked cars are ignored completely**, including cars parked for the whole clip.
- **Do not ignore the bus-stop region globally.** A bus stopped there in a live lane for **>=10 s** is a valid `stopped_vehicle` event.
- Scene configuration may still encode kerb-parking ignore zones, but bus-stop logic must remain eligible for stopped-vehicle detection.

### Exposure jump

- Camera auto-exposure reportedly changes brightness by roughly 35% around 52-67 s.
- Do not use raw/global brightness alone for `fire_smoke`; use spatial/temporal/local cues robust to exposure changes.

### YOLO tracks

- Hamid planned/provides YOLO11m tracks for all sample videos on the server with a consistent CSV format.
- When available in the canonical repo, inspect schema once and reuse them rather than recomputing identical EDA.

## 5. Model / runtime decisions already settled

- Final inference must be offline and reproducible.
- No OpenAI/Gemini/Anthropic/hosted inference APIs in the traffic pipeline.
- Download/install open weights locally and run on the user's GPU with CUDA when available.
- User's local PyTorch detects an RTX 3050 with 4 GB VRAM.
- Final implementation must also run on a T4-class 16 GB GPU and stay within total weight/runtime limits.
- Kaggle may be used for T4-like testing or fine-tuning.
- A local Linux VM is not currently required.
- GitHub Actions are not required and should not be a dependency.
- Never modify organizer `run_submission.py` or `evaluate.py`.
- Do not commit the raw multi-GB sample videos.

## 6. Annotation/dev-label state

A local annotation script exists at `scripts/label_video.py` / is being used for sample-video labeling.

Local GUI note:
- `opencv-python-headless` cannot use `cv2.imshow()`.
- For the manual labeling workstation, use normal `opencv-python` in the local/dev environment.
- Keep final/judge dependencies headless if appropriate; ideally separate dev-only GUI dependencies from submission dependencies.

Ground-truth dev annotations should use the official evaluator-compatible structure and be checked against the current organizer guide.

### Manual labeling rules to keep in mind

- Be conservative and use the official event boundary conventions.
- Because scoring is temporal IoU with **no frame tolerance**, place boundaries as accurately as possible. Do not deliberately pad events.
- `jaywalking`: a pedestrian on a marked zebra is **not** jaywalking merely because the pedestrian signal is red.
- `failure_to_yield`: if a vehicle drives through while a pedestrian is on/entering the zebra under the organizer's clarified condition, label `failure_to_yield`.
- `stopped_vehicle`: a bus stopped in a live lane at the bus stop for >=10 s counts; kerb-parked cars do not.
- `congestion`: normal red-light queues do not count. A queue must fail to clear on green / remain abnormally stopped or crawling to qualify.
- Do not label a traffic-law violation unless the relevant signal/lane/crossing rule is supported by the footage/scene geometry and current organizer guide.

## 7. Immediate next actions for Codex

1. Treat `abdulhamid-n/saleh-traffic-events` as the canonical team repo and inspect existing teammate EDA before recreating anything.
2. Update any existing stopped-vehicle rule/config so the bus-stop area is not blanket-ignored; retain kerb-parking ignore behavior.
3. Update jaywalking/failure-to-yield logic to match the organizer clarification for pedestrians on the zebra on red.
4. Update congestion logic so ordinary red-light queues are excluded unless they fail to clear on green.
5. Ensure annotation/evaluation tooling does not assume any frame-tolerance window; temporal boundaries should be exact.
6. Continue manual labeling of sample videos using the updated organizer guide.
7. Run the full 600-frame Kaggle OpenCV/BGR benchmark when useful; later repeat local Windows OpenCV with background downloads stopped.
8. Use local CUDA for detector work when needed, but detector sanity tests are lower priority than correct labels/geometry when connectivity or model availability is limited.
9. Only after runtime/geometry are understood, proceed with detector/tracker/event implementation and Part B TTC/conflict-risk logic.

## 8. Performance principle

Raw-video decode is a first-order constraint. Optimize model work around unavoidable decode cost:

- do not run YOLO on every 29.97-fps frame unless benchmarks justify it;
- sample detector inference and reuse/interpolate tracks;
- avoid independently decoding the same video multiple times inside Part A;
- keep Part B causal and light per frame;
- benchmark the real `run_submission.py` path early, not just isolated model inference.

## 9. Things not to rediscover

Unless new evidence changes them:
- `camera.md` is gone;
- C3905 signal ROI around `(2328, 780)` has already been identified by teammate EDA;
- Hamid has a direction field for C3905;
- **kerb-parked cars are ignored**, but **the bus-stop area is not a global stopped-vehicle ignore zone**;
- bus stopped in a live lane at the bus stop for >=10 s counts as `stopped_vehicle`;
- pedestrian on the zebra on red is not `jaywalking`; a car driving through while the pedestrian is on it is `failure_to_yield` under the clarified guide;
- normal red-light queue is not `congestion`; failure to clear on green is the important condition;
- temporal event scoring has no frame tolerance, so exact timing matters;
- exposure changes around 52-67 s make global brightness unreliable for fire/smoke;
- raw-video CPU decoding is a material part of the 3x runtime budget;
- local OpenCV/BGR decode currently measures roughly 22-30 fps depending on affinity/system load;
- Kaggle 4-vCPU FFmpeg-null decode measured ~22 fps / 0.74x realtime;
- Kaggle OpenCV 4.13.0 + FFMPEG can open and BGR-decode C3905.
