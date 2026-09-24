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

## 3. Current decode benchmarks

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

Important: this is **not** representative of the actual harness path because `ffmpeg -f null` avoids OpenCV's BGR ndarray conversion/copy path.

### Local Windows OpenCV BGR decode benchmark

Measured 2026-09-24 with `scripts/benchmark_decode.py` on the first 600 frames of `C3905.MP4`. This uses `cv2.VideoCapture.read()` and retains the BGR ndarray, matching the organizer harness much more closely.

A CUDA PyTorch wheel download was active during these measurements, so repeat later if a precise final runtime margin is needed.

| CPU affinity / run | Wall time | Decode rate | Wall / source duration |
|---|---:|---:|---:|
| 16 logical CPUs (default), first run | 23.78 s | 25.23 fps | 1.19x |
| first 8 logical CPUs | 27.17 s | 22.09 fps | 1.36x |
| 16 logical CPUs, user's exact `time.time()` script, later run | 19.82 s | 30.28 fps | 0.99x |

Interpretation:
- The real OpenCV/BGR path is dramatically slower than the earlier `ffmpeg -f null` result.
- On this laptop the practical decode rate is roughly **22-30 fps** across current runs, approximately **1.0-1.36x video duration**.
- The 8-logical-CPU result is **not equivalent to an 8-physical-core judge machine**. The i5-12500H is a hybrid CPU, and Windows affinity numbering may map those logical CPUs unevenly across P/E cores. Treat the 22.09 fps figure as a stress/reference point, not a direct judge prediction.
- Because the Part B harness must decode every frame, decode alone can plausibly consume around one video-duration unit or more. Model inference and Part A must therefore be aggressively sampled/cached.

### Kaggle Linux CPU benchmark

Dataset path used:

```text
/kaggle/input/datasets/jamoliddintoirov/firstvid/C3905.MP4
```

Kaggle CPU allocation:
- x86_64 KVM VM
- `Intel(R) Xeon(R) CPU @ 2.20GHz`
- **4 logical CPUs total**
- 2 cores / socket, 2 threads / core

FFmpeg version: Ubuntu FFmpeg 4.4.2.

Command:

```bash
ffmpeg -benchmark -i "/kaggle/input/datasets/jamoliddintoirov/firstvid/C3905.MP4" -frames:v 600 -f null -
```

Observed:
- 600 frames = ~20.02 s source video
- wall time: **27.072 s**
- ~**22 fps**
- reported speed: **0.74x realtime**
- decode cost: about **1.35x video duration**
- max RSS: ~619 MB

Interpretation:
- 4-vCPU Kaggle is substantially slower than the local i5-12500H on raw FFmpeg-null decode.
- This is useful evidence that CPU decode can dominate runtime on weaker CPUs.
- Do **not** linearly extrapolate to the judge's 8-core machine; scaling may not be linear.

### Kaggle OpenCV status

The earlier 0-frame OpenCV run was **not a codec/backend limitation**. A direct diagnostic on the same Kaggle notebook now succeeds:

```text
exists: True
opencv: 4.13.0
opened: True
backend: FFMPEG
first read: True
shape: (2160, 3840, 3)
```

Therefore Kaggle OpenCV 4.13.0 with its FFMPEG backend can open and BGR-decode this exact 4K 10-bit 4:2:2 H.264 sample. The earlier 0-frame result should be treated as a script/path/transient failure, not a performance result.

**Next required Kaggle benchmark:** run the full 600-frame `cv2.VideoCapture(..., cv2.CAP_FFMPEG)` loop and record wall time/fps. That will be the closest currently available Linux BGR-decode reference, although Kaggle only exposes 4 logical CPUs and therefore still does not match the judge's stated 8-core CPU.

Suggested robust cell:

```python
import cv2, time

p = "/kaggle/input/datasets/jamoliddintoirov/firstvid/C3905.MP4"
cap = cv2.VideoCapture(p, cv2.CAP_FFMPEG)
assert cap.isOpened(), "VideoCapture failed to open"

n = 0
t0 = time.perf_counter()
while n < 600:
    ok, frame = cap.read()
    if not ok:
        print("read failed at", n)
        break
    n += 1
elapsed = time.perf_counter() - t0
cap.release()

fps = n / elapsed if elapsed > 0 else 0.0
print("frames:", n)
print("elapsed:", elapsed)
print("decode fps:", fps)
print("realtime factor:", fps / 29.97 if fps else 0.0)
print("wall/source duration:", elapsed / (n / 29.97) if n else None)
```

## 4. Hamid's C3905 EDA handoff

Important availability note:
- Hamid's full EDA artifacts are **on Hamid's side only** right now.
- We do **not** currently have his `reports/eda/C3905/` directory, `signal.png`, or `direction_field.json` locally/in this staging repo.
- The facts he sent in chat are considered sufficient to guide current rule design. Do not block progress waiting for the full EDA files.
- If Hamid later shares those artifacts or track CSVs, consume them then.

### Traffic light

- Readable signal location in original 4K coordinates: approximately **`(2328, 780)`**.
- Hamid reports signal state over time exists in `signal.png`, but that file is not currently available to us.
- Reported correlation between signal state and moving cars: **0.88**.
- Practical `red_light` rule direction: **signal is red + vehicle crosses the relevant stop line**.

### Stop line / zebra region

For the near carriageway, cars reportedly wait around:
- **`(1500, 870)`**
- **`(1764, 985)`**

These are just before the zebra crossing and should inform the initial stop-line geometry. Verify exact line/polygon placement against local frames before hard-coding.

### Wrong-way direction field

- Hamid reports a `direction_field.json` with lane/traffic directions per roughly **80 px cell**.
- The file itself is not currently available to Jamoliddin/Codex.
- Until it is shared, do not invent detailed per-cell directions from memory. Use only geometry that can be verified from local footage.

### Stopped-vehicle ignore zones

Do not treat the following as `stopped_vehicle` anomalies:
- three cars parked for the whole clip at the **left edge**;
- far-kerb bus-stop region approximately **x = 1350..1700, y = 380..520**.

These zones should become explicit ignore masks/regions in scene configuration, not scattered conditionals.

### Exposure jump

- Camera auto-exposure reportedly changes brightness by roughly **35% around 52-67 s**.
- **Do not use raw/global brightness alone for `fire_smoke`.** Any smoke/fire detector should use spatial/temporal/local cues robust to exposure changes.

### YOLO tracks

- Hamid plans/provides **YOLO11m tracks for all sample videos** on the server.
- CSV format will be consistent across videos.
- These CSVs are not yet available locally.
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
- For the manual labeling workstation, use normal `opencv-python` in the local/dev environment.
- Keep final/judge dependencies headless if appropriate; ideally separate dev-only GUI dependencies from submission dependencies.

Ground-truth dev annotations should ultimately be saved in the official evaluator-compatible shape, e.g. `dev_labels.json`, then tested with `run_submission.py` and `evaluate.py`.

Manual labeling principle already established:
- be conservative;
- use official event start/end conventions;
- do not label a traffic-law violation unless the relevant signal/lane/crossing rule is actually supported by the footage/scene geometry.

## 7. Immediate next actions for Codex

1. Continue local work from the available sample videos; do **not** wait for Hamid's full EDA directory.
2. Run the **full 600-frame Kaggle OpenCV/BGR benchmark** now that `CAP_FFMPEG` is confirmed to work, and record the result here.
3. Later repeat the local Windows OpenCV benchmark when no large download/background load is running, to tighten the runtime estimate.
4. Use the numerical EDA facts already handed over (signal ROI, stop positions, ignore zones, exposure warning) to shape scene/rule code only where they are sufficient.
5. Build/verify any missing scene geometry directly from local video frames rather than pretending unavailable EDA files exist.
6. When Hamid shares `direction_field.json` and YOLO11m track CSVs, inspect and integrate them.
7. Continue manual dev labeling of the sample videos using official event boundary conventions.
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
- local OpenCV/BGR decode currently measures roughly 22-30 fps depending on affinity/system load, around 1.0-1.36x source duration;
- Kaggle 4-vCPU FFmpeg-null decode measured only ~22 fps / 0.74x realtime on C3905;
- Kaggle OpenCV 4.13.0 + FFMPEG **can** open and BGR-decode C3905; the earlier 0-frame run was not a codec limitation;
- Hamid's full EDA is not in our repo, and current work should proceed from the facts he already provided.
