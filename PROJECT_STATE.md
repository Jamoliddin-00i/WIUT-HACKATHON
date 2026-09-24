# WIUT Hackathon 2026 — Project State

_Last updated: 2026-09-24, Tashkent time_

This is the living handoff file for Codex and other development sessions. Keep it current. Do not duplicate facts that are already documented and still correct.

## 1. Ownership and working model

- **Jamoliddin:** ML Engineer / Model Lead.
- **Codex is the primary coding agent.** Repository Markdown is the canonical project memory. The user should not need to re-explain context in chat.
- Before work, sync `main`, read `AGENTS.md`, `README.md`, and this file, then inspect teammate artifacts already present.
- After a meaningful discovery, benchmark, decision, implementation change, or teammate handoff, update this file or the relevant Markdown doc in the same session.
- Commits from the user's machine should use the user's configured Git identity. Do not add AI `Co-authored-by:` trailers or intentionally set an AI/bot author.

### Repository coordination warning

Hamid said he is creating/using the project GitHub and has already referenced files under `reports/eda/C3905/`. At the moment this connected repository may not yet show those artifacts. **Before major implementation, verify that the local `origin` is the canonical team repository and that `main` contains Hamid's latest EDA.** Do not build a divergent copy.

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

### Immediate benchmark still required

Run the exact OpenCV path locally:

```python
import cv2, time
c = cv2.VideoCapture("C3905.MP4")
n = 0
t = time.time()
while n < 600:
    ok, f = c.read()
    if not ok:
        break
    n += 1
print(n / (time.time() - t), "fps")
```

Then repeat while limiting the process to roughly **8 CPU cores** using Windows Task Manager affinity (or another reliable process-affinity method). Record both results here.

Optional/secondary benchmark:
- Kaggle/Linux environment for a closer judge-like CPU/T4 setup.
- Record CPU model/core allocation, FFmpeg result, and OpenCV result.

## 4. Hamid's C3905 EDA handoff

Hamid reports EDA is under:

```text
reports/eda/C3905/
```

Pull/sync the canonical `main` and consume those artifacts instead of recreating them.

### Traffic light

- Readable signal location in original 4K coordinates: approximately **`(2328, 780)`**.
- Signal state over time is in `signal.png`.
- Reported correlation between signal state and moving cars: **0.88**.
- Practical `red_light` rule direction: **signal is red + vehicle crosses the relevant stop line**.

### Stop line / zebra region

For the near carriageway, cars reportedly wait around:
- **`(1500, 870)`**
- **`(1764, 985)`**

These are just before the zebra crossing and should inform the initial stop-line geometry. Verify exact line/polygon placement against frames before hard-coding.

### Wrong-way direction field

- `direction_field.json` contains lane/traffic directions per roughly **80 px cell**.
- Use this as the baseline geometry for `wrong_way` instead of inventing lane direction manually.

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
- When available, inspect schema once and integrate/reuse these tracks rather than recomputing identical EDA unnecessarily.

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

## 7. Immediate next actions for Codex

1. **Verify repository/remote first.** Pull canonical `main` and confirm Hamid's `reports/eda/C3905/` exists. If it does not, resolve the repo/remote mismatch before significant coding.
2. Run the **exact OpenCV 600-frame decode benchmark** on C3905 and record fps here.
3. Repeat with the process limited to ~8 CPU cores and record fps here.
4. Inspect Hamid's C3905 EDA artifacts and convert stable geometry into a clean scene config, preferably normalized coordinates where practical.
5. Incorporate `direction_field.json`, signal ROI/state logic, stop-line geometry, and stopped-vehicle ignore regions into rule design.
6. Get/inspect Hamid's YOLO11m track CSV schema when available.
7. Continue manual dev labeling of the sample videos using official event boundary conventions.
8. Only after runtime/geometry are understood, proceed with detector/tracker/event implementation and Part B TTC/conflict risk logic.

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
- a direction field already exists for C3905;
- known parked/bus-stop zones need to be ignored for stopped-vehicle logic;
- exposure changes around 52-67s make global brightness unreliable for fire/smoke;
- raw-video CPU decoding is a material part of the 3x runtime budget.
