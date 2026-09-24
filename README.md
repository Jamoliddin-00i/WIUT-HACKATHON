# WIUT Hackathon 2026 — Toyota Traffic Event Detection

> **Codex / AI agent handoff:** read this README completely before changing code. This repository is the official hackathon workspace. The goal is a strong, reproducible, offline submission, not a toy demo.

## 0. Current project state

- Repository: `Jamoliddin-00i/WIUT-HACKATHON`
- Starter kit has already been copied into this repo:
  - `solution.py`
  - `run_submission.py`
  - `evaluate.py`
  - `requirements.txt`
  - `examples/ground_truth.json`
  - `examples/predictions.json`
- **Do not modify `run_submission.py` or `evaluate.py`.** They are organizer files and the judges use the same interface.
- No real detector/tracker/event logic has been implemented yet. `solution.py` is still the starter interface and is where the submission hooks live.
- The user is downloading the four official sample videos locally because the originals are huge and should not be committed to GitHub.
- Official sample video names:
  - `C3896.MP4` (~6.24 GB)
  - `C3897.MP4` (~5.84 GB)
  - `C3902.MP4` (~5.84 GB)
  - `C3905.MP4` (~2.35 GB)
- These videos come from the **same fixed CCTV camera and angle** as the hidden test set.
- The organizer-described `camera.md` was not present in the resources we received. We will create our own scene configuration by inspecting the footage.
- GitHub Actions quota is currently exhausted. **Do not depend on GitHub Actions.** Run tests locally on the user's PC.
- Deadline: **27 September 2026, 23:59 Tashkent time**. Reliability and a working submission matter more than over-engineering.

## 1. Your first job when running locally

Do not immediately start writing a huge model pipeline. First inspect the local machine and sample footage.

1. Confirm the repository root and Python environment.
2. Find the local sample videos. They may be inside this repo in a gitignored folder such as `videos/` or `samples/`, or in a nearby folder such as `D:\wiut hackathon\videos`. If they are not obvious, ask the user for the exact path.
3. **Never commit or upload the original MP4 files to GitHub.**
4. Add/verify `.gitignore` entries for large/local assets:

```gitignore
videos/
samples/*.MP4
samples/*.mp4
*.MP4
*.mp4
.venv/
__pycache__/
*.pyc
```

5. Inspect each video with OpenCV/ffprobe and record:
   - resolution
   - fps
   - frame count
   - duration
   - codec if useful
6. Extract representative frames at sensible intervals, preferably into a local ignored debug folder.
7. Inspect the fixed road geometry and create a scene config rather than scattering coordinates through code.

Suggested structure:

```text
src/
  detector.py
  tracker.py
  scene.py
  trajectories.py
  event_rules.py
  risk.py
  postprocess.py
  video_utils.py
  visualize.py
config/
  scene.json        # or YAML; fixed-camera polygons/lines/directions
scripts/
  inspect_videos.py
  render_debug.py
  make_dev_labels.py
weights/
solution.py
```

Keep `solution.py` thin: it should call well-structured code in `src/`.

## 2. What we are building

The system receives a fixed-road-camera `.mp4` and must return traffic events as:

```python
[
    [start_sec, end_sec, "accident"],
    [start_sec, end_sec, "wrong_way"],
]
```

It also optionally outputs a **causal accident risk score per frame** through `RiskEstimator`. Causal means the estimator may use only the current and past frames, never future frames or Part A results computed using future frames.

The intended practical architecture is:

```text
video
  -> frame sampling
  -> pretrained object detector (YOLO or similarly lightweight open-weight detector)
  -> tracker (ByteTrack or equivalent)
  -> per-object trajectories
  -> fixed-camera scene geometry
  -> rule-based event detectors
  -> learned/heuristic interaction logic for accident + near_miss
  -> temporal post-processing
  -> [[start, end, class], ...]
```

For Part B:

```text
past/current tracks
  -> relative motion / conflict points / TTC / braking / trajectory crossing
  -> calibrated risk score in [0, 1]
```

### Important strategic principle

This is **one fixed camera**. Exploit that aggressively.

We are not trying to build a universal autonomous-driving model. Manually define scene geometry once:

- road polygon
- lane polygons
- lane legal directions
- stop lines
- solid lane markings
- pedestrian crossing polygons
- intersection polygon
- traffic-light ROI(s), if visible
- legal/illegal turn relationships if inferable

Then many classes become deterministic trajectory rules, which should be more reliable than asking a giant VLM to reason about every frame.

## 3. Official event classes

Use these IDs exactly:

```python
CLASSES = [
    "accident",
    "near_miss",
    "red_light",
    "wrong_way",
    "illegal_u_turn",
    "stopped_vehicle",
    "jaywalking",
    "failure_to_yield",
    "illegal_turn",
    "solid_line_crossing",
    "stop_line",
    "congestion",
    "road_obstacle",
    "fire_smoke",
]
```

Definitions and annotation boundaries:

| ID | Meaning | Start | End |
|---|---|---|---|
| `accident` | Collision between road users or road user and fixed object | first visible contact | involved objects stop moving or leave frame |
| `near_miss` | Sharp braking/swerving to avoid collision, no contact | evasive action starts | road users clear each other |
| `red_light` | Vehicle crosses stop line while signal is red | vehicle front crosses stop line | vehicle leaves intersection/frame |
| `wrong_way` | Vehicle moves against lane direction / into oncoming lane | enters opposing lane | returns to correct lane or leaves frame |
| `illegal_u_turn` | U-turn where prohibited | starts turning | completes turn |
| `stopped_vehicle` | Vehicle stationary on carriageway >=10 s, not a normal signal queue | vehicle stops | moves again or is removed |
| `jaywalking` | Pedestrian on carriageway outside a crossing | steps onto road | leaves road |
| `failure_to_yield` | Vehicle drives through crossing while pedestrian is on/entering it | vehicle enters crossing | vehicle leaves crossing |
| `illegal_turn` | Turn from wrong lane or prohibited direction | turn begins | turn completes |
| `solid_line_crossing` | Vehicle crosses solid road marking | wheel crosses line | vehicle fully enters new lane |
| `stop_line` | Vehicle stops beyond stop line on red without entering intersection | vehicle stops | signal turns green |
| `congestion` | Standstill/crawling across all lanes of a direction | queue stops/crawls | queue clears |
| `road_obstacle` | Debris, animal, fallen object on carriageway | obstacle appears | removed |
| `fire_smoke` | Visible fire/smoke from vehicle or roadway | first visible smoke/fire | clears or video ends |

Rules:

- One event = one contiguous segment with one class.
- Different classes may overlap.
- Same-class segments may **not** overlap.
- If two same-class events happen at once, return one segment covering both, matching organizer annotation convention.
- A video may contain no events: return `[]`.
- If an event continues past the video end, `end_sec = duration`.
- Do not add new class IDs. Speeding is intentionally excluded.

## 4. Starter-kit interface

`solution.py` must expose exactly:

```python
CLASSES = ["accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
           "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
           "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke"]


def detect_events(video_path: str) -> list[list]:
    """Part A: return [[start_sec, end_sec, label], ...] for one MP4."""


class RiskEstimator:
    def reset(self, meta: dict) -> None:
        ...

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        ...
```

`RiskEstimator.reset(meta)` receives:

```python
{
    "video_id": ...,
    "fps": ...,
    "width": ...,
    "height": ...,
    "n_frames": ...,
}
```

`step(frame, t_sec)` receives a BGR `uint8` OpenCV frame and must return a float in `[0, 1]`.

`step` is called for **every frame**. It is allowed to skip expensive inference internally and return the last score on skipped frames.

## 5. Non-negotiable competition rules

### Offline inference

The judges run the repository with **no internet**.

Allowed at inference:
- shipped open-weight models
- local Python code
- local model weights

Not allowed at inference:
- OpenAI API
- Gemini API
- Anthropic API
- any hosted/paid inference API
- downloading required weights during the actual offline run

AI coding tools may be used to develop the code, website, report, etc.

### Part B causality

`RiskEstimator.step()` may use only frames it has already received.

**Never:**
- open the video file from inside `RiskEstimator`
- look ahead
- reuse Part A outputs produced from future frames

Part A may use Part B's risk curve. Part B may not use non-causal Part A information.

### Determinism

Fix seeds. Two runs on the same machine should produce the same predictions up to tiny floating-point differences.

### Runtime

Organizer hardware target:
- 1 NVIDIA GPU, about T4 class
- 16 GB VRAM
- 8 CPU cores
- 32 GB RAM
- Python 3.10+

Time budget per video for **Part A + Part B combined**:

```text
<= 3 x video duration in wall-clock time
```

If a video exceeds the budget, it scores as empty.

Design for margin, not barely passing.

### Weights

Total shipped/downloaded model weights must be <= **5 GB**.

## 6. Recommended implementation order

### Phase 1 — Inspect footage and establish baseline

Before tuning event logic:

1. inspect all four videos
2. save video metadata
3. extract representative frames
4. identify camera geometry
5. run pretrained detector on representative frames
6. test tracking stability
7. measure runtime early

Use a reasonably strong pretrained detector that fits T4-class inference. Do not train from scratch unless evidence later says it is needed.

### Phase 2 — Tracking and scene geometry

Implement robust track records containing at least:

```text
track id
object class
bbox history
bottom-center / contact-point history
timestamps
velocity estimate
direction estimate
stationary duration
scene zones visited
```

Use bottom-center or another perspective-aware anchor for road geometry rather than bbox center where appropriate.

Create `config/scene.json` (or YAML) with normalized coordinates where possible so geometry is explicit and editable.

### Phase 3 — Easier rule-based event classes first

Prioritize classes where fixed-camera geometry gives high confidence:

1. `wrong_way`
2. `stopped_vehicle`
3. `jaywalking`
4. `solid_line_crossing`
5. `red_light` if signal state is visible/reliably detectable
6. `stop_line`
7. `failure_to_yield`
8. `illegal_turn`
9. `illegal_u_turn`
10. `congestion`

Do not emit a class merely because a weak heuristic can guess it. Macro scoring punishes false prediction classes.

### Phase 4 — Accident and near-miss

Use track interactions plus motion cues first:

- rapidly decreasing distance
- trajectory intersection/conflict point
- relative speed
- sudden deceleration
- abrupt heading change
- contact/overlap evidence
- post-contact stopping or trajectory discontinuity

A lightweight learned temporal classifier may be added later if useful, using public datasets and/or self-annotated samples. Do not start by training a giant model from scratch.

### Phase 5 — Part B accident anticipation

Implement a simple causal baseline as soon as stable tracks exist, not at the last minute.

Useful features:

- time-to-collision (TTC)
- distance to predicted trajectory conflict point
- closing speed
- sudden braking
- wrong-way trajectory
- pedestrian entering roadway in vehicle path
- red-light conflict trajectories

Risk must be calibrated because `0.5` is the organizer's alarm threshold.

### Phase 6 — Temporal post-processing

Boundary quality matters heavily. Implement:

- merge fragments separated by short gaps
- suppress tiny one-frame/sub-second noise where inappropriate
- enforce non-overlap within each class
- clamp times to `[0, duration]`
- exact class IDs
- conservative hysteresis for event start/end

Do not report a one-minute interval around a five-second event. Temporal IoU is part of the official score.

## 7. Development labels are essential

The organizer gives sample videos with **no labels**. We should manually annotate them ourselves using the official conventions.

Create something like:

```text
dev_labels.json
```

with the same shape as `examples/ground_truth.json`.

Then repeatedly run:

```bash
python run_submission.py --videos <local_sample_folder> --out predictions_samples.json --team <team-name>
python evaluate.py --pred predictions_samples.json --gt dev_labels.json --per-video
python evaluate.py --pred predictions_samples.json --validate-only
```

Without a self-labeled dev set, threshold tuning is guesswork.

## 8. Official scoring

### Part A — Event detection

For each class and each temporal IoU threshold:

```text
0.3, 0.5, 0.7
```

predicted and ground-truth segments of the same class/video are greedily matched by descending IoU.

Then F1 is computed per class and threshold, pooled across videos.

```text
Score A = mean over classes of mean(F1@0.3, F1@0.5, F1@0.7)
```

Important consequence: **boundary precision matters**.

Also, if we predict a class that never occurs in the hidden ground truth, that class is added to the evaluated set and scores 0. So avoid speculative rare-class predictions.

### Part B — Accident anticipation

Only `accident` events matter.

Constants:

```text
H = 5 s prediction horizon
W = 10 s alarm matching window
threshold theta = 0.5
```

A frame at time `t` is positive if an accident starts at `s` and:

```text
s - 5 <= t < s
```

Frames inside accidents and around near misses are ignored according to `evaluate.py`.

Score B:

```text
0.4 * chance-normalized AP
+ 0.4 * alarm F1
+ 0.2 * (mean TTA / 10)
```

A constant risk of 1.0 is not a useful hack and scores approximately zero.

### Overall model and elimination score

```text
Model score M = 0.7 * Score A + 0.3 * Score B

Elimination score =
0.60 * Model
+ 0.25 * Website
+ 0.15 * Code quality
```

The model is the user's primary responsibility. Website/product work may be handled separately, but our code should expose artifacts useful for the website: event JSON, annotated render, trajectories, risk curve, EDA outputs.

## 9. Efficiency guidance

Do not run expensive detector inference on every 25-fps frame unless measurements prove it is affordable and useful.

Likely baseline:

```text
detector at ~5-12 fps
tracker/interpolation between detector frames
```

Measure instead of guessing.

Reuse work. Ideally a video should not be independently decoded and fully detected multiple times just because Part A and visualization need the same tracks.

For final `run_submission.py`, respect its interface and time accounting. Optimize within our implementation, not by changing organizer files.

## 10. Local development and Git policy

### Large files

Do **not** commit:
- original MP4s
- generated debug videos unless intentionally small and useful
- temporary frames
- large caches
- arbitrary downloaded datasets

Weights may be committed only if reasonable and compliant with GitHub limits; otherwise use the allowed `weights/download.sh` approach and document it. Organizers can run that once with internet before offline evaluation.

### GitHub Actions

Do not add a dependency on Actions. The account's Actions quota is currently exhausted.

### Commits

Make small, descriptive commits after meaningful milestones. Do not rewrite organizer files. Before any large refactor, preserve a working checkpoint.

## 11. Immediate checklist for Codex

When you are opened on the user's local clone, do this in order:

- [ ] Confirm current branch and clean/dirty git state.
- [ ] Verify `run_submission.py` and `evaluate.py` are untouched organizer files.
- [ ] Add `.gitignore` for videos, env, caches, and debug outputs if absent.
- [ ] Locate the local sample MP4s; ask the user only if path cannot be determined safely.
- [ ] Create a virtual environment or use the user's preferred existing Python environment.
- [ ] Inspect video metadata.
- [ ] Extract a modest set of representative frames locally.
- [ ] Build an initial scene map from footage.
- [ ] Add a pretrained open-weight detector and tracker.
- [ ] Produce a debug render with IDs and trajectories on at least one short clip.
- [ ] Measure runtime.
- [ ] Implement the first high-confidence event rule(s).
- [ ] Keep `solution.py` compliant with the exact organizer interface.
- [ ] Add a causal TTC-based `RiskEstimator` baseline once tracking works.
- [ ] Create tooling for manual dev labels and repeated `evaluate.py` runs.
- [ ] Run `python evaluate.py --pred <file> --validate-only` before every submission-style checkpoint.

### Working style

Explain the plan briefly before making broad architectural changes, but do not stop for permission on every safe edit. Inspect first, make grounded changes, run tests, and report exact results. If a decision depends on the actual camera geometry, inspect the footage instead of inventing coordinates.

---

# Original starter-kit reference

Traffic events from a fixed road camera: **detect** them as time segments (`[start_sec, end_sec, label]`) and, as a bonus, **anticipate** accidents with a causal risk score.

```text
solution.py          <- submission interface (CLASSES, detect_events, RiskEstimator)
run_submission.py    <- organizers' harness: folder of videos -> predictions.json   (DO NOT MODIFY)
evaluate.py          <- format check + official metric                              (DO NOT MODIFY)
examples/            <- ground_truth.json and predictions.json in exact format
requirements.txt     <- starter dependencies; add project dependencies as needed
```

## Quickstart

```bash
pip install -r requirements.txt
python run_submission.py --videos samples --out predictions_samples.json --team <your-team>
python evaluate.py --pred predictions_samples.json --gt my_labels.json --per-video
python evaluate.py --pred predictions_samples.json --validate-only
```

## Final organizer-style run

```bash
pip install -r requirements.txt
python run_submission.py --videos /data/test --out predictions.json
python evaluate.py --pred predictions.json --gt ground_truth.json
```

A crash or timeout for a video is effectively scored as an empty prediction for that video, so robustness is part of model quality.

## Automatic sample annotation (development)

See [docs/auto_proposals.md](docs/auto_proposals.md) for the local CUDA
detector/tracker and scene-rule tool. It writes `auto_proposals.json` separately
from reviewed `dev_labels.json`. This development tool is not yet wired into
the organizer submission interface in `solution.py`.
