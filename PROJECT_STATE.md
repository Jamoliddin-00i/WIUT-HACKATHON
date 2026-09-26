# Current project state

Updated 2026-09-26. This is the primary handoff file for the next session.

User instruction (2026-09-26): commit our work in `Jamoliddin-00i/WIUT-HACKATHON`, not Hamid's Salen repo. This working snapshot was copied from local Salen commit `6a3b92e`. Source implementation and supporting EDA JSON/CSV metadata are included; large generated EDA images, raw tracks, debug outputs and model weights remain local in the original checkout. Earlier benchmark paths below refer to that checkout; rerunning here creates new local outputs.

Transfer verification: all 23 focused tests passed from WIUT-HACKATHON after
copying. Organizer `evaluate.py` and `run_submission.py` were preserved; their
normalized source matches the tested Salen copies. No heavy inference rerun was
needed for this file transfer. The existing empty local `asd` file is unrelated
and is left untracked. Model weights are available locally but excluded from Git.

## Active priority: Part B (user changed scope)

The user paused Part A development with about 30 hours left, and asked to
conserve limited Codex usage. Keep work in focused implementation/test batches;
do not resume Part A tuning unless requested. Its existing detector remains
connected because the official runtime budget covers both parts.

- `src/risk.py` now implements the Part B baseline; `solution.RiskEstimator`
  delegates to it. It no longer returns constant zero.
- Uses the SAME local YOLO26x at 2560 px / 2 Hz with a separate causal ByteTrack
  state. Every input frame gets a score; scores are held between sampled frames.
- Fits motion from the preceding 1.6 seconds (at least 3 observations), estimates
  contact within 5 seconds using approximate ground-footprint ellipses, requires
  persistent pair evidence, and smooths risk. Discard implausible ID jumps,
  clipped objects, person-in-car detections and already-overlapping footprints.
- Reads no video file, future frame, Part A result, saved sample tracks, manual
  labels, stored per-video homography, or sample signal timeline. State resets
  between videos. Shared camera translation cancels in relative velocity;
  rotation, perspective, occlusion and detector jitter remain limitations.
- This is a motion heuristic, NOT an accident-trained model or calibrated
  probability. It misses fixed-object crashes and can confuse close passing
  with collision risk. Do not advertise accident accuracy from synthetic tests.
- 23 focused tests pass, including 7 new risk tests: approaching vs separating/
  parallel traffic, future contact, history-only prefix consistency, ID jumps,
  missing-track decay, sampling and reset.
- Full C3905 run with the NEW Part B: 14 Part A events, 3,825 risk samples;
  A=109.1 s, B=146.4 s, total **255.5 s / 382.9 s budget**, no harness errors.
  Output: `debug/part_b_baseline_C3905.json`. Risk ranged 0-0.9386. Alarm starts
  at 3.5035, 11.011, 55.055, 97.097 and 121.6215 s: five false alarms relative
  to the current labels, all outside the ignored near-miss window. This confirms
  execution and exposes calibration problems; it does not establish accuracy.
- The stricter full-run wrapper FAILED because nine background socket attempts
  were blocked, although inference completed successfully without connections.
  Subsequently `src/__init__.py` sets YOLO_OFFLINE=true and
  YOLO_AUTOINSTALL=false before Ultralytics import. A fresh-process smoke test
  of public Part B initialization and real GPU inference passed with both DNS
  resolution and socket connections blocked and zero attempts. The full video
  has NOT been rerun after this environment-only fix. `tools/verify_offline.py`
  now checks DNS attempts too. Do not claim the stricter full-video check passed.
- Asked whether the user has accident clips with impact timestamps; no answer
  yet. Current four clips have no accident labels, so they assess false alarms
  and runtime only. Near-miss windows are ignored in the official Part B metric.
- Next: obtain labeled accident positives and non-accident controls, separate
  clips for tuning/evaluation, measure warning time and false alarms. Use data
  to decide whether a trained temporal head is justified in the remaining time.
  Preserve the existing working baseline while doing this.

Readiness: functional prototype, not submission-ready. Part A currently emits
3 of 14 classes; development Score A is 0.1521. Part B accuracy is unmeasured.
README's elimination weighting is 60% model / 25% website / 15% code; the
official evaluator weights A/B 70%/30% when accidents are present, giving
42%/18% of the overall score. With no test accidents the model score is A alone.
Website readiness has not been audited here. Do not present checklist completion
or synthetic tests as a percentage of competitive accuracy.

## Workspace and user preferences

- Active work/commit repo: `D:\wiut hackathon\code\WIUT-HACKATHON`, origin
  `https://github.com/Jamoliddin-00i/WIUT-HACKATHON.git`.
- Team source checkout: `D:\wiut hackathon\code\salen-traffic-events-main`;
  do not push changes to its Hamid/Salen origin.
- Working Python here: `.venv\Scripts\python.exe`.
- Videos: `D:\wiut hackathon\videos`. All four originals are local.
- Do not push to the team repository. Local commits use the user's configured
  human Git identity. Never add AI co-author trailers.
- User wants implementation to continue without repeated offers or permission
  questions. Explain results in simple words; do not make reports the answer.
- Use the large local model; no downgrade to a weaker detector. Traffic model
  inference must run locally on CUDA and work offline for the judges.
- Update this file at the end of every working session. Keep facts current.

## Task and constraints

- Part A: temporal traffic event intervals. Part B: causal accident risk.
- Judges: T4-class GPU, about 16 GB VRAM, <=5 GB weights, total wall time
  <=3 times video duration. Local GPU is an RTX 3050 Laptop with 4 GB VRAM.
- Original videos are 4K H.264 4:2:2 10-bit, approximately 29.97 fps. CPU decode
  is a significant cost; the harness also decodes every frame for Part B.
- Read `tools/scene/CONTRACT.md` before using EDA inputs. The known vehicle
  signal head D is readable, but which movement it controls remains UNKNOWN.
  Its correlation with traffic flow does not prove legal signal applicability.
- Manual zebra-edge convention: a person slightly beside the zebra while
  following it is exempt. Judge feet/path, not upper-body projection.

## Labels and models

- `labels/user_truth_2026-09-26.json`: the user's complete 40 annotations,
  preserved as supplied: C3905=10, C3896=14, C3897=8, C3902=8.
- Class counts: 16 stopped vehicles, 13 jaywalking, 6 congestion, 2 solid-line
  crossings, 1 red light, 1 near miss, 1 failure to yield.
- These are manual development labels, not independently adjudicated or held
  out from tuning. Do not overwrite them with predictions or hard-code their
  timestamps in inference. No accidents are labeled, so Part B cannot be scored.
- `labels/user_proposal_review_2026-09-26.json` preserves the user's latest
  review of 20 OLD C3905 proposals: 12 rejected, 7 confirmed with better manual
  timing, and 1 fragment to merge. The other three pasted videos were unreviewed.
  Comparing with current `auto_proposals.json`, 7 rejected intervals have no
  same-class overlap; 5 still overlap: yielding around 11, 40 and 110 seconds,
  jaywalking around 34 and 84 seconds. This is temporal matching, not proof of
  actor identity or that the current output has no new false alarms. Comparison
  details are in `debug/user_review_comparison.json`. Do not overwrite truth.
- Detector: official `weights/yolo26x.pt`, 118,667,365 bytes. Verified SHA-256:
  `9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92`.
- Optional diagnostic `weights/yolo26x-pose.pt` is local. The ankle experiment
  found ankles on pedestrians AND a motorcycle rider. Pose is not in runtime.
- No weights were retrained. Improvements so far are geometry/tracking rules.
- Observed environment: PyTorch 2.14.0+cu130, CUDA 13.0, Ultralytics 8.4.161,
  OpenCV 5.0.0. The base environment lacks pandas and requests; core inference
  currently avoids the pandas-dependent offline signal accessor.

## Working implementation

- `solution.detect_events` calls `src/pipeline.py`; it no longer returns
  the empty stub. `RiskEstimator` runs the causal baseline described above.
- Runtime runs YOLO26x/ByteTrack, saves fresh tracks to a temporary directory,
  and estimates registration from input frames using Hamid's SIFT registration.
  Late frames reduce initial camera shake; short clips use their first frame.
- Runtime never reads user labels, sample predictions, cached sample tracks,
  stored per-video registration, or sample signal timelines. It uses the shared
  reference image and provisional hand-drawn zones.
- `tools/auto_label_video.py` remains the development/replay CLI; its default
  sample-video geometry uses the stored homographies for fair comparisons.
- Pedestrians: vehicle-occupant suppression, box-bottom foot approximation,
  clipped-box rejection, motion over a two-second window, small allowance when
  following crossing/kerb edges, and protection against jitter/track jumps.
  Adjacent lane polygons are combined before finding kerbs. Rider and boundary
  errors remain. This is not actual ankle detection.
- Yielding: `tools/crossing_rules.py` checks moving vehicles and pedestrians
  sharing a road section of a crossing. A bottom-of-box contact region estimates
  vehicle entry/exit. C3905's annotated Cobalt event is recovered at
  81.582-85.085 s with the current cached tracks (user: 81.782-84.718).
- Bus dwell: >=10 seconds at the far-side bus stop, including buses whose bottom
  edge overlaps the stop while the bottom-center point falls just outside it.
  General stopped vehicles and signal-queue separation remain unfinished.
- `tools/download_weights.py` is setup-only and verifies the checksum.
- Comparison: `tools/evaluate_proposals.py` wraps the unchanged official metric.

## Measured results

Four-video comparison on the cached 2 fps tracks, after crossing and bus-edge
changes (`debug/bus_footprint_proposals.json`): 84 proposals; Part A **0.1521**.
At temporal IoU 0.5:

| Class | Matches | Extra predictions | Missed labels |
| --- | ---: | ---: | ---: |
| Jaywalking | 8 | 39 | 5 |
| Stopped vehicle | 12 | 5 | 4 |
| Failure to yield | 1 | 19 | 0 |

Congestion, red light, solid-line crossing, and near miss have no active rule.
The old occupant-only baseline scored 0.1263, with jaywalking 5/57/8.
Extras include wrong events and events with insufficient timing overlap; these
counts do not mean every unmatched actor was visually adjudicated.

Historical tests BEFORE the Part B implementation (zero-risk stub):

Full official harness test on C3905 with socket connections explicitly blocked:
14 events, 3,825 risk samples, **173.8 s total versus 382.9 s budget**, no harness
errors. Part A was approximately 104 s; peak CUDA reserve 1.30 GiB. Runtime
registration found 264 inliers, p90 error 1.9 reference pixels. Output:
`debug/offline_submission_C3905.json`. This used the crossing update before the
bus-edge change. Part B outputs were all zero; this verifies execution, not
accident prediction. No T4 measurement or fresh end-to-end four-video test yet.

The second full offline harness test, C3897, also finished: 26 events, 9,525 risk
samples, **418.4 s total versus 953.5 s budget**, no harness errors. Part A took
243.3 s and Part B 175.1 s; registration had 3,219 inliers, p90 error 1.2 pixels.
Output: `debug/offline_submission_C3897.json`. Both tests used the ignored
`debug/run_offline_harness.py` socket blocker. The tracked replacement
`tools/verify_offline.py` additionally counts attempted connections (including
ones swallowed by libraries); that stricter wrapper has not had a full run yet.

16 focused unit tests passed: pedestrian geometry/trajectories, crossing
entry/exit, opposite carriageways, stationary vehicles, occupants, public entry
point, and weak/short-clip registration. The Windows sandbox denied access to
Python TemporaryDirectory folders even inside the workspace; the full offline
test succeeded outside the sandbox with user approval.

## Running locally

From WIUT-HACKATHON in PowerShell:

```powershell
$env:SALEN_WORK_DIR = "$PWD\debug\runtime"
.\.venv\Scripts\python.exe tools\download_weights.py
.\.venv\Scripts\python.exe run_submission.py --videos "D:\wiut hackathon\videos\C3905.MP4" --out predictions.json --team Salen
.\.venv\Scripts\python.exe evaluate.py --pred predictions.json --validate-only
```

Fast rule replay and tests:

```powershell
.\.venv\Scripts\python.exe tools\auto_label_video.py "D:\wiut hackathon\videos" --reuse-tracks
.\.venv\Scripts\python.exe tools\evaluate_proposals.py
.\.venv\Scripts\python.exe -m unittest discover -s tools/tests -v
```

`--legacy-pedestrians` reproduces the earlier pedestrian rule;
`--legacy-crossings` reproduces the earlier yielding fragments. Models, tracks,
predictions, diagnostic images, and temporary output are ignored by Git.

## Part A backlog (paused) and eventual packaging

1. The 4 fps C3905 experiment finished (`debug/tracks4/`, `debug/proposals4.json`):
   196.6 s for Part A, 21 candidates instead of 14. The C3905-only Part A score
   fell from 0.3619 to 0.2892 with no additional matches at tIoU 0.5. Keep 2 fps;
   more frames alone did not solve tracking or geometry errors.
2. User clarification was requested asynchronously: identify the vehicles in
   C3902 red light 95.295-123.156, C3896 solid line 122.889-132.532, and C3897
   solid line 128.328-173.740. Frames are in `debug/missing_rules_review.jpg`.
   These labels lack actor IDs; some intervals are much longer than one crossing.
3. Red-light/congestion rules need a justified signal-to-movement association
   and a reader for NEW input video. Do not use sample signal timelines as
   predictions for new videos. Unknown signal applicability must stay unknown.
   A second user clarification is pending: C3905 congestion 72.139-111.612 is
   almost entirely amber/red in the D timeline (red 75.475-111.511; green begins
   114.514). Ask whether the relevant queue persisted after green; ordinary red
   queues are excluded by the organizer. Do not edit the label silently.
4. Solid-line rules need the actual prohibited line(s) mapped. The user's current
   zones contain a STOP line, not solid lane-divider annotations. Do not confuse
   those two markings.
5. Near miss needs visible evasive action and clearance, not proximity alone.
   The one user event is C3905 81.448-84.017; 2 fps and tracking breaks may miss
   the reported pedestrian jump-back. `debug/check_near_miss_evidence.py` tried
   abrupt pedestrian turning/braking plus predicted approach/clearance on both
   sampling rates. It missed the annotated event and produced unrelated samples
   at 4 fps, so it was NOT enabled. Cobalt track 2757 conflicts with pedestrians
   near x=3300-3700, y=800-1050 (C3905 4K coordinates); inspect ankles/occlusion
   around 81-84 s before promoting any near-miss heuristic.
6. Further reduce far-kerb, rider, and zebra-boundary false positives; improve
   track continuity/timing and missed stopped vehicles. Do not silently relabel
   the user's examples to make metrics look better.
   Prioritize the five remaining overlaps from the latest C3905 review above.
   User says the pedestrian let the car pass around 40 s, and had not entered
   the second crossing section around 110 s. Proximity/shared broad road polygon
   alone is insufficient. Do not blindly suppress all stationary pedestrians:
   stopping may itself be an evasive response. Inspect the actual actor/path.
   `debug/check_rider_crops.py` tested the SAME YOLO26x on close crops: at
   C3905 34.535 s, person track 1004, it detects a motorcycle (confidence .40),
   missed by whole-frame inference. A true walking child at 75.576 s did not
   produce a motorcycle. This is promising diagnostic evidence, not a verified
   general fix, and crop refinement is NOT integrated into runtime. Example
   image: `debug/rider_crop_check.jpg`.
7. Validate the final pipeline on all four videos, then a T4-like environment.
   Complete licensing/attribution, environment reproducibility, packaging,
   and optional causal risk model. The project is not submission-ready yet.
