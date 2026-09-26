# Local automatic event proposals

`tools/auto_label_video.py` uses a **locally stored YOLO26x detector** and
ByteTrack on CUDA. It samples 2 frames per second at 2560 px, then applies
provisional hand-drawn scene zones to propose `jaywalking`, `failure_to_yield`, and far-side
bus-stop `stopped_vehicle` intervals. A person detection enclosed high inside a
same-frame car, bus, or truck detection is suppressed before event rules run.
YOLO26x is the largest detection scale in the
[Ultralytics YOLO26 performance table](https://docs.ultralytics.com/models/yolo26);
its published COCO score is a model-selection guide, not traffic-event accuracy.
Inference does not call a hosted service. The 118,667,365-byte weight file is
`weights/yolo26x.pt` (SHA-256
`9FDD44A31C504547FFB81D2C6D9E6DAC3493C8EAA8B0398D3F43BAE6C7003E92`).
It was downloaded from the [official Ultralytics assets v8.4.0
release](https://github.com/ultralytics/assets/releases/tag/v8.4.0).
Weights and outputs are ignored by Git and are not part of this repository.

The existing development environment in the neighboring staging folder has
CUDA PyTorch, Ultralytics, OpenCV, and `lap` installed. On this machine run:

```powershell
cd "D:\wiut hackathon\code\salen-traffic-events-main"
..\WIUT-HACKATHON\.venv\Scripts\python.exe tools\auto_label_video.py "D:\wiut hackathon\videos" --reuse-tracks
```

For a single video, replace the video folder with its MP4 path. Omit
`--reuse-tracks` to rerun the local detector. `auto_proposals.json` contains proposed
intervals; `debug/<video>_auto_tracks.csv` stores detections and tracks. Rule
changes can be replayed without model inference:

```powershell
..\WIUT-HACKATHON\.venv\Scripts\python.exe tools\auto_label_video.py "D:\wiut hackathon\videos\C3905.MP4" --reuse-tracks
```

The 4 GB RTX 3050 Laptop GPU completed the 2560 px pass on all four clips:

| Clip | Video duration | Detector pass wall time | Current proposals | Peak CUDA reserved |
| --- | ---: | ---: | ---: | ---: |
| C3905 | 127.6 s | 105.6 s | 14 | 1.30 GiB |
| C3896 | 340.3 s | 268.3 s | 15 | 1.30 GiB |
| C3897 | 317.8 s | 245.2 s | 26 | 1.34 GiB |
| C3902 | 317.8 s | 260.5 s | 29 | 1.30 GiB |

The detector pass originally emitted 83 candidates using rough scene polygons.
Replaying its cached tracks through the added bus-dwell rule made 100 candidates.
After importing the hand-drawn zones, the initial replay produced 145 candidates.
The person-in-vehicle filter and one-second same-class merge reduced this to
97 candidates. The trajectory and boundary checks now produce
82 candidates. The subsequent crossing-episode and bus-footprint checks produce
**84 unverified candidates** (14 C3905, 15 C3896, 26 C3897, 29 C3902).
The 97-candidate historical baseline used `--legacy-pedestrians` before those
subsequent changes. `--legacy-crossings` reproduces the former yielding logic. The new
rule-only replay takes about 3-12 s per clip on this PC.
Detector wall times above predate this cheap replay. This measures Part A only;
the organizer's Part B full-frame decode still needs an end-to-end benchmark.
The detector weights are 118.7 MB, below the 5 GB model-weight limit.

## What the proposals mean

- `jaywalking`: a tracked person's bottom-center box point is inside a hand-drawn
  live-lane polygon, away from zebra polygons and excluded islands, while
  moving. Movement is measured over a two-second window, with short pauses
  retained and long stationary tails rejected. Require three observations and
  meaningful displacement, split large tracking jumps, and reject boxes clipped
  at the bottom or sides. Movement within 30 degrees of a nearby zebra edge is
  exempt within 0.15 times apparent person height; movement along the outer road
  edge is suppressed within 0.3 times that height to allow for uncertain feet
  near kerbs. Adjacent lane polygons are combined before finding road edges.
  These are pixel heuristics, not measured ground distances. The box bottom is
  still only an approximation of the feet; riders and boundary mistakes remain.
- `failure_to_yield`: a moving vehicle traverses a crossing while a pedestrian
  occupies the same road section nearby. Its ground-contact box region estimates
  entry/exit, extending the event beyond the original per-frame proximity flags.
- `stopped_vehicle`: a tracked COCO bus stays within roughly 1.2% of the frame
  width and height for at least 10 s, with its ground point in the far-side
  bus-stop polygon (with a 0.005 normalized boundary tolerance, about 10-20 px).
  The three bottom-edge points are checked, allowing a long bus partly outside
  the stop area to be included.
  This is a narrow first rule; other stopped vehicles remain
  unsupported. A bus at this stop appears to occupy a live lane, but the zone
  and bus-bay boundary still need verification.

The default scene geometry is Jamoliddin's hand-drawn `reports/eda/zones.json`
(C3897 reference, 4K pixels). The script maps it into each clip with Hamid's
homographies in `reports/eda/registration.json`. The original rough C3905
polygons remain in `config/scene.json` and can be selected with `--scene` for
comparison. The drawing aligns visually with C3897 and C3905, but its edges
have not been independently verified; one island has two points less than
1 px apart. C3905's bus track at 74-94 s falls only a few pixels beyond the
drawn bus-stop edge, so the bus rule uses a small tolerance. The camera also
shifts during the first seconds of some clips. See the EDA v4
[`tools/scene/CONTRACT.md`](../tools/scene/CONTRACT.md) for the provisional
rule-input status.

The latest user observations are in `labels/user_truth_2026-09-26.json` (all four
clips, 40 events). The C3905 proposals overlap all three marked jaywalking intervals,
and still contain
unmarked intervals, so overlap alone does not validate the rule. C3905 still
proposes a motorcycle rider around 34 s and a crossing-edge person around
85 s. The stop-line drawing is stored for later rules; this prototype does
not use it. See `docs/JAYWALKING_REVIEW.md` for the checked frames.

The official evaluator applied to these 40 manual labels and the current 84
proposals gives Part A score **0.1521** (previously 0.1263, then 0.1377).
At temporal IoU 0.5, stopped vehicles have 12 matches, 5 extra proposals, and 4 missed labels;
jaywalking has **8 matches, 39 extra proposals, and 5 missed labels** (previously
5/57/8); failure to yield has 1
match, 19 extra proposals, and 0 missed labels. These are development results
against user annotations, not held-out accuracy. The six congestion, two
solid-line crossing, one red-light, and one near-miss labels have no implemented
rules. No accidents are labeled, so Part B cannot be assessed here.
Part A development is paused at the user's request; Part B is now the priority
(see `PROJECT_STATE.md`). The Part A backlog is pedestrian identity and crossing trajectories, then yielding
geometry/timing, and implement the missing event rules. Preserve manual labels
when comparing changes; mere temporal overlap is not a successful match.

Reproduce the comparison with:

```powershell
..\WIUT-HACKATHON\.venv\Scripts\python.exe tools\evaluate_proposals.py
..\WIUT-HACKATHON\.venv\Scripts\python.exe -m unittest discover -s tools/tests -v
```

Sixteen targeted tests cover parallel/perpendicular movement, combined lane
boundaries, clipped boxes, pauses, stationary jitter/tails, missing samples, and
tracking jumps, crossing episodes, and runtime registration. Rules never read
the manual label file. See `PROJECT_STATE.md` for the full offline harness test.

**These outputs are not ground truth or final submission predictions.** The
rules now use shared road polygons for crossing-half association and an estimated
vehicle contact region for entry/exit. Those geometric approximations still need
review, especially where lane polygons overlap or feet are occluded. Detector
false positives and ID switches also occur. Do not feed `auto_proposals.json`
to evaluation as if it were reviewed labels. Other event classes are not yet
implemented.

Jamoliddin's earlier manual observations for C3905 and C3896 are preserved
in `labels/user_observations_2026-09-25.json`; the updated observations are in
`labels/user_truth_2026-09-26.json`. See
`docs/PROVISIONAL_LABEL_REVIEW.md` for a frame and signal-phase comparison.
The comparison exposed missed bus-stop dwell events in the earlier prototype;
the new bus rule recovers five of six user stopped-vehicle intervals across the
two reviewed clips at temporal IoU 0.81-0.90. The unmatched interval occurs in
a near-side signal queue. At C3905 around 82 s, the moving white Cobalt at
the right-hand crossing likely does fail to yield; the automatic 0.5 s proposal
captures only a fragment of the user's roughly 3 s interval. A pedestrian's
reported evasive step makes `near_miss` plausible, but that class has no
automatic rule yet. Neither file should be used as verified ground truth yet.

Ultralytics publishes its code and models under
[AGPL-3.0 or an Enterprise license](https://docs.ultralytics.com/help/contributing/#open-sourcing-your-yolo-project-under-agpl-30).
Resolve the team's license and attribution plan before distributing
the detector as part of a submission.
