# Automatic development-label proposals

`scripts/auto_label_video.py` runs an open-weight detector and tracker locally,
then applies the official class definitions to a fixed-camera scene map. It
currently proposes only `jaywalking` and `failure_to_yield`. The remaining
classes still need dedicated rules or models; an empty proposal list does not
prove that a video has no events.

The proposals use the same per-video shape as `examples/ground_truth.json`, but
they are **not verified ground truth**. Keep `auto_proposals.json` separate from
`dev_labels.json` when evaluating the model.

## Local setup

The tested Windows workstation has an RTX 3050 Laptop GPU (4 GB), PyTorch
2.14.0+cu130, Ultralytics 8.4.161, and `lap` 0.5.13. Install CUDA PyTorch from
the [official PyTorch selector](https://pytorch.org/get-started/locally/), then
install `ultralytics` and `lap` into an isolated Python environment. The script
refuses to run heavy detector inference when CUDA is unavailable.

The stronger YOLO26x detector is now the default at `weights/yolo26x.pt`.
Our local copy came
from [Ultralytics' v8.4.0 release](https://github.com/ultralytics/assets/releases/tag/v8.4.0)
and has SHA-256
`9FDD44A31C504547FFB81D2C6D9E6DAC3493C8EAA8B0398D3F43BAE6C7003E92`.
Model inference loads this local file and makes no weight download request.
The canonical team checkout at `D:\wiut hackathon\code\salen-traffic-events-main`
now has this prototype under `tools/auto_label_video.py`, plus the local weight
file. Use its `docs/AUTO_PROPOSALS.md` for the current run command.

```powershell
# From the repository root:
.venv\Scripts\python.exe scripts\auto_label_video.py "D:\wiut hackathon\videos" --skip-existing

# Tune scene rules without repeating video decode or model inference:
.venv\Scripts\python.exe scripts\auto_label_video.py "D:\wiut hackathon\videos\C3905.MP4" --reuse-tracks
```

Defaults: 2 detector samples per second, 2560 px inference size, and CUDA device
0. `--sample-fps`, `--image-size`, `--max-seconds`, `--scene`, and `--out` are
available for experiments. For `--reuse-tracks`, pass the same `--sample-fps`
used to create the cached track CSV. Outputs are:

- `auto_proposals.json`: per-video proposed events, kept out of Git.
- `debug/<video>_auto_tracks.csv`: detection and track diagnostics, kept out of Git.
- `config/scene.json`: normalized crossing, roadway, and exclusion polygons.

The staging scene polygons were traced from `C3905.MP4` and need validation on the
other clips. The team checkout at
`D:\wiut hackathon\code\salen-traffic-events-main\reports\eda` now provides
all four clips' crossing maps, signal reports, a merged reference scene, and
registration homographies. Its v3 summary shows substantial framing shifts
between clips, so these common polygons are unsuitable for final predictions
without registration. The underlying YOLO11m track CSVs are not in that
checkout. The local 2 fps /
960 px pass produced 23 candidates across four videos (1 C3905, 3 C3896,
10 C3897, 9 C3902). A midpoint frame review showed several sidewalk and
island false positives. The completed YOLO26x 2560 px detector run produced
83 initial candidates. The canonical team checkout has since added a narrow
far-side bus-dwell rule and replayed the same tracks, yielding 100 unverified
candidates (C3905 13, C3896 38, C3897 22, C3902 27). The detector pass wall
times were 105.6 s, 268.3 s, 245.2 s, and 260.5 s respectively; this excludes
Part B. Jamoliddin's later provisional manual observations are
preserved separately in the canonical checkout, with a focused comparison in
`docs/PROVISIONAL_LABEL_REVIEW.md`. Review the candidate's whole interval and actor track
before accepting it as a label. In particular, one frame showing a vehicle
and a pedestrian near the same crossing cannot establish failure to yield.

The rules require visible pedestrian motion for `jaywalking` and
a moving vehicle close to a pedestrian on the same crossing for
`failure_to_yield`. A car stopped to let a pedestrian cross is not a
`failure_to_yield` event under the official task definition.
The team guide clarifies that the pedestrian and vehicle must occupy the same
carriageway half of the crossing, with event boundaries at vehicle front entry
and rear exit. The current proximity heuristic does not enforce those details.

Ultralytics publishes its code and models under [AGPL-3.0 or an Enterprise
license](https://docs.ultralytics.com/help/contributing/#open-sourcing-your-yolo-project-under-agpl-30).
Check the final team repository's licensing plan before packaging this model.
