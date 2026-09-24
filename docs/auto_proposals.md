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

Store the official YOLO26n detector at `weights/yolo26n.pt`. Our local copy came
from [Ultralytics' v8.4.0 release](https://github.com/ultralytics/assets/releases/tag/v8.4.0)
and has SHA-256
`9B09CC8BF347F0FC8A5F7657480587F25DB09B34BF33B0652110FB03A8AD4FEF`.
Model inference loads this local file and makes no weight download request.

```powershell
# From the repository root:
.venv\Scripts\python.exe scripts\auto_label_video.py "D:\wiut hackathon\videos" --skip-existing

# Tune scene rules without repeating video decode or model inference:
python scripts\auto_label_video.py "D:\wiut hackathon\videos\C3905.MP4" --reuse-tracks
```

Defaults: 2 detector samples per second, 960 px inference size, and CUDA device
0. `--sample-fps`, `--image-size`, `--max-seconds`, `--scene`, and `--out` are
available for experiments. For `--reuse-tracks`, pass the same `--sample-fps`
used to create the cached track CSV. Outputs are:

- `auto_proposals.json`: per-video proposed events, kept out of Git.
- `debug/<video>_auto_tracks.csv`: detection and track diagnostics, kept out of Git.
- `config/scene.json`: normalized crossing, roadway, and exclusion polygons.

The scene polygons were traced from `C3905.MP4` and need validation on the
other clips. The rules require visible pedestrian motion for `jaywalking` and
a moving vehicle close to a pedestrian on the same crossing for
`failure_to_yield`. A car stopped to let a pedestrian cross is not a
`failure_to_yield` event under the official task definition.

Ultralytics publishes its code and models under [AGPL-3.0 or an Enterprise
license](https://docs.ultralytics.com/help/contributing/#open-sourcing-your-yolo-project-under-agpl-30).
Check the final team repository's licensing plan before packaging this model.
