# Rule-input contract (EDA v4, 2026-09-24)

This is the list of scene files the model's event rules may read, what is in
them, and which files they must NOT read. `tools/scene/tests/test_rule_inputs.py`
checks that every FROZEN and PROVISIONAL file below exists and that the key
invariants hold; run it after any change to these files:

```bash
uvx --with pandas --with numpy --with opencv-python-headless --with matplotlib pytest -q tools/scene/tests
```

Status words:

- **FROZEN**: checked independently; change only with a new version and a note in `reports/eda/SUMMARY.md`.
- **PROVISIONAL**: usable, but built from automatic readings or tracker output that no human has verified yet.
  Rules must respect the uncertainty fields and treat `unknown` as unknown.
- **DEPRECATED**: kept for traceability only. Rules must not read it.
- **PENDING**: the slot is reserved; the file does not exist yet.

## Conventions

- **Reference frame**: the C3897 median background (`tools/scene/reference_C3897.jpg`), in 4K pixels
  (3840 x 2160, x right, y down). All scene geometry lives here. Move points into a video with
  `H_ref_to_video` (and back with `H_video_to_ref`) from `reports/eda/registration.json` for the four
  sample videos, or from `tools/scene/register.py` (`register_frames`, `to_video`, `to_ref`) for any new
  video at run time. Use the homography, not the similarity.
- **Video frame**: each video's own 4K pixels. Ground point of a box = bottom centre.
- **Time base**: `t = frame / (30000/1001)` seconds, `frame` = 0-based index of the video's own frames
  (the 4K original and the 1080p proxy have the same frame count). Signal intervals are frame-indexed and
  half-open, `[start_frame, end_frame)`.
- **Headings**: image degrees in the reference frame, 0 = +x (right), 90 = +y (down the image). A velocity
  measured in a video is moved into the reference with the local Jacobian of `H_video_to_ref`
  (`tools/scene/scene_reference.py`, `jac_map`).

## Files the rules may consume

| File | What | Frame / time base | Version | Status |
|---|---|---|---|---|
| `tools/scene/signal.py` | accessor: `load(stem)`, `state_at(t) -> (state, observation, uncertainty_s)`, `state_at_frame`, `next_state`, `intervals()`; head D from the inventory; a new video without a timeline answers `unknown` everywhere | video frames, t as above | v4 | PROVISIONAL |
| `reports/eda/signals/v4/<STEM>_<HEAD>_intervals_v4.csv` | gap-free half-open intervals: `start_frame, end_frame, start_t, end_t, duration_s, state, n_samples, n_observed, n_imputed, n_unobserved, imputed, start_uncertainty_frames`. States: red, red_amber, amber, green, green_blink, dark, unknown (head A: no amber states). Occlusions over 1 s are `unknown`; shorter ones inside one state are bridged and flagged `imputed`. Schema in `tools/scene/signal_timeline.py` | video frames | v4 | PROVISIONAL |
| `reports/eda/signals/v4/<STEM>_<HEAD>_samples_v4.csv` | per sample (every 3rd frame): `frame, t_sec, state, observation (observed / occluded / low_confidence), confidence, margin_sd, reason, v3_label, <lamp>_on, <lamp>_z` | video frames | v4 | PROVISIONAL |
| `reports/eda/signals/v4/timeline_v4.json` | schema text, parameters, per-video coverage (share of time with a known state; this is NOT accuracy), unknown stretches over 1 s | video frames | v4 | PROVISIONAL |
| `reports/eda/signals/signal_heads.json` | head inventory. Identity fields (`box_ref`, `type`, `lamps`, `readable_from_camera`) are FROZEN: D = the one readable 3-lamp vehicle head, A = pedestrian head. The `controls` text is an interpretation (D is a timing proxy for the near-carriageway flow; what it controls is UNKNOWN) | reference px | v4 text | FROZEN |
| `reports/eda/registration.json` | `H_ref_to_video`, `H_video_to_ref` for C3896, C3902, C3905 (C3897 = identity), landmark checks | reference px <-> video px | v3 | FROZEN |
| `tools/scene/reference_C3897.jpg` | the reference image (1080p median; coordinates are 4K = 2 x its pixels) | reference | v3 | FROZEN |
| `tools/scene/register.py` | run-time registration of a new video to the reference | reference px <-> video px | v3 | FROZEN |
| `tools/scene/scene.py` | accessor: `heading_modes(x, y)`, `dominant_heading`, `wrong_way_safe(x, y)`, `against_flow(x, y, heading)` (None outside safe cells), `zones()` (raises if absent) | reference px | v2 | PROVISIONAL |
| `reports/eda/scene_reference_v2.json` | 80 px cells: heading modes with shares and per-video support, `dominant_heading`, `any_video_dissents`, `frame_edge`, `turning_or_conflict`, `wrong_way_safe`, `wrong_way_blockers`. Built from tracker output only | reference px | v2 | PROVISIONAL |
| `reports/eda/zones.json` | Jamoliddin's hand-drawn zones: stop line, zebra crossings, parking bays, live lanes, bus stop, non-carriageway. Exported 2026-09-26 from the zone editor; schema: `tools/scene/web/README.md`. The polygons are visually aligned on C3897 and C3905 but have not been independently reviewed at every edge. Event rules should use a small boundary tolerance and treat their output as provisional. | reference px | v290 | PROVISIONAL |

How rules should use them:

- Signal: a red-light or stop-line rule may fire only when `state_at(t)` is `red` (or `red_amber`) with
  observation `observed` or `imputed` and `uncertainty_s == 0`. `unknown`, `out_of_range` and
  `no_timeline` mean "no call". D is a timing proxy for the near-carriageway flow only; there is no readable
  head for the far carriageway, so far-carriageway signal state is unknown.
- Wrong way: call a vehicle wrong-way from its heading only where `wrong_way_safe(x, y)` is true (445 of 618
  accepted cells). Elsewhere the cell can legitimately hold several directions.
- Stop lines, zebras, parking bay and live lanes: use the hand-drawn `zones.json`
  with the reference-to-video homography, and review boundary cases against frames.

## Files the rules must NOT use

| File | Why | Replaced by | Status |
|---|---|---|---|
| `reports/eda/scene_reference.json` | v1 direction map: an accepted cell can hold two directions (5 accepted cells with a dissenting video, 483 of 52,760 contributions > 90 deg off) | `scene_reference_v2.json` | DEPRECATED |
| `reports/eda/signals/<STEM>_<HEAD>_phases.csv` | v3 phases: occlusions forward-filled (a 3 s "dark" phase in C3896 that nobody saw), one sampling interval missing per phase (62 gaps, 6.21 s over the four D files) | `v4/<STEM>_<HEAD>_intervals_v4.csv` | DEPRECATED |
| `reports/eda/signals/<STEM>_<HEAD>_samples.csv` | v3 per-sample labels, the input of v4; no observation status, `green_flash` and `unclear` labels | `v4/<STEM>_<HEAD>_samples_v4.csv` | DEPRECATED |
| `reports/eda/signals/lamp_summary.json` | calibration record of the v3 lamp reader; its `phases` lists are the forward-filled v3 phases | `v4/timeline_v4.json` | DEPRECATED |
| `reports/eda/signals/signal_validation.json` | v3 validation: "far carriageway: no effect" was wrong, onset counts were available not used, random-onset null only | `signal_validation_v2.json` | DEPRECATED |
| `reports/eda/C3905/signal_timeline_v1_DEPRECATED.csv` | v1 HSV lamp fractions of COCO boxes on the 1080p proxy | `v4/C3905_D_intervals_v4.csv` | DEPRECATED |
| `reports/eda/C3905/notes_v1_RETRACTED.json` | v1 reviewed notes with retracted claims (signal phases, "probably auto exposure", ...) | `reports/eda/SUMMARY.md` | DEPRECATED |
| `reports/eda/<STEM>/direction_field.json` | per-video v1 direction field in video px, not gated, not registered | `scene_reference_v2.json` | DEPRECATED |

Descriptive outputs (for people and the website, not rule inputs): `reports/eda/<STEM>/eda.json` (its
`extra.signal` block describes head D in that video; `extra.deprecated` says which legacy fields were
removed), the per-video figures, `reports/eda/signals/signal_validation_v2.json`,
`reports/eda/signals/cycle_phase_v2.png`, `reports/eda/SUMMARY.md`.

## Versions

- Signal timelines: v3 (lamp_states.py, 2026-09-24) -> v4 (signal_timeline.py, 2026-09-24, this contract).
- Direction map: v1 (scene_reference.py) -> v2 (scene_reference_v2.py).
- Signal validation: signal_validation.json -> signal_validation_v2.json (signal_validate.py v2).
- Zones: editor export v290 (2026-09-26, PROVISIONAL; edge review ongoing).
