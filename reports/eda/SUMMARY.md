# EDA summary across the 4 sample videos

2026-09-26 scene update: Jamoliddin exported 17 hand-drawn zones to
`reports/eda/zones.json`. They are provisional and visually aligned on the
C3897 reference and C3905. The v4 analysis below predates this export;
`tools/scene/CONTRACT.md` records the current rule-input status.

Version 4 (2026-09-24). Version 4 fixes what a second adversarial review found in
v3: a pedestrian head still exposed as "the vehicle light" in two eda.json files,
an unsupported claim that head D controls the near carriageway only, a direction
map whose accepted cells can hold two directions, and signal timelines that hid
imputation and had gaps. The files the model's rules may read are now listed in
`tools/scene/CONTRACT.md` and locked by tests (`tools/scene/tests`, 37 passing).
Every reviewer number below was reproduced from the committed code unless marked
otherwise. v3 is kept as `SUMMARY_v3.md`. Corrections are listed at the end.

Tracks: YOLO11m (imgsz 1280) + ByteTrack on a Kaggle T4, every 3rd frame, tracks
with at least 3 detections. All coordinates are 4K pixels (3840x2160). Scene
geometry is given in the REFERENCE frame (C3897) and moved into a video with
`H_ref_to_video` from `registration.json` (`tools/scene/register.py` at run time).
Time: `t = frame / (30000/1001)`, frames 0-based.

| | C3905 | C3896 | C3897 | C3902 |
|---|---|---|---|---|
| Recording start (camera clock, +06:00, 2026-09-18) | 18:22:21 | 12:18:24 | 12:24:47 | 17:58:18 |
| Duration (frames) | 2:08 (3825) | 5:40 (10200) | 5:18 (9525) | 5:18 (9525) |
| Exposure (constant over the clip) | f/5.6, 1/64 s, ISO 250 (+8 dB) | f/18, 1/64 s, ISO 100 | f/18, 1/64 s, ISO 100 | f/5.0, 1/64 s, ISO 100 |
| Lighting (mean luma, server decode) | lowest, 38.6 at start, 35.8 at end, +35% between 52 and 67 s | ~95, bright daylight | ~95, bright daylight | 69 falling to 58, evening |
| Framing vs reference C3897 | third set-up: 1.1 deg, 1.1% scale, points move 0.5 to 70 px | same (under 1 px) | reference | re-framed: points move 64 to 144 px |
| Signal cycle (head D, fitted to green onsets) | 79.98 s | 75.02 s | 75.02 s | 80.02 s |
| Head D timeline coverage (share of time with a known state; not accuracy) | 100% | 94.1% | 99.4% | 95.9% |
| Person tracklets | 541 | 770 | 692 | 1229 |
| Car tracklets | 211 | 458 | 425 | 502 |

Camera: Sony ILCE-6700, XAVC S 3840x2160 H.264 4:2:2 10-bit 140 Mbps, 29.97p,
rec709 (all four). Per-video details: `camera/camera_settings.json`.

## Rule-input status

| Input | File / accessor | Status for rules | Caveat |
|---|---|---|---|
| Registration of the 4 sample videos | `registration.json` | safe | homography, not similarity; lower-left corner has one passing landmark |
| Registration of a new video | `tools/scene/register.py` | safe with caveats | frames from the first ~30 s can be up to 30 px off; evening light halves the landmark checks |
| Which head is the vehicle light | `signals/signal_heads.json` (D) | safe | identity by appearance (3 lamps, faces the camera); A is a pedestrian head and is never used |
| Head D state over time | `signals/v4/*_D_intervals_v4.csv`, `tools/scene/signal.py` | safe with caveats | coverage 94.1 to 100%, NOT accuracy (0 of 240 spot-check crops human-verified); `unknown` 20.1 s in C3896, 13.0 s in C3902; weak amber lamp in daylight (d' 4.5 and 6.4); only `observed` or `imputed` with `uncertainty_s == 0` should trigger a rule |
| What D's state means | `signal_heads.json` `controls` | safe with caveats | a strong timing proxy for the NEAR-carriageway flow; which movements D physically controls is UNKNOWN |
| Far-carriageway signal state | none | not safe | no readable head faces the far carriageway; its flow is cycle-locked but its signal is unobserved |
| Head A state (pedestrian) | `signals/v4/*_A_intervals_v4.csv` | safe with caveats | which crossing A serves is MEDIUM (lower-left zebra, by facing); not for vehicle rules |
| Wrong-way direction | `scene_reference_v2.json`, `tools/scene/scene.py` | safe with caveats, only where `wrong_way_safe` (445 cells) | tracker output only, no labelled wrong_way event; 108 of 42,165 track-cell contributions inside safe cells are > 90 deg off (tracker errors or real events), so require persistence |
| Wrong-way anywhere else | same file, other cells | not safe | 63 accepted cells hold 2 or more heading modes; 13 have a dissenting video |
| v1 direction map, per-video direction fields | `scene_reference.json`, `<STEM>/direction_field.json` | not safe | DEPRECATED |
| v3 signal phases, lamp_summary phases, v3 validation | `signals/*_phases.csv`, `lamp_summary.json`, `signal_validation.json` | not safe | DEPRECATED (forward fill, gaps, wrong far-carriageway conclusion) |
| Stop line, zebras, parking bay, live lanes, bus stop | `zones.json` | not available | PENDING: to be drawn by a human in the zone editor |
| Stop-line and queue estimates, notes | `<STEM>/eda.json` | not a rule input | descriptive only |

## Findings and confidence

### 1. Framing differs between recordings, and the camera settles after start. Confidence: HIGH (not re-checked in v4).

Median backgrounds (60 frames of each 1080p proxy), SIFT + ratio test + RANSAC
against the C3897 median; independently checked with 14 named fixed landmarks
(normalized cross-correlation of gradient images, not SIFT) and with the
signal-head boxes YOLO found in each video.

| video | homography inliers / resid. median | landmark error, homography (landmarks used) | landmark error, similarity | displacement ref -> video |
|---|---|---|---|---|
| C3896 | 3962 / 0.32 px | median 0.14, max 0.99 px (14 of 14) | median 0.17, max 1.07 px | 0.5 to 0.9 px |
| C3902 | 307 / 0.89 px | median 0.95, max 2.05 px (7 of 14) | median 3.4, max 10.7 px | 64 to 144 px, median 101 |
| C3905 | 372 / 0.73 px | median 1.12, max 2.32 px (7 of 14) | median 5.9, max 7.7 px | 0.5 to 70 px, median 37 |

Drift after recording starts: C3896 starts 30 px off and is under 2 px only from
27 s on; C3902 7 px (settled from 17 s); C3905 4 px; C3897 under 1 px. Still
unverified: 7 of 14 landmarks failed the NCC gate in the evening videos; the
lower-left corner is covered by one passing landmark.

### 2. Lane directions in the reference frame (v2). Confidence: HIGH for the dominant heading of the two main flows; MEDIUM for `wrong_way_safe` as a rule input.

**What was wrong in v1.** An accepted ("ok") cell only needed coherence >= 0.7, 8
tracks and 2 videos that agree (`scene_reference.py`, gate at line ~129). That does
not mean one permissible direction. Reproduced: 5 of 635 accepted cells have a
video with at least 3 tracks whose own heading is more than 30 deg off, e.g. cell
(29,21) at (2360,1720): merged 32.4 deg, C3905 124.6 deg (3 tracks, median 150.9
deg); and 483 of 52,760 track-cell contributions in accepted cells are more than
90 deg off the merged heading, in 182 of the 635 accepted cells.

**v2** (`tools/scene/scene_reference_v2.py`, `scene_reference_v2.json`, figure
`scene_reference_v2.jpg`). Samples whose box touches the frame edge are masked
(7,757 of 141,994 moving samples: the ground point of a clipped box is wrong), and
motorcycle tracks (801 samples; they include bicycles and scooters crossing on the
zebras) are left out of the modes. Per cell, heading modes come from a greedy
circular clustering (von Mises kernel, about 20 deg; members within 45 deg), each
with its share and per-video support. Result: 746 cells, 618 accepted with the v1
gates, 63 of them with 2 or more modes, 13 with a dissenting video (10 of the 13
are C3905, 8 of them in the band x 1900 to 2600, y 1600 to 1840 where C3905 shows
a second flow at about 145 to 155 deg, down-left), 390 of 50,544
contributions > 90 deg off the dominant heading, in 158 accepted cells.

`wrong_way_safe` = accepted, dominant mode >= 95% of the cell's tracks, >= 12
tracks, >= 2 videos with >= 3 tracks, no video dissents, not within one cell of
any video's frame edge, not a turning or conflict cell (second mode >= 5%, or
dominant heading > 30 deg from an accepted neighbour, or a turn flow). 445 of 618
accepted cells pass: 281 near carriageway (towards the camera), 157 far carriageway
(away), 7 below the median line near the median nose. Blockers among accepted
cells (a cell can have several): turning or conflict 120, dominant share below 95%
83, frame edge 52, fewer than 12 tracks 26, a dissenting video 13. Cell (29,21) is
not safe (dominant 27.2 deg with 90%, a second mode at 154.6 deg from C3905, C3905
dissents).

**Still unverified.** Headings are tracker output; no labelled wrong_way event
exists. Inside safe cells 108 of 42,165 contributions (0.26%) are > 90 deg off: a
rule on single observations would fire on tracker errors, so it needs persistence
(several cells or seconds). Whether the C3905 down-left flow is a real turning
movement or a registration artefact of the third set-up is open (for a human).

### 3. Signal heads and what D means. Confidence: HIGH that D is a timing proxy for the near-carriageway flow; the far carriageway is also cycle-locked (MEDIUM-HIGH); what D controls is UNKNOWN; lamp states are coverage, not accuracy.

**Inventory** (`signals/signal_heads.json`, unchanged apart from D's `controls`
text): A = pedestrian head (2 lamps, red standing / green walking figure), D =
vehicle head (3 round lamps, faces the camera); B, C, E, F, G are not readable from
the camera. v2 picked A as "the vehicle light" in C3896 and C3902 by correlating
lamp colour with traffic; v3 fixed the inventory but the per-video eda.json still
exposed it. Reproduced: C3896 `extra.vehicle_light_guess` = L0 box
[514, 962, 551, 1037] and C3902 = L0 [414, 1040, 457, 1119], both head A.

**v4 fix.** `tools/eda/analyze.py` no longer searches for traffic lights or picks a
light by correlation. It takes the one readable 3-lamp vehicle head from
`signal_heads.json` (D), maps its reference box into the video with
`registration.json`, reads D's v4 intervals, and stops with an error if the
inventory, the registration or the intervals are missing (`--no-signal` is the
only way to run without a head). All four eda.json files were regenerated:
`vehicle_light_guess`, `signal_colour_summary` and `traffic_lights_detected` are
gone, `extra.deprecated` says why, `extra.signal` describes D; `signal.png` and
the stop map's background now show D's v4 states (grey hatched = unknown). As a
description only, D green vs (moving minus stationary vehicles) per second gives
r = 0.90 (C3905), 0.66 (C3896), 0.72 (C3897), 0.78 (C3902).

**Timelines v4** (`tools/scene/signal_timeline.py`, `signals/v4/`). Two v3 defects,
both reproduced: (a) `phases.csv` forward-filled occluded samples, so 29 occluded +
2 dark samples in C3896 (frames 5676 to 5766) became a 3.0 s "dark" phase nobody
saw; (b) phase duration = end - start with end = the last sample, so each phase
missed one sampling interval: 62 gaps totalling 6.21 s over the four D files (29
gaps, 2.91 s for A). v4 writes per sample a state, an observation status
(observed / occluded / low_confidence), a confidence and the per-lamp decisions,
and per video frame-indexed half-open intervals that tile the whole video with no
gap. Occlusions over 1 s stay `unknown`; shorter ones inside one steady state are
bridged and flagged `imputed`; a state change is known to one sample (3 frames,
0.1 s), after an unknown stretch to that stretch plus one sample. Low confidence
means: an impossible lamp combination, a lamp within one pooled SD of its
threshold, or a run shorter than 1 s with no long run of the same state (or a legal
change) next to it. C3896 frames 5667 to 5769 are now `unknown`, and no `dark`
interval remains in any video. Coverage of D: 100% (C3905), 94.1% (C3896, 20.1 s
unknown, 5 stretches over 1 s), 99.4% (C3897), 95.9% (C3902, 13.0 s unknown, one
10.6 s stretch at 224.5 to 235.1 s). Head A: 99.9 to 100%.

This is **coverage, not accuracy**: 0 of the 240 spot-check crops has a human label
yet.

**Grammar and plan (v4 intervals).** No illegal transition between known states in
any video. Complete phases with known neighbours: red 33.0 s (daylight; one 32.7 s) and
36.0 s (evening), red_amber 3.0 s, amber 3.0 s, blinking green 3.0 s, green 34.9 to
35.0 s before the blink in the evening; cycle 75.02 s (daylight) and 79.98 to 80.02 s (evening);
green onsets fit a fixed-time plan within 0.13 s. (v3 gave 32.9 / 35.9 / 2.9 s: one
sampling interval short, see defect (b).) A turns green in the same sample as D
(13 matched onsets, lag 0.0 to -0.1 s) and turns red exactly when D starts
blinking, 3.0 s before D's amber.

**What D means** (`signal_validate.py` v2, `signals/signal_validation_v2.json`,
`signals/cycle_phase_v2.png`). Vehicle crossings of four gates drawn on the
reference, folded onto the cycle phase (seconds since D green onset). Held-out: a
window is chosen on one lighting group (evening C3905 + C3902, daylight C3896 +
C3897) and tested on the other. Three nulls: random onset times (the v3 null),
each video's crossing sequence shifted circularly by a random offset, and one
shared shift per group (the most conservative). p-values below are in that order,
2000 draws each.

- *Near carriageway* (gate between the gantry and the zebra). Test A (rate after
  onset minus the 15 s before): +1.10 crossings/s in daylight with **8 onsets used**
  and +1.27 in the evening with **6 used** (v3 reported 9 available as if used).
  p = 0.0000 / 0.0000 / 0.0035 (daylight) and 0.0000 / 0.0025 / 0.011 (evening). The
  reviewer's periodic null gave 0.0107 / 0.0338; mine is smaller, most likely from
  details of how the shift is drawn; all are below 0.05. Over the full cycle the
  near flow rises within 5 s of D green, stops about 36 to 39 s after onset (when D
  turns amber and red) and stays near 0 until the next green; 2.5 to 6.3% of its
  crossings happen while D is red (45 to 53% of the time).
- *Far carriageway* (gate at the median nose). The v3 window [-20, +40) s showed "no
  effect" (reproduced: p 0.85 and 0.51), but over the full cycle the far flow is
  strongly cycle-locked too. Test B (the 10 s windows with the highest and lowest
  rate on the training group, contrast on the test group): daylight test, windows
  [35, 45) vs [47, 57) s after onset, 0.56 vs 0.11 crossings/s, p = 0.001 / 0.0025 /
  0.015; evening test, [18, 28) vs [45, 55), 0.72 vs 0.15, p = 0.0095 / 0.044 / 0.030
  (only 4 to 5 onsets). The reviewer's windows reproduce: daylight [15, 25) vs
  [47, 57) 0.738 vs 0.113 (reviewer 0.750 vs 0.088), evening [20, 30) vs [47, 57)
  0.780 vs 0.050 (reviewer 0.780 vs 0.100); the low-window differences most likely
  come from which onsets are included. The approach from the right edge behaves the
  same way.
- *Far offset, quantified.* Low-rate stretch (5 s smoothed rate below 25% of its
  peak; 95% intervals from a cycle-block bootstrap): far carriageway 47 to 61 s
  after D green onset in daylight (start 44 to 47, end 55 to 64) and 48 to 65 s in
  the evening (start 45 to 50, end 58 to 72), i.e. inside D red; near carriageway
  36 to 39 s until the next green onset. The far stop starts 11 s (daylight, 95%
  interval 2 to 16) and 9 s (evening, 6 to 16) after the near stop and ends 15 s
  before the near flow resumes (intervals -22 to -12 and -22 to -8). The far flow
  also dips 5 to 12 s after D green onset; not explained.
- *Lower-left exit*: 13 crossings per group, all early in D green (held-out p 0.0005
  to 0.044); too few to say more.

**What this means for the rules.** D is a strong timing proxy for the
near-carriageway flow. The far-carriageway flow is also cycle-locked, with a
different phase offset (its stop is about 14 to 17 s long and sits inside D red),
which fits coordinated signal groups or platoons released upstream. Which movements
D physically controls is UNKNOWN (its lamps face the camera; the near-carriageway
drivers face E and F, which are not readable). A red-light rule for the far
carriageway has no observed signal.

**Still unverified.** Human spot check of the lamp labels; what stops the far
carriageway 47 to 65 s after D green onset (its own signal, the right-hand zebra's
pedestrian phase, or upstream); amber in daylight rests on phase order more than on
colour; the v3 occlusion detector cannot keep a two-lamp reading (red_amber), which
makes several C3896 red_amber phases `unknown`.

### 4. Parked cars at the far left edge in every video. Confidence: HIGH (not re-checked in v4).

Other long-standing vehicles (far kerb, bus stop) appear in some videos only
(C3896, C3902); the reported durations are track lifetimes. Exclude the left-edge
parking bay from stopped_vehicle once it is drawn in `zones.json`; do not
blanket-exclude the bus stop area.

### 5. Stop line position. Confidence: LOW (not re-checked in v4).

Draw the painted stop lines by hand on the reference frame (`zones.json`).
Near-carriageway queues discharge through the gate between the gantry and the
zebra in every cycle, consistent with a stop line just upstream of that zebra. The
eda.json field is now `furthest_queued_stop_in_D_red` (stops that began while D
showed red, red_amber or amber); it is a rough upper bound, not a stop line.

### 6. Pedestrians on vehicle lanes. Confidence: LOW as a jaywalking measure (not re-checked in v4).

19 to 27% of person tracklets enter cells used by moving vehicles, but those cells
include zebras, kerbs and frame edges. Jaywalking needs the zebra and
non-carriageway polygons from `zones.json`.

### 7. C3905 brightness rise. Confidence: HIGH that it is not exposure control; cause otherwise UNVERIFIED (not re-checked in v4).

Iris, shutter, ISO, gain and white balance each have one value over the whole clip
in the Sony rtmd per-frame metadata. On static background a uniform ~20% change
remains. Brightness-based rules must normalise per frame. The eda.json lighting
note no longer says "probably camera auto exposure".

### 8. Stationary threshold is loose. Confidence in stop statistics: MEDIUM (not re-checked in v4).

### 9. Detector size matters for pedestrians. Confidence: MEDIUM (not re-checked in v4).

YOLO11m gave 2.0x more small person boxes per frame than YOLO11n on C3905; recall
needs a hand count.

## For human QA (in order of value)

1. Lamp spot check: 240 crops (heads D and A, 30 per video), in the spot-check page
   next to the labeling tool or `spotcheck/spotcheck.csv`. Until it is done every
   signal number above is coverage, not accuracy.
2. Zones: draw stop lines, zebras, the left-edge parking bay, live lanes and the bus
   stop in the zone editor; commit the result as `reports/eda/zones.json`.
3. Far carriageway: in C3897 watch the far carriageway 47 to 61 s after each D green
   onset (19.9, 94.7, 169.8, 244.9 s), i.e. about 67 to 81 s, 142 to 156 s, 217 to
   231 s and 292 to 306 s. What stops it? And which movements do D, E and F face?
4. C3905, reference x 1900 to 2600, y 1600 to 1840: is the down-left flow at 145 to
   155 deg a real turning movement or a registration artefact?
5. C3896 at 99.3 to 102.1 s, 122.3 to 127.7 s, 181.6 to 186.2 s, 189.1 to 192.5 s and
   325.8 to 327.3 s: head D is `unknown` in v4. Is a vehicle in front of the head?

## Verification status

| Claim | v4 check | Status |
|---|---|---|
| Framing (1) | v3 checks | done; lower-left corner thin |
| Lane directions (2) | v2 modes, per-video support, masking, wrong_way_safe gate; tests | done; no labelled event |
| Head identity and use (3) | analyze.py takes D from the inventory, fails loudly; eda.json regenerated; tests | done |
| Signal timelines (3) | v4 gap-free intervals, explicit unknown and imputed; tests | done; human spot check open |
| What D means (3) | full-cycle held-out tests, periodicity-preserving nulls, far offset with bootstrap | done; physical control unknown |
| Brightness cause (7) | per-frame exposure metadata | exposure ruled out; cause open |
| Stop line (5), zones (4, 6) | zone editor | open (PENDING) |
| Counts, recall (9) | hand-count 20 frames per video | open |

## Corrections

From v3 to v4:
- "D controls the near carriageway only" and "the far carriageway does not respond
  to any readable head" are retracted. The v3 test looked only at [-20, +40) s
  around green onset. Over the full cycle the far carriageway is cycle-locked with a
  different phase (low-rate stretch 47 to 61-65 s after D green onset). D is a timing
  proxy for the near-carriageway flow; what it controls is unknown.
- The near-carriageway test used 8 (daylight) and 6 (evening) onsets, not 9 and 6.
  Under a periodicity-preserving null its p-values are 0.0000 to 0.0035 (daylight)
  and 0.0025 to 0.011 (evening), not "< 0.001" for both.
- "Accepted" direction cells were described as reliable directions. 63 accepted
  cells hold 2 or more heading modes and 13 have a dissenting video; only the 445
  `wrong_way_safe` cells support a wrong-way call. `scene_reference.json` is
  superseded by `scene_reference_v2.json`.
- The per-video eda.json files for C3896 and C3902 still exposed
  `vehicle_light_guess` = head A (pedestrian). Removed from all four, together with
  `signal_colour_summary` and `traffic_lights_detected`; analyze.py no longer
  selects a light by correlation.
- The v3 phase files hid imputation (a 3 s "dark" phase in C3896 that was an
  occlusion) and had 62 gaps (6.21 s) in the D files. Replaced by v4 intervals.
- Phase durations were one sample short: red 33.0 / 36.0 s, red_amber, amber and
  blinking green 3.0 s (not 32.9 / 35.9 / 2.9 s).
- "Classified 99 to 100%" in v3 was coverage, not accuracy; no human label exists.
- `C3905/notes.json` (v1 reviewed notes with retracted claims) is renamed
  `notes_v1_RETRACTED.json` and no longer used; every eda.json now starts its notes
  with a pointer to this summary, and the automatic lighting note no longer blames
  auto exposure. `C3905/signal_timeline.csv` (v1 HSV reading) is renamed
  `signal_timeline_v1_DEPRECATED.csv`.
- Reproducibility note: the regenerated eda.json files were decoded on the Mac;
  mean-luma values differ from the v3 server decode by up to 0.7 of 255 (e.g. C3896
  95.5 -> 94.8 over the first 10 s). Tracks, flows and every non-luma number are
  identical. The lighting row of the table above keeps the server values.

From v2 to v3 (kept):
- C3905 framing is a third set-up (1.1 deg rotation, 1.1% scale), not "C3897 with a
  ~25 px shift".
- The v2 C3905 vehicle red "about 39 s" was red followed by red and amber lit
  together; the blinking end of green was read as "none".
- There is no single ~80 s cycle: 75 s at midday, 80 s in the evening.
- Daylight lamps are readable once the classifier is calibrated per head.
- The camera drifts up to 30 px during the first ~30 s of a recording.

From v1 to v2 (kept):
- The signal cycle row (~45 s red / ~35 s green) mixed pedestrian and vehicle heads.
- "A fifth to a quarter of pedestrians walk on vehicle lanes" was overstated.
- The bus stop exclusion zone was not supported in all videos.
- "Daylight makes C3897 40% unclear" referred to the pedestrian head.
