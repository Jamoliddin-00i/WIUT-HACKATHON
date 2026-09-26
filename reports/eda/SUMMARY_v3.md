> SUPERSEDED by SUMMARY.md (v4, 2026-09-24). Kept unchanged below for traceability; several claims here were corrected in v4 (see its Corrections list).

# EDA summary across the 4 sample videos

Version 3 (2026-09-24). Version 3 runs the automatic checks from the v2
verification plan: registration against independent landmarks, signal heads
mapped by appearance and read from the 4K originals, camera metadata, and a
merged lane-direction map in one reference frame. Each finding says what was
checked, the new confidence, and what is still unverified. Corrections to v2 are
listed at the end.

Tracks: YOLO11m (imgsz 1280) + ByteTrack on a Kaggle T4, every 3rd frame, tracks
with at least 3 detections. All coordinates are 4K pixels (3840x2160). Scene
geometry is now given in the REFERENCE frame (C3897) and moved into a video with
`H_ref_to_video` from `registration.json` (`tools/scene/register.py` does this at
run time).

| | C3905 | C3896 | C3897 | C3902 |
|---|---|---|---|---|
| Recording start (camera clock, +06:00, 2026-09-18) | 18:22:21 | 12:18:24 | 12:24:47 | 17:58:18 |
| Duration | 2:08 | 5:40 | 5:18 | 5:18 |
| Exposure (constant over the clip) | f/5.6, 1/64 s, ISO 250 (+8 dB) | f/18, 1/64 s, ISO 100 | f/18, 1/64 s, ISO 100 | f/5.0, 1/64 s, ISO 100 |
| Lighting (mean luma) | lowest, 38.6 at start, 35.8 at end, +35% between 52 and 67 s | ~95, bright daylight | ~95, bright daylight | 69 falling to 58, evening |
| Framing vs reference C3897 | third set-up: 1.1 deg, 1.1% scale, points move 0.5 to 70 px | same (under 1 px) | reference | re-framed: points move 64 to 144 px |
| Signal cycle (vehicle head D) | 80 s | 75 s | 75 s | 80 s |
| Person tracklets | 541 | 770 | 692 | 1229 |
| Car tracklets | 211 | 458 | 425 | 502 |

Camera: Sony ILCE-6700, XAVC S 3840x2160 H.264 4:2:2 10-bit 140 Mbps, 29.97p,
rec709 (all four). Shutter is given as exiftool reports it (0.015625 s). Per-video
details: `camera/camera_settings.json`.

## Findings and confidence

### 1. Framing differs between recordings, and the camera settles after start. Confidence: HIGH.

**Checked.** Median backgrounds (60 frames of each 1080p proxy), SIFT + ratio test
+ RANSAC against the C3897 median (`registration.json`, overlays in
`registration/`). Independently: 14 named fixed landmarks (signs, kerb and island
corners, a manhole cover, a building sign, a lamp head, ...) found in each video by
normalized cross-correlation of gradient images, not by SIFT; and the signal-head
boxes that YOLO found in each video on its own.

| video | homography inliers / resid. median | landmark error, homography (landmarks used) | landmark error, similarity | displacement ref -> video |
|---|---|---|---|---|
| C3896 | 3962 / 0.32 px | median 0.14, max 0.99 px (14 of 14) | median 0.17, max 1.07 px | 0.5 to 0.9 px |
| C3902 | 307 / 0.89 px | median 0.95, max 2.05 px (7 of 14) | median 3.4, max 10.7 px | 64 to 144 px, median 101; (-80, +56) at the junction |
| C3905 | 372 / 0.73 px | median 1.12, max 2.32 px (7 of 14) | median 5.9, max 7.7 px | 0.5 to 70 px, median 37; (+14, +18) at the junction |

Heads C, D, E, G mapped with the homography land within 0.6 to 2.5 px of the box
YOLO found independently in each video. (A and B: 2.3 to 5.4 px, except C3905
where YOLO merged A and B into one box; F: 8 to 11 px, the YOLO box covers only
part of the housing.) C3905 relative to C3902: 93 to 120 px, so C3905 is a third
set-up, not the evening set-up of C3902.

**Answer to "is the C3905 ~25 px shift real":** yes, and v2 undersold it. It is a
1.1 deg rotation with 1.1% scale, so the shift depends on position: about 22 px
at the junction (the v2 number), up to 70 px at the frame edges.

**New: drift after recording starts.** Single frames registered to their own
video's median: C3896 starts 30 px off and is under 2 px only from 27 s on; C3902
starts 7 px off (settled from 17 s); C3905 4 px (from 3 s); C3897 under 1 px.
Short 2 to 9 px jolts also appear in the last second of C3897 and C3902. The
gyro in the camera's metadata track spikes at the same moments (button presses).

**Use.** Draw zones once on the reference, map with the homography (not the
similarity). Registration costs 10 to 15 s decode + 3 to 4.5 s matching on the
2-core server for 15 proxy frames. If the first ~30 s matter, re-register in a
sliding window.

**Still unverified.** 7 of 14 landmarks failed the NCC gate in the evening
videos (shadow and light changes); two of them (gantry post foot, left island
vertex) sit 23 and 8 px from the homography prediction, most likely template
mismatches, but the lower-left corner is covered by only one passing landmark.

### 2. Lane directions in the reference frame. Confidence: HIGH for the two main flows and for gated cells.

**Checked.** `scene_reference.json` / `.jpg`: every video's vehicle ground points
and velocities are mapped into the reference (velocity through the local
Jacobian of the homography) and binned on the same 48 x 27 grid. A cell is "ok"
only with coherence >= 0.7, at least 8 tracks, and at least 2 videos whose own
heading is within 30 deg. 635 of 778 cells pass; 143 are rejected (kept in the
file with the reason, not dropped). In 99.2% of ok cells every video with 3 or
more tracks agrees within 30 deg. Flows named by geometry: near carriageway,
top-left to bottom-right, towards the camera (384 cells, mean 26 deg); far
carriageway, right to left, away from the camera (196 cells, 198 deg); right to
left below the median line near the median nose (45 cells, 177 deg); down-left
turn into the lower-left road (10 cells, 135 deg). The per-video
`direction_field.json` warped into the reference agrees with the merged map
(median 0.6 to 1.0 deg), but that only checks the warp, not the data.

**Still unverified.** Headings come from tracker output only; no labelled
wrong_way event exists to test against.

### 3. Signal heads: inventory by appearance, readings from 4K. Confidence: HIGH for the two readable heads.

**Inventory** (`signals/signal_heads.json`, `signals/signal_heads_C3897.jpg`).
Appearance was read by eye from 4K crops of C3897 at six moments of one cycle,
not from correlation with traffic. Reference boxes in 4K px.

| head | reference box | type (appearance) | faces | most plausible control | confidence | readable |
|---|---|---|---|---|---|---|
| A | 517,964,554,1038 | pedestrian: 2 lamps, red standing figure over green walking figure | camera | pedestrians on the lower-left zebra (across the side road) | MEDIUM | yes |
| B | 549,957,594,1034 | 2-lamp, side / back view (next to A) | right | pedestrians crossing the near carriageway | LOW | no |
| C | 2244,719,2281,779 | 2-lamp, side / back view | left | pedestrians crossing the near carriageway | LOW | no |
| D | 2291,719,2340,827 | vehicle: 3 round lamps, red / amber / green | camera | signal group of the near carriageway (see validation) | HIGH for the group, LOW for which movement it physically faces | yes |
| E | 1442,493,1502,615 | tall housing hanging from the gantry, seen from behind | away, up-left | near carriageway (drivers approaching the gantry) | MEDIUM | no |
| F | 799,530,835,640 | as E | away, up-left | near carriageway | MEDIUM | no |
| G | 3782,708,3816,753 | 2-lamp, side view at the right edge | left | pedestrians on the right-hand zebra | LOW | no |

Automatic support: for A and D, pixels inside the head follow the cycle
(|corr| 0.87 to 0.99 with D's green, null 95th percentile 0.51 to 0.82; for D the
test is circular). For B, C, E, F, G the best pixel stays at or near the null in
all four videos (a few marginal exceedances at dusk, e.g. G in C3905, 0.54 vs
0.42, probably visor glow); none is usable. v2 picked A, a pedestrian head, as
"the vehicle signal" in C3896 and C3902.

**Reading.** One decode pass per 4K original on Kaggle cropped every head (A, D
every 3rd frame, the others every 15th). `tools/scene/lamp_states.py` calibrates
per head and per video: lamp positions from where each colour switches on and
off, a per-lamp colour score at the lamp centre, and a two-cluster on / off
threshold per lamp. Crops are re-aligned for drift; samples with a vehicle in
front of the head are marked occluded. Per-sample CSVs, phase CSVs and a
timeline figure per video are in `signals/`.

| head | video | BEFORE: v2 as published (1080p), unclear | BEFORE: v2 rule on the same 4K crops, classified | AFTER: classified, all samples | AFTER: classified, not occluded | lamp separation d' (red / amber / green) |
|---|---|---|---|---|---|---|
| A | C3905 | 1.7% | 98.7% | 100% | 100% | 41 / - / 32 |
| A | C3896 | 3.8% | 54.9% | 99.97% | 100% | 47 / - / 32 |
| A | C3897 | 39.6% | 50.3% | 100% | 100% | 53 / - / 32 |
| A | C3902 | 1.4% | 98.0% | 99.97% | 99.97% | 88 / - / 53 |
| D | C3905 | 1.2% | 98.1% | 98.8% | 100% | 42 / 46 / 46 |
| D | C3896 | 100% | 2.5% | 95.9% | 99.0% | 28 / 4.0 / 19 |
| D | C3897 | 100% | 2.2% | 99.4% | 100% | 32 / 6.4 / 29 |
| D | C3902 | 5.9% | 91.1% | 95.5% | 100% | 56 / 33 / 22 |

The occluded share of D is 0.6 to 4.5%. The daylight failure in v2 was the
classifier, not the video: daylight lamps are readable.

**Validation without humans** (`signals/signal_validation.json`,
`signals/discharge_vs_green_onset.png`):

- *Phase grammar of D.* Sequence red -> red+amber -> green (blinking for its last
  ~3 s) -> amber -> red. C3897, C3902, C3905: no illegal transition. C3896: three
  interruptions inside green (unclear 3.9 s and 1.2 s, dark 3.0 s); the first is
  a green bus in front of the head (checked in the crops), the other two are not
  checked. Durations are constant to 0.1 s: midday (C3896, C3897) red 32.9,
  red+amber 2.9, green 35.9, amber 2.9, cycle 74.8 to 75.2 s; evening (C3902,
  C3905) red 35.9, red+amber 2.9, green 37.9, amber 2.9, cycle 80.0 to 80.1 s.
  Two signal plans by time of day.
- *A vs D.* Synchronous, not complementary: A turns green in the same sample as
  D (lag -0.1 to +0.1 s, 15 cycles) and turns red exactly 3.0 s before D's amber,
  when D starts blinking. A is never green while D is red.
- *Queue discharge, held out.* Vehicle crossings of four gates drawn on the
  reference, per second, around D's green onsets. The response window was chosen
  on the evening videos and tested on the daylight videos, and the reverse;
  null = random onset times. Near carriageway (gate between the gantry and the
  zebra): +1.10 crossings/s in daylight (null 95th percentile 0.32, p < 0.001, 9
  onsets) and +1.27 in the evening (null 0.52, p < 0.001, 6 onsets); 86 to 96% of
  its crossings happen during D green while D is green 40 to 48% of the time.
  Far carriageway (at the median nose and at the right edge): no effect (p 0.49
  to 1.0; 50 to 60% of crossings during D green). Lower-left exit: 4 to 9
  crossings per video, mostly during D green (p 0.006 and 0.022).

**What this means for the rules.** D's state is the near-carriageway signal
(red-light running, stop line). The far carriageway does not respond to any
readable head; treat its signal state as unknown.

**Still unverified.** Which movement D physically faces (its lamps face the
camera, but the near-carriageway drivers face E and F, which are not readable);
which crossing A and the other pedestrian heads serve; amber is weak in daylight
(d' 4.0 and 6.4) and the phase order supports it more than its colour does; the
human spot check (below) is not done yet.

### 4. Parked cars at the far left edge in every video. Confidence: HIGH (not re-checked in v3).

Other long-standing vehicles (far kerb, bus stop) appear in some videos only
(C3896, C3902), and the reported durations are track lifetimes.
*Consequence:* exclude the left-edge parking bay from stopped_vehicle. Do not
blanket-exclude the bus stop area.

### 5. Stop line position. Confidence: LOW (not re-checked in v3).

Draw the painted stop line by hand on the reference frame. Near-carriageway
queues discharge through the gate between the gantry and the zebra in every
cycle, consistent with a stop line just upstream of that zebra.

### 6. Pedestrians on vehicle lanes. Confidence: LOW as a jaywalking measure (not re-checked in v3).

19 to 27% of person tracklets enter cells used by moving vehicles, but those
cells include zebras, kerbs and frame edges. Jaywalking needs hand-drawn
crossing and sidewalk polygons on the reference frame.

### 7. C3905 brightness rise. Confidence: HIGH that it is not exposure control; cause otherwise UNVERIFIED.

**Checked.** The Sony rtmd track read with
`exiftool -api LargeFileSupport=1 -ee -G3 -n` gives one record per frame
(3825 in C3905). F-number, shutter, ISO, gain and white balance each have exactly
one value over the whole clip in all four videos, so iris, shutter and gain did
not change (`camera/C3905_brightness_vs_exposure.png`). Measured on the proxy:
whole-frame luma +31% (39 to 51), bottom two thirds 43 to 58, but pixels that
are static over the clip only +19% (33.5 to 40), then all fall back at 110 to
115 s. Part of the whole-frame rise is traffic content (a white truck and buses
in the lower frame around 70 to 100 s); a uniform ~20% change remains on static
background. Without LargeFileSupport exiftool stops at the first 64-bit atom and
shows no per-frame data.

**Still unverified.** Whether the remaining ~20% is sky light (clouds at dusk)
or in-camera tone processing that the metadata does not record.
Brightness-based rules must still normalise per frame.

### 8. Stationary threshold is loose. Confidence in stop statistics: MEDIUM (not re-checked in v3).

### 9. Detector size matters for pedestrians. Confidence: MEDIUM (not re-checked in v3).

YOLO11m gave 2.0x more small person boxes per frame than YOLO11n on C3905;
recall needs a hand count.

## For human QA (in order of value)

1. `spotcheck/<STEM>_sheet.jpg`: 30 random moments per video, heads D and A,
   automatic label printed on each crop; fill `human_label` in
   `spotcheck/spotcheck.csv` (240 rows). The rare labels are the most useful:
   C3905_D_26, C3896_D_01, C3896_D_06, C3896_D_29, C3897_D_01, C3902_D_16
   (red_amber); C3896_D_10, C3896_D_11, C3902_D_07 (amber); C3897_A_26,
   C3902_D_29, C3902_A_10 (green_flash); C3896_D_17, C3897_D_12, C3902_D_19,
   C3902_D_26, C3902_D_27 (occluded).
2. C3896 at 184.5 to 185.7 s and 189.4 to 192.4 s: D reads unclear / dark inside
   green. Is a vehicle in front of the head?
3. `signals/signal_heads_C3897.jpg`: confirm the lamp counts and facing of B, C,
   E, F, G, and which approach D physically faces.
4. `registration/*_overlay.jpg`, lower-left corner (gantry post foot, left
   island): the two landmarks with 8 and 23 px error.
5. C3905 at 50 to 70 s: sky and cloud cover, for the brightness question.

## Verification status

| Claim | v3 check | Status |
|---|---|---|
| Framing (1) | SIFT vs 14 NCC landmarks + YOLO head boxes + drift | done; lower-left corner thin |
| Lane directions (2) | merged, gated reference map, per-video agreement | done |
| Signal (3) | appearance inventory, calibrated 4K reading, grammar, A vs D, held-out discharge | done; human spot check open |
| Brightness cause (7) | per-frame rtmd exposure metadata | exposure ruled out; cause open |
| Stop line (5), zones (4, 6) | draw on the reference frame by hand | open |
| Counts, recall (9) | hand-count 20 frames per video | open |

## Corrections

From v2 to v3:
- C3905 framing is not "close to C3897 with a ~25 px shift": it is a third
  set-up (1.1 deg rotation, 1.1% scale); the shift is 22 px at the junction and up
  to 70 px at the edges, and 93 to 120 px from C3902.
- The v2 C3905 vehicle red "about 39 s" is 35.9 s of red followed by 2.9 s of red
  and amber lit together; v2 read red+amber as red and the blinking end of green
  as "none". The same holds for C3902.
- There is no single ~80 s cycle: 75 s at midday (C3896, C3897), 80 s in the
  evening (C3902, C3905).
- "UNKNOWN for daylight": daylight lamps read cleanly once the classifier is
  calibrated per head (99 to 100% of unoccluded samples).
- The readable vehicle head D belongs to the near-carriageway signal group; the
  far carriageway does not respond to it.
- New: the camera drifts up to 30 px during the first ~30 s of a recording.

From v1 to v2 (kept):
- The signal cycle row (~45 s red / ~35 s green) mixed pedestrian and vehicle heads.
- The C3902 shift direction was not stated; reference-to-video is about (-129, +78)
  translation plus rotation and scale.
- "A fifth to a quarter of pedestrians walk on vehicle lanes" was overstated.
- The bus stop exclusion zone was not supported in all videos.
- C3905 lighting start and end means are 38.6 and 35.8, with the 35% rise in the middle.
- "Daylight makes C3897 40% unclear" referred to the pedestrian head.
