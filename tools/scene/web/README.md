# Scene tools: zone editor and signal spot-check

Two browser pages that run next to the labeling tool (`tools/labeler/`): same
server, same token, same service (`salen-labeler`). No build step, no CDNs.

| File | What |
|---|---|
| `zones.html` | zone editor: draw the scene zones once on the reference frame, check them warped into all 4 videos |
| `spotcheck.html` | signal spot-check: a human checks the automatic lamp readings against 4K crops |
| `tools.html` | tiny index of the team tools |
| `build_data.py` | stdlib; writes `scene/scene_data.json`, `spotcheck/items.json` and copies the pack crops |
| `make_backgrounds.py` | server (OpenCV); 4K median of the C3897 original (`ref_4k.jpg`) and the per-video background JPEGs |
| `context_strips.py` | server (ffmpeg); 13 crops 0.2 s apart around every spot-check moment |
| `deploy.sh` | builds the data and copies pages, data and `server.py` to the server, restarts the service |

The zone evidence overlay comes from `tools/scene/zone_evidence.py`
(`reports/eda/zone_evidence.json`).

## Zone editor (`zones.html`)

1. Click a zone type under **Add a zone** (hover it to see what the model uses it for).
2. Click the corners on the frame. Zoom with the wheel or `+` `-`, pan by dragging (or
   Space + drag), `F` fits. The loupe (bottom right) shows the 4K pixels under the
   cursor while you place or drag points. For a stop line or live lane pick the
   approach or direction in the side panel.
3. `Enter`, a double-click or a click on the first point closes the shape.
   `Backspace` removes the last point while drawing.
4. To fix a zone: click it, drag a point, click an edge to insert a point, `Delete`
   removes the selected point, arrow keys nudge it by 1 px (Shift: 10 px). Rename in
   the list. `Ctrl+Z` / `Ctrl+Shift+Z` undo and redo.
5. Press **Verify in 4 videos** (`V`): every zone is warped into each video's own
   background with `H_ref_to_video`. The signal-head boxes and the landmarks are fixed
   objects: if they sit on the heads in all four panels, the alignment is right. Clear
   the **Check** list, then **Export JSON**.

Saving: about 2 s after every change (`saved at hh:mm:ss`); every version goes to
the history (the **History** button can load any old version); every change is also
mirrored to the browser, and on load the page offers to restore changes that never
reached the server. Incomplete shapes are saved too and listed under **Check**.

### zones.json schema

`~/wiut/scene/zones.json` (and the **Export JSON** download):

```json
{
 "version": 12,
 "frame": "C3897 reference",
 "image_size": [3840, 2160],
 "coords": "reference 4K px",
 "zones": [
  {"id": "zmufui880zs8", "type": "stop_line", "name": "near stop line",
   "points": [[1210.5, 1402.0], [2231.0, 1275.5]], "attrs": {"approach": "near_carriageway"}},
  {"id": "zmufuk1m2abc", "type": "live_lane", "name": "near lanes",
   "points": [[...], [...], [...]], "attrs": {"direction": "near_carriageway"}}
 ],
 "_meta": {"saved_at": "...", "edited_by": "...", "editors": ["..."],
           "types": {"stop_line": "polyline", "zebra_crossing": "polygon", "...": "..."}, "note": "..."}
}
```

| type | geometry | attrs | used for |
|---|---|---|---|
| `stop_line` | polyline, 2+ points | `approach`: `near_carriageway` / `far_carriageway` / `side_road` | red_light, stop_line |
| `zebra_crossing` | polygon, 3+ | none | jaywalking, failure_to_yield (answer 3) |
| `parking_bay` | polygon, 3+ | none | ignore for stopped_vehicle (answer 4) |
| `live_lane` | polygon, 3+ | `direction`: `near_carriageway` (towards camera) / `far_carriageway` (away) / `turn` | stopped_vehicle only in live lanes (answer 4), wrong_way |
| `bus_stop` | polygon, 3+ | none | a bus standing here in a live lane counts (answer 2) |
| `non_carriageway` | polygon, 3+ | `kind` (optional): `median` / `island` / `sidewalk` / `other` | not carriageway (jaywalking bounds) |

Polygons are stored open (the last point connects back to the first). Every zone may
also carry `attrs.note` (text). Points are reference C3897 4K pixels; move them into a
video with `tools/scene/register.py` (`to_video(points, r)`) or with `H_ref_to_video`
from `reports/eda/registration.json`. Signal heads are not stored here; they stay in
`reports/eda/signals/signal_heads.json` and are shown read-only.

## Signal spot-check (`spotcheck.html`)

1. Open the page; it starts at the first unanswered item. The 17 rare automatic labels
   (red+amber, amber, blinking green, occluded) come first.
2. Look at the large crop (nearest-neighbour upscaled), the previous and next sample of
   the same head, and the strip of 13 frames from -1.2 s to +1.2 s (hover a frame to see
   it large, `P` plays them; this is how blinking green and amber show).
3. Press one key: `1`/`R` red, `2`/`M` red+amber, `3`/`A` amber, `4`/`G` green,
   `5`/`B` blinking green, `6`/`D` dark/off, `7`/`O` occluded, `8`/`U` unsure. It saves
   and moves on. The automatic label is hidden until you answer (untick to show it).
4. `Left` / Backspace goes back to change an answer; `Right` skips; `N` jumps to the
   first unanswered. Answers are saved one by one; if the server is unreachable they
   wait in the browser and are sent when it is back.
5. **Results**: agreement overall, per head, per video, per automatic label, confusion
   table (auto vs human), all disagreements with crops, export JSON or CSV
   (`unsure` answers are left out of the agreement).

Answers: `~/wiut/scene/spotcheck_answers.json`
(`{"version", "answers": {id: {"label", "at", "by", "auto_hidden"}}, "_meta"}`), every
change appended to `~/wiut/scene/history/spotcheck_log.jsonl`, a full snapshot every
25 versions in `~/wiut/scene/history/spotcheck/`.

## API (same token rules as the labeling API: wrong or missing token = 404)

| Method | Path | What |
|---|---|---|
| GET | `/api/zones` | current zones.json (`&download=1` as a file) |
| PUT | `/api/zones` | body `{zones, base_version, edited_by, force?}`; 409 with the server copy on a version conflict; 400 on malformed structure (unknown type or attribute, point outside 0..3840 x 0..2160) |
| GET | `/api/zones/history` | saved versions, newest first; `?file=<name>` returns one |
| GET | `/api/spotcheck` | all answers |
| PUT | `/api/spotcheck` | body `{id, label, by, auto_hidden}`; `label: null` clears |
| GET | `/api/spotcheck/export` | `?format=csv` or JSON, `&download=1` as a file |

## Server layout

```
/var/www/salen-label/<TOKEN>/zones.html, spotcheck.html, tools.html
/var/www/salen-label/<TOKEN>/scene/            ref_1080.jpg, ref_4k.jpg, bg_<STEM>.jpg, scene_data.json
/var/www/salen-label/<TOKEN>/spotcheck/        items.json, crops/<STEM>/*.jpg, strips/<STEM>/*_strip.jpg
~/wiut/scene/assets/                           ref_4k.jpg, bg_<STEM>.jpg, zone_evidence.json (server-made)
~/wiut/scene/strips/                           context strips + strips.json (server-made)
~/wiut/scene/zones.json, spotcheck_answers.json, spotcheck_items.json, history/
```

## Rebuild and deploy

```bash
# server, once (outputs are kept; --force redoes them)
~/wiut/.venv/bin/python make_backgrounds.py --original ~/wiut/originals/C3897.MP4 \
    --ref-1080 reference_C3897.jpg --medians ~/wiut/verify/bg --out ~/wiut/scene/assets   # ~1 min
python3 context_strips.py --csv spotcheck.csv --heads signal_heads.json \
    --originals ~/wiut/originals --proxies ~/wiut/proxies --out ~/wiut/scene/strips      # ~25 min, nice 19
~/wiut/.venv/bin/python zone_evidence.py --tracks-dir ~/wiut/eda \
    --registration registration.json --out zone_evidence.json                            # ~5 s
# Mac, repo root
bash tools/scene/web/deploy.sh
```

Local test: run `server.py` with `LABELER_BASE` pointing at a folder with
`labeler/token` and `proxies/<STEM>.probe.json`, and serve a folder holding the
`build_data.py` output plus the pages under `/<token>/`, with `/api/*` proxied to it.

## Limitations

- The reference image in the editor is a 15-frame 4K median of C3897 (moving traffic
  removed; long-parked cars remain). Zones are exact in the reference; in the other
  videos they are as good as the homography: landmark error median 0.14 px (C3896),
  0.95 px (C3902), 1.12 px (C3905), max 2.3 px, and the lower-left corner of the evening
  videos is covered by one landmark only.
- The verification backgrounds are 1080p medians (2 px of 4K per pixel), not 4K.
- The camera settles during the first ~30 s of a clip (C3896 starts 30 px off); the
  verification view shows the settled position only.
- Evidence overlays are detector output (YOLO11m + ByteTrack, stop rule 25 px/s for
  2 s), mapped with the same homographies; they guide, they do not decide.
- Context strips of C3905 and the other videos come from the 4K originals; if an
  original is missing, `context_strips.py` falls back to the 1080p proxy (upscaled 2x),
  and the page says which source a strip came from.
- One editor at a time: two open tabs saving zones get a conflict prompt, not a merge.
