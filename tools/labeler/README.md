# Event labeling tool

Browser tool for marking temporal event segments `[start_sec, end_sec, label]`
on the sample videos. It writes labels in the official ground-truth shape used
by `evaluate.py`.

- `server.py`: stdlib Python API (no dependencies), listens on 127.0.0.1:8766.
- `index.html`: the whole page (HTML, CSS, JS in one file, no build step, no CDNs).

## How it runs on the server

```
~/wiut/labeler/server.py        API, systemd unit salen-labeler (User=deployer, Restart=always)
~/wiut/labeler/index.html       source copy of the page
~/wiut/labeler/token            secret token (chmod 600, not in git)
/var/www/salen-label/<TOKEN>/index.html          page, served by Caddy
/var/www/salen-label/<TOKEN>/media/<STEM>_720p.mp4  hard links to ~/wiut/proxies/<STEM>_720p.mp4
~/wiut/labels/<STEM>.json                        current labels per video
~/wiut/labels/history/<STEM>/<utc-time>_v<N>.json  every saved version
```

Caddy site block (`label-salen.<host>`): `/api/*` goes to 127.0.0.1:8766, everything
else is `file_server` rooted at `/var/www/salen-label`. The page and media live in a
directory named after the token, so a wrong token is a 404 and Caddy handles video
Range requests (206). The API checks the token on every request (`X-Token` header or
`?t=`) with `hmac.compare_digest` and answers 404 when it is wrong or missing.

The page URL is `https://label-<host>/<TOKEN>/`.

### Setup from scratch

```bash
mkdir -p ~/wiut/labeler ~/wiut/labels/history
(umask 077; python3 -c "import secrets;print(secrets.token_urlsafe(18))" > ~/wiut/labeler/token)
cp server.py index.html ~/wiut/labeler/
T=$(cat ~/wiut/labeler/token)
sudo mkdir -p /var/www/salen-label && sudo chown deployer:deployer /var/www/salen-label
mkdir -p /var/www/salen-label/$T/media
for s in C3905 C3896 C3897 C3902; do ln -f ~/wiut/proxies/${s}_720p.mp4 /var/www/salen-label/$T/media/; done
cp ~/wiut/labeler/index.html /var/www/salen-label/$T/
sudo systemctl enable --now salen-labeler
```

After editing `index.html`, copy it to `/var/www/salen-label/$T/index.html` again.
If a proxy file is regenerated (new inode), redo the `ln -f` line.

Local test: `LABELER_BASE=/path/with/labeler+proxies+eda python3 server.py`
(port via `LABELER_PORT`).

## API

All endpoints need the token. Unknown paths and bad tokens return 404.

| Method | Path | What |
|---|---|---|
| GET | `/api/videos` | list: name, duration, fps, media url (relative to the page), assigned_to, n_events, version |
| GET | `/api/labels?video=C3897.MP4` | `{events: [{start, end, label, note}], version, saved_at, labeled_by, ...}` |
| PUT | `/api/labels?video=C3897.MP4` | body `{events: [{start, end, label, note}], labeled_by, base_version, force?}` |
| GET | `/api/export` | merged ground_truth.json of all videos that have a labels file (`&all=1` adds empty ones, `&download=1` sets a file name) |
| GET | `/api/signal?video=...` | signal phases of one auto-detected traffic light, for the hint strip |
| GET, PUT | `/api/zones`, `/api/zones/history` | zone editor (`tools/scene/web/zones.html`) |
| GET, PUT | `/api/spotcheck`, `/api/spotcheck/export` | signal spot-check (`tools/scene/web/spotcheck.html`) |

The zone editor and the spot-check pages live in the same token folder
(`zones.html`, `spotcheck.html`, `tools.html`) and are linked from the header of the
labeling page. Their data files are in `~/wiut/scene/`; details, schema and deploy
steps in `tools/scene/web/README.md`.

PUT validation: label must be one of the 14 classes, start/end numbers with
`0 <= start < end <= duration` (rounded to 3 decimals), else 400 with a problem list.
Same-class overlaps are accepted and returned as `warnings`. If `base_version` is not
the current version the answer is 409 with the server copy (the page then asks the
user which version to keep; `force: true` overwrites). A save with identical events
returns `unchanged` and writes nothing.

## File format

`~/wiut/labels/C3897.json`:

```json
{
 "C3897.MP4": {"duration": 317.82, "fps": 29.97, "events": [[3.0, 9.25, "red_light"]]},
 "_meta": {"labeled_by": "Abdulhamid", "editors": ["Abdulhamid"], "notes": [""],
           "version": 4, "saved_at": "2026-09-24T13:43:58.062+00:00", "assigned_to": "Abdulhamid"}
}
```

`_meta.notes[i]` belongs to `events[i]` (events are stored sorted by start).
`/api/export` drops `_meta`, so its output is exactly the organizers' ground-truth shape.
Writes are atomic (temp file, fsync, rename) and every version is also written to
`history/`.

## Page behaviour

- Autosave about 2 s after each change; status shows `saved at hh:mm:ss`, `saving...`
  or `OFFLINE, retrying`. Every change is mirrored to localStorage first. On load, if
  the browser holds unsaved changes newer than the server copy, the page offers to
  restore them.
- Marks snap to the start of the frame on screen (29.97 fps, frame n starts at n*1001/30000 s).
- The signal strip under the timeline comes from the EDA signal detector (one light,
  `vehicle_light_guess` or the most active one). It is only a hint for red_light and
  stop_line; no model detections are shown as event suggestions.
