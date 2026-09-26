"""Labeling API for the Salen traffic event videos.

Stdlib only. Caddy serves the page and the 720p media from an unguessable
directory named after the token; this service answers /api/* on 127.0.0.1 and
checks the token (X-Token header or ?t=) on every request. A wrong or missing
token gets 404, same as any unknown path.

Labels live in <base>/labels/<STEM>.json in the official ground-truth shape for
that one video, plus a sibling "_meta" key (labeler name, per-event notes,
version). Every save is written atomically and also copied to
<base>/labels/history/<STEM>/<timestamp>.json, so no version is ever lost.

The same service also answers the scene tools (zone editor, signal spot-check):
/api/zones, /api/zones/history, /api/spotcheck, /api/spotcheck/export. Their
files live in <base>/scene/ (see the "scene tools" section below).
"""
import hmac
import json
import math
import os
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

BASE = Path(os.environ.get("LABELER_BASE", str(Path.home() / "wiut")))
TOKEN = (BASE / "labeler" / "token").read_text().strip()
LABELS = BASE / "labels"
HISTORY = LABELS / "history"
PROXIES = BASE / "proxies"
EDA = BASE / "eda"
PORT = int(os.environ.get("LABELER_PORT", "8766"))
MAX_BODY = 2 * 1024 * 1024

# (official file name, assigned labeler)
VIDEOS = [
    ("C3905.MP4", "Jamoliddin"),
    ("C3896.MP4", "Jamoliddin"),
    ("C3897.MP4", "Abdulhamid"),
    ("C3902.MP4", "Abdulhamid"),
]
CLASSES = {
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle",
    "fire_smoke",
}
LOCK = threading.Lock()


def stem_of(name: str) -> str:
    return name.rsplit(".", 1)[0]


def video_info(name: str) -> dict:
    stem = stem_of(name)
    probe = json.loads((PROXIES / f"{stem}.probe.json").read_text())
    duration = round(float(probe["format"]["duration"]), 2)
    fps = 29.97
    for s in probe.get("streams", []):
        if s.get("codec_type") == "video":
            num, _, den = s.get("r_frame_rate", "30000/1001").partition("/")
            if float(den or 1):
                fps = round(float(num) / float(den or 1), 2)
            break
    return {"duration": duration, "fps": fps}


INFO = {name: video_info(name) for name, _ in VIDEOS}
ASSIGNED = dict(VIDEOS)


def label_path(name: str) -> Path:
    return LABELS / f"{stem_of(name)}.json"


def read_labels(name: str) -> dict:
    p = label_path(name)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def public_view(name: str, doc: dict) -> dict:
    """Labels as the page wants them: events with notes attached."""
    info = INFO[name]
    gt = doc.get(name, {})
    meta = doc.get("_meta", {})
    notes = meta.get("notes", [])
    events = []
    for i, (s, e, lab) in enumerate(gt.get("events", [])):
        events.append({"start": s, "end": e, "label": lab,
                       "note": notes[i] if i < len(notes) else ""})
    return {
        "video": name,
        "duration": info["duration"],
        "fps": info["fps"],
        "events": events,
        "version": meta.get("version", 0),
        "saved_at": meta.get("saved_at"),
        "labeled_by": meta.get("labeled_by", ""),
        "editors": meta.get("editors", []),
    }


def overlap_warnings(events: list) -> list:
    out = []
    by_cls = {}
    for ev in events:
        by_cls.setdefault(ev["label"], []).append(ev)
    for lab, evs in sorted(by_cls.items()):
        evs.sort(key=lambda x: (x["start"], x["end"]))
        for a, b in zip(evs, evs[1:]):
            if b["start"] < a["end"]:
                out.append(f"{lab}: [{a['start']}, {a['end']}] overlaps "
                           f"[{b['start']}, {b['end']}]; same-class segments "
                           f"must not overlap, merge them into one")
    return out


def validate(name: str, body) -> tuple:
    """Returns (clean_events, errors)."""
    dur = INFO[name]["duration"]
    errors = []
    if not isinstance(body, dict) or not isinstance(body.get("events"), list):
        return [], ["body must be an object with an 'events' list"]
    clean = []
    for i, ev in enumerate(body["events"]):
        w = f"event {i + 1}"
        if not isinstance(ev, dict):
            errors.append(f"{w}: not an object")
            continue
        lab = ev.get("label")
        s, e = ev.get("start"), ev.get("end")
        note = ev.get("note", "") or ""
        if lab not in CLASSES:
            errors.append(f"{w}: unknown label {lab!r}")
            continue
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
               for x in (s, e)):
            errors.append(f"{w}: start/end must be numbers")
            continue
        s, e = round(float(s), 3), round(float(e), 3)
        if e > dur and e <= dur + 0.001:
            e = dur
        if not (0 <= s < e <= dur):
            errors.append(f"{w} ({lab}): need 0 <= start < end <= {dur}, got [{s}, {e}]")
            continue
        if not isinstance(note, str):
            errors.append(f"{w}: note must be text")
            continue
        clean.append({"start": s, "end": e, "label": lab, "note": note[:2000]})
    clean.sort(key=lambda x: (x["start"], x["end"], x["label"]))
    return clean, errors


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Handler(BaseHTTPRequestHandler):
    server_version = "salen-labeler"
    sys_version = ""

    def _q(self):
        u = urlparse(self.path)
        return u.path, {k: v[0] for k, v in parse_qs(u.query).items()}

    def _authed(self, q) -> bool:
        given = self.headers.get("X-Token") or q.get("t", "")
        return hmac.compare_digest(given.encode(), TOKEN.encode())

    def _send(self, code, body, ctype="application/json", extra=None):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _json(self, code, obj, extra=None):
        self._send(code, json.dumps(obj), extra=extra)

    def _nf(self):
        return self._send(404, "not found", "text/plain")

    def _video(self, q):
        v = q.get("video", "")
        return v if v in INFO else None

    def do_GET(self):
        path, q = self._q()
        if not self._authed(q):
            return self._nf()
        if path == "/api/videos":
            out = []
            for name, who in VIDEOS:
                stem = stem_of(name)
                doc = read_labels(name)
                meta = doc.get("_meta", {})
                out.append({
                    "name": name,
                    "duration": INFO[name]["duration"],
                    "fps": INFO[name]["fps"],
                    "media": f"media/{stem}_720p.mp4",
                    "assigned_to": who,
                    "n_events": len(doc.get(name, {}).get("events", [])),
                    "version": meta.get("version", 0),
                    "saved_at": meta.get("saved_at"),
                    "labeled_by": meta.get("labeled_by", ""),
                })
            return self._json(200, {"videos": out, "classes": sorted(CLASSES)})
        if path == "/api/labels":
            name = self._video(q)
            if not name:
                return self._json(400, {"error": "unknown video"})
            return self._json(200, public_view(name, read_labels(name)))
        if path == "/api/signal":
            name = self._video(q)
            if not name:
                return self._json(400, {"error": "unknown video"})
            return self._json(200, signal_for(name))
        if path == "/api/export":
            merged = {}
            for name, _ in VIDEOS:
                doc = read_labels(name)
                if name in doc:
                    merged[name] = doc[name]
                elif q.get("all") == "1":
                    merged[name] = {**INFO[name], "events": []}
            extra = None
            if q.get("download") == "1":
                extra = {"Content-Disposition": 'attachment; filename="ground_truth.json"'}
            return self._send(200, json.dumps(merged, indent=1), extra=extra)
        if path in SCENE_GET:
            code, body, ctype, extra = SCENE_GET[path](q)
            if code == 404:
                return self._nf()
            return self._send(code, body, ctype, extra=extra)
        return self._nf()

    def _body(self):
        """(parsed JSON body, None) or (None, error message)."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if length <= 0 or length > MAX_BODY:
            return None, "bad body size"
        try:
            return json.loads(self.rfile.read(length)), None
        except (ValueError, UnicodeDecodeError):
            return None, "body is not JSON"

    def do_PUT(self):
        path, q = self._q()
        if not self._authed(q):
            return self._nf()
        if path in SCENE_PUT:
            body, err = self._body()
            if err:
                return self._json(400, {"error": err})
            code, obj = SCENE_PUT[path](body)
            return self._json(code, obj)
        if path != "/api/labels":
            return self._nf()
        name = self._video(q)
        if not name:
            return self._json(400, {"error": "unknown video"})
        body, err = self._body()
        if err:
            return self._json(400, {"error": err})
        clean, errors = validate(name, body)
        if errors:
            return self._json(400, {"error": "validation failed", "problems": errors})
        who = str(body.get("labeled_by", "") or "").strip()[:60]
        with LOCK:
            doc = read_labels(name)
            meta = doc.get("_meta", {})
            cur = int(meta.get("version", 0))
            base = body.get("base_version")
            if not body.get("force") and base is not None and int(base) != cur:
                return self._json(409, {"error": "conflict", "server": public_view(name, doc)})
            old = public_view(name, doc)["events"] if doc else None
            if old is not None and old == clean:
                return self._json(200, {"ok": True, "unchanged": True, "version": cur,
                                        "saved_at": meta.get("saved_at"),
                                        "warnings": overlap_warnings(clean)})
            editors = list(meta.get("editors", []))
            if who and who not in editors:
                editors.append(who)
            ts = now_iso()
            new_doc = {
                name: {
                    "duration": INFO[name]["duration"],
                    "fps": INFO[name]["fps"],
                    "events": [[ev["start"], ev["end"], ev["label"]] for ev in clean],
                },
                "_meta": {
                    "labeled_by": who,
                    "editors": editors,
                    "notes": [ev["note"] for ev in clean],
                    "version": cur + 1,
                    "saved_at": ts,
                    "assigned_to": ASSIGNED[name],
                },
            }
            atomic_write(label_path(name), new_doc)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            atomic_write(HISTORY / stem_of(name) / f"{stamp}_v{cur + 1}.json", new_doc)
        return self._json(200, {"ok": True, "version": cur + 1, "saved_at": ts,
                                "warnings": overlap_warnings(clean)})

    def do_POST(self):
        return self._nf()

    def do_DELETE(self):
        return self._nf()

    def log_message(self, fmt, *args):
        pass


def signal_for(name: str) -> dict:
    """Signal phases of one detected traffic light, as a hint strip only."""
    try:
        extra = json.loads((EDA / stem_of(name) / "eda.json").read_text())["extra"]
        summ = extra.get("signal_colour_summary") or []
        guess = (extra.get("vehicle_light_guess") or {}).get("light")
        pick = next((s for s in summ if s.get("light") == guess), None)
        why = "vehicle_light_guess"
        if pick is None:
            active = [s for s in summ if s.get("active")] or summ
            if not active:
                return {"available": False}
            pick = max(active, key=lambda s: s.get("share_red", 0) + s.get("share_green", 0))
            why = "most active light"
        phases = [{"state": p["state"] if p["state"] in ("red", "green", "amber") else "unclear",
                   "start": p["start"], "end": p["end"]} for p in pick.get("phases", [])]
        return {"available": True, "light": pick.get("light"), "box": pick.get("box"),
                "chosen_by": why, "phases": phases}
    except (OSError, ValueError, KeyError, TypeError):
        return {"available": False}


# ---------------------------------------------------------------- scene tools
# Zone editor (zones.html) and signal spot-check (spotcheck.html). Same token,
# same service. Files:
#   <base>/scene/zones.json                     current zones (schema below)
#   <base>/scene/history/zones/<utc>_v<N>.json  every saved zones version
#   <base>/scene/spotcheck_answers.json         current spot-check answers
#   <base>/scene/history/spotcheck_log.jsonl    every answer change, append only
#   <base>/scene/history/spotcheck/<utc>_v<N>.json  full snapshot every 25 versions
#   <base>/scene/spotcheck_items.json           copy of the page's item list (ids, auto labels)
SCENE = BASE / "scene"
SCENE_HIST = SCENE / "history"
ZONES_FILE = SCENE / "zones.json"
ANSWERS_FILE = SCENE / "spotcheck_answers.json"
SC_ITEMS_FILE = SCENE / "spotcheck_items.json"
SCENE_LOCK = threading.Lock()
IMG_W, IMG_H = 3840, 2160
ZONE_TYPES = {  # type -> (geometry, minimum points, {attr: allowed values})
    "stop_line": ("polyline", 2, {"approach": {"", "near_carriageway", "far_carriageway", "side_road"}}),
    "zebra_crossing": ("polygon", 3, {}),
    "parking_bay": ("polygon", 3, {}),
    "live_lane": ("polygon", 3, {"direction": {"", "near_carriageway", "far_carriageway", "turn"}}),
    "bus_stop": ("polygon", 3, {}),
    "non_carriageway": ("polygon", 3, {"kind": {"", "median", "island", "sidewalk", "other"}}),
}
ZONE_ID_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
SC_LABELS = {"red", "red_amber", "amber", "green", "green_flash", "dark", "occluded", "unsure"}


def _num_ok(x) -> bool:
    return not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x)


def empty_zones_doc() -> dict:
    return {"version": 0, "frame": "C3897 reference", "image_size": [IMG_W, IMG_H],
            "coords": "reference 4K px", "zones": [], "_meta": {}}


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def validate_zones(body) -> tuple:
    """Structure only. Incomplete shapes (too few points, missing attributes) are
    saved anyway so no work is lost; the page lists them as warnings."""
    if not isinstance(body, dict) or not isinstance(body.get("zones"), list):
        return [], ["body must be an object with a 'zones' list"]
    if len(body["zones"]) > 500:
        return [], ["more than 500 zones"]
    errors, clean, seen = [], [], set()
    for i, z in enumerate(body["zones"]):
        w = f"zone {i + 1}"
        if not isinstance(z, dict):
            errors.append(f"{w}: not an object")
            continue
        zid, typ, name = z.get("id"), z.get("type"), z.get("name", "")
        if not isinstance(zid, str) or not (1 <= len(zid) <= 40) or not set(zid) <= ZONE_ID_OK:
            errors.append(f"{w}: bad id {zid!r}")
            continue
        if zid in seen:
            errors.append(f"{w}: duplicate id {zid}")
            continue
        seen.add(zid)
        if typ not in ZONE_TYPES:
            errors.append(f"{w}: unknown type {typ!r}")
            continue
        if not isinstance(name, str):
            errors.append(f"{w}: name must be text")
            continue
        pts = z.get("points")
        if not isinstance(pts, list) or len(pts) > 1000:
            errors.append(f"{w} ({name}): points must be a list (at most 1000)")
            continue
        cp = []
        for p in pts:
            if (not isinstance(p, list) or len(p) != 2 or not all(_num_ok(v) for v in p)
                    or not (0 <= p[0] <= IMG_W and 0 <= p[1] <= IMG_H)):
                errors.append(f"{w} ({name}): point {p!r} is not [x, y] inside 0..{IMG_W} x 0..{IMG_H}")
                break
            cp.append([round(float(p[0]), 1), round(float(p[1]), 1)])
        else:
            attrs = z.get("attrs") or {}
            allowed = ZONE_TYPES[typ][2]
            ca = {}
            if not isinstance(attrs, dict):
                errors.append(f"{w} ({name}): attrs must be an object")
                continue
            for k, v in attrs.items():
                if k == "note" and isinstance(v, str):
                    ca["note"] = v[:500]
                elif k in allowed and v in allowed[k]:
                    ca[k] = v
                else:
                    errors.append(f"{w} ({name}): attribute {k}={v!r} not allowed for {typ}")
            clean.append({"id": zid, "type": typ, "name": name.strip()[:80], "points": cp, "attrs": ca})
    return clean, errors


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def get_zones(q):
    doc = read_json(ZONES_FILE, None) or empty_zones_doc()
    extra = None
    if q.get("download") == "1":
        extra = {"Content-Disposition": 'attachment; filename="zones.json"'}
    return 200, json.dumps(doc, indent=1), "application/json", extra


def put_zones(body):
    clean, errors = validate_zones(body)
    if errors:
        return 400, {"error": "validation failed", "problems": errors}
    who = str(body.get("edited_by", "") or "").strip()[:60]
    with SCENE_LOCK:
        doc = read_json(ZONES_FILE, None) or empty_zones_doc()
        cur = int(doc.get("version", 0))
        base = body.get("base_version")
        if not body.get("force") and base is not None and int(base) != cur:
            return 409, {"error": "conflict", "server": doc}
        meta = doc.get("_meta", {})
        if doc.get("zones") == clean and cur > 0:
            return 200, {"ok": True, "unchanged": True, "version": cur, "saved_at": meta.get("saved_at")}
        editors = list(meta.get("editors", []))
        if who and who not in editors:
            editors.append(who)
        ts = now_iso()
        new = empty_zones_doc()
        new.update({"version": cur + 1, "zones": clean, "_meta": {
            "saved_at": ts, "edited_by": who, "editors": editors,
            "types": {t: g for t, (g, _, _) in ZONE_TYPES.items()},
            "note": "points are [x, y] in reference C3897 4K px; move into a video with H_ref_to_video "
                    "(reports/eda/registration.json). Signal heads are not stored here: "
                    "reports/eda/signals/signal_heads.json",
        }})
        atomic_write(ZONES_FILE, new)
        atomic_write(SCENE_HIST / "zones" / f"{_stamp()}_v{cur + 1}.json", new)
    return 200, {"ok": True, "version": cur + 1, "saved_at": ts}


def get_zones_history(q):
    d = SCENE_HIST / "zones"
    files = sorted(d.glob("*.json"), reverse=True) if d.exists() else []
    want = q.get("file")
    if want:
        f = d / want
        if "/" in want or not want.endswith(".json") or not f.exists():
            return 400, json.dumps({"error": "unknown version"}), "application/json", None
        return 200, f.read_text(), "application/json", None
    out = []
    for f in files[:300]:
        try:
            doc = json.loads(f.read_text())
        except ValueError:
            continue
        m = doc.get("_meta", {})
        out.append({"file": f.name, "version": doc.get("version"), "saved_at": m.get("saved_at"),
                    "edited_by": m.get("edited_by", ""), "n_zones": len(doc.get("zones", []))})
    return 200, json.dumps({"versions": out, "total": len(files)}), "application/json", None


def sc_items() -> dict:
    items = read_json(SC_ITEMS_FILE, {}).get("items", [])
    return {it["id"]: it for it in items}


def empty_answers() -> dict:
    return {"version": 0, "answers": {}, "_meta": {}}


def get_spotcheck(q):
    return 200, json.dumps(read_json(ANSWERS_FILE, None) or empty_answers()), "application/json", None


def put_spotcheck(body):
    """One answer per request: {id, label (null clears), by, auto_hidden}."""
    if not isinstance(body, dict):
        return 400, {"error": "body must be an object"}
    sid, lab = body.get("id"), body.get("label")
    items = sc_items()
    if not isinstance(sid, str) or (items and sid not in items) or len(sid) > 40:
        return 400, {"error": f"unknown item {sid!r}"}
    if lab is not None and lab not in SC_LABELS:
        return 400, {"error": f"unknown answer {lab!r}"}
    who = str(body.get("by", "") or "").strip()[:60]
    with SCENE_LOCK:
        doc = read_json(ANSWERS_FILE, None) or empty_answers()
        prev = doc["answers"].get(sid)
        if (prev or {}).get("label") == lab:
            return 200, {"ok": True, "unchanged": True, "version": doc["version"],
                         "saved_at": doc["_meta"].get("saved_at")}
        ts = now_iso()
        if lab is None:
            doc["answers"].pop(sid, None)
        else:
            doc["answers"][sid] = {"label": lab, "at": ts, "by": who,
                                   "auto_hidden": bool(body.get("auto_hidden", False))}
        doc["version"] = int(doc["version"]) + 1
        doc["_meta"] = {"saved_at": ts, "n_answers": len(doc["answers"]),
                        "labels": sorted(SC_LABELS)}
        atomic_write(ANSWERS_FILE, doc)
        SCENE_HIST.mkdir(parents=True, exist_ok=True)
        with open(SCENE_HIST / "spotcheck_log.jsonl", "a") as f:
            f.write(json.dumps({"at": ts, "version": doc["version"], "id": sid, "label": lab,
                                "previous": (prev or {}).get("label"), "by": who}) + "\n")
            f.flush()
            os.fsync(f.fileno())
        if doc["version"] % 25 == 0:
            atomic_write(SCENE_HIST / "spotcheck" / f"{_stamp()}_v{doc['version']}.json", doc)
    return 200, {"ok": True, "version": doc["version"], "saved_at": ts}


def get_spotcheck_export(q):
    doc = read_json(ANSWERS_FILE, None) or empty_answers()
    items = read_json(SC_ITEMS_FILE, {}).get("items", [])
    rows = []
    for it in items:
        a = doc["answers"].get(it["id"], {})
        h = a.get("label", "")
        rows.append({"id": it["id"], "stem": it["stem"], "head": it["head"], "frame": it["frame"],
                     "t_s": it["t_s"], "auto_label": it["auto"], "image": it.get("pack_image", ""),
                     "human_label": h, "answered_at": a.get("at", ""), "by": a.get("by", ""),
                     "agree": "" if h in ("", "unsure") else int(h == it["auto"])})
    if q.get("format") == "csv":
        cols = list(rows[0].keys()) if rows else ["id", "human_label"]
        def cell(v):
            v = str(v)
            return '"' + v.replace('"', '""') + '"' if any(c in v for c in ',"\n') else v
        body = "\n".join([",".join(cols)] + [",".join(cell(r[c]) for c in cols) for r in rows]) + "\n"
        ctype, fname = "text/csv; charset=utf-8", "spotcheck_answers.csv"
    else:
        body = json.dumps({"version": doc["version"], "saved_at": doc["_meta"].get("saved_at"), "rows": rows},
                          indent=1)
        ctype, fname = "application/json", "spotcheck_answers.json"
    extra = {"Content-Disposition": f'attachment; filename="{fname}"'} if q.get("download") == "1" else None
    return 200, body, ctype, extra


SCENE_GET = {
    "/api/zones": get_zones,
    "/api/zones/history": get_zones_history,
    "/api/spotcheck": get_spotcheck,
    "/api/spotcheck/export": get_spotcheck_export,
}
SCENE_PUT = {
    "/api/zones": put_zones,
    "/api/spotcheck": put_spotcheck,
}


if __name__ == "__main__":
    LABELS.mkdir(parents=True, exist_ok=True)
    HISTORY.mkdir(parents=True, exist_ok=True)
    SCENE_HIST.mkdir(parents=True, exist_ok=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
