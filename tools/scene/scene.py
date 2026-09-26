"""Scene accessor for the event rules: lane-direction modes and the wrong-way
gate, in REFERENCE coordinates (C3897, 4K px, 3840 x 2160).

    from tools.scene import scene
    from tools.scene.register import to_ref
    x, y = to_ref([[gx, gy]], registration)[0]      # a ground point from the video -> reference
    scene.heading_modes(x, y)   # [{"heading_deg", "share", "tracks", "per_video_tracks"}, ...], largest first
    scene.wrong_way_safe(x, y)  # True only where a wrong-way call from heading alone is allowed
    scene.against_flow(x, y, heading_deg)   # True / False in safe cells, None elsewhere

Headings are image-coordinate degrees in the reference frame: 0 = +x (right),
90 = +y (down the image). A vehicle heading measured in a video must be moved
into the reference with the local Jacobian of H_video_to_ref (see
tools/scene/scene_reference.py, jac_map) before comparing.

Data: reports/eda/scene_reference_v2.json (PROVISIONAL; built from tracker
output of four videos, no labelled wrong_way event exists). The v1 file
scene_reference.json is SUPERSEDED and must not be used by rules: its accepted
cells can hold two directions.

Zones (stop line, zebras, parking bay, live lanes) are human-drawn and
PROVISIONAL in reports/eda/zones.json (see tools/scene/CONTRACT.md).
zones() raises if the file is absent.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SCENE_V2 = os.path.join(ROOT, "reports", "eda", "scene_reference_v2.json")
ZONES = os.path.join(ROOT, "reports", "eda", "zones.json")
W, H = 3840, 2160


class Scene:
    def __init__(self, path=SCENE_V2):
        d = json.load(open(path))
        if d.get("version") != "v2":
            raise ValueError(f"{path}: expected scene_reference v2, got {d.get('version')}")
        self.meta = {k: v for k, v in d.items() if k != "cells"}
        self.cw, self.ch = d["cell_px"]
        self.cells = {(c["i"], c["j"]): c for c in d["cells"]}

    def cell_index(self, x, y):
        if not (0 <= x < W and 0 <= y < H):
            return None
        return int(x // self.cw), int(y // self.ch)

    def cell(self, x, y):
        """The v2 cell record at reference point (x, y), or None (outside the frame or no moving vehicles)."""
        k = self.cell_index(x, y)
        return None if k is None else self.cells.get(k)

    def heading_modes(self, x, y):
        c = self.cell(x, y)
        return [] if c is None else [dict(m) for m in c["modes"]]

    def dominant_heading(self, x, y):
        c = self.cell(x, y)
        return None if c is None or not c["modes"] else c["dominant_heading"]

    def wrong_way_safe(self, x, y):
        c = self.cell(x, y)
        return bool(c is not None and c["wrong_way_safe"])

    def against_flow(self, x, y, heading_deg, tol_deg=90.0):
        """True if heading_deg is more than tol_deg from the cell's single permitted
        direction, False if not; None where the cell is not wrong_way_safe (no call)."""
        c = self.cell(x, y)
        if c is None or not c["wrong_way_safe"]:
            return None
        d = abs((heading_deg - c["dominant_heading"] + 180.0) % 360.0 - 180.0)
        return bool(d > tol_deg)


_default = None


def _scene():
    global _default
    if _default is None:
        _default = Scene()
    return _default


def heading_modes(x, y):
    return _scene().heading_modes(x, y)


def dominant_heading(x, y):
    return _scene().dominant_heading(x, y)


def wrong_way_safe(x, y):
    return _scene().wrong_way_safe(x, y)


def against_flow(x, y, heading_deg, tol_deg=90.0):
    return _scene().against_flow(x, y, heading_deg, tol_deg)


def zones(path=ZONES):
    """Hand-drawn provisional zones in reference px; raises if absent."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"zones missing ({path}); see tools/scene/CONTRACT.md")
    return json.load(open(path))
