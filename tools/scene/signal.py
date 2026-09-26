"""Signal-state accessor for the event rules (reads the v4 timelines only).

    from tools.scene import signal
    sig = signal.load("C3896")                 # vehicle head D by default
    st = sig.state_at(123.4)                   # SignalState(state, observation, uncertainty_s)
    if st.state == "red" and st.observation in ("observed", "imputed") and st.uncertainty_s == 0: ...
    iv = sig.intervals()                       # DataFrame, half-open [start_frame, end_frame)

state        red | red_amber | amber | green | green_blink | dark | unknown
             (pedestrian head A: red | green | green_blink | dark | unknown)
observation  observed        the sample covering this frame was read cleanly
             imputed         short (<= 1 s) occlusion or weak reading bridged inside one state
             occluded / low_confidence   the state is unknown here, and why
             out_of_range    t before 0 or after the end of the video
             no_timeline     no v4 timeline exists for this video (see load())
uncertainty_s  > 0 when t lies in the last sample period before a state change:
             the next state (sig.next_state(t)) may already hold. 0 otherwise.
             inf for out_of_range / no_timeline.

Time base: t in seconds from the first frame, t = frame / (30000/1001). state_at
converts t to the NEAREST frame index; use state_at_frame() when you have the
frame index. Rules must treat "unknown" as unknown: never as red, never as green.

The head is taken from reports/eda/signals/signal_heads.json: the one readable
3-lamp vehicle head (D). The pedestrian head A is never used as a vehicle light.
D is a timing proxy for the near-carriageway flow; which movements it physically
controls is unknown (reports/eda/SUMMARY.md, finding 3).

New video: load(stem, registration=r) maps D's reference box into that video
(r from tools/scene/register.py) and returns a Signal with available = False
until its lamps have been read (kaggle_head_crops.py -> lamp_states.py ->
signal_timeline.py). Such a Signal answers "unknown" / "no_timeline" everywhere,
so no rule fires on a signal nobody read. Pass strict=True to raise instead.

Name clash guard: Python puts the folder of a script it runs first on sys.path,
so any script run as `python tools/scene/<x>.py` that (directly or through
matplotlib, subprocess, multiprocessing, ...) imports the standard library
module `signal` would get this file. When this file is imported under the bare
name "signal", it loads the real standard-library module and hands it over.
Import this accessor as `from tools.scene import signal`.
"""
if __name__ == "signal":
    import importlib.util as _ilu
    import os as _os
    import sys as _sys
    import sysconfig as _sc

    _spec = _ilu.spec_from_file_location("signal", _os.path.join(_sc.get_paths()["stdlib"], "signal.py"))
    _real = _ilu.module_from_spec(_spec)
    _sys.modules["signal"] = _real
    _spec.loader.exec_module(_real)

import bisect
import json
import math
import os
from collections import namedtuple

import numpy as np
import pandas as pd

FPS = 30000 / 1001
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SIGNALS_V4 = os.path.join(ROOT, "reports", "eda", "signals", "v4")
HEADS_FILE = os.path.join(ROOT, "reports", "eda", "signals", "signal_heads.json")
VEHICLE_STATES = ("red", "red_amber", "amber", "green", "green_blink", "dark", "unknown")
PED_STATES = ("red", "green", "green_blink", "dark", "unknown")

SignalState = namedtuple("SignalState", "state observation uncertainty_s")


def _heads(heads_file=HEADS_FILE):
    if not os.path.exists(heads_file):
        raise FileNotFoundError(f"signal head inventory missing: {heads_file}")
    return json.load(open(heads_file))["heads"]


def vehicle_head_id(heads_file=HEADS_FILE):
    """The one readable 3-lamp vehicle head in the inventory (D). Raises otherwise."""
    heads = _heads(heads_file)
    veh = [h for h, d in heads.items() if d.get("type") == "vehicle" and d.get("lamps") == 3
           and d.get("readable_from_camera")]
    if len(veh) != 1:
        raise ValueError(f"expected exactly one readable 3-lamp vehicle head, found {veh}")
    return veh[0]


class Signal:
    def __init__(self, stem, head, kind, intervals=None, samples=None, box_video=None):
        self.stem, self.head, self.kind = stem, head, kind
        self.box_video = box_video
        self.available = intervals is not None
        self._iv = intervals
        self._smp = samples
        if self.available:
            self._starts = intervals.start_frame.to_numpy()
            self._ends = intervals.end_frame.to_numpy()
            self.n_frames = int(self._ends[-1])
            self._sfr = samples.frame.to_numpy() if samples is not None else None

    def __repr__(self):
        return (f"Signal({self.stem}, head {self.head} ({self.kind}), "
                + (f"{len(self._iv)} intervals, {self.n_frames} frames)" if self.available else "no timeline)"))

    def intervals(self):
        """Copy of the v4 intervals (empty DataFrame if no timeline)."""
        return self._iv.copy() if self.available else pd.DataFrame()

    def _index(self, frame):
        return bisect.bisect_right(self._starts, frame) - 1

    def state_at_frame(self, frame):
        if not self.available:
            return SignalState("unknown", "no_timeline", math.inf)
        frame = int(frame)
        if frame < 0 or frame >= self.n_frames:
            return SignalState("unknown", "out_of_range", math.inf)
        k = self._index(frame)
        r = self._iv.iloc[k]
        state = str(r.state)
        if self._sfr is not None:
            s = self._smp.iloc[bisect.bisect_right(self._sfr, frame) - 1]
            if state == "unknown":
                obs = str(s.observation)
            else:
                obs = "observed" if (s.observation == "observed" and s.state == state) else "imputed"
        else:
            obs = "unknown" if state == "unknown" else ("imputed" if bool(r.imputed) else "observed")
        unc = 0.0
        if k + 1 < len(self._iv) and state != "unknown":
            # the change to the next interval happened after this interval's last
            # sample: frames after that sample may already be in the next state
            # (for a known state the next interval's start_uncertainty_frames is one sample step)
            step = int(self._iv.iloc[k + 1].start_uncertainty_frames)
            if self._ends[k] - frame < step:
                unc = step / FPS
        return SignalState(state, obs, round(unc, 3))

    def state_at(self, t):
        """State at time t (s). t is converted to the nearest frame index."""
        if not self.available:
            return SignalState("unknown", "no_timeline", math.inf)
        if not np.isfinite(t):
            return SignalState("unknown", "out_of_range", math.inf)
        return self.state_at_frame(int(round(t * FPS)))

    def next_state(self, t):
        """State of the interval after the one containing t (None at the end)."""
        if not self.available:
            return None
        k = self._index(int(round(t * FPS)))
        return str(self._iv.iloc[k + 1].state) if 0 <= k < len(self._iv) - 1 else None


def load(stem, head=None, registration=None, signals_dir=SIGNALS_V4, heads_file=HEADS_FILE, strict=False):
    """Signal accessor for one video.

    stem          video stem (e.g. "C3896"); v4 files <stem>_<head>_{intervals,samples}_v4.csv
    head          None = the vehicle head from the inventory (D); "A" for the pedestrian head
    registration  optional dict from tools/scene/register.py for this video; maps the head's
                  reference box into the video (box_video, 4K px)
    strict        raise FileNotFoundError when no v4 timeline exists instead of returning a
                  Signal that answers "unknown" everywhere
    """
    heads = _heads(heads_file)
    head = head or vehicle_head_id(heads_file)
    if head not in heads:
        raise KeyError(f"head {head} not in {heads_file}")
    kind = "vehicle" if heads[head]["type"] == "vehicle" else "pedestrian"
    box = None
    if registration is not None:
        from tools.scene.register import to_video  # needs OpenCV; only imported when used
        x1, y1, x2, y2 = heads[head]["box_ref"]
        p = to_video([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], registration)
        box = [float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())]
    fi = os.path.join(signals_dir, f"{stem}_{head}_intervals_v4.csv")
    fs = os.path.join(signals_dir, f"{stem}_{head}_samples_v4.csv")
    if not os.path.exists(fi):
        if strict:
            raise FileNotFoundError(f"no v4 timeline for {stem} head {head}: {fi}")
        return Signal(stem, head, kind, box_video=box)
    iv = pd.read_csv(fi)
    smp = pd.read_csv(fs) if os.path.exists(fs) else None
    allowed = VEHICLE_STATES if kind == "vehicle" else PED_STATES
    bad = set(iv.state) - set(allowed)
    if bad:
        raise ValueError(f"{fi}: states outside the {kind} vocabulary: {bad}")
    if iv.start_frame.iloc[0] != 0 or (iv.start_frame.to_numpy()[1:] != iv.end_frame.to_numpy()[:-1]).any():
        raise ValueError(f"{fi}: intervals do not tile the video")
    return Signal(stem, head, kind, iv, smp, box)
