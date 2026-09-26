"""Lock-in tests for the files the model's rules consume (tools/scene/CONTRACT.md).

Run from the repo root:
  uvx --with pandas --with numpy --with opencv-python-headless --with matplotlib pytest -q tools/scene/tests
"""
import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from tools.scene import scene, signal  # noqa: E402

EDA = os.path.join(ROOT, "reports", "eda")
V4 = os.path.join(EDA, "signals", "v4")
STEMS = ["C3905", "C3896", "C3897", "C3902"]
FPS = 30000 / 1001


def n_frames(stem):
    return json.load(open(os.path.join(EDA, stem, "eda.json")))["extra"]["video"]["frames"]


def intervals(stem, head):
    return pd.read_csv(os.path.join(V4, f"{stem}_{head}_intervals_v4.csv"))


@pytest.mark.parametrize("stem", STEMS)
@pytest.mark.parametrize("head", ["D", "A"])
def test_intervals_tile_the_video(stem, head):
    iv = intervals(stem, head)
    assert iv.start_frame.iloc[0] == 0
    assert (iv.end_frame > iv.start_frame).all()
    # no gaps, no overlaps
    assert (iv.start_frame.to_numpy()[1:] == iv.end_frame.to_numpy()[:-1]).all()
    # coverage = the whole video
    assert iv.end_frame.iloc[-1] == n_frames(stem)
    assert (iv.end_frame - iv.start_frame).sum() == n_frames(stem)
    assert abs(iv.duration_s.sum() - n_frames(stem) / FPS) < 0.01
    # consecutive intervals differ in state (maximal runs)
    assert (iv.state.to_numpy()[1:] != iv.state.to_numpy()[:-1]).all()


@pytest.mark.parametrize("stem", STEMS)
@pytest.mark.parametrize("head", ["D", "A"])
def test_samples_agree_with_intervals(stem, head):
    iv = intervals(stem, head)
    smp = pd.read_csv(os.path.join(V4, f"{stem}_{head}_samples_v4.csv"))
    vocab = signal.VEHICLE_STATES if head == "D" else signal.PED_STATES
    assert set(iv.state) <= set(vocab) and set(smp.state) <= set(vocab)
    assert set(smp.observation) <= {"observed", "occluded", "low_confidence"}
    assert (smp[smp.observation == "occluded"].state == "unknown").all()
    k = np.searchsorted(iv.start_frame.to_numpy(), smp.frame.to_numpy(), side="right") - 1
    st = iv.state.to_numpy()[k]
    obs = smp.observation.to_numpy() == "observed"
    # an observed reading is never overwritten or hidden
    assert (st[obs] == smp.state.to_numpy()[obs]).all()
    # a known interval contains at least one observed sample; imputed samples are counted
    known = iv[iv.state != "unknown"]
    assert (known.n_observed >= 1).all()
    assert ((known.n_observed + known.n_imputed) == known.n_samples).all()
    assert (iv[iv.state == "unknown"].n_observed == 0).all()


@pytest.mark.parametrize("stem", STEMS)
def test_no_long_occlusion_is_forward_filled(stem):
    """A non-observed stretch longer than 1 s must never carry a state."""
    smp = pd.read_csv(os.path.join(V4, f"{stem}_D_samples_v4.csv"))
    sig = signal.load(stem)
    bad = (smp.observation != "observed").to_numpy()
    fr = smp.frame.to_numpy()
    i = 0
    while i < len(bad):
        if not bad[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(bad) and bad[j + 1]:
            j += 1
        end = fr[j + 1] if j + 1 < len(fr) else sig.n_frames
        if end - fr[i] > 30:
            for f in range(fr[i], end):
                assert sig.state_at_frame(f).state == "unknown", (stem, f)
        i = j + 1


def test_head_used_is_D_never_A():
    heads = json.load(open(os.path.join(EDA, "signals", "signal_heads.json")))["heads"]
    assert signal.vehicle_head_id() == "D"
    assert heads["D"]["type"] == "vehicle" and heads["D"]["lamps"] == 3
    assert heads["A"]["type"] == "pedestrian"
    for stem in STEMS:
        s = signal.load(stem)
        assert s.head == "D" and s.kind == "vehicle"
        e = json.load(open(os.path.join(EDA, stem, "eda.json")))["extra"]
        assert e["signal"]["vehicle_head"]["head"] == "D"
        assert e["signal"]["vehicle_head"]["lamps"] == 3
        for legacy in ("vehicle_light_guess", "signal_colour_summary", "traffic_lights_detected"):
            assert legacy not in e, (stem, legacy)
    src = open(os.path.join(ROOT, "tools", "eda", "analyze.py")).read()
    assert "corrcoef" not in src.split("def load_signal")[1].split("def state_per_second")[0]
    assert not re.search(r"best_light|vehicle_light_guess\"\] =", src)


def test_analyze_fails_loudly_without_head_inventory(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "tools", "eda"))
    import analyze
    with pytest.raises(SystemExit, match="signal heads file missing"):
        analyze.load_signal("C3896", str(tmp_path / "nope.json"), os.path.join(EDA, "registration.json"), V4)
    # and with an inventory whose vehicle head is ambiguous
    heads = json.load(open(os.path.join(EDA, "signals", "signal_heads.json")))
    heads["heads"]["A"].update({"type": "vehicle", "lamps": 3})
    p = tmp_path / "heads.json"
    p.write_text(json.dumps(heads))
    with pytest.raises(SystemExit, match="exactly one"):
        analyze.load_signal("C3896", str(p), os.path.join(EDA, "registration.json"), V4)


def test_c3896_fake_dark_phase_is_unknown():
    sig = signal.load("C3896")
    for f in range(5676, 5767):
        st = sig.state_at_frame(f)
        assert st.state == "unknown", f
        assert st.state != "dark"
    iv = sig.intervals()
    assert "dark" not in set(iv.state)


@pytest.mark.parametrize("stem", STEMS)
@pytest.mark.parametrize("head", ["D", "A"])
def test_state_at_every_boundary(stem, head):
    sig = signal.load(stem, head=head)
    vocab = signal.VEHICLE_STATES if head == "D" else signal.PED_STATES
    iv = sig.intervals()
    for r in iv.itertuples():
        for t in (r.start_t, r.start_frame / FPS, (r.end_frame - 1) / FPS):
            st = sig.state_at(t)
            assert st.state in vocab
            assert st.state == r.state, (stem, head, t)
            assert st.observation in ("observed", "imputed", "occluded", "low_confidence")
        if r.Index + 1 < len(iv) and r.state != "unknown":
            # the last frame before a change carries the one-sample uncertainty
            assert sig.state_at_frame(r.end_frame - 1).uncertainty_s > 0
    assert sig.state_at(-1).observation == "out_of_range"
    assert sig.state_at(sig.n_frames / FPS + 1).observation == "out_of_range"


def test_new_video_without_timeline_is_unknown():
    sig = signal.load("NEW_VIDEO_WITHOUT_TIMELINE")
    assert not sig.available
    assert sig.state_at(10.0).state == "unknown"
    with pytest.raises(FileNotFoundError):
        signal.load("NEW_VIDEO_WITHOUT_TIMELINE", strict=True)


def test_stdlib_signal_not_shadowed():
    """Scripts run from tools/scene have that folder first on sys.path."""
    code = ("import sys; sys.path.insert(0, %r); import signal, multiprocessing; "
            "assert hasattr(signal, 'SIGINT') and callable(signal.signal)") % os.path.join(ROOT, "tools", "scene")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_wrong_way_safe_cell_29_21_false():
    s = scene.Scene()
    assert s.cell_index(2360, 1720) == (29, 21)
    c = s.cell(2360, 1720)
    assert c is not None and c["any_video_dissents"]
    assert len(c["modes"]) >= 2
    assert scene.wrong_way_safe(2360, 1720) is False
    assert scene.against_flow(2360, 1720, 200.0) is None


def test_wrong_way_safe_cells_meet_the_gate():
    d = json.load(open(os.path.join(EDA, "scene_reference_v2.json")))
    p = d["params"]
    safe = [c for c in d["cells"] if c["wrong_way_safe"]]
    assert safe
    for c in safe:
        assert c["accepted"] and not c["any_video_dissents"] and not c["frame_edge"]
        assert not c["turning_or_conflict"] and c["tracks"] >= p["safe_tracks"]
        assert c["modes"][0]["share"] >= p["safe_share"]
        assert sum(1 for v in c["per_video"].values() if v["tracks"] >= 3) >= 2


def test_v1_outputs_marked_superseded():
    assert json.load(open(os.path.join(EDA, "scene_reference.json")))["superseded_by"] == "scene_reference_v2.json"
    v = json.load(open(os.path.join(EDA, "signals", "signal_validation.json")))
    assert v["superseded_by"] == "signal_validation_v2.json"


def test_contract_files_exist():
    """Every FROZEN / PROVISIONAL file listed in CONTRACT.md exists."""
    txt = open(os.path.join(ROOT, "tools", "scene", "CONTRACT.md")).read()
    rows = re.findall(r"^\| `([^`]+)` \|.*\| (FROZEN|PROVISIONAL|DEPRECATED|PENDING) \|", txt, re.M)
    assert len(rows) >= 8
    for path, status in rows:
        paths = [path.replace("<STEM>", s).replace("<HEAD>", h) for s in STEMS for h in ("D", "A")] \
            if "<STEM>" in path else [path]
        for q in paths:
            exists = os.path.exists(os.path.join(ROOT, q))
            if status in ("FROZEN", "PROVISIONAL"):
                assert exists, q
            if status == "PENDING":
                assert not exists or q.endswith("zones.json"), q
