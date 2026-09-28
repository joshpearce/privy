"""Regression tests: python -m pytest tests/  (renders are not exercised here)."""
import copy
import json
import os

import numpy as np
import pytest

from privy import bom
from privy.geometry import Box, Plate, Prism, Tube, clip
from privy.model import build, load
from privy.units import fmt_ftin, parse_len

HERE = os.path.dirname(__file__)
DESIGN = os.path.join(HERE, "..", "design", "privy.json")


def vol(geom):
    t, _ = geom.triangles()
    return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6)


@pytest.fixture(scope="module")
def cfg():
    return load(DESIGN)


@pytest.fixture(scope="module")
def model(cfg):
    return build(cfg)


def test_units():
    assert parse_len("6'-2 1/2\"") == 74.5
    assert parse_len("3/4\"") == 0.75
    assert parse_len(12) == 12.0
    assert fmt_ftin(74.5) == "6'-2 1/2\""
    assert fmt_ftin(11.75) == '11 3/4"'


def test_closed_meshes_have_correct_volume():
    assert vol(Box(0, 0, 0, 2, 3, 4)) == pytest.approx(24)
    notch = Prism("y", [[0, 0], [10, 0], [10, 2], [6, 2], [6, 1], [4, 1], [4, 2], [0, 2]], 0, 1.5)
    assert vol(notch) == pytest.approx(27)
    assert vol(Tube(0, 0, 18, 15, 0, 10)) == pytest.approx(np.pi * (18 ** 2 - 15 ** 2) * 10, rel=0.01)
    plate = Plate(("rect", 0, 0, 40, 40), 0, 1, hole=("ellipse", 20, 20, 15, 15))
    assert vol(plate) == pytest.approx(1600 - np.pi * 225, rel=0.01)


def test_sections_subtract_holes():
    plate = Plate(("rect", -70, -60, 214, 120), -36, 0, hole=("ellipse", 22, 24, 18, 18), segments=96)
    secs = plate.section(1, 24.0)          # cut through the hole centre (vertices lie on the line)
    assert len(secs) == 2
    xs = sorted((s.outer[:, 0].min(), s.outer[:, 0].max()) for s in secs)
    assert xs[0][1] == pytest.approx(4) and xs[1][0] == pytest.approx(40)


def test_clip_fast_paths():
    t = Tube(0, 0, 5, 4, -10, 10)
    c = clip(t, 2, 0.0, -1)
    assert c.bbox()[0][2] == pytest.approx(0)
    b = Box(0, 0, 0, 10, 10, 10)
    assert clip(b, 0, 20, 1) is b
    assert clip(b, 0, -5, 1) is None


def test_default_design_passes_all_checks(model):
    fails = [c["title"] for c in model.checks if c["ok"] is False]
    assert not fails, fails


def test_pipe_path_is_clear(model):
    k = model.key
    (a0, a1), (b0, b1) = k["skid_ys"]
    assert (b0 - a1) > k["OD"]
    assert k["header_z"][0] > k["pipe_top"]
    # nothing below the pipe top may occupy the slide path behind the bench-front joist
    for p in model.parts:
        if p.group.split(".")[0] in ("pit", "ground") or "removable_panel" in p.tags:
            continue
        lo, hi = p.bbox()
        if hi[0] <= 0 or lo[0] >= k["x_bj"] - 1e-6 or lo[2] >= k["pipe_top"]:
            continue
        overlap_y = min(hi[1], k["pipe_cy"] + k["OD"] / 2) - max(lo[1], k["pipe_cy"] - k["OD"] / 2)
        assert overlap_y <= 1e-6, p.id


def test_cut_list_and_purchase(model):
    rows = bom.cut_list(model)
    assert rows and all(r["qty"] > 0 and r["length"] > 0 for r in rows)
    pr = bom.purchase(model, rows)
    for r in rows:
        stock_lens = model.stock[r["stock"]].lengths
        assert r["length"] <= max(stock_lens) + 1e-6 or r["label"].startswith("4x6")


@pytest.mark.parametrize("change", [
    {"building": {"width": 54}},
    {"building": {"depth": 78}},
    {"deck": {"depth": 60}},
    {"roof": {"pitch": 4}},
    {"building": {"walls": {"stud_spacing": 16}}},
    {"foundation": {"skids": {"orientation": "flat"}}, "building": {"width": 56}},
])
def test_parameter_sweep_builds(cfg, change):
    c = copy.deepcopy(cfg)

    def merge(a, b):
        for key, val in b.items():
            if isinstance(val, dict):
                merge(a[key], val)
            else:
                a[key] = val
    merge(c, change)
    m = build(c)
    assert len(m.parts) > 200
    assert not [x["title"] for x in m.checks if x["ok"] is False and x["cat"] == "Fit"]


def test_drawings_build(model, tmp_path):
    from privy.drawing.sheets import build_sheets
    sheets = build_sheets(model, [], log=lambda *_: None, only=["A-101", "A-301", "S-201"])
    assert len(sheets) == 3
    for n, t, svg in sheets:
        assert svg.startswith("<svg") and len(svg) > 10000


def test_inswing_door_clears_bench_and_fits_under_plate(model):
    k = model.key
    assert k["door_prehung"] and k["door_swing"] == "in"
    titles = {c["title"]: c for c in model.checks}
    assert titles["In-swing door clears bench when open 90 deg"]["ok"]
    assert titles["Door header fits under sloped top plate"]["ok"]


def test_service_pedestal_and_devices(model):
    k = model.key
    tags = {d["tag"] for d in k["electrical"]["devices"]}
    assert {"MP", "DS", "S1", "S2", "R1", "R2", "L1", "L2"} <= tags
    # the pedestal is site equipment: it must not move the building's bounding box
    lo, hi = model.bbox()
    assert lo[1] > k["service"]["y"]


def test_standing_seam_panels_cover_roof(model):
    k = model.key
    panels = [p for p in model.parts if p.name == "Roofing panel"]
    assert len(panels) == k["roof_sheets"]
    xs = sorted((p.bbox()[0][0], p.bbox()[1][0]) for p in panels)
    assert xs[0][0] == pytest.approx(k["roof_x"][0]) and xs[-1][1] == pytest.approx(k["roof_x"][1])
    assert all(b[0] == pytest.approx(a[1]) for a, b in zip(xs, xs[1:]))
