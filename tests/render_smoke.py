"""Smoke test for the photoreal renderer on a small toy model.

    python tests/render_smoke.py [--out DIR] [--quality 0.4] [--views hero rear cutaway]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from privy.geometry import Box, CorrugatedSheet, Plate, Tube  # noqa: E402
from privy.parts import Model, Part  # noqa: E402
from privy.render import render_views  # noqa: E402


# Wooded NC Piedmont site.  Used only while design/privy.json predates the "trees" schema;
# these are the entries proposed for design/privy.json (renders.environment + materials).
_LITTER = {"leaf_colors": ["#7a5230", "#8f6a3c", "#5f4028", "#a58050", "#6e5a38"], "needle_color": "#8a4f2a",
           "contrast": 0.3}
_CLAY = {"type": "soil", "tile": [60, 60], "base_color": "#8a5a3c", "contrast": 0.3, "seed": 5}
PIEDMONT_MATERIALS = {
    "soil": {
        "description": "Packed brown dirt pad (slight red-clay cast), ~50% covered by oak leaves and pine straw",
        "base_color": "#6e5238", "roughness": 1.0, "specular": 0.1,
        "texture": {"type": "litter", "tile": [48, 48], "dirt_color": "#6e5238", **_LITTER,
                    "leaf_density": 0.4, "needle_density": 0.45, "seed": 4},
        "side_texture": _CLAY, "drawing": {"fill": "#ffffff", "cut": "earth"}},
    "forest_floor": {
        "description": "Woodland floor: dense pine straw and brown oak leaves, some red clay showing",
        "base_color": "#7a5230", "roughness": 1.0, "specular": 0.1,
        "texture": {"type": "litter", "tile": [96, 96], "dirt_color": "#4e3624", **_LITTER,
                    "leaf_density": 1.3, "needle_density": 1.3, "seed": 6},
        "side_texture": _CLAY, "drawing": {"fill": "#ffffff", "cut": "earth"}},
    "pine_bark": {
        "description": "Loblolly pine bark: platy, reddish-brown, dark fissures",
        "base_color": "#5b4535", "roughness": 0.95, "specular": 0.2, "variation": 0.08,
        "texture": {"type": "bark", "fissure_color": "#2a1f18", "plate": [7, 3.5], "tile": [48, 24],
                    "contrast": 0.4, "seed": 12},
        "drawing": {"fill": "#ffffff", "cut": "wood"}},
    "oak_bark": {
        "description": "Post / white oak bark: grey, furrowed",
        "base_color": "#6a655c", "roughness": 0.95, "specular": 0.2, "variation": 0.08,
        "texture": {"type": "bark", "fissure_color": "#34302b", "plate": [14, 1.6], "tile": [48, 16],
                    "contrast": 0.35, "seed": 13},
        "drawing": {"fill": "#ffffff", "cut": "wood"}},
    "pine_needles": {
        "description": "Loblolly pine foliage (needle-cluster blobs; texture = per-cluster shade variation)",
        "base_color": "#3f5a2c", "roughness": 0.9, "specular": 0.3, "variation": 0.12,
        "texture": {"type": "noise", "scale": 8, "contrast": 0.3, "seed": 3},
        "drawing": {"fill": "#ffffff", "cut": "wood"}},
    "oak_leaves": {
        "description": "Oak foliage (leaf-cluster blobs; texture = per-cluster shade variation)",
        "base_color": "#4a6329", "roughness": 0.9, "specular": 0.3, "variation": 0.18,
        "texture": {"type": "noise", "scale": 8, "contrast": 0.4, "seed": 4},
        "drawing": {"fill": "#ffffff", "cut": "wood"}},
}
PIEDMONT_ENVIRONMENT = {
    "dirt_pad": {"margin": 48, "material": "soil"},
    "surround": {"size": 6000, "material": "forest_floor"},
    "ground_depth": 60,
    "trees": {
        "seed": 7, "clear_radius": 150, "outer_radius": 1100, "camera_corridor": 70,
        "species": [
            {"name": "loblolly pine", "count": 16, "height": [600, 960], "trunk_diameter": [9, 18], "form": "pine",
             "crown_fraction": 0.3, "bark_material": "pine_bark", "foliage_material": "pine_needles"},
            {"name": "small oak (post / white oak)", "count": 12, "height": [180, 360], "trunk_diameter": [4, 9],
             "form": "oak", "crown_fraction": 0.65, "bark_material": "oak_bark", "foliage_material": "oak_leaves"},
        ],
        "understory": {"count": 40, "height": [24, 72], "foliage_material": "oak_leaves"},
        "backdrop": {"count": 70, "outer_radius": 2600}},
}


def load_cfg() -> dict:
    cfg = json.load(open(os.path.join(ROOT, "design", "privy.json")))
    if "trees" not in cfg["renders"].get("environment", {}):
        cfg["renders"]["environment"] = PIEDMONT_ENVIRONMENT
        cfg["materials"].update(PIEDMONT_MATERIALS)
    return cfg


def toy_model() -> Model:
    m = Model(cfg=load_cfg(), stock={})
    P = lambda pid, geom, mat, tags=(), grain=None: m.add(  # noqa: E731
        Part(pid, pid, pid.split(".")[0], geom, mat, tags=set(tags), grain_axis=grain))
    # skids and floor framing (treated), 72" deep x 48" wide shed footprint
    P("skid.low", Box(-6, 1, 0, 78, 4.5, 5.5), "treated_lumber")
    P("skid.high", Box(-6, 43.5, 0, 78, 47, 5.5), "treated_lumber")
    for i, x in enumerate((0, 70.5)):
        P(f"floor.rim.{i}", Box(x, 0, 5.5, x + 1.5, 48, 11), "treated_lumber")
    # low-side wall left open as bare framing
    for i, x in enumerate((0, 24, 46, 70.5)):
        P(f"wall.low.stud.{i}", Box(x, 0, 11, x + 1.5, 3.5, 90), "framing_lumber")
    P("wall.low.plate", Box(0, 0, 90, 72, 3.5, 91.5), "framing_lumber")
    P("wall.low.sill", Box(0, 0, 11, 72, 3.5, 12.5), "framing_lumber")
    # high-side wall: rough board-and-batten siding (vertical grain)
    for i in range(7):
        x = i * 10.25
        P(f"wall.high.board.{i}", Box(x, 48, 6, min(x + 9.5, 72), 49, 92), "rough_siding")
        if i:
            P(f"wall.high.batten.{i}", Box(x - 1.625, 49, 6, x + 0.875, 50, 92), "rough_siding")
    # back wall (x < 0): removable siding panel
    for i in range(5):
        y = i * 9.75
        P(f"panel.board.{i}", Box(-1, y, 6, 0, min(y + 9.5, 48), 60), "rough_siding", ("removable_panel",))
    # front wall with a glazed window
    P("wall.front.sill", Box(70.5, 3.5, 44, 72, 44.5, 45.5), "framing_lumber")
    P("window.frame", Box(71.5, 12, 46, 72.5, 36, 48), "window_frame")
    P("window.glass", Box(71.9, 13, 48, 72.1, 35, 74), "glass")
    # bench top with seat hole over the pit, and the corrugated HDPE pipe
    P("bench.top", Plate(("rect", 1.5, 3.5, 42, 44.5), 22.5, 23.25, hole=("ellipse", 22, 24, 5.75, 4.5)),
      "bench_plywood")
    P("pit.pipe", Tube(22, 24, 18, 15, -36, 21.75, corrugation=(3.6, 1.25)), "hdpe")
    # covered deck in front: 4 posts + small corrugated roof
    for i, (x, y) in enumerate(((80, 0), (136, 0), (80, 44), (136, 44))):
        top = 80 + (y + 6) * 0.25
        P(f"deck.post.{i}", Box(x, y, 0, x + 3.5, y + 3.5, top), "treated_lumber")
    for i, x in enumerate(range(78, 142, 6)):
        P(f"deck.board.{i}", Box(x, 0, 6, x + 5.5, 47.5, 7), "decking")
    P("roof.sheet", CorrugatedSheet(76, 144, -6, 54, 80, 0.25, 2.67, 0.5), "galvanized")
    # ground: dirt pad (with a hole for the pipe) over a solid soil block
    P("ground.pad", Plate(("rect", -120, -120, 260, 170), -36, 0, hole=("ellipse", 22, 24, 18, 18)), "soil",
      ("ground",))
    P("ground.base", Box(-120, -120, -60, 260, 170, -36), "soil", ("ground",))
    m.key = {"pipe_cx": 22, "pipe_cy": 24}
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "render_smoke"))
    ap.add_argument("--quality", type=float, default=0.4)
    ap.add_argument("--views", nargs="*")
    a = ap.parse_args(argv)
    t = time.time()
    res = render_views(toy_model(), a.out, names=a.views, quality=a.quality)
    for r in res:
        assert os.path.getsize(r["path"]) > 10_000, r
        print(f'{r["name"]:10s} {r["title"]}  ->  {r["path"]}')
    print(f"total {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
