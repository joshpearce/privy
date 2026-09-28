# privy: parametric outhouse, JSON to SVG blueprints to PDF

A skid-mounted pit privy with a covered deck, defined by a single JSON file.
One build turns that file into:

- a 3-D part model: every stud, board, batten, rafter, skid and fastener-bearing piece is a solid;
- architectural sheets as SVG (plans, elevations, sections, framing, details, installation sequence,
  schedules), drawn from the model with hidden-line removal and hatched section cuts;
- path-traced renderings (Mitsuba 3), with materials, sun, sky, site and cameras taken from the JSON;
- one PDF with bookmarks, a cut list CSV and a design-check report.

```
design/privy.json  ──►  privy.model  ──►  drawing/ (SVG sheets) ──► PDF
                                    ├──►  render/  (Mitsuba)    ──► PNG (embedded in the PDF)
                                    └──►  checks, BOM (cut list, purchase list, hardware)
```

## The design

| | |
|---|---|
| Pit | 30" ID (36" OD) dual-wall corrugated HDPE pipe, set vertically 36" into the ground, open bottom |
| Building | 4'-0" x 6'-0" framed, 2x4 @ 24" o.c., rough-sawn 1x10 board-and-batten |
| Foundation | two 4x6 PT (UC4B) skids on edge, chamfered both ends, with tow holes. Building and deck ride on one sled |
| Bench | full-width bench over the pipe, seat hole inside the pipe ID. A flange plate + collar + EPDM skirt closes pipe to bench after placement |
| Removable panel | the back wall below the bench is a lagged panel. Remove it and the privy slides back over the standing pipe: the pipe passes between the skids and under the 4x6 bench header |
| Roof | single slope 3:12, draining away from the door and the pit. 2x6 rafters on continuous 4x6 beams, 2x4 purlins, 29 ga corrugated steel |
| Window | 18" x 27" double-hung, sill 4'-8" above the floor, on the high side wall |
| Deck | 4'-0" x 6'-0" in front of the door, under the same roof, two 4x4 posts with knee braces |
| Vent | 4" PVC from the flange up through the roof |

## Usage

```bash
pip install -r requirements.txt           # numpy, cairosvg (needs libcairo), pypdf, pillow, mitsuba, jsonschema
python -m privy build design/privy.json -o build
python -m privy build --no-render         # drawings only (reuses existing renders if present), ~10 s
python -m privy build --quality 0.35      # quick draft renders
python -m privy build --only A-101 A-301  # just some sheets
python -m privy build --strict            # exit 1 if a design check fails
```

Outputs: `build/privy.pdf`, `build/svg/*.svg`, `build/renders/*.png`, `build/cut_list.csv`, `build/checks.json`.

## Editing the design

All lengths are inches, either as numbers or as feet-inch strings (`"6'-2 1/2\""`).
Model axes: x runs from the back wall (bench and removable panel) toward the door and deck, y from the low (eave) side to
the high side, and z points up with z = 0 at grade.

| Section | Controls |
|---|---|
| `stock` | every lumber / sheet size used (nominal, surfacing `S4S`/`rough`, actual size override, species, treatment, render material) |
| `pit` | pipe ID/OD, bury depth, clearances, flange, vent, usage (for the pit service-life estimate) |
| `foundation.skids` | stock, `on_edge`/`flat`, overhangs, chamfer, tow hole |
| `building` | width, depth, low bearing height, floor framing, walls (stud spacing, blocking, strap bracing), siding |
| `bench`, `removable_panel`, `door`, `window` | heights, hole setback, seat, rough openings, hardware text |
| `roof` | pitch, overhangs, rafter/purlin spacing, roofing profile |
| `deck` | depth, joists, decking, posts, knee braces |
| `materials` | per material: base color, roughness, metallic, specular, procedural texture, per-part variation, drawing fill and section hatch |
| `lighting` | sun azimuth/elevation (plan angle from +x toward +y), sky turbidity, exposure, tone map |
| `renders` | environment (dirt pad, surrounding ground, trees) and camera views (orbit or absolute; optional `hide_tags` and `cutaway`) |
| `drawings` | sheet size, theme (`white` / `blueprint`), plan cut height, allowed scales, general notes |

The schema is in `schema/privy.schema.json` and is validated on every build. Derived geometry is computed, not
hand-entered: the pipe top sits a set clearance below the bench header, the flange collar fills the remaining gap,
the window centers over the standing area, the vent position is searched inside the pipe ID clear of the seat,
rafters and purlins, and so on. If a parameter change breaks a relationship, a check on sheet A-602 fails. Examples:
the pipe no longer clears the skids, the seat hole leaves the pipe ID, or the door header no longer fits under the
sloped plate.

## Code map

| File | Role |
|---|---|
| `privy/model.py` | JSON to parts (the parametric template) |
| `privy/geometry.py` | primitives: prism, tube (with corrugation), plate with hole, corrugated sheet; clipping, sections |
| `privy/checks.py` | fit checks, NDS-style screening checks, snow load, weight and towing force, pit capacity |
| `privy/bom.py` | cut list, first-fit-decreasing purchase list, sheet goods, hardware |
| `privy/drawing/view.py` | orthographic projection, section caps, and painter's ordering: faces are ordered pairwise by depth tests and then topologically sorted |
| `privy/drawing/sheets.py` | sheet layouts, keynotes, dimensions, title block |
| `privy/render/` | Mitsuba scene, procedural textures, trees |
| `privy/pdf.py` | SVG to PDF |

The structural rows are screening calculations: simplified NDS allowable stress, No. 2 lumber, snow per ASCE 7 for an
unheated Risk Category I structure. They are not a sealed design. Siting (setbacks from wells and water, soil, and
water table) is governed by the county health department.
