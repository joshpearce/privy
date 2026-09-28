"""Parametric privy model: JSON design -> positioned parts + named coordinates.

Conventions (inches):
  x : 0 at the outside face of back-wall framing (bench / removable panel side),
      increasing toward the door, the front wall (x = D) and the deck (to D + DD).
  y : 0 at the outside face of the low (eave) side-wall framing, W at the high side.
  z : 0 at grade.
Walls are named back (x=0), front (x=D), left (low side, y=0), right (high side, y=W).
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict

import numpy as np

from .geometry import Box, CorrugatedSheet, Plate, Prism, Tube, clip_poly3, rect
from .parts import Model, Part, make_stock
from .units import parse_len

L = parse_len


# --------------------------------------------------------------------------- helpers

class Builder:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.stock = {k: make_stock(k, v) for k, v in cfg["stock"].items()}
        self.m = Model(cfg, self.stock)
        self.k = self.m.key
        self._n = defaultdict(int)

    def add(self, group, name, geom, stock=None, material=None, length=None, note="", tags=(), grain=None):
        s = self.stock.get(stock) if stock else None
        mat = material or (s.material if s else "framing_lumber")
        slug = name.lower().replace(" ", "_")
        self._n[(group, slug)] += 1
        pid = f"{group}.{slug}.{self._n[(group, slug)]:02d}"
        tags = set(tags) | {group.split(".")[0]}
        if length is None and s is not None and not s.sheet:
            lo, hi = geom.bbox()
            ext = hi - lo
            length = float(ext[grain] if grain is not None else ext.max())
        return self.m.add(Part(pid, name, group, geom, mat, stock, length, note, tags=tags, grain_axis=grain))

    def s(self, key):
        return self.stock[key]


class WallFrame:
    """Maps wall-plane coordinates (u along the wall, v = z, d = depth outward
    from the exterior framing face) to model prisms."""

    def __init__(self, name, normal_axis, face, outward, u_axis):
        self.name, self.n, self.face, self.o, self.u_axis = name, normal_axis, face, outward, u_axis

    def prism(self, poly_uv, d0, d1):
        a, b = self.face + self.o * d0, self.face + self.o * d1
        return Prism(self.n, poly_uv, min(a, b), max(a, b))

    def box(self, u0, v0, u1, v1, d0, d1):
        return self.prism(rect(min(u0, u1), min(v0, v1), max(u0, u1), max(v0, v1)), d0, d1)


def seg_hits_rect(p, q, r, pad=0.0):
    """Liang-Barsky: does segment p-q intersect axis-aligned rect r=(u0,v0,u1,v1)?"""
    u0, v0, u1, v1 = r[0] - pad, r[1] - pad, r[2] + pad, r[3] + pad
    t0, t1 = 0.0, 1.0
    d = (q[0] - p[0], q[1] - p[1])
    for pk, qk in ((-d[0], p[0] - u0), (d[0], u1 - p[0]), (-d[1], p[1] - v0), (d[1], v1 - p[1])):
        if abs(pk) < 1e-12:
            if qk < 0:
                return False
        else:
            t = qk / pk
            if pk < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
    return t0 <= t1


def even_positions(a, b, spacing, t):
    """Left edges of members of thickness t from a to b (both ends flush), spacing <= spacing."""
    span = b - a - t
    n = max(1, math.ceil(span / spacing - 1e-9))
    return [a + span * i / n for i in range(n + 1)]


def stable_hash(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:10]


def clip2d(poly, u0, v0, u1, v1):
    """Clip a 2-D convex polygon to an axis-aligned rectangle."""
    p = np.column_stack([np.asarray(poly, float), np.zeros(len(poly))])
    for axis, val, keep in ((0, u0, -1), (0, u1, 1), (1, v0, -1), (1, v1, 1)):
        p = clip_poly3(p, axis, val, keep)
        if len(p) < 3:
            return np.zeros((0, 2))
    return p[:, :2]


def crescent(R, r, off, n=48):
    """Crescent = disc(R, origin) minus disc(r, off), as a simple polygon."""
    off = np.asarray(off, float)
    t = np.linspace(0, 2 * np.pi, 4 * n, endpoint=False)
    outer = np.stack([R * np.cos(t), R * np.sin(t)], 1)
    keep_o = np.linalg.norm(outer - off, axis=1) >= r
    inner = off + np.stack([r * np.cos(t), r * np.sin(t)], 1)
    keep_i = np.linalg.norm(inner, axis=1) < R

    def arc(pts, mask):
        i0 = int(np.argmax(mask & ~np.roll(mask, 1)))   # first kept point after a gap
        idx = [(i0 + j) % len(pts) for j in range(len(pts))]
        return np.array([pts[i] for i in idx if mask[i]])

    a = arc(outer, keep_o)
    b = arc(inner, keep_i)
    if np.linalg.norm(b[0] - a[-1]) > np.linalg.norm(b[-1] - a[-1]):
        b = b[::-1]
    return np.vstack([a, b])


# --------------------------------------------------------------------------- build

def build(cfg: dict) -> Model:
    B = Builder(cfg)
    k = B.k
    m = B.m
    k["cfg_hash"] = stable_hash(json.dumps(cfg, sort_keys=True))

    bld, fnd, pit, bench, roof, deck = (cfg[x] for x in ("building", "foundation", "pit", "bench", "roof", "deck"))
    W, D, DD = L(bld["width"]), L(bld["depth"]), L(deck["depth"])
    k.update(W=W, D=D, DD=DD)

    # ---------------------------------------------------------------- foundation: skids
    sk = fnd["skids"]
    sks = B.s(sk["stock"])
    on_edge = sk.get("orientation", "on_edge") == "on_edge"
    skid_w, skid_h = (sks.t, sks.w) if on_edge else (sks.w, sks.t)
    inset = L(sk.get("inset", 0))
    xs0, xs1 = -L(sk["overhang_back"]), D + DD + L(sk["overhang_front"])
    cl, cr = L(sk["chamfer_length"]), L(sk["chamfer_rise"])
    both = sk.get("chamfer_both_ends", True)
    prof = [[xs0 + cl, 0], [xs1 - (cl if both else 0), 0], [xs1, cr if both else 0], [xs1, skid_h], [xs0, skid_h], [xs0, cr]]
    skid_ys = [(inset, inset + skid_w), (W - inset - skid_w, W - inset)]
    for y0, y1 in skid_ys:
        B.add("foundation", "Skid", Prism("y", prof, y0, y1), sk["stock"], grain=0, length=xs1 - xs0,
              note=f"{cl:g}\" x {cr:g}\" chamfer {'both ends' if both else 'front end'}; tow hole", tags={"skid"})
    k.update(skid_w=skid_w, skid_h=skid_h, skid_x0=xs0, skid_x1=xs1, skid_ys=skid_ys, chamfer=(cl, cr))

    # ---------------------------------------------------------------- floor framing
    fl = bld["floor"]
    rim, joist = B.s(fl["rim"]), B.s(fl["joist"])
    sheath_t = B.s(fl["sheathing"]).t
    rim_plies = int(fl.get("rim_plies", 2))
    rim_th = rim_plies * rim.t
    z_fj0, z_fj1 = skid_h, skid_h + joist.w
    FF = z_fj1 + sheath_t
    k.update(FF=FF, z_floor_framing=(z_fj0, z_fj1))

    st_d = B.s(bld["walls"]["stud"]).w        # wall depth (3.5)
    st_t = B.s(bld["walls"]["stud"]).t        # 1.5

    # pit pipe position
    pp = pit["pipe"]
    OD, ID = L(pp["outside_diameter"]), L(pp["inside_diameter"])
    pcy = W / 2 + L(pp.get("lateral_offset", 0))
    pcx = st_d + L(pp["clearance_to_back_wall"]) + OD / 2
    pipe_front = pcx + OD / 2
    k.update(OD=OD, ID=ID, pipe_cx=pcx, pipe_cy=pcy, pipe_front=pipe_front)

    x_bj = pipe_front + L(fl.get("joist_clear_to_pipe", 0.5))   # bench-front (doubled) floor joist
    for i in range(rim_plies):
        for side in (0, 1):
            y0 = i * rim.t if side == 0 else W - (i + 1) * rim.t
            B.add("floor", "Rim joist", Box(0, y0, z_fj0, D, y0 + rim.t, z_fj1), fl["rim"], grain=0)
    jy0, jy1 = rim_th, W - rim_th
    jx = [x_bj, x_bj + joist.t] + even_positions(x_bj + 2 * joist.t, D, L(fl["joist_spacing"]), joist.t)[1:]
    for x in jx:
        B.add("floor", "Floor joist", Box(x, jy0, z_fj0, x + joist.t, jy1, z_fj1), fl["joist"], grain=1)
    k["floor_joists"] = jx
    k["x_bj"] = x_bj
    B.add("floor", "Floor sheathing", Box(x_bj, 0, z_fj1, D, W, FF), fl["sheathing"], grain=1)
    for y0, y1 in ((0, st_d), (W - st_d, W)):
        B.add("floor", "Floor sheathing strip", Box(0, y0, z_fj1, x_bj, y1, FF), fl["sheathing"], grain=0)

    # ---------------------------------------------------------------- bench + pit
    z_bt = FF + L(bench["height"])
    bt_t = B.s(bench["top"]).t
    z_bu = z_bt - bt_t
    hdr = B.s(bench["header"])
    hdr_h = hdr.w
    z_h0, z_h1 = z_bu - hdr_h, z_bu
    pipe_top = z_h0 - L(pp["slide_clearance_under_header"])
    pipe_bot = -L(pp["bury_depth"])
    bf = B.s(bench["framing"])
    z_bf0 = z_bu - bf.w
    fp_t = B.s(bench["front_panel"]).t
    x_fp0 = pipe_front + L(bench["front_panel_clearance_to_pipe"])
    x_bf = x_fp0 + fp_t
    k.update(z_bt=z_bt, z_bu=z_bu, header_z=(z_h0, z_h1), pipe_top=pipe_top, pipe_bot=pipe_bot,
             x_fp0=x_fp0, x_bf=x_bf, z_bf0=z_bf0)

    yi0, yi1 = st_d, W - st_d        # interior faces of side-wall framing
    B.add("bench", "Bench header", Box(0, yi0, z_h0, st_d, yi1, z_h1), bench["header"], grain=1,
          note="hang from doubled corner studs w/ face-mount hangers")
    rpl = int(bench.get("front_rail_plies", 1))
    for i in range(rpl):
        B.add("bench", "Bench front rail", Box(x_fp0 - (i + 1) * bf.t, yi0, z_bf0, x_fp0 - i * bf.t, yi1, z_bu),
              bench["framing"], grain=1)
    x_rail = x_fp0 - rpl * bf.t
    for y0 in (yi0, yi1 - bf.t):
        B.add("bench", "Bench side ledger", Box(st_d, y0, z_bf0, x_rail, y0 + bf.t, z_bu), bench["framing"], grain=0)
    hole = bench["hole"]
    hl, hw = L(hole["length"]), L(hole["width"])
    hx = x_bf - L(hole["setback"])
    jc = L(bench.get("joist_clear_to_hole", 2))
    bj_y = [pcy - hw / 2 - jc - bf.t, pcy + hw / 2 + jc]
    for y0 in bj_y:
        B.add("bench", "Bench joist", Box(st_d, y0, z_bf0, x_rail, y0 + bf.t, z_bu), bench["framing"], grain=0)
    k.update(hole=(hx, pcy, hl / 2, hw / 2), bench_joist_y=bj_y, rail_plies=rpl)
    B.add("bench", "Bench top", Plate(("rect", 0, yi0, x_bf, yi1), z_bu, z_bt, hole=("ellipse", hx, pcy, hl / 2, hw / 2)),
          bench["top"], grain=1, note="seat hole cut to template")
    B.add("bench", "Bench front panel", Box(x_fp0, yi0, FF, x_bf, yi1, z_bu), bench["front_panel"], grain=1)
    seat = bench["seat"]
    sl, sw, st = L(seat["length"]), L(seat["width"]), L(seat["thickness"])
    ol, ow = L(seat["opening_length"]), L(seat["opening_width"])
    B.add("bench", "Toilet seat", Plate(("ellipse", hx, pcy, sl / 2, sw / 2), z_bt, z_bt + st * 0.55,
                                        hole=("ellipse", hx, pcy, ol / 2, ow / 2)), material=seat.get("material", "seat"),
          tags={"hardware", "seat"})
    if seat.get("lid", True):
        B.add("bench", "Toilet seat lid", Plate(("ellipse", hx, pcy, sl / 2, sw / 2), z_bt + st * 0.55, z_bt + st),
              material=seat.get("material", "seat"), tags={"hardware", "seat"})
    k["seat"] = (hx, pcy, sl / 2, sw / 2)

    # pit pipe + flange
    corr = (L(pp["corrugation_pitch"]), L(pp["corrugation_depth"]))
    B.add("pit", "HDPE pipe", Tube(pcx, pcy, OD / 2, ID / 2, pipe_bot, pipe_top, corrugation=corr),
          material=pp.get("material", "hdpe"), tags={"pipe", "below_grade"}, grain=2)
    fg = pit["flange"]
    fpl = B.s(fg["plate"])
    z_fp1 = z_bf0
    z_fp0 = z_fp1 - fpl.t
    col_z0 = pipe_top + L(fg["gap_to_pipe"])
    B.add("pit", "Flange plate", Plate(("rect", st_d, yi0, x_fp0, yi1), z_fp0, z_fp1, hole=("ellipse", pcx, pcy, ID / 2, ID / 2)),
          fg["plate"], grain=1, tags={"flange"})
    if z_fp0 - col_z0 > 0.05:
        B.add("pit", "Flange collar", Tube(pcx, pcy, OD / 2, ID / 2, col_z0, z_fp0), material=fg.get("collar_material", "treated_plywood"),
              tags={"flange"})
    skw = L(fg["skirt_width"])
    zc = 0.5 * (pipe_top + col_z0)
    B.add("pit", "EPDM skirt", Tube(pcx, pcy, OD / 2 + 0.09, OD / 2 + 0.01, zc - skw / 2, zc + skw / 2),
          material=fg.get("skirt_material", "epdm"), tags={"flange"})
    k.update(flange_z=(z_fp0, z_fp1), collar_z0=col_z0)

    # ---------------------------------------------------------------- walls
    wl = bld["walls"]
    spacing = L(wl["stud_spacing"])
    blk_sp = L(wl["blocking_spacing"])
    s_ = L(roof["pitch"]) / 12.0
    c_ = math.sqrt(1 + s_ * s_)
    beam = B.s(roof["beam"])
    beam_w, beam_h = beam.t, beam.w
    z_Lb = FF + L(bld["low_bearing_height"])
    z_Hb = z_Lb + (W - beam_w) * s_
    raf = B.s(roof["rafter"])
    tv = raf.w * c_
    z_rb = lambda y: z_Lb + (y - beam_w) * s_          # noqa: E731  rafter bottom line
    z_rt = lambda y: z_rb(y) + tv                       # noqa: E731  rafter top line
    substrate = roof.get("substrate", "purlins")
    sub = B.s(roof["purlin"] if substrate == "purlins" else roof["sheathing"])
    z_pt = lambda y: z_rt(y) + sub.t * c_               # noqa: E731  top of purlins / sheathing (roofing underside)
    rfg = roof["roofing"]
    profile = rfg.get("profile", "corrugated")
    pan_t = L(rfg.get("thickness", 0.024))
    top_add = L(rfg["profile_depth"]) if profile == "corrugated" else pan_t
    z_roof = lambda y: z_pt(y) + top_add               # noqa: E731
    pv = B.s(wl["plate"]).t * c_                        # vertical depth of sloped plate
    k.update(slope=s_, c=c_, z_Lb=z_Lb, z_Hb=z_Hb, beam_w=beam_w, beam_h=beam_h, tv=tv,
             z_rb=z_rb, z_rt=z_rt, z_pt=z_pt, z_roof=z_roof, st_d=st_d)

    plate_t = B.s(wl["plate"]).t
    bp_t = B.s(wl["bottom_plate"]).t

    def frame_wall(wf: WallFrame, u0, u1, v_bot, top, sloped, openings, bottom_plate=True, first_double=None,
                   group=None):
        """Frame one wall.  top(u) = top of the top plate.  openings: dicts with u0,u1,v0,v1,header(stock,plies),kind."""
        g = group or f"wall.{wf.name}"
        verts = []      # (u0,u1,v0,v1) of vertical members for blocking

        def vbox(name, a, b, v0, v1, stock, note=""):
            if sloped and v1 is None:
                poly = [[a, v0], [b, v0], [b, top(b) - (pv if sloped else plate_t)], [a, top(a) - (pv if sloped else plate_t)]]
                geom = wf.prism(poly, -st_d, 0)
                length = max(poly[2][1], poly[3][1]) - v0
            else:
                v1 = v1 if v1 is not None else top(a) - plate_t
                geom = wf.box(a, v0, b, v1, -st_d, 0)
                length = v1 - v0
            verts.append((a, b, v0, max(p[1] for p in geom.poly)))
            B.add(g, name, geom, stock, grain=2, length=length, note=note, tags={"framing", "wall"})

        vb0 = v_bot + (bp_t if bottom_plate else 0)
        if bottom_plate:
            cuts = sorted((o["u0"], o["u1"]) for o in openings if o["kind"] == "door")
            a = u0
            for c0, c1 in cuts + [(u1, u1)]:
                if c0 - a > 0.1:
                    B.add(g, "Bottom plate", wf.box(a, v_bot, c0, vb0, -st_d, 0), wl["bottom_plate"], grain=wf.u_axis,
                          tags={"framing", "wall"})
                a = c1
        # top plate
        if sloped:
            poly = [[u0, top(u0) - pv], [u1, top(u1) - pv], [u1, top(u1)], [u0, top(u0)]]
            B.add(g, "Top plate", wf.prism(poly, -st_d, 0), wl["plate"], grain=wf.u_axis,
                  length=(u1 - u0) * c_, note=f"{L(roof['pitch']):g}:12 slope", tags={"framing", "wall"})
        else:
            B.add(g, "Top plate", wf.box(u0, top(u0) - plate_t, u1, top(u0), -st_d, 0), wl["plate"], grain=wf.u_axis,
                  tags={"framing", "wall"})
        # studs
        commons = [u0] + ([u0 + st_t] if first_double == "start" else [])
        kmark = 1
        while True:
            e = u0 + kmark * spacing - st_t / 2
            if e + st_t > u1 - st_t - 0.25:
                break
            commons.append(e)
            kmark += 1
        commons.append(u1 - st_t)
        if first_double == "end":
            commons.append(u1 - 2 * st_t)
        # openings
        zones = []
        for o in openings:
            ku0, ku1 = o["u0"] - 2 * st_t, o["u1"] + 2 * st_t
            zones.append((ku0, ku1))
            hs = B.s(o["header"]["stock"])
            plies = int(o["header"].get("plies", 2))
            h0, h1 = o["v1"], o["v1"] + hs.w
            for a in (o["u0"] - 2 * st_t, o["u1"] + st_t):
                vbox("King stud", a, a + st_t, vb0, None, wl["stud"])
            for a in (o["u0"] - st_t, o["u1"]):
                B.add(g, "Jack stud", wf.box(a, vb0, a + st_t, o["v1"], -st_d, 0), wl["stud"], grain=2, tags={"framing", "wall"})
                verts.append((a, a + st_t, vb0, o["v1"]))
            for p in range(plies):
                B.add(g, "Header", wf.box(o["u0"] - st_t, h0, o["u1"] + st_t, h1, -hs.t * (p + 1), -hs.t * p),
                      o["header"]["stock"], grain=wf.u_axis, tags={"framing", "wall", "header"})
            o["header_top"] = h1
            if o["kind"] == "window":
                B.add(g, "Rough sill", wf.box(o["u0"], o["v0"] - plate_t, o["u1"], o["v0"], -st_d, 0), wl["plate"],
                      grain=wf.u_axis, tags={"framing", "wall"})
            # cripples on layout inside the opening
            for c in commons:
                if o["u0"] <= c and c + st_t <= o["u1"]:
                    top_c = top(c) - (pv if sloped else plate_t)
                    if top_c - h1 > 1.0:
                        poly = [[c, h1], [c + st_t, h1], [c + st_t, top(c + st_t) - (pv if sloped else plate_t)], [c, top_c]]
                        B.add(g, "Cripple stud", wf.prism(poly, -st_d, 0), wl["stud"], grain=2, tags={"framing", "wall"})
                        verts.append((c, c + st_t, h1, top_c))
                    if o["kind"] == "window" and o["v0"] - plate_t - vb0 > 1.0:
                        B.add(g, "Cripple stud", wf.box(c, vb0, c + st_t, o["v0"] - plate_t, -st_d, 0), wl["stud"], grain=2,
                              tags={"framing", "wall"})
                        verts.append((c, c + st_t, vb0, o["v0"] - plate_t))
        for c in commons:
            if any(c + st_t > z0 + 0.01 and c < z1 - 0.01 for z0, z1 in zones):
                continue
            vbox("Stud", c, c + st_t, vb0, None, wl["stud"])
        # blocking rows (flat, flush to exterior) for board-and-batten nailing
        verts.sort()
        row = 1
        while True:
            vc = v_bot + row * blk_sp
            r0, r1 = vc - B.s(wl["blocking"]).w / 2, vc + B.s(wl["blocking"]).w / 2
            if r1 > max(top(u0), top(u1)) - (pv if sloped else plate_t) - 0.5:
                break
            row += 1
            cover = sorted((a, b) for a, b, v0, v1 in verts if v0 <= r0 + 0.01 and v1 >= r1 - 0.01)
            for (a0, a1), (b0, b1) in zip(cover, cover[1:]):
                g0, g1 = a1, b0
                if g1 - g0 < 0.5:
                    continue
                if any(o["u0"] - st_t - 0.01 <= g0 and g1 <= o["u1"] + st_t + 0.01 and
                       r1 > o["v0"] - plate_t - 0.01 and r0 < o["header_top"] + 0.01 for o in openings):
                    continue
                if r1 > min(top(g0), top(g1)) - (pv if sloped else plate_t) - 0.25:
                    continue
                B.add(g, "Blocking", wf.box(g0, r0, g1, r1, -B.s(wl["blocking"]).t, 0), wl["blocking"],
                      grain=wf.u_axis, tags={"framing", "wall", "blocking"})
        # let-in diagonal strap bracing: an X in an end bay, clear of openings
        if wl.get("strap_bracing", True):
            ob = [(o["u0"] - 2 * st_t, o["v0"] - 2, o["u1"] + 2 * st_t, o["header_top"] + 1) for o in openings]
            tl = lambda u: top(u) - (pv if sloped else plate_t) - 0.5   # noqa: E731
            va = vb0 + 0.5
            made = 0

            def run_for(ua, sgn, t_):
                lo_, hi_ = 0.0, u1 - u0
                for _ in range(60):
                    mid = 0.5 * (lo_ + hi_)
                    if va + mid * t_ < tl(ua + sgn * mid):
                        lo_ = mid
                    else:
                        hi_ = mid
                return lo_

            for ua, sgn in ((u0 + 2, 1), (u1 - 2, -1)):
                for ang in (45, 50, 55, 60, 65):
                    t_ = math.tan(math.radians(ang))
                    rn = run_for(ua, sgn, t_)
                    ub = ua + sgn * rn
                    if rn < 12 or not (u0 < ub < u1):
                        continue
                    diag = [((ua, va), (ub, va + rn * t_)), ((ua, tl(ua)), (ub, va + (tl(ua) - va) * 0 + 0))]
                    # falling diagonal of the same bay: from the top at ua down to the bottom at ub
                    diag[1] = ((ua, tl(ua)), (ub, va))
                    if any(seg_hits_rect(p, q, r) for p, q in diag for r in ob):
                        continue
                    for P, Q in diag:
                        P, Q = np.array(P), np.array(Q)
                        n = np.array([-(Q[1] - P[1]), Q[0] - P[0]])
                        n = n / np.linalg.norm(n) * 0.625
                        B.add(g, "Strap brace", wf.prism([P - n, Q - n, Q + n, P + n], 0.0, 0.04), material="steel",
                              note="let-in steel wall-bracing strap", tags={"framing", "wall", "strap", "hardware"})
                        made += 1
                    break
                if made:
                    break
            k[f"straps_{wf.name}"] = made

    # side walls
    beam_bot_L, beam_bot_H = z_Lb - beam_h, z_Hb - beam_h
    win = cfg["window"]
    ro_c = L(win["rough_opening_clearance"])
    wro_w, wro_h = L(win["unit_width"]) + ro_c, L(win["unit_height"]) + ro_c
    # auto: centred over the standing area between the bench front and the front wall
    wcx = 0.5 * (x_bf + D - st_d) if win.get("center", "auto") == "auto" else L(win["center"])
    w_open = dict(kind="window", u0=wcx - wro_w / 2, u1=wcx + wro_w / 2, v0=FF + L(win["sill_height"]),
                  v1=FF + L(win["sill_height"]) + wro_h, header=win["header"])
    win_wall = win.get("wall", "right")
    k["window"] = dict(w_open, wall=win_wall)
    sides = {
        "left": (WallFrame("left", 1, 0.0, -1, 0), beam_bot_L),
        "right": (WallFrame("right", 1, W, 1, 0), beam_bot_H),
    }
    for nm, (wf, tp) in sides.items():
        ops = [dict(w_open)] if win_wall == nm else []
        frame_wall(wf, 0, D, FF, lambda u, tp=tp: tp, False, ops, first_double="start")
        if ops:
            k["window"]["header_top"] = ops[0]["header_top"]
    k["stud_top_low"], k["stud_top_high"] = beam_bot_L - plate_t, beam_bot_H - plate_t

    # front wall with door
    dr = cfg["door"]
    prehung = dr.get("type", "board") == "prehung"
    if prehung:
        ra = dr.get("rough_opening_allowance", [2, 2.5])
        dro_w, dro_h = L(dr["slab_width"]) + L(ra[0]), L(dr["slab_height"]) + L(ra[1])
    else:
        dro_w, dro_h = L(dr["rough_opening_width"]), L(dr["rough_opening_height"])
    hinge_hi = dr.get("hinge_side", "right") == "right"     # as seen from outside: right = +y (high side)
    if dr.get("position", "center") == "hinge_corner":
        # king + jack against the end stud on the hinge side, so an in-swinging leaf folds flat to the side wall
        dcy = (yi1 - 3 * st_t - dro_w / 2) if hinge_hi else (yi0 + 3 * st_t + dro_w / 2)
    else:
        dcy = W / 2 + L(dr.get("offset", 0))
    d_open = dict(kind="door", u0=dcy - dro_w / 2, u1=dcy + dro_w / 2, v0=FF, v1=FF + dro_h, header=dr["header"])
    front = WallFrame("front", 0, D, 1, 1)
    ops = [d_open]
    frame_wall(front, yi0, yi1, FF, z_rb, True, ops)
    k["door"] = dict(d_open)

    # back wall (upper part, above the bench header / removable panel)
    back = WallFrame("back", 0, 0.0, -1, 1)
    frame_wall(back, yi0, yi1, z_bt, z_rb, True, [])

    # side-wall corner studs for the bench header are handled by first_double="start"

    # ---------------------------------------------------------------- roof framing
    x_end = D + DD
    for y0, zt_ in ((0, z_Lb), (W - beam_w, z_Hb)):
        B.add("roof", "Beam", Box(0, y0, zt_ - beam_h, x_end, y0 + beam_w, zt_), roof["beam"], grain=0,
              tags={"framing"})
    oh_lo, oh_hi = L(roof["overhang_low"]), L(roof["overhang_high"])
    y_lo, y_hi = -oh_lo, W + oh_hi
    raf_x = even_positions(0, D, L(roof["rafter_spacing"]), raf.t)
    raf_x += even_positions(D - raf.t, x_end, L(roof["rafter_spacing"]), raf.t)[1:]
    rpoly = [[y_lo, z_rb(y_lo)], [0, z_rb(0)], [0, z_Lb], [beam_w, z_Lb], [W - beam_w, z_rb(W - beam_w)],
             [W - beam_w, z_Hb], [W, z_Hb], [y_hi, z_rb(y_hi)], [y_hi, z_rt(y_hi)], [y_lo, z_rt(y_lo)]]
    raf_len = (y_hi - y_lo) * c_
    for x in raf_x:
        B.add("roof", "Rafter", Prism("x", rpoly, x, x + raf.t), roof["rafter"], grain=1, length=raf_len,
              note=f"{L(roof['pitch']):g}:12, birdsmouth both beams, plumb-cut tails", tags={"framing"})
    k.update(rafter_x=raf_x, y_lo=y_lo, y_hi=y_hi, rafter_poly=rpoly, raf_len=raf_len)
    if roof.get("frieze_blocking", True):
        for a, b in zip(raf_x, raf_x[1:]):
            if b > D:
                break
            for poly in ([[0, z_Lb], [raf.t, z_Lb], [raf.t, z_rt(raf.t)], [0, z_rt(0)]],
                         [[W - raf.t, z_Hb], [W, z_Hb], [W, z_rt(W)], [W - raf.t, z_rt(W - raf.t)]]):
                B.add("roof", "Frieze block", Prism("x", poly, a + raf.t, b), roof["rafter"], grain=0, tags={"framing"})
    th = math.atan(s_)
    ct, stn = math.cos(th), math.sin(th)
    fas = B.s(roof["fascia"])
    rb, rf = L(roof["rake_back"]), L(roof["rake_front"])
    purlins = []
    if substrate == "purlins":            # flat 2x4 purlins on the rafters (exposed-fastener panels)
        run = (y_hi - y_lo) - sub.w * ct
        n_p = max(1, math.ceil(run * c_ / L(roof["purlin_spacing"]) - 1e-9))
        for i in range(n_p + 1):
            a = y_lo + run * i / n_p
            p0 = np.array([a, z_rt(a)])
            p1 = p0 + sub.w * np.array([ct, stn])
            nrm = sub.t * np.array([-stn, ct])
            B.add("roof", "Purlin", Prism("x", [p0, p1, p1 + nrm, p0 + nrm], -rb, x_end + rf), roof["purlin"], grain=0,
                  tags={"framing"})
            purlins.append((a, a + sub.w * ct))
    else:                                 # plywood deck on rafters, fly rafters carry the rake overhangs
        sy0, sy1 = y_lo - fas.t, y_hi + fas.t
        spoly = [[sy0, z_rt(sy0)], [sy1, z_rt(sy1)], [sy1, z_pt(sy1)], [sy0, z_pt(sy0)]]
        B.add("roof", "Roof sheathing", Prism("x", spoly, -rb, x_end + rf), roof["sheathing"], grain=0,
              note="H-clips between rafters; synthetic underlayment over", tags={"roofing_deck"})
        if roof.get("fly_rafters", True) and (rb > 0 or rf > 0):
            fpoly = [[y_lo, z_rt(y_lo) - tv], [y_hi, z_rt(y_hi) - tv], [y_hi, z_rt(y_hi)], [y_lo, z_rt(y_lo)]]
            for x0 in ((-rb,) if rb > raf.t else ()) + ((x_end + rf - raf.t,) if rf > raf.t else ()):
                B.add("roof", "Fly rafter", Prism("x", fpoly, x0, x0 + raf.t), roof["rafter"], grain=1,
                      length=(y_hi - y_lo) * c_, note="hung from sheathing + fascia; plumb-cut ends", tags={"framing"})
    k["purlins"] = purlins
    # fascia + barge
    B.add("roof", "Fascia", Box(-rb, y_lo - fas.t, z_rt(y_lo) - fas.w, x_end + rf, y_lo, z_rt(y_lo)), roof["fascia"], grain=0,
          tags={"trim"})
    B.add("roof", "Fascia", Box(-rb, y_hi, z_rt(y_hi) - fas.w, x_end + rf, y_hi + fas.t, z_rt(y_hi)), roof["fascia"], grain=0,
          tags={"trim"})
    by0, by1 = y_lo - fas.t, y_hi + fas.t
    bpoly = [[by0, z_pt(by0) - fas.w * c_], [by1, z_pt(by1) - fas.w * c_], [by1, z_pt(by1)], [by0, z_pt(by0)]]
    for x0 in (-rb - fas.t, x_end + rf):
        B.add("roof", "Barge board", Prism("x", bpoly, x0, x0 + fas.t), roof["fascia"], grain=1,
              length=(by1 - by0) * c_, tags={"trim"})
    # roofing
    rx0, rx1 = -rb - fas.t - L(rfg["rake_overhang"]), x_end + rf + fas.t + L(rfg["rake_overhang"])
    ry0, ry1 = by0 - L(rfg["eave_overhang"]), by1 + 0.5
    mat_r = rfg.get("material", "galvanized")
    seams = []
    if profile == "standing_seam":
        cov = L(rfg["panel_width"])
        n_sh = math.ceil((rx1 - rx0) / cov - 1e-9)
        sh_, sw_ = L(rfg["seam_height"]), L(rfg["seam_width"])
        pan = [[ry0, z_pt(ry0)], [ry1, z_pt(ry1)], [ry1, z_roof(ry1)], [ry0, z_roof(ry0)]]
        for i in range(n_sh):
            a = rx0 + i * cov
            b = min(rx1, a + cov)
            B.add("roof", "Roofing panel", Prism("x", pan, a, b), material=mat_r, grain=1, length=(ry1 - ry0) * c_,
                  tags={"roofing"})
            if i:
                seams.append(a)
        spoly = [[ry0, z_roof(ry0)], [ry1, z_roof(ry1)], [ry1, z_roof(ry1) + sh_ * c_], [ry0, z_roof(ry0) + sh_ * c_]]
        for sx in seams:
            B.add("roof", "Standing seam", Prism("x", spoly, sx - sw_ / 2, sx + sw_ / 2), material=mat_r, grain=1,
                  tags={"roofing", "seam"})
    else:
        cov = L(rfg["coverage_width"])
        n_sh = math.ceil((rx1 - rx0) / cov - 1e-9)
        for i in range(n_sh):
            a = rx0 + i * cov
            b = min(rx1, a + cov)
            B.add("roof", "Roofing sheet", CorrugatedSheet(a, b, ry0, ry1, z_pt(ry0), s_, L(rfg["profile_pitch"]),
                                                           L(rfg["profile_depth"])),
                  material=mat_r, grain=1, length=(ry1 - ry0) * c_, tags={"roofing"})
    k.update(roof_x=(rx0, rx1), roof_y=(ry0, ry1), roof_sheets=n_sh, roof_sheet_len=(ry1 - ry0) * c_,
             fascia=(fas.t, fas.w), rake=(rb, rf), seams=seams, roof_profile=profile, substrate=substrate)

    # ---------------------------------------------------------------- deck
    dk_rim, dk_j = B.s(deck["rim"]), B.s(deck["joist"])
    post = B.s(deck["post"])
    dplies = int(deck.get("rim_plies", 2))
    drt = dplies * dk_rim.t
    z_dj0, z_dj1 = skid_h, skid_h + dk_j.w
    dkb = B.s(deck["decking"])
    deck_top = z_dj1 + dkb.t
    xp0 = x_end - post.t
    for i in range(dplies):
        for side in (0, 1):
            y0 = i * dk_rim.t if side == 0 else W - (i + 1) * dk_rim.t
            B.add("deck", "Deck rim", Box(D, y0, z_dj0, xp0, y0 + dk_rim.t, z_dj1), deck["rim"], grain=0)
    djx = even_positions(D, x_end, L(deck["joist_spacing"]), dk_j.t)
    for x in djx:
        y0, y1 = (post.t, W - post.t) if x + dk_j.t > xp0 else (drt, W - drt)
        B.add("deck", "Deck joist", Box(x, y0, z_dj0, x + dk_j.t, y1, z_dj1), deck["joist"], grain=1)
    gap = L(deck["decking_gap"])
    y = 0.0
    boards = []
    while y < W - 0.5:
        w_ = min(dkb.w, W - y)
        boards.append((y, y + w_))
        y += w_ + gap
    for y0, y1 in boards:
        pts = [[D, y0], [x_end, y0], [x_end, y1], [D, y1]]
        notch_lo = y0 < post.t
        notch_hi = y1 > W - post.t
        if notch_lo and notch_hi:
            pts = [[D, y0], [xp0, y0], [xp0, y1], [D, y1]]
        elif notch_lo:
            pts = [[D, y0], [xp0, y0], [xp0, post.t], [x_end, post.t], [x_end, y1], [D, y1]]
        elif notch_hi:
            pts = [[D, y0], [x_end, y0], [x_end, W - post.t], [xp0, W - post.t], [xp0, y1], [D, y1]]
        note = "rip to width" if y1 - y0 < dkb.w - 0.01 else ""
        if notch_lo or notch_hi:
            note = (note + "; " if note else "") + "notch at post"
        B.add("deck", "Deck board", Prism("z", pts, deck_top - dkb.t, deck_top), deck["decking"], grain=0, note=note,
              length=x_end - D)
    posts = []
    for (y0, y1), zt_ in (((0, post.t), beam_bot_L), ((W - post.t, W), beam_bot_H)):
        B.add("deck", "Post", Box(xp0, y0, skid_h, x_end, y1, zt_), deck["post"], grain=2,
              note="bear on skid; galv. post base + (2) 1/2\" through-bolts to rim", tags={"framing"})
        posts.append((y0, y1, skid_h, zt_))
        kb = deck.get("knee_brace")
        if kb:
            kbs = B.s(kb["stock"])
            leg = L(kb["leg"])
            dd = kbs.t * math.sqrt(2)
            poly = [[xp0, zt_ - leg], [xp0, zt_ - leg + dd], [xp0 - leg + dd, zt_], [xp0 - leg, zt_]]
            B.add("deck", "Knee brace", Prism("y", poly, y0, y1), kb["stock"], grain=None, length=leg * math.sqrt(2),
                  note="45 deg both ends", tags={"framing"})
    k.update(deck_top=deck_top, deck_joists=djx, posts=posts, xp0=xp0, deck_boards=boards)

    # ---------------------------------------------------------------- siding
    sd = bld["siding"]
    bstk, batk, trk = B.s(sd["board"]), B.s(sd["batten"]), B.s(sd["trim"])
    tb, tbat = bstk.t, batk.t
    sgap = L(sd["gap"])
    min_clear = L(sd.get("min_clearance_to_grade", 6.0))      # IRC R317.1: untreated siding >= 6" above grade
    side_bot = max(skid_h - L(sd.get("lap_over_skid", 1.0)), min_clear)
    k.update(tb=tb, tbat=tbat, side_bot=side_bot, siding_min_clear=min_clear)

    def siding(wf, grp, u0, u1, v_bot, top, cutouts=(), obstacles=(), tags=(), edge_battens=(True, True),
               edge_ext=(0.0, 0.0), rails=None):
        """Board-and-batten on wall `wf`; top(u) linear.  cutouts: (u0,u1,v0,v1).  obstacles: (u0,u1,vtop)."""
        cutouts = [tuple(c) + (("door",) if len(c) == 4 else ()) for c in cutouts]
        cas = [(c0 - trk.w, c1 + trk.w, cv0 - (trk.w + 1.5 if kind == "window" else 0), cv1 + trk.w)
               for c0, c1, cv0, cv1, kind in cutouts]
        cutouts = [c[:4] for c in cutouts]
        Lw = u1 - u0
        n = max(1, math.ceil((Lw + sgap) / (bstk.w + sgap) - 1e-9))
        bw = (Lw - (n - 1) * sgap) / n
        tags = set(tags) | {"siding"}
        edges = []

        def bottom_at(a, b):
            v = v_bot
            for o0, o1, vt in obstacles:
                if a < o1 - 0.01 and b > o0 + 0.01:
                    v = max(v, vt)
            return v

        def pieces(a, b, vb_fn, vt_fn, extra_cut=()):
            brks = {a, b}
            for c0, c1, *_ in list(cutouts) + list(extra_cut):
                for c in (c0, c1):
                    if a < c < b:
                        brks.add(c)
            for o0, o1, _ in obstacles:
                for c in (o0, o1):
                    if a < c < b:
                        brks.add(c)
            brks = sorted(brks)
            out = []
            for p, q in zip(brks, brks[1:]):
                vb = vb_fn(p, q)
                ivs = [(vb, None)]
                for c0, c1, cv0, cv1 in list(cutouts) + list(extra_cut):
                    if p < c1 - 0.01 and q > c0 + 0.01:
                        nxt = []
                        for lo_, hi_ in ivs:
                            hi_v = hi_ if hi_ is not None else 1e9
                            if cv1 <= lo_ or cv0 >= hi_v:
                                nxt.append((lo_, hi_))
                            else:
                                if cv0 > lo_:
                                    nxt.append((lo_, cv0))
                                if cv1 < hi_v:
                                    nxt.append((cv1, hi_))
                        ivs = nxt
                for lo_, hi_ in ivs:
                    if hi_ is None:
                        poly = [[p, lo_], [q, lo_], [q, vt_fn(q)], [p, vt_fn(p)]]
                    else:
                        poly = [[p, lo_], [q, lo_], [q, hi_], [p, hi_]]
                    if max(pt[1] for pt in poly[2:]) - lo_ > 0.1:
                        out.append(poly)
            return out

        for i in range(n):
            a = u0 + i * (bw + sgap)
            b = a + bw
            for poly in pieces(a, b, bottom_at, top):
                B.add(grp, "Siding board", wf.prism(poly, 0, tb), sd["board"], grain=2, tags=tags,
                      length=max(pt[1] for pt in poly) - min(pt[1] for pt in poly))
            if i < n - 1:
                edges.append(b + sgap / 2)
        # battens
        rail_cuts = []
        if rails:
            for rv0, rv1 in rails:
                rail_cuts.append((u0 - edge_ext[0] - 1, u1 + edge_ext[1] + 1, rv0, rv1))
        bats = [(e - batk.w / 2, e + batk.w / 2) for e in edges]
        if edge_battens[0]:
            bats.insert(0, (u0 - edge_ext[0], u0 - edge_ext[0] + batk.w))
        if edge_battens[1]:
            bats.append((u1 + edge_ext[1] - batk.w, u1 + edge_ext[1]))
        for a, b in bats:
            for poly in pieces(a, b, bottom_at, top, extra_cut=cas + rail_cuts):
                B.add(grp, "Batten", wf.prism(poly, tb, tb + tbat), sd["batten"], grain=2, tags=tags,
                      length=max(pt[1] for pt in poly) - min(pt[1] for pt in poly))
        # rails (removable panel)
        for rv0, rv1 in rails or []:
            B.add(grp, "Panel rail", wf.box(u0 - edge_ext[0], rv0, u1 + edge_ext[1], rv1, tb, tb + trk.t), sd["trim"],
                  grain=wf.u_axis, tags=tags)
        return n, bw

    # side walls: boards x in [0, D], corner battens wrap the corners
    wrap = tb + tbat
    top_left = z_rb(-tb)
    n_l, _ = siding(sides["left"][0], "siding.left", 0, D, side_bot, lambda u: top_left, edge_ext=(wrap, wrap))
    wo = k["window"]
    cut_win = [(wo["u0"], wo["u1"], wo["v0"], wo["v1"], "window")] if win_wall == "right" else []
    cut_win_l = [(wo["u0"], wo["u1"], wo["v0"], wo["v1"])] if win_wall == "left" else []
    if cut_win_l:
        raise ValueError("window on the low (left) wall is not supported by this template yet")
    siding(sides["right"][0], "siding.right", 0, D, side_bot, lambda u: z_Hb, cutouts=cut_win, edge_ext=(wrap, wrap))
    # front wall
    front_bot = deck_top + 0.25
    do = k["door"]
    siding(front, "siding.front", -tb, W + tb, front_bot, z_rt,
           cutouts=[(do["u0"], do["u1"], FF, do["v1"])], edge_battens=(False, False))
    # back wall: fixed siding above the removable panel
    rp = cfg["removable_panel"]
    panel_top = z_h0 + L(rp["head_lap"])
    siding(back, "siding.back", -tb, W + tb, panel_top, z_rt, edge_battens=(False, False))
    B.add("siding.back", "Z-flashing", back.box(-tb, panel_top - 0.06, W + tb, panel_top, 0, tb + tbat + 0.25),
          material="galvanized", tags={"siding", "hardware"})
    # removable panel
    obst = [(y0, y1, skid_h + 0.25) for y0, y1 in skid_ys]
    rail = B.s(rp["rail"])
    p_bot = L(rp["bottom_above_grade"])
    rails = [(skid_h + 0.5, skid_h + 0.5 + rail.w), (panel_top - 0.25 - rail.w, panel_top - 0.25)]
    siding(back, "removable_panel", -tb, W + tb, max(p_bot, min_clear), lambda u: panel_top - 0.25, obstacles=obst,
           tags={"removable_panel"}, edge_battens=(False, False), rails=rails)
    if p_bot < min_clear:
        kb_key = rp.get("kick_board", "floor_rim")
        kbs = B.s(kb_key)
        B.add("removable_panel", "Kick board", back.box(skid_ys[0][1] + 0.25, p_bot, skid_ys[1][0] - 0.25,
                                                        min_clear, 0, kbs.t), kb_key,
              grain=1, note="rip to fit; part of the removable panel, closes the gap between the skids",
              tags={"removable_panel"})
    for hy in (W * 0.28, W * 0.72)[: int(rp.get("handles", 2))]:
        B.add("removable_panel", "Pull handle", back.box(hy - 3, rails[0][1] - 2.4, hy + 3, rails[0][1] - 1.4,
                                                         tb + rail.t, tb + rail.t + 1.25),
              material="steel", tags={"removable_panel", "hardware"})
    k.update(panel=(p_bot, panel_top), panel_rails=rails)

    # casing / trim at door and window
    def casing(wf, grp, o, sill=False):
        u0, u1, v0, v1 = o["u0"], o["u1"], o["v0"], o["v1"]
        d0, d1 = tb, tb + trk.t
        B.add(grp, "Head casing", wf.box(u0 - trk.w, v1, u1 + trk.w, v1 + trk.w, d0, d1), sd["trim"], grain=wf.u_axis,
              tags={"trim"})
        vb = v0 if not sill else v0
        for a in (u0 - trk.w, u1):
            B.add(grp, "Side casing", wf.box(a, vb, a + trk.w, v1, d0, d1), sd["trim"], grain=2, tags={"trim"})
        if sill:
            B.add(grp, "Sill", wf.box(u0 - trk.w - 0.75, v0 - 1.5, u1 + trk.w + 0.75, v0, d0, d1 + 1.0), sd["trim"],
                  grain=wf.u_axis, tags={"trim"})
            B.add(grp, "Apron", wf.box(u0 - trk.w, v0 - 1.5 - trk.w, u1 + trk.w, v0 - 1.5, d0, d1), sd["trim"],
                  grain=wf.u_axis, tags={"trim"})

    casing(front, "siding.front", dict(do, v0=FF))
    casing(sides[win_wall][0], f"siding.{win_wall}", wo, sill=True)

    # ---------------------------------------------------------------- door leaf
    if prehung:
        jt = 0.75
        fu0, fu1 = do["u0"] + 0.25, do["u1"] - 0.25           # frame outside, 1/4" shim space each side
        fv1 = do["v1"] - 0.5
        dfm = dr.get("frame_material", "window_frame")
        jd0, jd1 = -st_d, tb                                   # jamb depth: studs + siding (4-1/2")
        for a, b in ((fu0, fu0 + jt), (fu1 - jt, fu1)):
            B.add("door", "Door jamb", front.box(a, FF, b, fv1, jd0, jd1), material=dfm, tags={"door"})
        B.add("door", "Door head jamb", front.box(fu0, fv1 - jt, fu1, fv1, jd0, jd1), material=dfm, tags={"door"})
        B.add("door", "Door sill", front.box(fu0, FF, fu1, FF + 1.0, jd0, jd1 + 1.25), material="steel",
              tags={"door", "hardware"})
        du0, du1 = fu0 + jt + 0.125, fu1 - jt - 0.125
        dv0, dv1 = FF + 1.25, fv1 - jt - 0.125
        slab_t = L(dr.get("slab_thickness", 1.75))
        s0, s1 = -st_d, -st_d + slab_t                         # in-swing: slab against the exterior stop
        dmat = dr.get("material", "door_fiberglass")
        B.add("door", "Door slab", front.box(du0, dv0, du1, dv1, s0, s1), material=dmat, tags={"door"})
        # two-panel face, both sides (shallow relief)
        st_w, top_r, mid_r, bot_r = 4.75, 4.75, 7.0, 9.0
        vm = dv0 + (dv1 - dv0) * 0.42
        for (pv0, pv1) in ((dv0 + bot_r, vm - mid_r / 2), (vm + mid_r / 2, dv1 - top_r)):
            for dd0, dd1 in ((s1, s1 + 0.06), (s0 - 0.06, s0)):
                B.add("door", "Door panel", front.box(du0 + st_w, pv0, du1 - st_w, pv1, dd0, dd1), material=dmat,
                      tags={"door"})
        lu = du0 + 2.6 if hinge_hi else du1 - 2.6              # latch side
        for dd0, dd1 in ((s1, s1 + 2.6), (s0 - 2.6, s0)):
            B.add("door", "Lever handle", front.box(lu - 1.1, FF + 35, lu + 1.1, FF + 37.5, dd0, dd1), material="steel",
                  tags={"door", "hardware"})
            B.add("door", "Deadbolt", front.box(lu - 1.0, FF + 41, lu + 1.0, FF + 43, dd0, dd0 + 0.6 if dd0 == s1
                                                  else dd1), material="steel", tags={"door", "hardware"})
        hu = du1 - 0.6 if hinge_hi else du0 + 0.6
        for hv in (dv0 + 10, 0.5 * (dv0 + dv1), dv1 - 7):
            B.add("door", "Hinge", front.box(hu - 0.6, hv - 2, hu + 0.6, hv + 2, s0 - 0.12, s0), material="steel",
                  tags={"door", "hardware"})
    else:
        dcl = L(dr["clearance"])
        du0, du1 = do["u0"] + dcl, do["u1"] - dcl
        dv0, dv1 = FF + 0.5, do["v1"] - dcl
        nb = int(dr.get("boards", 3))
        dbw = (du1 - du0) / nb
        for i in range(nb):
            B.add("door", "Door board", front.box(du0 + i * dbw, dv0, du0 + (i + 1) * dbw, dv1, 0, tb), sd["board"],
                  grain=2, tags={"door"})
        zb = B.s("door_brace") if "door_brace" in B.stock else trk
        zk = "door_brace" if "door_brace" in B.stock else sd["trim"]
        rails_v = [(dv0 + 6, dv0 + 6 + zb.w), (dv1 - 6 - zb.w, dv1 - 6)]
        for rv0, rv1 in rails_v:
            B.add("door", "Z-brace rail", front.box(du0 + 1, rv0, du1 - 1, rv1, -zb.t, 0), zk, grain=1, tags={"door"})
        ub_, ut_ = (du1 - 1, du0 + 1) if hinge_hi else (du0 + 1, du1 - 1)
        P = np.array([ub_, rails_v[0][1]])
        Q = np.array([ut_, rails_v[1][0]])
        dvec = (Q - P) / np.linalg.norm(Q - P)
        nrm = np.array([-dvec[1], dvec[0]]) * zb.w / 2
        band = [P - 20 * dvec - nrm, Q + 20 * dvec - nrm, Q + 20 * dvec + nrm, P - 20 * dvec + nrm]
        poly = clip2d(band, du0 + 1, rails_v[0][1], du1 - 1, rails_v[1][0])
        B.add("door", "Z-brace diagonal", front.prism(poly, -zb.t, 0), zk, grain=None,
              length=float(np.linalg.norm(Q - P)), tags={"door"})
        hu = du1 if hinge_hi else du0
        for hv in (rails_v[0][0] + zb.w / 2, 0.5 * (dv0 + dv1), rails_v[1][0] + zb.w / 2):
            a, b = (hu - 10, hu + 1.0) if hinge_hi else (hu - 1.0, hu + 10)
            B.add("door", "Strap hinge", front.box(a, hv - 0.6, b, hv + 0.6, tb, tb + 0.1), material="steel",
                  tags={"door", "hardware"})
        lu = du0 + 2 if hinge_hi else du1 - 2
        B.add("door", "Thumb latch", front.box(lu - 0.6, FF + 38, lu + 0.6, FF + 44, tb, tb + 0.4), material="steel",
              tags={"door", "hardware"})
        if dr.get("moon_cutout", False):
            cres = crescent(3.4, 2.9, (1.4, 0.8)) + np.array([0.5 * (du0 + du1), dv1 - 13])
            B.add("door", "Crescent cut-out", front.prism(cres, tb - 0.02, tb + 0.03), material="void",
                  tags={"door", "void"})
    k["door_leaf"] = (du0, du1, dv0, dv1, hinge_hi)
    k["door_swing"] = dr.get("swing", "out" if not prehung else "in")
    k["door_prehung"] = prehung

    # ---------------------------------------------------------------- window unit
    wf = sides[win_wall][0]
    uw, uh = L(win["unit_width"]), L(win["unit_height"])
    a0, b0 = wcx - uw / 2, wcx + uw / 2
    v0, v1 = wo["v0"] + ro_c / 2, wo["v0"] + ro_c / 2 + uh
    fm, gm = win.get("frame_material", "window_frame"), win.get("glass_material", "glass")
    fw = 1.25
    for box_ in ((a0, v0, a0 + fw, v1), (b0 - fw, v0, b0, v1), (a0, v1 - fw, b0, v1), (a0, v0, b0, v0 + fw)):
        B.add("window", "Window frame", wf.box(*box_, -st_d, tb), material=fm, tags={"window"})
    vm = 0.5 * (v0 + v1)
    for (sv0, sv1), (d0, d1) in (((v0 + fw, vm + 0.75), (-2.25, -1.0)), ((vm - 0.75, v1 - fw), (-1.0, 0.25))):
        sa0, sb0 = a0 + fw, b0 - fw
        for box_ in ((sa0, sv0, sa0 + 1.5, sv1), (sb0 - 1.5, sv0, sb0, sv1), (sa0, sv1 - 1.5, sb0, sv1), (sa0, sv0, sb0, sv0 + 1.5)):
            B.add("window", "Sash", wf.box(*box_, d0, d1), material=fm, tags={"window"})
        B.add("window", "Glass", wf.box(sa0 + 1.5, sv0 + 1.5, sb0 - 1.5, sv1 - 1.5, 0.5 * (d0 + d1) - 0.06, 0.5 * (d0 + d1) + 0.06),
              material=gm, tags={"window", "glass"})
    k["window_unit"] = (a0, b0, v0, v1)

    # ---------------------------------------------------------------- vent
    vt = pit["vent"]
    if vt.get("enabled", True):
        vod = L(vt["outside_diameter"])
        vr = vod / 2
        pos = vt.get("position", "auto")
        if pos == "auto":
            best = None
            for vx in np.arange(st_d + vr + 1.5, x_fp0, 0.5):
                for vy in np.arange(yi0 + vr + 1, yi1 - vr - 1, 0.5):
                    if math.hypot(vx - pcx, vy - pcy) > ID / 2 - vr - 0.75:
                        continue
                    if any(y0 - vr - 0.25 < vy < y0 + bf.t + vr + 0.25 for y0 in bj_y):
                        continue
                    ex, ey, erx, ery = hx, pcy, sl / 2 + vr + 1.5, sw / 2 + vr + 1.5
                    if ((vx - ex) / erx) ** 2 + ((vy - ey) / ery) ** 2 < 1:
                        continue
                    if any(x - vr - 0.5 < vx < x + raf.t + vr + 0.5 for x in raf_x):
                        continue
                    if any(p0 - vr - 0.5 < vy < p1 + vr + 0.5 for p0, p1 in purlins):
                        continue
                    if any(sx - vr - 2.0 < vx < sx + vr + 2.0 for sx in seams):
                        continue
                    score = vx + 0.35 * (yi1 - vy)
                    if best is None or score < best[0]:
                        best = (score, float(vx), float(vy))
            pos = best[1:] if best else None
        if pos:
            vx, vy = L(pos[0]), L(pos[1])
            zr = z_roof(vy + vr)
            vtop = zr + L(vt["height_above_roof"])
            B.add("pit", "Vent pipe", Tube(vx, vy, vr, vr - L(vt.get("wall", 0.24)), z_fp0, vtop),
                  material=vt.get("material", "vent_pipe"), tags={"vent"}, grain=2)
            B.add("pit", "Vent cap", Tube(vx, vy, vr + 0.6, 0, vtop, vtop + 1.5), material=vt.get("material", "vent_pipe"),
                  tags={"vent", "hardware"})
            B.add("pit", "Roof boot", Tube(vx, vy, vr + 1.25, vr, z_roof(vy) - 0.5, z_roof(vy) + 2.0), material="epdm",
                  tags={"vent", "hardware"})
            k["vent"] = (vx, vy, vr, vtop)
        else:
            k["vent"] = None

    # ---------------------------------------------------------------- electrical, lights, room vent
    el = cfg.get("electrical", {})
    k["electrical"] = None
    latch_hi = not hinge_hi
    do = k["door"]

    def rafter_bay(xc):
        bays = [(a + raf.t, b) for a, b in zip(raf_x, raf_x[1:]) if b - a > raf.t + 1]
        return min(bays, key=lambda ab: abs(0.5 * (ab[0] + ab[1]) - xc))

    def ceiling_light(tag, xc, desc):
        a, b = rafter_bay(xc)
        ya, yb = W / 2 - 2.75, W / 2 + 2.75
        bpoly = [[ya, z_rb(ya)], [yb, z_rb(yb)], [yb, z_rb(yb) + 1.5 * c_], [ya, z_rb(ya) + 1.5 * c_]]
        B.add("electrical", "Fixture block", Prism("x", bpoly, a, b), roof["rafter"], grain=0,
              note="2x6 flat between rafters for light fixture", tags={"framing", "electrical"})
        xm, zb_ = 0.5 * (a + b), z_rb(W / 2)
        B.add("electrical", f"Light {tag}", Tube(xm, W / 2, 2.9, 0, zb_ - 1.0, zb_ + 0.1), material="fixture",
              tags={"electrical", "hardware", "light"})
        B.add("electrical", f"Light {tag} globe", Tube(xm, W / 2, 2.4, 0, zb_ - 4.5, zb_ - 1.0), material="fixture_globe",
              tags={"electrical", "hardware", "light"})
        return dict(tag=tag, kind="light", x=xm, y=W / 2, z=zb_ - 4.5, desc=desc)

    if el.get("enabled", False):
        dev = []
        swh = L(el.get("switch_height", 48))
        rhi = L(el.get("receptacle_height_interior", 44))
        rhe = L(el.get("receptacle_height_exterior", 20))
        # interior switch S1: inside face of the front wall, latch side of the door
        u1_ = do["u1"] + 3 * st_t + 1.75 if latch_hi else do["u0"] - 3 * st_t - 1.75
        B.add("electrical", "Switch S1", front.box(u1_ - 1.4, FF + swh - 2.25, u1_ + 1.4, FF + swh + 2.25, -st_d - 0.3,
                                                   -st_d), material="device", tags={"electrical", "hardware"})
        dev.append(dict(tag="S1", kind="switch", x=D - st_d, y=u1_, z=FF + swh, controls="L1",
                        desc=el.get("switch", "Single-pole switch")))
        # interior receptacle R1: latch-side wall near the front corner
        wall_r = sides["right" if latch_hi else "left"][0]
        xr = D - st_d - 9
        B.add("electrical", "Receptacle R1", wall_r.box(xr - 1.4, FF + rhi - 2.25, xr + 1.4, FF + rhi + 2.25, -st_d - 0.3,
                                                        -st_d), material="device", tags={"electrical", "hardware"})
        dev.append(dict(tag="R1", kind="receptacle", x=xr, y=(W - st_d) if latch_hi else st_d, z=FF + rhi,
                        desc=el.get("receptacle_interior", "20 A GFCI receptacle")))
        # exterior switch S2 + receptacle R2 on a rough-sawn mounting board beside the door (latch side)
        mb = B.s(el.get("mounting_board", "trim"))
        ue = do["u1"] + trk.w + 1.5 + mb.w / 2 if latch_hi else do["u0"] - trk.w - 1.5 - mb.w / 2
        d0 = tb + tbat
        B.add("electrical", "Mounting board", front.box(ue - mb.w / 2, FF + rhe - 5, ue + mb.w / 2, FF + swh + 5, d0,
                                                        d0 + mb.t), el.get("mounting_board", "trim"), grain=2,
              tags={"electrical", "trim"})
        B.add("electrical", "Switch S2", front.box(ue - 1.6, FF + swh - 2.6, ue + 1.6, FF + swh + 2.6, d0 + mb.t,
                                                   d0 + mb.t + 2.25), material="wp_box", tags={"electrical", "hardware"})
        B.add("electrical", "Receptacle R2", front.box(ue - 2.1, FF + rhe - 3.2, ue + 2.1, FF + rhe + 3.2, d0 + mb.t,
                                                       d0 + mb.t + 3.25), material="wp_box", tags={"electrical", "hardware"})
        dev.append(dict(tag="S2", kind="switch_wp", x=D + d0 + mb.t, y=ue, z=FF + swh, controls="L2",
                        desc=el.get("switch_exterior", "Weatherproof single-pole switch")))
        dev.append(dict(tag="R2", kind="receptacle_wp", x=D + d0 + mb.t, y=ue, z=FF + rhe,
                        desc=el.get("receptacle_exterior", "20 A WR GFCI receptacle, in-use cover")))
        # lights
        dev.append(ceiling_light("L1", 0.5 * (x_bf + D - st_d), el.get("interior_light", "Damp-rated LED ceiling light")))
        dev.append(ceiling_light("L2", D + DD / 2, el.get("deck_light", "Damp-rated LED deck light")))
        # service entry: disconnect on the low side wall near the front corner, conduit to grade
        wl_ = sides["left"][0]
        xd = D - 10
        zd = L(el.get("disconnect_height", 36))
        B.add("electrical", "Mounting board", wl_.box(xd - mb.w / 2, zd - 8, xd + mb.w / 2, zd + 8, d0, d0 + mb.t),
              el.get("mounting_board", "trim"), grain=2, tags={"electrical", "trim"})
        B.add("electrical", "Disconnect DS", wl_.box(xd - 2.2, zd - 3.5, xd + 2.2, zd + 3.5, d0 + mb.t, d0 + mb.t + 3.0),
              material="wp_box", tags={"electrical", "hardware"})
        cy_ = -(d0 + mb.t + 1.2)
        B.add("electrical", "Conduit", Tube(xd, cy_, 0.525, 0.41, -L(el.get("burial_depth", 24)), zd - 3.5, segments=24),
              material="pvc_gray", tags={"electrical", "hardware", "below_grade"})
        dev.append(dict(tag="DS", kind="disconnect", x=xd, y=cy_, z=zd,
                        desc=el.get("disconnect", "Weatherproof disconnect switch")))
        # dedicated service: meter-main pedestal on a PT post, off the low side (utility service ends here)
        svc = el.get("service", {})
        if svc.get("type", "pedestal") == "pedestal":
            px_, py_ = xd + L(svc.get("offset_x", 0)), -L(svc.get("distance", 72))
            pst = B.s(svc.get("post", "post"))
            pz0 = -L(svc.get("post_embed", 36))
            ptop = L(svc.get("post_height", 66))
            B.add("site", "Service post", Box(px_ - pst.t / 2, py_ - pst.t / 2, pz0, px_ + pst.t / 2, py_ + pst.t / 2, ptop),
                  svc.get("post", "post"), grain=2, note="set in tamped gravel / concrete", tags={"site", "electrical"})
            ew, eh, ed = 12.0, 28.0, 5.0
            ez0 = L(svc.get("meter_center", 60)) - 8
            fy = py_ + pst.t / 2
            B.add("site", "Meter-main enclosure", Box(px_ - ew / 2, fy, ez0, px_ + ew / 2, fy + ed, ez0 + eh),
                  material="wp_box", tags={"site", "electrical", "hardware"})
            B.add("site", "Meter", Tube(px_, fy + ed, 3.4, 0, ez0 + eh - 11.5, ez0 + eh - 4.5, segments=32), material="glass",
                  tags={"site", "electrical", "hardware"})
            for dx_ in (-3.0, 3.0):
                B.add("site", "Conduit", Tube(px_ + dx_, fy + 2.0, 0.66, 0.52, -L(el.get("burial_depth", 24)), ez0,
                                              segments=24), material="pvc_gray", tags={"site", "electrical", "below_grade"})
            dev.append(dict(tag="MP", kind="panel", x=px_, y=fy, z=ez0 + eh / 2,
                            desc=svc.get("description", "Meter-main pedestal")))
            k["service"] = dict(x=px_, y=py_, run_ft=(math.hypot(px_ - xd, py_ - cy_) + (ez0 - 0) + (zd + 12)) / 12)
        k["electrical"] = dict(devices=dev)

    lv = cfg.get("room_vent", {})
    if lv.get("enabled", False):
        vw_, vh_ = L(lv.get("width", 12)), L(lv.get("height", 8))
        yc = W - st_d - vw_ / 2 - 4
        zc_ = z_rb(yc) - vh_ / 2 - 6
        B.add("siding.back", "Louver vent", back.box(yc - vw_ / 2, zc_ - vh_ / 2, yc + vw_ / 2, zc_ + vh_ / 2, tb + tbat,
                                                     tb + tbat + 1.25), material=lv.get("material", "louver"),
              tags={"hardware", "vent"})
        k["louver"] = (yc, zc_, vw_, vh_)

    # ---------------------------------------------------------------- ground (dirt pad around the pit)
    env = cfg.get("renders", {}).get("environment", {})
    margin = L(env.get("dirt_pad", {}).get("margin", 60))
    gdepth = L(env.get("ground_depth", 60))
    lo, hi = m.bbox()
    gx0, gy0, gx1, gy1 = lo[0] - margin, lo[1] - margin, hi[0] + margin, hi[1] + margin
    B.add("ground", "Soil", Plate(("rect", gx0, gy0, gx1, gy1), pipe_bot, 0, hole=("ellipse", pcx, pcy, OD / 2, OD / 2),
                                  segments=96), material="soil", tags={"ground"})
    B.add("ground", "Soil", Box(gx0, gy0, -gdepth, gx1, gy1, pipe_bot), material="soil", tags={"ground"})
    k["ground_rect"] = (gx0, gy0, gx1, gy1)

    from .checks import run_checks
    run_checks(m)
    return m


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
