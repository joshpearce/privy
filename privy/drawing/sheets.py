"""Sheet compositions: cover, plans, elevations, sections, framing, details, schedules, renderings."""
from __future__ import annotations

import datetime as _dt
import math
import os
import textwrap

import numpy as np

from .. import bom
from ..units import fmt_elev, fmt_ftin, fmt_in, parse_len
from . import svg as S
from .view import View

L = parse_len


# ============================================================================ helpers

def wrap(text, width, size, caps=False):
    cw = size * (0.60 if caps else 0.50)
    n = max(8, int(width / cw))
    out = []
    for para in str(text).split("\n"):
        out += textwrap.wrap(para, n) or [""]
    return out


def table(svg: S.SVG, x, y, cols, rows, title=None, size=0.072, rh=None, zebra=True, max_y=None, overflow=None):
    """cols: [(header, width, align)] ; rows: list of lists of str.  Returns y below table.
    Rows that do not fit above max_y are appended to `overflow` (if given) instead of being dropped."""
    rh = rh or size * 1.75
    t = svg.t
    W = sum(c[1] for c in cols)
    if title:
        svg.text(x, y, title.upper(), size * 1.35, weight="bold")
        y += 0.08
    svg.rect(x, y, W, rh, fill=t["table"], lw="light")
    cx = x
    for h, w, a in cols:
        svg.text(cx + (0.04 if a == "start" else (w / 2 if a == "middle" else w - 0.04)), y + rh * 0.68, h.upper(),
                 size * 0.95, a, weight="bold")
        cx += w
    y += rh
    for i, r in enumerate(rows):
        # wrap long cells
        cells = []
        nl = 1
        for (h, w, a), c in zip(cols, r):
            ln = wrap(c, w - 0.08, size)
            cells.append(ln)
            nl = max(nl, len(ln))
        hh = rh + (nl - 1) * size * 1.2
        if max_y and y + hh > max_y:
            if overflow is not None:
                overflow.extend(rows[i:])
            svg.text(x, y + rh * 0.7, "(continued)" if overflow is not None else "(continued / truncated)", size,
                     italic=True)
            return y + rh
        if zebra and i % 2 == 1:
            svg.rect(x, y, W, hh, fill=t["table"], stroke=False)
        cx = x
        for (h, w, a), ln in zip(cols, cells):
            for j, s_ in enumerate(ln):
                svg.text(cx + (0.04 if a == "start" else (w / 2 if a == "middle" else w - 0.04)),
                         y + rh * 0.68 + j * size * 1.2, s_, size, a)
            cx += w
        svg.line(x, y + hh, x + W, y + hh, lw="fine", color=t["faint"])
        y += hh
    svg.rect(x, y - (y - (y)), 0, 0)
    return y


def pick_scale(ctx, views, w, h, max_scale=None):
    scales = sorted((S.parse_scale(s) for s in ctx["cfg"]["drawings"]["scales"]), reverse=True)
    for s in scales:
        if max_scale and s > max_scale + 1e-9:
            continue
        if all(v.size(s)[0] <= w and v.size(s)[1] <= h for v in views):
            return s
    return scales[-1]


def place(v, x, y, w, h, scale, valign="center"):
    vw, vh = v.size(scale)
    x0 = x + (w - vw) / 2
    y0 = y + ((h - vh) / 2 if valign == "center" else 0)
    return v.place(x0, y0, scale)


def finish_view(svg, v, num, title, below=0.45):
    x0, y0, x1, y1 = v.paper_box()
    S.view_title(svg, x0, y1 + below, num, title, S.scale_label(v.scale))


# ============================================================================ sheet frame

class Sheet:
    def __init__(self, ctx, number, title):
        cfg = ctx["cfg"]
        dr = cfg["drawings"]
        self.ctx, self.number, self.title = ctx, number, title
        w, h = dr["sheet"]["size"]
        self.W, self.H = w, h
        self.m_ = dr["sheet"].get("margin", 0.4)
        self.tbw = dr["sheet"].get("title_block_width", 2.3)
        self.svg = S.SVG(w, h, dr.get("theme", "white"), dr.get("font", "sans-serif"))
        # drawing area
        self.x0, self.y0 = self.m_ + 0.2, self.m_ + 0.2
        self.x1, self.y1 = w - self.m_ - self.tbw - 0.2, h - self.m_ - 0.2

    @property
    def area(self):
        return self.x0, self.y0, self.x1 - self.x0, self.y1 - self.y0

    def frame(self):
        s, m, W, H = self.svg, self.m_, self.W, self.H
        ctx, cfg, k = self.ctx, self.ctx["cfg"], self.ctx["model"].key
        s.rect(m, m, W - 2 * m, H - 2 * m, lw="border")
        tx = W - m - self.tbw
        s.line(tx, m, tx, H - m, lw="heavy")
        p = cfg["project"]
        x = tx + 0.15
        wdt = self.tbw - 0.3
        y = m + 0.35
        for ln in wrap(p["name"].upper(), wdt, 0.16, caps=True):
            s.text(x, y, ln, 0.16, weight="bold")
            y += 0.2
        y += 0.02
        for ln in wrap(p.get("subtitle", ""), wdt, 0.085):
            s.text(x, y, ln, 0.085)
            y += 0.12
        y += 0.1
        s.line(tx, y, W - m, y, lw="light")
        y += 0.2
        info = [("SITE", p.get("site") or "—"), ("OWNER", p.get("owner") or "—"), ("DESIGN", p.get("designer") or "—")]
        for a, b in info:
            s.text(x, y, a, 0.06, weight="bold", color=s.t["light"])
            for ln in wrap(b, wdt - 0.55, 0.08):
                s.text(x + 0.55, y, ln, 0.08)
                y += 0.12
            y += 0.02
        y += 0.05
        s.line(tx, y, W - m, y, lw="light")
        y += 0.2
        s.text(x, y, "KEY DIMENSIONS", 0.07, weight="bold")
        y += 0.15
        kd = [("Footprint (framing)", f"{fmt_ftin(k['W'])} x {fmt_ftin(k['D'])}"),
              ("Deck", f"{fmt_ftin(k['W'])} x {fmt_ftin(k['DD'])}"),
              ("Skids", f"{fmt_ftin(k['skid_x1'] - k['skid_x0'])} long"),
              ("Roof", f"{cfg['roof']['pitch']}:12 single slope, metal"),
              ("Pit", f"{fmt_in(k['ID'])} ID HDPE, {fmt_ftin(-k['pipe_bot'])} deep"),
              ("Height to roof", fmt_ftin(k['z_roof'](k['roof_y'][1])))]
        for a, b in kd:
            s.text(x, y, a, 0.068, color=s.t["light"])
            s.text(W - m - 0.12, y, b, 0.068, "end")
            y += 0.125
        y += 0.05
        s.line(tx, y, W - m, y, lw="light")
        y += 0.2
        s.text(x, y, "PARAMETRIC SOURCE", 0.065, weight="bold")
        y += 0.13
        s.text(x, y, ctx.get("source_name", "design/privy.json"), 0.07)
        y += 0.12
        s.text(x, y, f"model hash {k['cfg_hash']}", 0.065, color=s.t["light"])
        y += 0.12
        s.text(x, y, f"generated {ctx['generated']}", 0.065, color=s.t["light"])
        y += 0.15
        s.line(tx, y, W - m, y, lw="light")
        y += 0.2
        s.text(x, y, "REV", 0.06, weight="bold")
        s.text(x + 0.35, y, "DATE", 0.06, weight="bold")
        s.text(x + 1.05, y, "DESCRIPTION", 0.06, weight="bold")
        y += 0.14
        s.text(x, y, p.get("revision", "A"), 0.07)
        s.text(x + 0.35, y, p.get("date", ""), 0.07)
        s.text(x + 1.05, y, "Issued for review", 0.07)
        # bottom: sheet title and number
        yb = H - m - 1.75
        s.line(tx, yb, W - m, yb, lw="heavy")
        yy = yb + 0.25
        s.text(x, yy, "SHEET TITLE", 0.06, weight="bold", color=s.t["light"])
        yy += 0.2
        for ln in wrap(self.title.upper(), wdt, 0.13, caps=True):
            s.text(x, yy, ln, 0.13, weight="bold")
            yy += 0.17
        s.line(tx, H - m - 0.85, W - m, H - m - 0.85, lw="light")
        s.text(x, H - m - 0.66, "SCALE: AS NOTED", 0.07)
        s.text(x, H - m - 0.52, f"DATE: {p.get('date', '')}", 0.07)
        s.text(x, H - m - 0.38, f"SHEET {ctx['index'][self.number][0]} OF {len(ctx['index'])}", 0.07)
        s.text(W - m - 0.12, H - m - 0.18, self.number, 0.42, "end", weight="bold")

    def done(self):
        self.frame()
        return self.number, self.title, self.svg.tostring()


# ============================================================================ annotation helpers

class Keynotes:
    """Numbered keynote tags with leaders, plus a legend.  Identical text shares a number."""

    def __init__(self, svg: S.SVG):
        self.svg = svg
        self.num: dict[str, int] = {}
        self.tags: list[tuple[float, float]] = []

    def __call__(self, v, pt3, text, d=(0.35, -0.3)):
        s = self.svg
        if text not in self.num:
            self.num[text] = len(self.num) + 1
        n = self.num[text]
        ax, ay = v.Pm(pt3)
        tx, ty = ax + d[0], ay + d[1]
        step = 0.21 if d[1] >= 0 else -0.21
        for _ in range(10):
            if all(math.hypot(tx - a, ty - b) > 0.2 for a, b in self.tags):
                break
            ty += step
        self.tags.append((tx, ty))
        r = 0.075
        L_ = math.hypot(tx - ax, ty - ay) or 1.0
        ex, ey = tx - (tx - ax) / L_ * r, ty - (ty - ay) / L_ * r
        s.line(ax, ay, ex, ey, lw="dim")
        s.circle(ax, ay, 0.013, fill=s.t["ink"], stroke=False)
        hexagon(s, tx, ty, r)
        s.text(tx, ty + 0.027, str(n), 0.07, "middle", weight="bold")

    def legend(self, x, y, w, h, cols=2, size=0.072, title="Keynotes"):
        s = self.svg
        if not self.num:
            return y
        s.text(x, y, title.upper(), 0.095, weight="bold")
        y0 = y + 0.2
        cw = (w - 0.2 * (cols - 1)) / cols
        col, yy = 0, y0
        for text, n in sorted(self.num.items(), key=lambda kv: kv[1]):
            lines = wrap(text, cw - 0.3, size)
            need = len(lines) * size * 1.3 + 0.06
            if yy + need > y + h and col < cols - 1:
                col += 1
                yy = y0
            cx = x + col * (cw + 0.2)
            hexagon(s, cx + 0.08, yy - size * 0.35, 0.068)
            s.text(cx + 0.08, yy - size * 0.35 + 0.025, str(n), 0.062, "middle", weight="bold")
            s.mtext(cx + 0.22, yy, lines, size, lead=1.3)
            yy += need
        return yy


def hexagon(s, cx, cy, r):
    pts = [(cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))) for a in range(0, 360, 60)]
    s.path(S.pts_to_d(pts), fill=s.t["bg"], lw="light")


def row_layout(ctx, views, x, y, w, h, pads, max_scale=None, valign="top"):
    """Place views side by side at one common scale.  pads: [(left, right)] paper inches."""
    aw = w - sum(a + b for a, b in pads)
    scales = sorted((S.parse_scale(t) for t in ctx["cfg"]["drawings"]["scales"]), reverse=True)
    sc = scales[-1]
    for s_ in scales:
        if max_scale and s_ > max_scale + 1e-9:
            continue
        if sum(v.size(s_)[0] for v in views) <= aw and all(v.size(s_)[1] <= h for v in views):
            sc = s_
            break
    cx = x
    for v, (a, b) in zip(views, pads):
        vw, vh = v.size(sc)
        cx += a
        v.place(cx, y if valign == "top" else y + h - vh, sc)
        cx += vw + b
    return sc


def _thin(vals, min_gap):
    vals = sorted(vals)
    out = [vals[0]]
    for a in vals[1:]:
        if a - out[-1] >= min_gap:
            out.append(a)
        elif a == vals[-1]:
            out[-1] = a
    return out


def dims_below(v, s, us, v_from, off=0.3, min_gap=1.9):
    """Chain of horizontal dims under the view box; extension lines start at model height v_from."""
    x0, y0, x1, y1 = v.paper_box()
    yl = y1 + off
    us = _thin(us, min_gap)
    for a, b in zip(us, us[1:]):
        if b - a < 1e-6:
            continue
        pa, pb = v.P(a, v_from), v.P(b, v_from)
        S.dim_h(s, pa[0], pb[0], yl, fmt_ftin(b - a), ext_from=(pa[1], pb[1]))
    return yl


def dims_above(v, s, us, v_from, off=0.3, min_gap=1.9):
    x0, y0, x1, y1 = v.paper_box()
    yl = y0 - off
    us = _thin(us, min_gap)
    for a, b in zip(us, us[1:]):
        if b - a < 1e-6:
            continue
        pa, pb = v.P(a, v_from), v.P(b, v_from)
        S.dim_h(s, pa[0], pb[0], yl, fmt_ftin(b - a), ext_from=(pa[1], pb[1]))
    return yl


def dims_left(v, s, vs, u_from, off=0.3, min_gap=1.4):
    x0, y0, x1, y1 = v.paper_box()
    xl = x0 - off
    vs = _thin(vs, min_gap)
    for a, b in zip(vs, vs[1:]):
        if b - a < 1e-6:
            continue
        pa, pb = v.P(u_from, a), v.P(u_from, b)
        S.dim_v(s, pa[1], pb[1], xl, fmt_ftin(b - a), ext_from=(pa[0], pb[0]))
    return xl


def dims_right(v, s, vs, u_from, off=0.3, min_gap=1.4):
    x0, y0, x1, y1 = v.paper_box()
    xl = x1 + off
    vs = _thin(vs, min_gap)
    for a, b in zip(vs, vs[1:]):
        if b - a < 1e-6:
            continue
        pa, pb = v.P(u_from, a), v.P(u_from, b)
        S.dim_v(s, pa[1], pb[1], xl, fmt_ftin(b - a), ext_from=(pa[0], pb[0]), side=1)
    return xl


def title_under(s, v, num, title, below=0.55):
    x0, y0, x1, y1 = v.paper_box()
    S.view_title(s, x0, y1 + below, num, title, S.scale_label(v.scale))


def notes_block(s, x, y, w, title, notes, size=0.075):
    s.text(x, y, title.upper(), 0.095, weight="bold")
    y += 0.2
    for i, n in enumerate(notes, 1):
        s.text(x, y, f"{i}.", size, weight="bold")
        y = s.mtext(x + 0.2, y, wrap(n, w - 0.22, size), size) + 0.05
    return y


def arrow(s, x0, y0, x1, y1, head=0.08, lw="med"):
    s.line(x0, y0, x1, y1, lw=lw)
    a = math.atan2(y1 - y0, x1 - x0)
    p1 = (x1 - head * math.cos(a - 0.4), y1 - head * math.sin(a - 0.4))
    p2 = (x1 - head * math.cos(a + 0.4), y1 - head * math.sin(a + 0.4))
    s.path(S.pts_to_d([(x1, y1), p1, p2]), fill=s.t["ink"], lw="fine")


def sec_marker(s, v, a, b, letter, sheet, look):
    """Section cut line between plan points a, b with bubbles; look = (du, dv) plan direction."""
    pa, pb = v.P(*a), v.P(*b)
    s.line(*pa, *pb, lw="light", dash=(0.18, 0.05, 0.03, 0.05))
    for p in (pa, pb):
        r = 0.14
        dx, dy = look[0], -look[1]
        tip = (p[0] + dx * (r + 0.1), p[1] + dy * (r + 0.1))
        base = (p[0] + dx * r, p[1] + dy * r)
        nx, ny = -dy, dx
        s.path(S.pts_to_d([tip, (base[0] + nx * 0.08, base[1] + ny * 0.08), (base[0] - nx * 0.08, base[1] - ny * 0.08)]),
               fill=s.t["ink"], lw="fine")
        s.circle(p[0], p[1], r, fill=s.t["bg"], lw="med")
        s.line(p[0] - r, p[1], p[0] + r, p[1], lw="light")
        s.text(p[0], p[1] - 0.025, letter, 0.1, "middle", weight="bold")
        s.text(p[0], p[1] + 0.09, sheet, 0.055, "middle")


def roof_text(m):
    rf, k = m.cfg["roof"], m.key
    if k.get("substrate", "purlins") == "purlins":
        return f"{rf['roofing']['description']} on 2x4 purlins @ {fmt_in(L(rf['purlin_spacing']))} max"
    return f"{rf['roofing']['description']}; {m.stock[rf['sheathing']].description} on the rafters"


def door_text(m):
    d = m.cfg["door"]
    if m.key.get("door_prehung"):
        return f"D1: {d['description']}. {d['hardware']}"
    return "D1: 3-board Z-braced rough-sawn door, strap hinges, thumb latch"


def swing_arc(s, v, hinge, open_end, closed_end):
    """Door leaf (open, heavy) + dashed swing arc, all in plan model coords."""
    ph, po, pc = v.P(*hinge), v.P(*open_end), v.P(*closed_end)
    s.line(*ph, *po, lw="heavy")
    r = math.hypot(po[0] - ph[0], po[1] - ph[1])
    cross = (po[0] - ph[0]) * (pc[1] - ph[1]) - (po[1] - ph[1]) * (pc[0] - ph[0])
    s.path(f"M{S.f(po[0])} {S.f(po[1])} A{S.f(r)} {S.f(r)} 0 0 {1 if cross > 0 else 0} {S.f(pc[0])} {S.f(pc[1])}",
           lw="light", dash=(0.04, 0.03))


# ============================================================================ view factories

def elevation_view(m, side):
    return View(m, side, hidden=lambda p: "pipe" in p.tags)


def plan_view(m, site=False):
    k = m.key
    zc = k["FF"] + L(m.cfg["drawings"].get("plan_cut_height", 60))
    return View(m, "top", cut=(2, zc, 1), include=lambda p: p.group != "door", hidden=lambda p: "pipe" in p.tags,
                site=site)


def long_section(m):
    return View(m, "left", cut=(1, m.key["pipe_cy"], -1))


def trans_section(m):
    return View(m, "front", cut=(0, m.key["pipe_cx"], 1))


def std_levels(v, svg, m, which, off=0.3):
    k = m.key
    lv = {
        "grade": (0.0, "Grade"), "deck": (k["deck_top"], "Top of deck"), "ff": (k["FF"], "Finish floor"),
        "bench": (k["z_bt"], "Top of bench"), "low": (k["z_Lb"], "Brg. low beam"), "high": (k["z_Hb"], "Brg. high beam"),
        "roof": (k["z_roof"](k["roof_y"][1]), "Top of roof"), "pit": (k["pipe_bot"], "Bottom of pit"),
        "pipe": (k["pipe_top"], "Top of pipe"),
    }
    x0, y0, x1, y1 = v.paper_box()
    placed = []
    for key in which:
        z, lab = lv[key]
        y = v.P(0, z)[1]
        if any(abs(y - yy) < 0.22 for yy in placed):
            continue
        placed.append(y)
        S.level_mark(svg, x1 + off, y, lab, fmt_elev(z))
        svg.line(x1 + 0.04, y, x1 + off - 0.06, y, lw="fine", dash=(0.03, 0.02))


# ============================================================================ sheets

def sheet_cover(ctx):
    sh = Sheet(ctx, "G-001", "Cover, sheet index & general notes")
    s, cfg, m, k = sh.svg, ctx["cfg"], ctx["model"], ctx["model"].key
    x, y, w, h = sh.area
    p = cfg["project"]
    s.text(x, y + 0.3, p["name"].upper(), 0.32, weight="bold")
    s.text(x, y + 0.56, p.get("subtitle", ""), 0.13)
    iw = 8.9
    ty = y + 0.78
    hero = next((r for r in ctx["renders"] if r["name"] == "hero"), ctx["renders"][0] if ctx["renders"] else None)
    ih = iw / 1.5
    if hero and os.path.exists(hero["path"]):
        data, ar = S.jpeg_bytes(hero["path"])
        ih = iw / ar
        s.image(x, ty, iw, ih, data)
        s.rect(x, ty, iw, ih, lw="light")
        s.text(x, ty + ih + 0.15, hero["title"], 0.075, italic=True)
    else:
        s.rect(x, ty, iw, ih, lw="light")
        s.text(x + iw / 2, ty + ih / 2, "Rendering not generated (build without --no-render)", 0.12, "middle")
    ny = ty + ih + 0.42
    s.text(x, ny, "GENERAL NOTES", 0.1, weight="bold")
    ny += 0.17
    notes = cfg["drawings"].get("general_notes", [])
    colw = (iw - 0.25) / 2
    col, yy = 0, ny
    for i, note in enumerate(notes, 1):
        lines = wrap(note, colw - 0.25, 0.07)
        if yy + len(lines) * 0.095 > sh.y1 and col == 0:
            col, yy = 1, ny
        cx = x + col * (colw + 0.25)
        s.text(cx, yy, f"{i}.", 0.07, weight="bold")
        yy = s.mtext(cx + 0.2, yy, lines, 0.07, lead=1.33) + 0.03
    cx = x + iw + 0.35
    cw = sh.x1 - cx
    yy = ty + 0.05
    rows = [[n, t] for n, (i, t) in ctx["index"].items()]
    yy = table(s, cx, yy, [("Sheet", 0.55, "start"), ("Title", cw - 0.55, "start")], rows, title="Sheet index", size=0.068)
    yy += 0.3
    ld = cfg["loads"]
    crit = [["Ground snow pg", f"{ld['ground_snow_psf']} psf"], ["Roof snow pf", f"{k['snow_pf']:.1f} psf"],
            ["Roof dead", f"{ld['roof_dead_psf']} psf"], ["Floor / deck live", f"{ld['floor_live_psf']} / {ld['deck_live_psf']} psf"],
            ["Bench occupant", f"{ld['occupant_point_lb']} lb point"], ["Est. empty weight", f"{k['weight']:,.0f} lb"],
            ["Tow force on dirt", f"{0.5 * k['weight']:,.0f}-{0.7 * k['weight']:,.0f} lb"]]
    if "pit_volume_gal" in k:
        crit.append(["Usable pit volume", f"{k['pit_volume_gal']:.0f} gal"])
        if k.get("pit_years"):
            crit.append(["Est. pit service life", f"{k['pit_years']:.1f} yr"])
    yy = table(s, cx, yy, [("Item", cw * 0.58, "start"), ("Value", cw * 0.42, "end")], crit, title="Design basis",
               size=0.068)
    yy += 0.3
    n_bad = sum(1 for c in m.checks if c["ok"] is False)
    s.text(cx, yy, "DESIGN CHECKS", 0.095, weight="bold")
    yy += 0.18
    s.text(cx, yy, f"{sum(1 for c in m.checks if c['ok'] is True)} pass, {n_bad} fail, "
           f"{sum(1 for c in m.checks if c['ok'] is None)} info (A-602)", 0.075,
           color=(s.t["accent"] if n_bad else s.t["ink"]), weight="bold" if n_bad else "normal")
    yy += 0.3
    s.text(cx, yy, "HOW TO REVISE", 0.095, weight="bold")
    yy += 0.17
    msg = ("Edit the JSON (dimensions, stock, materials, lighting, cameras) and run "
           "python -m privy build. Sheets, cut list, checks and renderings regenerate from it.")
    s.mtext(cx, yy, wrap(msg, cw, 0.07), 0.07)
    return sh.done()


def sheet_floor_plan(ctx):
    sh = Sheet(ctx, "A-101", "Floor plan")
    s, m, k = sh.svg, ctx["model"], ctx["model"].key
    kn = Keynotes(s)
    x, y, w, h = sh.area
    W, D, DD = k["W"], k["D"], k["DD"]
    rx0, rx1 = k["roof_x"]
    ry0, ry1 = k["roof_y"]
    vp = plan_view(m)
    ext = vp.extents()
    vp.crop = (min(ext[0], rx0) - 1, min(ext[1], ry0) - 1, max(ext[2], rx1) + 1, max(ext[3], ry1) + 1)
    row_layout(ctx, [vp], x, y + 0.85, w, h - 3.1, [(1.1, 1.1)])
    vp.draw(s)
    vp.draw_hidden(s)
    s.path(vp._d([(rx0, ry0), (rx1, ry0), (rx1, ry1), (rx0, ry1)]), lw="light", dash=(0.14, 0.05, 0.03, 0.05))
    # door swing
    du0, du1, dv0, dv1, hinge_hi = k["door_leaf"]
    hy, ly = (du1, du0) if hinge_hi else (du0, du1)
    wl = du1 - du0
    if k.get("door_swing") == "in":
        hx = D - k["st_d"]
        swing_arc(s, vp, (hx, hy), (hx - wl, hy), (hx, ly))
    else:
        hx = D + k["tb"]
        swing_arc(s, vp, (hx, hy), (hx + wl, hy), (hx, ly))
    for u, v_, t in (((k["x_bf"] + D) / 2, W * 0.3, "PRIVY"), (D + DD / 2, W * 0.66, "COVERED DECK")):
        px, py = vp.P(u, v_)
        s.text(px, py, t, 0.11, "middle", weight="bold")
    # dims
    dims_below(vp, s, [k["skid_x0"], 0, D, D + DD, k["skid_x1"]], 0, off=0.3)
    dims_below(vp, s, [k["skid_x0"], k["skid_x1"]], 0, off=0.62)
    dims_above(vp, s, [0, k["x_bf"], D - k["st_d"], D], W, off=0.3)
    dims_above(vp, s, [rx0, 0, D + DD, rx1], ry1, off=0.62)
    dims_left(vp, s, [0, W], 0, off=0.3)
    dims_right(vp, s, [ry0, 0, W, ry1], rx1, off=0.3)
    # section markers outside the dims
    x0, y0, x1, y1 = vp.paper_box()
    ua = vp.ext[0] - 0.95 / vp.scale
    ub = vp.ext[2] + 0.95 / vp.scale
    sec_marker(s, vp, (ua, k["pipe_cy"]), (ub, k["pipe_cy"]), "A", "A-301", (0, 1))
    va_ = vp.ext[1] - 0.95 / vp.scale
    vb_ = vp.ext[3] + 0.8 / vp.scale
    sec_marker(s, vp, (k["pipe_cx"], va_), (k["pipe_cx"], vb_), "B", "A-302", (-1, 0))
    sec_marker(s, vp, (D + DD * 0.55, va_), (D + DD * 0.55, vb_), "C", "A-302", (-1, 0))
    # keynotes
    zc = k["FF"] + 60
    wo = k["window"]
    kn(vp, ((wo["u0"] + wo["u1"]) / 2, W + 1.5, zc), "W1 window: 18\" x 27\" double-hung, sill at "
       f"{fmt_ftin(wo['v0'] - k['FF'])} above floor (see schedule)", (0.25, -0.4))
    kn(vp, (D - 1, (du0 + du1) / 2, zc), door_text(m) + (" In-swing; the leaf folds against the side wall."
                                                        if k.get("door_swing") == "in" else ""), (0.45, 0.35))
    kn(vp, (k["pipe_cx"] - 12, k["pipe_cy"] - 11, 0), "30\" HDPE pit below the bench (hidden)", (-0.5, 0.35))
    if k.get("vent"):
        vx, vy, vr_, _ = k["vent"]
        kn(vp, (vx, vy + vr_, zc), "4\" PVC vent from flange up through roof", (-0.4, -0.45))
    kn(vp, (k["seat"][0] + 5, k["seat"][1] + 4, k["z_bt"]), f"Toilet seat on 3/4\" plywood bench, top at "
       f"{fmt_ftin(k['z_bt'] - k['FF'])} above floor", (0.35, -0.55))
    kn(vp, (-1.5, W * 0.62, zc), "Removable back panel below bench (see A-202, A-501)", (-0.45, -0.3))
    kn(vp, (D * 0.3, -0.5, zc), "2x4 studs @ 24\" o.c.; 1x10 rough-sawn boards + 1x3 battens", (0.1, 0.45))
    kn(vp, (D + DD - 1.75, 1.75, zc), "4x4 PT post on skid; knee brace to beam above", (0.35, 0.45))
    kn(vp, (D + DD * 0.72, W * 0.4, k["deck_top"]), "5/4x6 PT decking, 3/16\" gaps, notched at posts", (0.35, 0.55))
    kn(vp, (k["skid_x1"] - 3, 1.75, 5), "4x6 PT skid, chamfered ends with tow hole", (0.35, 0.45))
    kn(vp, (rx1, ry1 - 10, 0), "Line of roof overhang above", (0.35, -0.2))
    x0_, y0_, x1_, y1_ = vp.paper_box()
    S.view_title(s, x0_ + 1.2, y1_ + 1.28, 1, "Floor plan", S.scale_label(vp.scale))
    # bottom band: notes + schedule + keynotes
    by = sh.y1 - 1.3
    kn.legend(x, by, w * 0.56, 1.3, cols=3, size=0.066)
    cx = x + w * 0.58
    do = k["door"]
    rows = [["D1", "Door", f"{fmt_in(do['u1'] - do['u0'])} x {fmt_in(do['v1'] - do['v0'])}",
             door_text(m)[4:] + f"; swing {k.get('door_swing', 'out')}, hinge "
             f"{m.cfg['door'].get('hinge_side', 'right')} (from outside)"],
            ["W1", "Window", f"{fmt_in(wo['u1'] - wo['u0'])} x {fmt_in(wo['v1'] - wo['v0'])}",
             f"{m.cfg['window']['description']}; sill +{fmt_ftin(wo['v0'] - k['FF'])} AFF"]]
    table(s, cx, by, [("Tag", 0.35, "start"), ("Type", 0.55, "start"), ("R.O.", 0.95, "start"),
                      ("Remarks", w * 0.42 - 1.85, "start")], rows, title="Door & window schedule", size=0.064)
    return sh.done()


def sheet_roof(ctx):
    sh = Sheet(ctx, "A-102", "Roof plan & roof framing plan")
    s, m, k = sh.svg, ctx["model"], ctx["model"].key
    kn = Keynotes(s)
    x, y, w, h = sh.area
    W, D, DD = k["W"], k["D"], k["DD"]
    vr = View(m, "top", include=lambda p: "ground" not in p.tags)
    vf = View(m, "top", include=lambda p: p.group == "roof" and "roofing" not in p.tags or p.name == "Post")
    colw = w * 0.66
    hh = (h - 1.9) / 2
    sc = row_layout(ctx, [vr], x, y + 0.45, colw, hh, [(0.6, 1.1)])
    vr.draw(s)
    s.path(vr._d([(0, 0), (D, 0), (D, W), (0, W)]), lw="light", dash=(0.05, 0.03))
    rx0, rx1 = k["roof_x"]
    ry0, ry1 = k["roof_y"]
    dims_below(vr, s, [rx0, 0, D + DD, rx1], ry0, off=0.3)
    dims_right(vr, s, [ry0, 0, W, ry1], rx1, off=0.3)
    ax_, ay_ = vr.P((D + DD) * 0.62, W * 0.7)
    bx_, by_ = vr.P((D + DD) * 0.62, W * 0.15)
    arrow(s, ax_, ay_, bx_, by_)
    s.text(ax_ + 0.08, (ay_ + by_) / 2, f"{m.cfg['roof']['pitch']}:12", 0.1, weight="bold")
    s.text(ax_ + 0.08, (ay_ + by_) / 2 + 0.14, "DOWN", 0.075)
    zt = k["z_roof"](W * 0.5)
    kn(vr, ((D + DD) * 0.3, W * 0.45, zt), roof_text(m), (-0.2, 0.75))
    if k.get("vent"):
        vx, vy, vr_, _ = k["vent"]
        kn(vr, (vx, vy, zt), "4\" vent through EPDM roof boot, screened cap 18\" above roof", (-0.45, -0.5))
    kn(vr, (rx1 - 2, ry1 - 1, zt), "High edge: outside foam closure + peak (shed) flashing", (0.4, -0.35))
    kn(vr, (rx1 - 2, ry0 + 1, zt), "Low eave: sheets overhang fascia 1 1/2\"; inside foam closure", (0.4, 0.3))
    kn(vr, (rx0 + 0.5, W * 0.5, zt), "Rake: sheets overhang 1x6 barge board 1\"; optional rake trim", (-0.4, 0.4))
    title_under(s, vr, 1, "Roof plan", below=0.7)
    y2 = y + hh + 1.4
    row_layout(ctx, [vf], x, y2, colw, hh, [(0.6, 1.1)], max_scale=sc)
    vf.draw(s)
    raf = k["rafter_x"]
    dims_below(vf, s, [raf[0]] + [r + 0.75 for r in raf[1:-1]] + [raf[-1] + 1.5], k["y_lo"], off=0.3)
    pur = [p[0] for p in k["purlins"]]
    dims_right(vf, s, [k["y_lo"]] + (pur[1:] if pur else [0, W]) + [k["y_hi"]], rx1, off=0.3)
    zr = k["z_Lb"]
    kn(vf, (raf[2] + 0.75, W * 0.62, zr), "2x6 rafters @ 24\" o.c. max, birdsmouth on both beams, plumb-cut tails "
       "(see A-501)", (0.3, -0.4))
    if pur:
        kn(vf, (D + DD * 0.45, pur[1] + 1.5, zr), "2x4 purlins laid flat @ 24\" max along the slope, (2) #10 x 3\" "
           "screws per rafter", (0.35, 0.45))
    else:
        kn(vf, (-k["rake"][0] + 0.75, W * 0.62, zr), "2x6 fly rafter at each rake, hung from the sheathing and fascia",
           (-0.35, -0.4))
    kn(vf, (D * 0.5, 1.75, zr), "4x6 beam, low side, continuous from back wall to deck post", (0.2, 0.5))
    kn(vf, (D * 0.5, W - 1.75, zr), "4x6 beam, high side, continuous from back wall to deck post", (0.2, -0.45))
    kn(vf, (rx0 + 0.5, W * 0.4, zr), "1x6 rough-sawn barge board", (-0.4, 0.3))
    kn(vf, (D * 0.25, k["y_lo"] - 0.5, zr), "1x6 rough-sawn fascia on rafter tails", (-0.2, 0.4))
    kn(vf, (raf[1] + 8, W - 0.75, zr), "2x6 frieze blocks between rafters over building walls", (0.3, -0.4))
    title_under(s, vf, 2, "Roof framing plan", below=0.7)
    cx = x + colw + 0.2
    cw = sh.x1 - cx
    yy = notes_block(s, cx, y + 0.3, cw, "Roof notes", [
        "Single-slope roof drains to the low (left) side, away from the door and the pit.",
        f"Snow: pg = {m.cfg['loads']['ground_snow_psf']} psf, pf = {k['snow_pf']:.1f} psf (ASCE 7, unheated, Risk Cat. I).",
        "Every rafter: hurricane tie to beam. The whole roof is part of the towed assembly.",
        "Keep 10' from dead limbs; a metal roof tolerates small branch and cone falls.",
    ])
    kn.legend(cx, yy + 0.3, cw, sh.y1 - yy - 0.3, cols=1, size=0.07)
    return sh.done()


def sheet_foundation(ctx):
    sh = Sheet(ctx, "S-101", "Foundation & floor framing plan")
    s, m, k = sh.svg, ctx["model"], ctx["model"].key
    kn = Keynotes(s)
    x, y, w, h = sh.area
    W, D, DD = k["W"], k["D"], k["DD"]
    skip = {"Floor sheathing", "Floor sheathing strip", "Deck board", "Flange plate", "Flange collar", "EPDM skirt",
            "Vent pipe", "Vent cap", "Roof boot", "Knee brace"}
    fnd = View(m, "top", include=lambda p: (p.group in ("foundation", "floor", "deck", "pit") and p.name not in skip)
               or p.name == "Post")
    row_layout(ctx, [fnd], x, y + 0.9, w, h - 4.2, [(1.0, 1.0)])
    fnd.draw(s)
    cl, cr = k["chamfer"]
    for y0, y1 in k["skid_ys"]:
        for xe in (k["skid_x0"] + cl, k["skid_x1"] - cl):
            a, b = fnd.P(xe, y0), fnd.P(xe, y1)
            s.line(*a, *b, lw="light", dash=(0.04, 0.025))
    jx = k["floor_joists"]
    dims_below(fnd, s, [k["skid_x0"], 0, jx[0]] + [j + 0.75 for j in jx[2:-1]] + [D, D + DD, k["skid_x1"]], 0, off=0.3)
    dims_below(fnd, s, [k["skid_x0"], k["skid_x1"]], 0, off=0.62)
    dims_above(fnd, s, [0, k["pipe_cx"], D] + [j + 0.75 for j in k["deck_joists"][1:-1]] + [D + DD], W, off=0.3)
    (a0, a1), (b0, b1) = k["skid_ys"]
    dims_left(fnd, s, [0, a1, k["pipe_cy"] - k["OD"] / 2, k["pipe_cy"] + k["OD"] / 2, b0, W], k["skid_x0"], off=0.3)
    pcx, pcy, OD = k["pipe_cx"], k["pipe_cy"], k["OD"]
    kn(fnd, (pcx + 12, pcy - 12, 20), f"30\" dual-wall HDPE pipe ({fmt_in(OD)} OD) set before the privy arrives; "
       f"passes between the skids with {fmt_in(((b0 - a1) - OD) / 2)} clear each side", (0.45, 0.55))
    kn(fnd, (D * 0.8, a0 + 1.75, 5), "4x6 PT UC4B skid on edge, full length of privy + deck, chamfered both ends, "
       "tow hole each end (A-501)", (0.3, 0.55))
    kn(fnd, (k["x_bj"] + 1.5, W * 0.75, 8), "(2) 2x6 PT bench-front joist; no floor framing behind it (pipe path)",
       (0.25, -0.65))
    kn(fnd, (D * 0.85, W - 0.75, 8), "(2) 2x6 PT rim joists on the skid, screwed down with 1/4\" x 6\" structural "
       "screws @ 16\" o.c.", (0.25, -0.5))
    kn(fnd, ((jx[-2] + jx[-1]) / 2, W * 0.3, 8), "2x6 PT floor joists @ 16\" max, joist hangers; 3/4\" PT plywood floor",
       (0.3, 0.6))
    kn(fnd, (D + DD * 0.45, W * 0.5, 8), "2x4 PT deck joists @ 16\" o.c. between (2) 2x4 PT rims, hangers", (0.35, -0.7))
    kn(fnd, (D + DD - 1.75, W - 1.75, 8), "4x4 PT post on skid, galv. post base, (2) 1/2\" bolts to rim", (0.4, -0.4))
    title_under(s, fnd, 1, "Foundation & floor framing plan", below=1.0)
    by = sh.y1 - 2.2
    kn.legend(x, by, w * 0.55, 2.2, cols=2, size=0.07)
    notes_block(s, x + w * 0.58, by, w * 0.42, "Foundation notes", [
        "No foundation: the privy and deck ride on two continuous skids (a sled) on a level dirt pad.",
        "Grade the skid paths flat and compact them; shim with PT blocks or flat stones, never with the pipe.",
        "Red clay (NC Piedmont) holds water: crown the pad slightly so it drains away from the pit and skids.",
        "Tow only from both skid ends at once (bridle), with the back panel installed and lagged.",
    ], size=0.07)
    return sh.done()


def _elev_keynotes(kn, v, m, side):
    k = m.key
    W, D, DD = k["W"], k["D"], k["DD"]
    tb, tbat = k["tb"], k["tbat"]
    roof_note = roof_text(m) + f"; {m.cfg['roof']['pitch']}:12"
    board = "1x10 rough-sawn boards, 1/2\" gaps, 1x3 battens (board-and-batten)"
    skid = "4x6 PT skid on edge, chamfered ends"
    if side == "left":
        kn(v, (D * 0.45, 12, k["z_roof"](12)), roof_note, (-0.3, -0.45))
        kn(v, (D * 0.55, -tb - tbat, k["FF"] + 40), board, (0.45, -0.3))
        kn(v, (D + DD * 0.35, 0, 2.5), skid, (0.3, 0.45))
        kn(v, (D + DD - 1.75, 0, 50), "4x4 PT post with 4x4 knee brace to beam", (-0.5, 0.2))
        kn(v, (D + DD * 0.45, 0, k["z_Lb"] - 2.75), "4x6 beam, continuous over deck", (0.1, 0.55))
        kn(v, (k["pipe_cx"], k["pipe_cy"], -18), "30\" HDPE pit, 3'-0\" below grade (hidden)", (0.55, 0.1))
        kn(v, (D * 0.8, k["y_lo"] - 0.5, k["z_rt"](k["y_lo"]) - 2.5), "1x6 rough-sawn fascia on plumb-cut rafter tails",
           (0.35, 0.35))
        if k.get("vent"):
            vx, vy, vr_, vtop = k["vent"]
            kn(v, (vx, vy - vr_, vtop - 6), "4\" PVC vent, screened cap", (0.35, -0.1))
    elif side == "right":
        wo = k["window"]
        kn(v, ((wo["u0"] + wo["u1"]) / 2, W + tb + 1, wo["v1"] + 2), "W1 double-hung window, 1x4 rough-sawn casing, "
           "sill and apron", (0.45, -0.35))
        kn(v, (D * 0.2, W + tb + tbat, k["FF"] + 30), board, (-0.4, 0.3))
        kn(v, (D * 0.85, W + tb, k["side_bot"] + 0.5), f"Siding stops {fmt_in(k['side_bot'])} above grade "
           "(IRC R317.1 decay / termite clearance); PT rim exposed below", (0.3, 0.45))
        kn(v, (D + DD * 0.4, W, k["z_Hb"] - 2.75), "4x6 beam, continuous over deck", (0.1, 0.55))
        kn(v, (D + DD - 1.75, W, 50), "4x4 PT post with 4x4 knee brace to beam", (0.45, 0.2))
        kn(v, (D + DD * 0.5, W, 2.5), skid, (0.3, 0.45))
    elif side == "front":
        du0, du1, dv0, dv1, _ = k["door_leaf"]
        kn(v, (D - 1.5, (du0 + du1) / 2 + 5, dv0 + 28), door_text(m), (0.55, 0.2))
        kn(v, (D + DD, W * 0.35, k["deck_top"] - 0.5), "5/4x6 PT deck on 2x4 PT joists", (0.35, 0.45))
        kn(v, (D + tb + 1, k["door"]["u1"] + 1.75, dv1 - 20), "1x4 rough-sawn door casing", (0.4, 0.1))
        kn(v, (D + tb, W - 3, k["FF"] + 60), board, (0.45, -0.2))
        kn(v, (k["skid_x1"], 1.75, 3), skid, (-0.2, 0.45))
    elif side == "back":
        p0, p1 = k["panel"]
        kn(v, (-tb - tbat, W * 0.55, (p0 + p1) / 2 + 3), "Removable panel: boards on 1x4 rails, (10) 1/4\" lags; remove "
           "to slide the privy over the pit, then re-install", (-0.55, 0.35))
        kn(v, (-tb - 0.5, W * 0.85, p1), "Galvanized Z-flashing at panel head", (0.45, -0.25))
        kn(v, (-tb - 2, W * 0.28, k["panel_rails"][0][1] - 1.9), "Pull handles", (-0.4, 0.4))
        kn(v, (-tb - tbat, W * 0.3, k["FF"] + 50), board, (-0.45, -0.2))
        kn(v, (k["skid_x0"], 1.75, 2.5), skid, (-0.2, 0.45))


def sheet_elevations(ctx, number, sides, title):
    sh = Sheet(ctx, number, title)
    s, m, k = sh.svg, ctx["model"], ctx["model"].key
    kn = Keynotes(s)
    x, y, w, h = sh.area
    views = [elevation_view(m, sd) for sd in sides]
    names = {"left": "Left (low / eave side) elevation", "right": "Right (high / window side) elevation",
             "front": "Front (door / deck) elevation", "back": "Back (removable panel) elevation"}
    legend_h = 1.15
    row_layout(ctx, views, x, y + 0.3, w, h - legend_h - 1.3, [(0.55, 0.5), (0.6, 1.3)])
    for i, (v, sd) in enumerate(zip(views, sides)):
        v.draw(s)
        v.draw_ground(s)
        v.draw_hidden(s)
        if i == len(views) - 1:
            std_levels(v, s, m, ["grade", "ff", "low", "high", "roof", "pit"])
        if sd in ("left", "right"):
            us = [v.U((xx, 0, 0))[0] for xx in (k["skid_x0"], 0, k["D"], k["D"] + k["DD"], k["skid_x1"])]
            top = k["z_Lb"] if sd == "left" else k["z_Hb"]
            dims_left(v, s, [0, k["FF"], top], v.ext[0], off=0.25)
        else:
            us = [v.U((0, yy, 0))[0] for yy in (0, k["W"])]
            dims_left(v, s, [0, k["FF"], k["z_Lb"]], v.ext[0], off=0.25)
        dims_below(v, s, us, 0, off=0.25)
        _elev_keynotes(kn, v, m, sd)
        title_under(s, v, i + 1, names[sd], below=0.72)
    kn.legend(x, sh.y1 - legend_h + 0.1, w, legend_h - 0.1, cols=3, size=0.068)
    return sh.done()


def _section_keynotes(kn, v, m, which):
    k = m.key
    W, D, DD = k["W"], k["D"], k["DD"]
    pcx, pcy = k["pipe_cx"], k["pipe_cy"]
    fz = k["flange_z"]
    if which == "A":
        c = pcy
        kn(v, (k["seat"][0] + 6, c, k["z_bt"] + 0.6), "Toilet seat with lid on 3/4\" BC plywood bench top, sealed",
           (0.45, -0.55))
        kn(v, (k["x_fp0"] - 0.75, c, k["z_bf0"] + 1.75), "2x4 bench framing: front rail, joists, side ledgers; 3/4\" "
           "plywood front panel", (0.6, -0.15))
        kn(v, (pcx + 6, c, (fz[0] + fz[1]) / 2), "Flange: 3/4\" PT plywood plate with 30\" opening + 2-ply collar; EPDM "
           "skirt with SS band clamps (A-501)", (0.75, 0.15))
        kn(v, (1.75, c, (k["header_z"][0] + k["header_z"][1]) / 2), "4x6 PT bench header over the panel opening",
           (-0.45, -0.35))
        kn(v, (pcx - 12, c, -18), "30\" dual-wall HDPE pipe, open bottom, compacted native backfill", (-0.55, 0.2))
        kn(v, (k["x_bj"] + 1.5, c, 8), "(2) 2x6 PT bench-front joist; open behind for the pipe", (0.35, 0.5))
        kn(v, ((D + k["x_bj"]) / 2 + 6, c, k["FF"] - 0.4), "2x6 PT joists @ 16\", 3/4\" PT plywood floor", (0.2, 0.55))
        kn(v, (D + DD * 0.5, c, k["deck_top"] - 0.5), "5/4x6 PT decking on 2x4 PT joists @ 16\" o.c.", (0.3, 0.5))
        kn(v, (D * 0.3, W, k["FF"] + 48), "2x4 studs @ 24\" o.c., flat blocking @ 24\", let-in strap bracing", (0.4, -0.5))
        if k.get("vent"):
            vx, vy, vr_, vtop = k["vent"]
            kn(v, (vx + vr_, c, vtop - 10), "4\" PVC vent, screened cap", (0.35, 0.1))
        kn(v, (D + DD * 0.3, c, k["z_roof"](c)), "2x6 rafters @ 24\" max; " + roof_text(m), (0.3, -0.4))
        kn(v, (k["skid_x0"] + 3, W, 2.75), "4x6 PT skid beyond", (-0.3, 0.45))
    else:
        c = pcx
        kn(v, (c, W - 1.75, k["z_Hb"] - 2.75), "4x6 beam on 2x4 top plate, birdsmouth rafter seat", (0.45, -0.25))
        kn(v, (c, 1.75, k["z_Lb"] - 2.75), "4x6 beam on 2x4 top plate, birdsmouth rafter seat", (-0.45, -0.25))
        kn(v, (c, W * 0.3, k["z_rt"](W * 0.3) + 1), "2x6 rafters @ 24\" max; " + roof_text(m), (-0.35, -0.5))
        kn(v, (c, pcy + 3, k["z_bt"] + 0.6), "Toilet seat with lid on 3/4\" BC plywood bench top, sealed", (0.5, -0.4))
        kn(v, (c, pcy + 12, (fz[0] + fz[1]) / 2), "Flange: 3/4\" PT plywood plate with 30\" opening + 2-ply collar; "
           "EPDM skirt with SS band clamps (A-501)", (0.6, 0.25))
        kn(v, (c, pcy - 12, -18), "30\" dual-wall HDPE pipe, open bottom, compacted native backfill", (-0.45, 0.25))
        kn(v, (c, 1.75, 2.75), "4x6 PT skid on edge", (-0.4, 0.3))
        kn(v, (c, 1.5, 8), "(2) 2x6 PT rim joists", (-0.45, -0.1))
        kn(v, (c, W - 1.75, k["FF"] + 40), "2x4 studs @ 24\" o.c., flat blocking @ 24\", let-in strap bracing", (0.4, 0.1))


def sheet_sections(ctx, number, which):
    title = "Section A: longitudinal through pit" if which == "A" else "Sections B & C: transverse"
    sh = Sheet(ctx, number, title)
    s, m, k = sh.svg, ctx["model"], ctx["model"].key
    kn = Keynotes(s)
    x, y, w, h = sh.area
    if which == "A":
        v = long_section(m)
        legend_h = 0
        row_layout(ctx, [v], x, y + 0.3, w * 0.7, h - 1.2, [(0.9, 1.4)])
    else:
        v = trans_section(m)
        legend_h = 0
        vc = View(m, "front", cut=(0, k["D"] + k["DD"] * 0.55, 1))
        row_layout(ctx, [v, vc], x, y + 0.3, w * 0.72, h - 1.2, [(0.8, 0.25), (0.45, 1.35)])
        vc.draw(s)
        vc.draw_ground(s)
        std_levels(vc, s, m, ["grade", "deck", "ff", "bench", "pipe", "low", "high", "roof", "pit"])
        cxs = k["D"] + k["DD"] * 0.55
        dims_below(vc, s, [vc.U((0, yy, 0))[0] for yy in (k["y_lo"], 0, k["W"], k["y_hi"])], 0, off=0.3)
        kn(vc, (cxs, k["W"] - 1.75, 60), "4x4 PT post on skid, galv. base, (2) 1/2\" bolts to deck rim", (0.35, 0.3))
        kn(vc, (cxs, k["W"] * 0.5, k["deck_top"] - 0.5), "5/4x6 PT decking on 2x4 PT joists @ 16\" between (2) 2x4 "
           "PT rims", (0.35, 0.55))
        do = k["door"]
        kn(vc, (cxs, (do["u0"] + do["u1"]) / 2, do["v1"] - 20), "D1 door beyond (see A-503)", (0.4, -0.3))
        kn(vc, (cxs, 1.75, k["z_Lb"] - 2.75), "4x6 beam on post, 4x4 knee brace (beyond)", (-0.45, -0.3))
        title_under(s, vc, 2, "Section C: through the deck", below=0.7)
    v.draw(s)
    v.draw_ground(s)
    if which == "A":
        std_levels(v, s, m, ["grade", "ff", "bench", "pipe", "low", "high", "roof", "pit"])
    D, DD, W = k["D"], k["DD"], k["W"]
    if which == "A":
        dims_below(v, s, [k["skid_x0"], 0, k["pipe_cx"] - k["OD"] / 2, k["pipe_cx"] + k["OD"] / 2, k["x_bf"], D - k["st_d"],
                          D, D + DD], 0, off=0.3)
    else:
        dims_below(v, s, [v.U((0, yy, 0))[0] for yy in (k["y_lo"], 0, W, k["y_hi"])], 0, off=0.3)
    dims_left(v, s, [k["pipe_bot"], 0, k["FF"], k["pipe_top"], k["z_bt"]], v.ext[0], off=0.3)
    _section_keynotes(kn, v, m, which)
    title_under(s, v, 1, title if which == "A" else "Section B: transverse through pit", below=0.7)
    if which == "A":
        kn.legend(x + w * 0.72, y + 0.3, w * 0.28, h - 0.5, cols=1, size=0.07)
    else:
        cx = x + w * 0.74
        yy = kn.legend(cx, y + 0.3, w * 0.26, h * 0.6, cols=1, size=0.068)
        notes_block(s, cx, yy + 0.3, w * 0.26, "Section notes", [
            f"The pipe top ({fmt_elev(k['pipe_top'])}) sits {fmt_in(k['header_z'][0] - k['pipe_top'])} below the "
            "bench header so the privy can slide over it; the flange closes the gap afterwards.",
            f"The rafters bear on 4x6 beams at {fmt_elev(k['z_Lb'])} (low) and {fmt_elev(k['z_Hb'])} (high); "
            "the front and back walls are non-bearing.",
            "Everything above the flange plate is removable with the building; the pipe stays in the ground.",
        ], size=0.072)
    return sh.done()


def framing_view(m, side):
    k = m.key
    if side in ("left", "right"):
        lowside = side == "left"

        def inc(p):
            if p.group == f"wall.{side}":
                return True
            lo, hi = p.bbox()
            on_side = hi[1] <= k["W"] / 2 if lowside else lo[1] >= k["W"] / 2
            return on_side and (p.name in ("Beam", "Post", "Knee brace", "Skid", "Rim joist", "Deck rim"))
        return View(m, side, include=inc, grade=True)
    return View(m, side, include=lambda p: p.group == f"wall.{side}" or (side == "back" and p.name == "Bench header"),
                grade=False)


def _stud_centers(m, wall, axis):
    return sorted({round(p.bbox()[0][axis] + 0.75, 3) for p in m.parts if p.group == f"wall.{wall}" and p.name in
                   ("Stud", "King stud")})


def sheet_side_framing(ctx, number, wall):
    title = f"{'Left (low)' if wall == 'left' else 'Right (high)'} side wall framing"
    sh = Sheet(ctx, number, title)
    s, m, k = sh.svg, ctx["model"], ctx["model"].key
    kn = Keynotes(s)
    x, y, w, h = sh.area
    v = framing_view(m, wall)
    row_layout(ctx, [v], x, y + 0.5, w, h - 2.9, [(0.9, 0.5)])
    v.draw(s)
    v.draw_ground(s)
    us = [v.U((xx, 0, 0))[0] for xx in _stud_centers(m, wall, 0)]
    dims_below(v, s, [v.U((0, 0, 0))[0]] + us + [v.U((k["D"], 0, 0))[0]], k["FF"], off=0.3)
    dims_below(v, s, [v.U((xx, 0, 0))[0] for xx in (0, k["D"], k["D"] + k["DD"])], k["FF"], off=0.62)
    top = k["stud_top_low"] if wall == "left" else k["stud_top_high"]
    dims_left(v, s, [0, k["FF"], k["FF"] + 1.5, top, top + 1.5, (k["z_Lb"] if wall == "left" else k["z_Hb"])],
              v.ext[0], off=0.3)
    yw = 0 if wall == "left" else k["W"]
    sgn = 1 if wall == "left" else -1
    kn(v, (k["D"] * 0.18, yw, k["FF"] + 24), "2x4 flat blocking @ 24\" o.c., flush to outside face (siding nailers)",
       (0.3 * sgn, -0.35))
    kn(v, (k["D"] * 0.5, yw, k["FF"] + 50), "Let-in steel wall-bracing strap, X in the solid bay; nail at each stud "
       "and plate", (0.35, 0.3))
    kn(v, (k["D"] * 0.9, yw, k["FF"] + 0.75), "PT 2x4 bottom plate on 3/4\" PT plywood", (0.3 * sgn, 0.5))
    kn(v, (1.5, yw, k["FF"] + 60), "Doubled corner studs at the back: bench-header hangers", (-0.4 * sgn, -0.45))
    kn(v, (k["D"] + k["DD"] - 1.75, yw, 55), "4x4 PT post on skid", (0.4 * sgn, 0.25))
    kn(v, (k["D"] + k["DD"] - 14, yw, (k["posts"][0][3] if wall == "left" else k["posts"][1][3]) - 10),
       "4x4 PT knee brace, 45 deg, 24\" legs", (-0.35 * sgn, -0.4))
    kn(v, (k["D"] + k["DD"] * 0.5, yw, (k["z_Lb"] if wall == "left" else k["z_Hb"]) - 2.75),
       "4x6 beam on top plate, continuous to the post", (0.2, -0.45))
    kn(v, (k["D"] * 0.3, yw, top + 0.75), "Single 2x4 top plate (beam above acts as second plate)", (0.3, -0.4))
    kn(v, (k["D"] * 0.6, yw, 2.75), "4x6 PT skid and (2) 2x6 PT rim joists", (0.3, 0.5))
    wo = k["window"]
    if wo["wall"] == wall:
        kn(v, ((wo["u0"] + wo["u1"]) / 2, yw, wo["v1"] + 1.75), f"(2) 2x4 header over W1, R.O. "
           f"{fmt_in(wo['u1'] - wo['u0'])} x {fmt_in(wo['v1'] - wo['v0'])}; king + jack studs; flat 2x4 rough sill",
           (0.35, -0.45))
        a, b = sorted(v.U((xx, 0, 0))[0] for xx in (wo["u0"], wo["u1"]))
        dims_right(v, s, [k["FF"], wo["v0"], wo["v1"]], b, off=0.35)
    title_under(s, v, 1, f"{'Left' if wall == 'left' else 'Right'} side frame (from outside)", below=1.0)
    by = sh.y1 - 1.55
    kn.legend(x, by, w * 0.6, 1.55, cols=2, size=0.07)
    notes_block(s, x + w * 0.63, by, w * 0.37, "Framing notes", [
        "2x4 SPF #2 studs @ 24\" o.c.; the side walls carry the roof via one top plate plus the 4x6 beam.",
        "Toe-screw and strap the beam to each corner stud pack and to the post.",
        "Studs differ in length between the low and high walls; cut from 92-5/8\" precuts where possible.",
    ], size=0.07)
    return sh.done()


def sheet_end_framing(ctx):
    sh = Sheet(ctx, "S-203", "End wall framing elevations")
    s, m, k = sh.svg, ctx["model"], ctx["model"].key
    kn = Keynotes(s)
    x, y, w, h = sh.area
    vf, vb = framing_view(m, "front"), framing_view(m, "back")
    row_layout(ctx, [vf, vb], x, y + 0.6, w * 0.62, h - 1.6, [(0.7, 0.3), (0.7, 0.3)], valign="bottom")
    for i, (v, wall) in enumerate(((vf, "front"), (vb, "back"))):
        v.draw(s)
        us = [v.U((0, yy, 0))[0] for yy in _stud_centers(m, wall, 1)]
        base = k["FF"] if wall == "front" else k["z_bt"]
        W = k["W"]
        dims_below(v, s, us, base, off=0.25)
        xw = k["D"] if wall == "front" else 0.0
        if wall == "front":
            do = k["door"]
            dims_left(v, s, [k["FF"], do["v1"], do["header_top"]], v.ext[0], off=0.3)
            kn(v, (xw, (do["u0"] + do["u1"]) / 2, do["v1"] + 2.75), f"(2) 2x6 header over D1, R.O. "
               f"{fmt_in(do['u1'] - do['u0'])} x {fmt_in(do['v1'] - do['v0'])}; king + jack studs", (0.35, -0.5))
            kn(v, (xw, do["u0"] - 0.75, k["FF"] + 40), "Bottom plate cut out at door after framing", (-0.4, 0.4))
        else:
            hz = k["header_z"]
            dims_left(v, s, [hz[0], hz[1], k["z_bt"], k["z_bt"] + 1.5], v.ext[0], off=0.3)
            kn(v, (xw, W * 0.5, (hz[0] + hz[1]) / 2), "4x6 PT bench header, full interior width, face-mount hangers "
               "to doubled corner studs; removable panel opening below", (0.4, 0.45))
            kn(v, (xw, W * 0.3, k["z_bt"] + 0.75), "2x4 PT bottom plate on bench top over header", (-0.45, 0.3))
        kn(v, (xw, W * 0.6, k["z_rb"](W * 0.6) - 1), "Sloped 2x4 top plate under end rafter, 3:12", (0.3, -0.4))
        kn(v, (xw, W * 0.2, base + 30), "2x4 flat blocking @ 24\" (siding nailers)", (-0.35, -0.3))
        title_under(s, v, i + 1, f"{'Front' if wall == 'front' else 'Back'} wall framing (from outside)", below=0.7)
    cx = x + w * 0.64
    yy = kn.legend(cx, y + 0.3, w * 0.36, h * 0.55, cols=1, size=0.072)
    notes_block(s, cx, yy + 0.3, w * 0.36, "End wall notes", [
        "End walls are non-bearing: the rafters span side to side between the 4x6 beams.",
        "The back wall has no bottom plate below the bench header. With the panel off, the bench box and the header "
        "brace the back; keep the panel lagged whenever the privy is moved.",
        "The front wall is too narrow for strap bracing beside the door; the roof diaphragm and the floor carry racking.",
    ], size=0.072)
    return sh.done()


def sheet_details(ctx):
    sh = Sheet(ctx, "A-501", "Details: pit & flange, skid, rafter")
    s, m, k = sh.svg, ctx["model"], ctx["model"].key
    kn = Keynotes(s)
    x, y, w, h = sh.area
    W = k["W"]
    pcy, OD, ID = k["pipe_cy"], k["OD"], k["ID"]
    pt, hz, fz = k["pipe_top"], k["header_z"], k["flange_z"]
    # 1. pit & flange, enlarged: upper part of the pit only
    crop = (-2.5, -14, W + 2.5, k["z_bt"] + 4)
    v1 = View(m, "front", cut=(0, k["pipe_cx"], 1), crop=crop)
    row_layout(ctx, [v1], x, y + 0.3, w * 0.5, h * 0.62, [(1.3, 0.3)])
    v1.draw(s)
    v1.draw_ground(s)
    dims_left(v1, s, [0, pt, k["collar_z0"], fz[0], fz[1], k["z_bt"]], v1.ext[0] + 0.5, off=0.3)
    dims_below(v1, s, [v1.U((0, pcy - OD / 2, 0))[0], v1.U((0, pcy + OD / 2, 0))[0]], -14, off=0.2)
    c = pcx = k["pipe_cx"]
    kn(v1, (c, pcy + 3, k["z_bt"] + 0.6), "Toilet seat, stainless hinge bolts through bench top", (0.5, -0.35))
    kn(v1, (c, pcy - 14, k["z_bt"] - 0.3), "3/4\" BC plywood bench top, sealed; hole cut to the seat template", (-0.4, -0.45))
    kn(v1, (c, W - 4.25, k["z_bf0"] + 1.75), "2x4 bench side ledger screwed to studs", (0.55, -0.3))
    kn(v1, (c, pcy + ID / 2 + 1, (fz[0] + fz[1]) / 2), "Flange plate: 3/4\" PT plywood with 30\" hole; slid in from the "
       "back after the privy is placed; (12) #10 x 3\" SS screws up into bench framing", (0.6, 0.1))
    kn(v1, (c, pcy - OD / 2 + 0.5, (k["collar_z0"] + fz[0]) / 2), "Collar: 2 plies 3/4\" PT plywood rings (half rings "
       "glued & screwed), OD = pipe OD", (-0.55, 0.05))
    kn(v1, (c, pcy + OD / 2 + 0.05, pt), "EPDM skirt over the gap, (2) stainless band clamps; foam gasket under "
       "collar", (0.55, 0.35))
    kn(v1, (c, pcy - OD / 2 + 1, -6), "30\" dual-wall HDPE pipe, cut square, plumb", (-0.5, 0.3))
    kn(v1, (c, pcy + OD / 2 + 4, -8), "Backfill: native soil in 6\" lifts, hand tamped; no concrete", (0.5, 0.4))
    kn(v1, (c, 1.75, 2.75), "4x6 PT skid, clear of pipe", (-0.4, 0.45))
    title_under(s, v1, 1, "Pit & flange detail (section B)", below=0.5)
    # 2. skid end
    cx = x + w * 0.55
    cw = x + w - cx
    x0s = k["skid_x0"]
    v2 = View(m, "left", include=lambda p: p.name == "Skid" and p.bbox()[0][1] < 1,
              crop=(x0s - 2, -1.5, x0s + 22, k["skid_h"] + 2))
    row_layout(ctx, [v2], cx, y + 0.35, cw - 0.2, 1.5, [(0.55, 0.3)])
    v2.draw(s)
    v2.draw_ground(s)
    cl, cr = k["chamfer"]
    fs = m.cfg["foundation"]["skids"]
    hx, hz_ = v2.P(x0s + L(fs["tow_hole_from_end"]), k["skid_h"] / 2)
    s.circle(hx, hz_, L(fs["tow_hole_diameter"]) / 2 * v2.scale, lw="med")
    dims_below(v2, s, [x0s, x0s + cl], 0, off=0.2)
    dims_left(v2, s, [0, cr, k["skid_h"]], x0s, off=0.25)
    kn(v2, (x0s + L(fs["tow_hole_from_end"]), 0, k["skid_h"] / 2), f"{fmt_in(L(fs['tow_hole_diameter']))} tow hole "
       f"{fmt_in(L(fs['tow_hole_from_end']))} from end, for a 3/4\" shackle", (0.55, -0.35))
    kn(v2, (x0s + cl * 0.5, 0, cr * 0.5), f"Chamfer {fmt_in(cl)} x {fmt_in(cr)} on the bottom edge, both ends, so the "
       "sled rides up over roots and ruts", (0.9, 0.3))
    title_under(s, v2, 2, "Skid end (both ends)", below=0.45)
    # 3. rafter
    rid = [p for p in m.parts if p.name == "Rafter"][0]
    v3 = View(m, "front", include=lambda p: p is rid or p.name in ("Beam", "Purlin", "Fascia", "Roof sheathing")
              or "roofing" in p.tags,
              cut=(0, rid.bbox()[0][0] + 0.75, 1), grade=False,
              crop=(k["y_lo"] - 3, k["z_Lb"] - 7, k["y_hi"] + 3, k["z_roof"](k["y_hi"]) + 3))
    row_layout(ctx, [v3], cx, y + 2.6, cw, 2.6, [(0.35, 0.35)])
    v3.draw(s)
    ys = [v3.U((0, yy, 0))[0] for yy in (k["y_lo"], 0, k["beam_w"], k["W"] - k["beam_w"], k["W"], k["y_hi"])]
    dims_below(v3, s, ys, k["z_Lb"] - 5.5, off=0.25)
    bm = k["beam_w"] * k["slope"]
    kn(v3, (0, 1.75, k["z_Lb"]), f"Birdsmouth: {fmt_in(k['beam_w'])} seat, {fmt_in(bm)} heel; hurricane tie each rafter",
       (-0.35, -0.55))
    kn(v3, (0, k["W"] - 1.75, k["z_Hb"]), "Birdsmouth at high beam, heel cut on the inside face", (0.3, 0.45))
    kn(v3, (0, k["y_lo"] - 0.5, k["z_rt"](k["y_lo"]) - 2.5), "Plumb-cut tail, 1x6 rough-sawn fascia", (-0.3, 0.45))
    kn(v3, (0, k["W"] * 0.55, k["z_roof"](k["W"] * 0.55)), roof_text(m) + f"; {m.cfg['roof']['pitch']}:12",
       (0.2, -0.4))
    title_under(s, v3, 3, f"Rafter layout (typical) - {fmt_ftin(k['raf_len'])} 2x6", below=0.45)
    kn.legend(cx, y + 5.95, cw, h - 5.95, cols=1, size=0.07)
    return sh.done()


def sheet_panel_door(ctx):
    sh = Sheet(ctx, "A-503", "Details: removable panel, door, bench")
    s, m, k = sh.svg, ctx["model"], ctx["model"].key
    kn = Keynotes(s)
    x, y, w, h = sh.area
    W = k["W"]
    v4 = View(m, "back", include=lambda p: p.group == "removable_panel" or p.name in ("Skid", "Bench header", "Z-flashing"),
              crop=(-W - 4, -1.5, 4, k["panel"][1] + 4))
    du0, du1, dv0, dv1, _ = k["door_leaf"]
    do = k["door"]
    if k.get("door_prehung"):
        v5 = View(m, "front", include=lambda p: p.group == "door" or (p.name in ("Head casing", "Side casing") and
                  p.bbox()[0][0] > k["D"] - 1), grade=False, crop=(do["u0"] - 6, k["FF"] - 2, do["u1"] + 6, do["v1"] + 6))
    else:
        v5 = View(m, "back", include=lambda p: p.group == "door", grade=False, crop=(-du1 - 2, dv0 - 2, -du0 + 2, dv1 + 2))
    row_layout(ctx, [v4], x, y + 0.4, w * 0.55, 3.2, [(0.6, 0.6)])
    v4.draw(s)
    v4.draw_ground(s)
    r0, r1 = k["panel_rails"]
    for yy in (1.75, W / 2, W - 1.75):
        for zz in ((r0[0] + r0[1]) / 2, (r1[0] + r1[1]) / 2):
            px, py = v4.Pm((-2, yy, zz))
            s.circle(px, py, 0.025, lw="light")
    dims_below(v4, s, [v4.U((0, yy, 0))[0] for yy in (-k["tb"], W + k["tb"])], 0, off=0.25)
    dims_left(v4, s, [0, k["panel"][0], k["skid_h"], k["panel"][1]], v4.ext[0] + 0.5, off=0.3)
    kn(v4, (-2.5, W - 1.75, (r1[0] + r1[1]) / 2), "1/4\" x 3\" galv. lag + fender washer: 3 per rail into corner studs "
       "and header, plus 2 into the rims (10 total)", (0.45, -0.3))
    kn(v4, (-3.5, W * 0.28, r0[1] - 1.9), "Galvanized pull handles", (-0.4, 0.4))
    kn(v4, (-1, W * 0.62, (r0[1] + r1[0]) / 2), "1x10 rough-sawn boards and 1x3 battens to match the walls", (0.4, 0.3))
    kn(v4, (-1, W * 0.5, 2.5), "PT 2x6 kick board between the skids (critter guard, part of the panel); untreated "
       "boards stop 6\" above grade", (0.4, 0.45))
    kn(v4, (-1.5, W * 0.85, k["panel"][1]), "Z-flashing over the panel head", (0.4, -0.35))
    title_under(s, v4, 1, "Removable back panel (from outside)", below=0.5)
    row_layout(ctx, [v5], x + w * 0.56, y + 0.4, w * 0.24, h - 1.2, [(0.5, 0.4)])
    v5.draw(s)
    if k.get("door_prehung"):
        dims_below(v5, s, [do["u0"], do["u1"]], k["FF"], off=0.25)
        dims_left(v5, s, [k["FF"], do["v1"]], do["u0"] - 6, off=0.3)
        kn(v5, (k["D"], (du0 + du1) / 2, (dv0 + dv1) / 2 + 10), door_text(m), (0.45, -0.3))
        kn(v5, (k["D"] + 1.5, do["u0"] - 1.75, k["FF"] + 50), "Rough-sawn 1x4 casing over the jamb; flash the head with "
           "Z-flashing and tape the jamb legs", (-0.4, 0.3))
        kn(v5, (k["D"] + 0.5, (du0 + du1) / 2, k["FF"] + 0.5), "Adjustable sill on sill-pan flashing; 1 3/4\" step down "
           "to the deck", (0.4, 0.35))
    else:
        dims_below(v5, s, [-du1, -du0], dv0, off=0.25)
        dims_left(v5, s, [dv0, dv1], -du1, off=0.3)
        kn(v5, (0, (du0 + du1) / 2, (dv0 + dv1) / 2), "1x6 rough-sawn Z-brace, diagonal runs down to the hinge side",
           (0.45, 0.3))
    title_under(s, v5, 2, "Door D1 (exterior)" if k.get("door_prehung") else "Door D1 (inside face)", below=0.5)
    # bench plan detail
    vb = View(m, "top", include=lambda p: p.group == "bench" or p.name in ("Vent pipe",), grade=False,
              crop=(-1, k["st_d"] - 1, k["x_bf"] + 2, W - k["st_d"] + 1))
    row_layout(ctx, [vb], x, y + 4.8, w * 0.55, 2.6, [(0.8, 0.6)])
    vb.draw(s)
    hx, hy, hrx, hry = k["hole"]
    px, py = vb.P(hx, hy)
    s.path(f"M{S.f(px - hrx * vb.scale)} {S.f(py)} a{S.f(hrx * vb.scale)} {S.f(hry * vb.scale)} 0 1 0 "
           f"{S.f(2 * hrx * vb.scale)} 0 a{S.f(hrx * vb.scale)} {S.f(hry * vb.scale)} 0 1 0 {S.f(-2 * hrx * vb.scale)} 0",
           lw="light", dash=(0.04, 0.02))
    dims_below(vb, s, [0, hx, k["x_bf"]], k["st_d"], off=0.25)
    dims_left(vb, s, [k["st_d"], hy, W - k["st_d"]], 0, off=0.25)
    kn(vb, (hx, hy + hry, k["z_bt"]), f"Seat hole {fmt_in(2 * hrx)} x {fmt_in(2 * hry)}, centered "
       f"{fmt_in(k['x_bf'] - hx)} back from the bench front, inside the pipe ID", (0.45, -0.35))
    kn(vb, (k["x_bf"] - 0.4, W * 0.3, k["z_bt"]), "Bench front 3/4\" plywood panel on 2x4 front rail", (0.4, 0.4))
    title_under(s, vb, 3, "Bench top (seat lid closed)", below=0.5)
    kn.legend(x + w * 0.8, y + 0.3, w * 0.2, h - 0.5, cols=1, size=0.07)
    return sh.done()


class _Staged:
    """The model with the building translated by dx (pipe and ground stay put)."""
    FIXED = ("HDPE pipe",)

    def __init__(self, m, dx, include):
        import copy
        from ..geometry import Translated
        self.cfg, self.key, self.stock = m.cfg, m.key, m.stock
        self.parts = []
        for p in m.parts:
            if not include(p):
                continue
            if "ground" in p.tags or p.name in self.FIXED or abs(dx) < 1e-9:
                self.parts.append(p)
            else:
                q = copy.copy(p)
                q.geom = Translated(p.geom, (dx, 0.0, 0.0))
                self.parts.append(q)


def sheet_install(ctx):
    sh = Sheet(ctx, "A-502", "Installation & relocation sequence")
    s, m, k = sh.svg, ctx["model"], ctx["model"].key
    x, y, w, h = sh.area
    W, D, DD = k["W"], k["D"], k["DD"]
    pcx, pcy, OD = k["pipe_cx"], k["pipe_cy"], k["OD"]
    gap = ((k["skid_ys"][1][0] - k["skid_ys"][0][1]) - OD) / 2
    stage = OD + 30
    later = {"Flange plate", "Flange collar", "EPDM skirt", "Vent pipe", "Vent cap", "Roof boot"}
    no_bldg = lambda p: "ground" in p.tags or p.name == "HDPE pipe"                     # noqa: E731
    hole_only = lambda p: "ground" in p.tags                                              # noqa: E731
    open_back = lambda p: p.name not in later and "removable_panel" not in p.tags         # noqa: E731
    closed = lambda p: True                                                               # noqa: E731
    steps = [
        ("Lay out & excavate", f"Clear and level the pad, crowned slightly so water runs away. Dig or auger a "
         f"{fmt_in(OD + 8)} hole {fmt_ftin(-k['pipe_bot'])} deep; keep the bottom undisturbed.", 0.0, hole_only),
        ("Set the pipe", f"Cut the pipe to {fmt_ftin(k['pipe_top'] - k['pipe_bot'])}. Set it plumb, top "
         f"{fmt_in(k['pipe_top'])} above the skid path. Backfill in 6\" lifts and tamp.", 0.0, no_bldg),
        ("Stage the privy", f"Grade two skid paths level with the pad. Set the privy in line, back wall toward the pipe. "
         "Remove the back panel (10 lags) and the flange.", stage + OD / 2 + 6, open_back),
        ("Slide over the pipe", f"Pull on both skid ends with a bridle. The pipe passes between the skids "
         f"({fmt_in(gap)} each side) and under the bench header ({fmt_in(k['header_z'][0] - k['pipe_top'])} clear).",
         OD * 0.55, open_back),
        ("Stop at the mark", f"Stop with the pipe center {fmt_ftin(pcx)} from the outside of the back-wall framing "
         "(chalk a mark on each skid). Level the sled with PT shims under the skids.", 0.0, open_back),
        ("Flange, vent, panel", "Slide the flange plate + collar in from the back and screw it up to the bench framing; "
         "wrap and clamp the EPDM skirt. Fit the vent through the roof boot. Re-install the back panel.", 0.0, closed),
    ]
    cols = 3
    cw_, ch_ = w / cols, (h - 0.35) / 2
    u0 = pcx - OD / 2 - 14
    u1 = k["skid_x1"] + stage + OD / 2 + 10
    crop = (u0, k["pipe_bot"] - 5, u1, k["z_roof"](k["roof_y"][1]) + 26)
    sc = min((cw_ - 0.35) / (u1 - u0), (ch_ - 1.6) / (crop[3] - crop[1] + 60))
    for i, (title, text, dx, inc) in enumerate(steps):
        cx = x + (i % cols) * cw_
        cy = y + (i // cols) * ch_
        s.circle(cx + 0.18, cy + 0.2, 0.14, lw="med")
        s.text(cx + 0.18, cy + 0.245, str(i + 1), 0.11, "middle", weight="bold")
        s.text(cx + 0.42, cy + 0.245, title.upper(), 0.11, weight="bold")
        sm = _Staged(m, dx, inc)
        v = View(sm, "left", cut=(1, pcy, -1), crop=crop)
        v.place(cx + 0.15, cy + 0.45, sc)
        v.draw(s, lw_face="light")
        v.draw_ground(s)
        # plan strip at foundation level
        pl = View(_Staged(m, dx, lambda p, inc=inc: inc(p) and "ground" not in p.tags and (p.name in (
                      "Skid", "HDPE pipe", "Rim joist", "Floor joist", "Deck rim", "Deck joist", "Bench header", "Post"))),
                  "top", crop=(u0, -4, u1, W + 4))
        pl.place(cx + 0.15, cy + 0.55 + (crop[3] - crop[1]) * sc + 0.1, sc)
        if pl.items:
            pl.draw(s, lw_face="light")
        if i == 0:
            c = pl.P(pcx, pcy)
            s.circle(*c, (OD / 2 + 4) * sc, lw="light", dash=(0.03, 0.02))
        if i in (2, 3):
            ya = pl.P(0, W / 2)[1]
            xa = pl.P(k["skid_x1"] + dx, 0)[0]
            arrow(s, xa + 0.45, ya, xa + 0.05, ya)
            s.text(xa + 0.05, ya - 0.08, "PULL", 0.065)
        if i == 4:
            a_, b_ = pl.P(0, W), pl.P(pcx, W)
            S.dim_h(s, a_[0], b_[0], a_[1] - 0.15, fmt_ftin(pcx), ext_from=(a_[1], b_[1]))
        ty = cy + 0.55 + (crop[3] - crop[1]) * sc + 0.1 + (W + 8) * sc + 0.25
        s.mtext(cx + 0.1, ty, wrap(text, cw_ - 0.35, 0.074), 0.074)
    s.text(x + 0.1, y + h - 0.02, "RELOCATION: reverse steps 6 to 3. Fill the old pit with at least 2' of clean soil, mound "
           "it and mark it. Pull the pipe for reuse or leave it and set a new one.", 0.08, weight="bold")
    return sh.done()


def _elec_symbol(s, x, y, kind, tag):
    t = s.t
    if kind in ("switch", "switch_wp"):
        s.text(x, y + 0.045, "S", 0.13, "middle", weight="bold")
        if kind == "switch_wp":
            s.text(x + 0.07, y + 0.085, "WP", 0.055, "start", weight="bold")
    elif kind in ("receptacle", "receptacle_wp"):
        r = 0.07
        s.circle(x, y, r, fill=t["bg"], lw="med")
        s.line(x - r * 1.5, y - 0.025, x + r * 1.5, y - 0.025, lw="med")
        s.line(x - r * 1.5, y + 0.025, x + r * 1.5, y + 0.025, lw="med")
        s.text(x, y + r + 0.1, "GFCI" + (" WP" if kind == "receptacle_wp" else ""), 0.055, "middle", weight="bold")
    elif kind == "light":
        r = 0.1
        s.circle(x, y, r, fill=t["bg"], lw="med")
        d = r * 0.7
        s.line(x - d, y - d, x + d, y + d, lw="med")
        s.line(x - d, y + d, x + d, y - d, lw="med")
    elif kind == "disconnect":
        s.rect(x - 0.1, y - 0.07, 0.2, 0.14, fill=t["bg"], lw="med")
        s.line(x - 0.1, y + 0.07, x + 0.1, y - 0.07, lw="light")
    elif kind == "panel":
        s.rect(x - 0.13, y - 0.08, 0.26, 0.16, fill=t["bg"], lw="med")
        s.path(S.pts_to_d([(x - 0.13, y + 0.08), (x + 0.13, y - 0.08), (x + 0.13, y + 0.08)]), fill=t["ink"], lw="fine")
    s.text(x + 0.13, y - 0.08, tag, 0.075, "start", weight="bold", color=t["accent"])


def _wire(s, a, b, bulge=0.25, dash=(0.06, 0.04)):
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    dx, dy = b[0] - a[0], b[1] - a[1]
    ln = math.hypot(dx, dy) or 1.0
    cx, cy = mx - dy / ln * bulge, my + dx / ln * bulge
    s.path(f"M{S.f(a[0])} {S.f(a[1])} Q{S.f(cx)} {S.f(cy)} {S.f(b[0])} {S.f(b[1])}", lw="light", dash=dash)


def sheet_electrical(ctx):
    sh = Sheet(ctx, "E-101", "Electrical plan")
    s, m, k = sh.svg, ctx["model"], ctx["model"].key
    x, y, w, h = sh.area
    el = m.cfg.get("electrical", {})
    if not k.get("electrical"):
        s.text(x + w / 2, y + h / 2, "No electrical service in this design (electrical.enabled = false).", 0.14, "middle")
        return sh.done()
    W, D, DD = k["W"], k["D"], k["DD"]
    vp = plan_view(m, site=True)
    ext = vp.extents()
    vp.crop = (ext[0] - 1, ext[1] - 30, ext[2] + 1, ext[3] + 4)
    row_layout(ctx, [vp], x, y + 0.5, w * 0.64, h - 1.4, [(0.7, 0.5)])
    vp.draw(s, lw_face="light")
    du0, du1, dv0, dv1, hinge_hi = k["door_leaf"]
    hy, ly = (du1, du0) if hinge_hi else (du0, du1)
    if k.get("door_swing") == "in":
        swing_arc(s, vp, (D - k["st_d"], hy), (D - k["st_d"] - (du1 - du0), hy), (D - k["st_d"], ly))
    else:
        swing_arc(s, vp, (D + k["tb"], hy), (D + k["tb"] + (du1 - du0), hy), (D + k["tb"], ly))
    devs = {d["tag"]: d for d in k["electrical"]["devices"]}
    off = {"S1": (-6, -2), "R1": (-9, -6), "S2": (7, -4), "R2": (7, 6), "DS": (0, -7), "MP": (0, 9)}
    pos = {}
    for tag, d in devs.items():
        ox, oy = off.get(tag, (0, 0))
        pos[tag] = vp.P(d["x"] + ox, d["y"] + oy)
        _elec_symbol(s, *pos[tag], d["kind"], tag)
    for a, b in (("MP", "DS"), ("DS", "R1"), ("R1", "S1"), ("S1", "L1"), ("DS", "R2"), ("R2", "S2"), ("S2", "L2")):
        if a in pos and b in pos:
            _wire(s, pos[a], pos[b])
    svc = el.get("service", {})
    src = "MP" if "MP" in pos else "DS"
    if src in pos:
        px, py = pos[src]
        arrow(s, px + 0.9, py + 0.55, px + 0.18, py + 0.1)
        fd = el.get("feed", {})
        lines = ([f"UTILITY SERVICE LATERAL ({svc.get('utility', 'serving utility').upper()}) TO METER-MAIN MP",
                  f"MP TO DS: {k.get('feed_awg', 12)}-2 W/ GND UF-B DIRECT BURIED, "
                  f"{fmt_in(L(el.get('burial_depth', 24)))} MIN COVER (NEC TABLE 300.5)"] if src == "MP" else
                 [f"HOME RUN TO {fd.get('source', 'PANEL').upper()}"])
        s.mtext(px + 0.95, py + 0.6, lines, 0.068, weight="bold")
    title_under(s, vp, 1, "Electrical plan", below=0.95)
    # legend + schedule + notes
    cx = x + w * 0.66
    cw = x + w - cx
    rows = []
    for tag in ("MP", "DS", "S1", "R1", "L1", "S2", "R2", "L2"):
        d = devs.get(tag)
        if not d:
            continue
        mount = (f"{fmt_ftin(d['z'] - k['FF'])} AFF" if d["kind"] not in ("light", "disconnect", "panel")
                 else ("ceiling" if d["kind"] == "light" else f"{fmt_ftin(d['z'])} above grade"))
        rows.append([tag, d["desc"], mount])
    yy = table(s, cx, y + 0.1, [("Tag", 0.35, "start"), ("Device", cw - 1.35, "start"), ("Mount", 1.0, "start")], rows,
               title="Device schedule", size=0.066)
    yy += 0.3
    s.text(cx, yy, "SYMBOLS", 0.09, weight="bold")
    yy += 0.22
    for kind, label in (("panel", "Meter-main / panelboard"), ("disconnect", "Disconnect switch"), ("switch", "Switch"),
                        ("switch_wp", "Switch, weatherproof"), ("receptacle", "Duplex receptacle, GFCI"),
                        ("light", "Ceiling light")):
        _elec_symbol(s, cx + 0.15, yy - 0.03, kind, "")
        s.text(cx + 0.45, yy, label, 0.068)
        yy += 0.36
    s.line(cx + 0.02, yy - 0.05, cx + 0.3, yy - 0.05, lw="light", dash=(0.06, 0.04))
    s.text(cx + 0.45, yy, "Circuit run (switch leg / feed)", 0.068)
    yy += 0.35
    fd = el.get("feed", {})
    notes = [
        f"Dedicated service: {svc.get('description', 'meter-main pedestal')} on a PT post "
        f"{fmt_ftin(L(svc.get('distance', 72)))} off the low side. The utility service lateral, meter and main "
        "stay put when the privy moves. Coordinate the meter location and lateral route with the serving utility.",
        f"Grounding at MP: {svc.get('grounding', '(2) 5/8 in x 8 ft rods 6 ft apart, #6 Cu GEC')}; neutral bonded "
        "only at the service disconnect (MP main).",
        f"Branch circuit to the privy: {fd.get('breaker_a', 20)} A GFCI breaker in MP, {k.get('feed_awg', 12)} AWG UF-B "
        f"with ground (voltage drop on A-602). Spare breaker spaces for a later circuit.",
        "DS at the point of entry is the building disconnect (NEC 225.31/225.36) and the unhook point for relocation. "
        "No grounding electrode at the privy: single branch circuit with an EGC (NEC 250.32(A) exception).",
        "All receptacles GFCI (NEC 210.8); R2 weather- and tamper-resistant with an extra-duty in-use cover (406.9).",
        el.get("wiring", "UF-B with ground") + ". Unheated, damp building: NM-B is not permitted.",
        "Fixtures damp-rated. Keep 30 in x 36 in working space clear in front of MP (NEC 110.26).",
        "Permit and inspections: Rockingham County (NEC as adopted in NC). Utility releases the meter after inspection.",
        "Relocation: MP branch breaker off, open DS, disconnect the UF-B in the entry box, pull it from the conduit.",
    ]
    notes_block(s, cx, yy, cw, "Electrical notes", notes, size=0.068)
    return sh.done()


def sheet_schedules(ctx):
    sh = Sheet(ctx, "A-601", "Cut list & material schedules")
    s, m, k = sh.svg, ctx["model"], ctx["model"].key
    x, y, w, h = sh.area
    cl = bom.cut_list(m)
    rows = [[r["group"].replace("_", " "), r["label"], r["name"], str(r["qty"]), fmt_ftin(r["length"]), r["note"]] for r in cl]
    colw = [0.75, 1.05, 1.1, 0.3, 0.62, 1.35]
    cols = list(zip(["Area", "Stock", "Piece", "Qty", "Length", "Note"], colw,
                    ["start", "start", "start", "middle", "end", "start"]))
    half = (len(rows) + 1) // 2
    size, rh = 0.056, 0.088
    table(s, x, y + 0.1, cols, rows[:half], title="Cut list (1 of 2)", size=size, rh=rh, max_y=sh.y1)
    x2 = x + sum(colw) + 0.2
    y2 = table(s, x2, y + 0.1, cols, rows[half:], title="Cut list (2 of 2)", size=size, rh=rh, max_y=sh.y1)
    x3 = x2 + sum(colw) + 0.2
    cw = sh.x1 - x3
    pr = bom.purchase(m, cl)
    prow = [[r["label"], (r["treatment"] or r["species"]).replace(" ground contact", " GC"), fmt_ftin(r["length"]),
             str(r["qty"]), f"{r['bf']:.0f}"] for r in pr]
    yy = table(s, x3, y + 0.1, [("Lumber", 0.8, "start"), ("Grade / treat.", cw - 1.9, "start"), ("Len", 0.45, "end"),
                                ("Qty", 0.3, "middle"), ("BF", 0.35, "end")], prow, title="Lumber to buy", size=size, rh=rh)
    s.text(x3, yy + 0.13, f"Total {bom.totals(pr):.0f} board feet (nominal). Rough-sawn stock is sold by the board foot.",
           size, italic=True)
    yy += 0.35
    sg = bom.sheet_goods(m)
    yy = table(s, x3, yy, [("Sheet goods / pieces", cw - 0.35, "start"), ("Qty", 0.35, "middle")],
               [[f"{r['label']}: {r['pieces']}", str(r["qty"])] for r in sg], title="Sheet goods", size=size, rh=rh)
    yy += 0.3
    hw = bom.hardware(m)
    hcols = [("Item", cw - 1.3, "start"), ("Qty", 0.55, "middle"), ("Note", 0.75, "start")]
    rest = []
    table(s, x3, yy, hcols, [[r["item"], str(r["qty"]), r["note"]] for r in hw], title="Pit, roofing & hardware",
          size=size, rh=rh, max_y=sh.y1, overflow=rest)
    if rest:        # continue under the second half of the cut list
        table(s, x2, y2 + 0.3, hcols, rest, title="Pit, roofing & hardware (cont.)", size=size, rh=rh, max_y=sh.y1)
    return sh.done()


def sheet_checks(ctx):
    sh = Sheet(ctx, "A-602", "Design checks, parameters & render setup")
    s, m, k, cfg = sh.svg, ctx["model"], ctx["model"].key, ctx["cfg"]
    x, y, w, h = sh.area
    rows = []
    for c in m.checks:
        st = "OK" if c["ok"] is True else ("FAIL" if c["ok"] is False else "info")
        rows.append([c["cat"], c["title"], c["value"], c["limit"], st, c["note"]])
    cols = [("Cat.", 0.6, "start"), ("Check", 2.45, "start"), ("Value", 1.85, "start"), ("Limit", 1.1, "start"),
            ("", 0.4, "middle"), ("Note", 1.9, "start")]
    yy = table(s, x, y + 0.1, cols, rows, title="Design checks (computed from the JSON)", size=0.066, rh=0.112,
               max_y=sh.y1 - 0.2)
    s.text(x, min(yy + 0.15, sh.y1), "Structural rows are simplified NDS allowable-stress screening (No. 2 lumber, CD by "
           "load case, Cr 1.15 for repetitive members). Not a sealed design.", 0.058, italic=True)
    x2 = x + sum(c[1] for c in cols) + 0.3
    cw = sh.x1 - x2
    b, r = cfg["building"], cfg["roof"]
    params = [
        ["building.width x depth", f"{fmt_ftin(k['W'])} x {fmt_ftin(k['D'])}"],
        ["building.low_bearing_height", fmt_ftin(L(b["low_bearing_height"]))],
        ["deck.depth", fmt_ftin(k["DD"])],
        ["roof.pitch, overhangs", f"{r['pitch']}:12; eave {fmt_in(L(r['overhang_low']))}, high "
         f"{fmt_in(L(r['overhang_high']))}, rakes {fmt_in(L(r['rake_back']))} / {fmt_in(L(r['rake_front']))}"],
        ["pit.pipe ID / OD / bury", f"{fmt_in(k['ID'])} / {fmt_in(k['OD'])} / {fmt_ftin(-k['pipe_bot'])}"],
        ["bench.height, hole setback", f"{fmt_in(L(cfg['bench']['height']))}, {fmt_in(L(cfg['bench']['hole']['setback']))}"],
        ["foundation.skids", f"{cfg['foundation']['skids']['stock']} {cfg['foundation']['skids']['orientation']}, "
         f"chamfer {fmt_in(k['chamfer'][0])} x {fmt_in(k['chamfer'][1])}"],
        ["walls.stud_spacing", fmt_in(L(b["walls"]["stud_spacing"]))],
        ["roof.roofing", f"{r['roofing'].get('profile', 'corrugated')} on {k.get('substrate', 'purlins')}"],
        ["pit.usage", f"{cfg['pit'].get('usage', {}).get('persons', '?')} persons x "
         f"{cfg['pit'].get('usage', {}).get('days_per_year', '?')} days/yr"],
        ["door R.O.", f"{fmt_in(k['door']['u1'] - k['door']['u0'])} x {fmt_in(k['door']['v1'] - k['door']['v0'])}, "
         f"{cfg['door'].get('type', 'board')}, swing {k.get('door_swing', 'out')}"],
        ["window unit, sill", f"{fmt_in(L(cfg['window']['unit_width']))} x {fmt_in(L(cfg['window']['unit_height']))}, "
         f"{fmt_ftin(L(cfg['window']['sill_height']))}"],
    ]
    yy = table(s, x2, y + 0.1, [("Parameter", cw * 0.45, "start"), ("Value", cw * 0.55, "start")], params,
               title="Key parameters", size=0.064, rh=0.108)
    yy += 0.3
    s.text(x2, yy, "RENDER MATERIALS", 0.084, weight="bold")
    yy += 0.1
    for key, mt in cfg["materials"].items():
        if yy > sh.y1 - 1.2:
            break
        col = mt.get("base_color", mt.get("tint", "#888888"))
        s.rect(x2, yy, 0.18, 0.1, fill=col, lw="fine")
        desc = mt.get("description", mt.get("type", ""))
        s.text(x2 + 0.25, yy + 0.08, key, 0.058, weight="bold")
        s.text(x2 + 1.15, yy + 0.08, desc[:52], 0.055)
        yy += 0.125
    yy += 0.2
    lt = cfg["lighting"]
    s.text(x2, yy, "LIGHTING", 0.084, weight="bold")
    yy += 0.16
    txt = (f"Sun azimuth {lt['sun']['azimuth_deg']} deg, elevation {lt['sun']['elevation_deg']} deg; sky "
           f"{lt['sky'].get('model', 'sunsky')}, turbidity {lt['sky'].get('turbidity')}; exposure {lt.get('exposure_ev', 0)} "
           f"EV; tone map {lt.get('tonemap', 'aces')}. {lt.get('note', '')}")
    s.mtext(x2, yy, wrap(txt, cw, 0.06), 0.06)
    return sh.done()


def sheet_renders(ctx):
    sh = Sheet(ctx, "R-101", "Renderings")
    s = sh.svg
    x, y, w, h = sh.area
    rs = [r for r in ctx["renders"] if os.path.exists(r["path"])]
    if not rs:
        s.text(x + w / 2, y + h / 2, "Renderings were not generated for this build.", 0.14, "middle")
        return sh.done()
    main, others = rs[0], rs[1:3]
    mw = w * 0.64
    data, ar = S.jpeg_bytes(main["path"])
    mh = min(mw / ar, h - 0.6)
    mw = mh * ar
    s.image(x, y + 0.1, mw, mh, data)
    s.rect(x, y + 0.1, mw, mh, lw="light")
    s.text(x, y + mh + 0.32, f"1  {main['title'].upper()}", 0.1, weight="bold")
    s.mtext(x, y + mh + 0.52, wrap("Path traced with Mitsuba 3 from the same part model as the drawings. Materials, "
                                   "sun, sky, site and cameras come from the JSON (materials / lighting / renders).",
                                   mw, 0.075), 0.075, italic=True)
    ox = x + mw + 0.3
    ow = x + w - ox
    oy = y + 0.1
    for i, r in enumerate(others, 2):
        d, a = S.jpeg_bytes(r["path"], max_w=1400)
        hh = ow / a
        s.image(ox, oy, ow, hh, d)
        s.rect(ox, oy, ow, hh, lw="light")
        for j, ln in enumerate(wrap(f"{i}  {r['title'].upper()}", ow, 0.085, caps=True)):
            s.text(ox, oy + hh + 0.18 + j * 0.12, ln, 0.085, weight="bold")
        oy += hh + 0.55
    return sh.done()


# ============================================================================ driver

SHEETS = [
    ("G-001", "Cover, sheet index & general notes", sheet_cover),
    ("A-101", "Floor plan", sheet_floor_plan),
    ("A-102", "Roof plan & roof framing plan", sheet_roof),
    ("S-101", "Foundation & floor framing plan", sheet_foundation),
    ("A-201", "Exterior elevations: left & front",
     lambda c: sheet_elevations(c, "A-201", ["left", "front"], "Exterior elevations: left & front")),
    ("A-202", "Exterior elevations: right & back",
     lambda c: sheet_elevations(c, "A-202", ["right", "back"], "Exterior elevations: right & back")),
    ("A-301", "Section A: longitudinal through pit", lambda c: sheet_sections(c, "A-301", "A")),
    ("A-302", "Sections B & C: transverse", lambda c: sheet_sections(c, "A-302", "B")),
    ("S-201", "Left (low) side wall framing", lambda c: sheet_side_framing(c, "S-201", "left")),
    ("S-202", "Right (high) side wall framing", lambda c: sheet_side_framing(c, "S-202", "right")),
    ("S-203", "End wall framing elevations", sheet_end_framing),
    ("A-501", "Details: pit & flange, skid, rafter", sheet_details),
    ("A-502", "Installation & relocation sequence", sheet_install),
    ("A-503", "Details: removable panel, door, bench", sheet_panel_door),
    ("E-101", "Electrical plan", sheet_electrical),
    ("A-601", "Cut list & material schedules", sheet_schedules),
    ("A-602", "Design checks, parameters & render setup", sheet_checks),
    ("R-101", "Renderings", sheet_renders),
]


def build_sheets(model, renders, source_name="design/privy.json", only=None, log=print):
    ctx = dict(model=model, cfg=model.cfg, renders=renders, source_name=source_name,
               generated=_dt.date.today().isoformat(),
               index={n: (i + 1, t) for i, (n, t, _) in enumerate(SHEETS)})
    out = []
    for n, t, fn in SHEETS:
        if only and n not in only:
            continue
        log(f"  sheet {n}  {t}")
        out.append(fn(ctx))
    return out
