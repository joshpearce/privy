"""Orthographic views of the part model: elevations, plans and sections.

Hidden-surface removal is a painter's algorithm at part granularity.  Parts
whose projections overlap are ordered pairwise: by disjoint depth ranges when
possible, otherwise by comparing their face depths at sample points inside the
overlap (Newell-style), then topologically sorted far-to-near.  Section views
clip every part to the kept half-space and draw the cut faces ("caps") last
with a material hatch.
"""
from __future__ import annotations

import heapq
import math
from collections import defaultdict

import numpy as np

from ..geometry import CorrugatedSheet, Tube, clip, convex_hull, lift
from ..units import fmt_elev, fmt_ftin
from . import svg as S

DIRS = {  # right (u), up (v), view direction d (into the page)
    "front": ((0, 1, 0), (0, 0, 1), (-1, 0, 0)),
    "back": ((0, -1, 0), (0, 0, 1), (1, 0, 0)),
    "left": ((1, 0, 0), (0, 0, 1), (0, 1, 0)),
    "right": ((-1, 0, 0), (0, 0, 1), (0, -1, 0)),
    "top": ((1, 0, 0), (0, 1, 0), (0, 0, -1)),
}


def clip_poly_rect(poly, u0, v0, u1, v1):
    p = [tuple(x) for x in poly]
    for axis, val, keep in ((0, u0, -1), (0, u1, 1), (1, v0, -1), (1, v1, 1)):
        out = []
        n = len(p)
        for i in range(n):
            a, b = p[i], p[(i + 1) % n]
            da, db = keep * (a[axis] - val), keep * (b[axis] - val)
            if da <= 0:
                out.append(a)
            if da * db < 0:
                t = da / (da - db)
                out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
        p = out
        if len(p) < 3:
            return []
    return p


def pip(pts, poly):
    """Even-odd point-in-polygon for many points."""
    poly = np.asarray(poly, float)
    x, y = pts[:, 0:1], pts[:, 1:2]
    x1, y1 = poly[None, :, 0], poly[None, :, 1]
    x2, y2 = np.roll(poly[:, 0], -1)[None], np.roll(poly[:, 1], -1)[None]
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = (x2 - x1) * (y - y1) / (y2 - y1) + x1
    c = ((y1 > y) != (y2 > y)) & (x < xi)
    return (c.sum(1) % 2) == 1


class Item:
    __slots__ = ("part", "polys", "lines", "bb", "dmin", "dmax", "dmean", "tube", "rank")

    def __init__(self, part):
        self.part = part
        self.polys = []   # (poly2d, holes2d, plane(n, p0) | None, edge_mask | None)
        self.lines = []
        self.tube = None


class View:
    def __init__(self, model, direction, include=None, cut=None, hidden=None, above_grade=None, crop=None,
                 roof_strips=6.0, grade=True, face_fill=None):
        self.m = model
        self.cfg = model.cfg
        ru, rv, d = (np.array(v, float) for v in DIRS[direction])
        self.ru, self.rv, self.d = ru, rv, d
        self.va = int(np.argmax(np.abs(d)))
        self.direction = direction
        self.cut = cut
        self.crop = crop
        self.grade = grade and direction != "top"
        self.above_grade = (cut is None) if above_grade is None else above_grade
        self.face_fill = face_fill
        self.items: list[Item] = []
        self.caps = []
        self.hidden = []
        self.extra_hidden_circles = []
        self._build(include or (lambda p: True), hidden or (lambda p: False), roof_strips)
        self._order()

    # ------------------------------------------------------------ projection
    def uv(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        return np.stack([P @ self.ru, P @ self.rv], 1)

    def depth(self, P):
        return np.atleast_2d(np.asarray(P, float)) @ self.d

    def _faces2d(self, g, internal=()):
        """Viewer-facing polygons of g: [(poly2d, holes2d, plane|None, edge_mask|None, pts3d|None, tube)]."""
        out = []
        if getattr(g, "smooth_axis", None) is not None and g.smooth_axis != self.va:
            if isinstance(g, Tube) and abs(self.ru[2]) < 1e-9:
                c = self.uv([[g.cx, g.cy, 0]])[0][0]
                cd = float(self.depth([[g.cx, g.cy, 0]])[0])
                poly = [(c - g.ro, g.z0), (c + g.ro, g.z0), (c + g.ro, g.z1), (c - g.ro, g.z1)]
                return [(poly, [], None, None, None, (c, cd, g.ro))]
            t, _ = g.triangles()
            if len(t):
                pts = t.reshape(-1, 3)
                hull = convex_hull(self.uv(pts))
                if len(hull) >= 3:
                    return [([tuple(p) for p in hull], [], None, None, pts, None)]
            return []
        for fc in g.faces():
            if float(fc.normal @ self.d) < -1e-6:
                if internal and np.ptp(fc.pts[:, 1]) < 1e-6 and any(abs(fc.pts[0, 1] - y) < 1e-6 for y in internal):
                    continue       # strip end cap inside a split sheet
                mask = None
                if internal:
                    n = len(fc.pts)
                    mask = [not (any(abs(fc.pts[i][1] - y) < 1e-6 and abs(fc.pts[(i + 1) % n][1] - y) < 1e-6
                                     for y in internal)) for i in range(n)]
                out.append(([tuple(p) for p in self.uv(fc.pts)], [[tuple(p) for p in self.uv(h)] for h in fc.holes],
                            (fc.normal, fc.pts[0]), mask, fc.pts, None))
        return out

    def _build(self, include, hidden, roof_strips):
        axis = value = keep = None
        if self.cut:
            axis, value, keep = self.cut
        for part in self.m.parts:
            if not include(part):
                continue
            geoms = [(part.geom, ())]
            if isinstance(part.geom, CorrugatedSheet) and roof_strips:
                g0 = part.geom
                n = max(1, math.ceil((g0.y1 - g0.y0) / roof_strips))
                ys = [g0.y0 + (g0.y1 - g0.y0) * i / n for i in range(n + 1)]
                internal = tuple(ys[1:-1])
                geoms = [(CorrugatedSheet(g0.x0, g0.x1, ya, yb, g0.z0 + (ya - g0.y0) * g0.slope, g0.slope, g0.pitch,
                                          g0.depth), internal) for ya, yb in zip(ys, ys[1:])]
            is_ground = "ground" in part.tags
            if hidden(part):
                self._hidden(part.geom)
            for g, internal in geoms:
                if self.above_grade and not is_ground:
                    lo, hi = g.bbox()
                    if hi[2] <= 0:
                        continue
                    if lo[2] < 0:
                        g = clip(g, 2, 0.0, -1)
                if self.cut:
                    lo, hi = g.bbox()
                    if lo[axis] + 1e-6 < value < hi[axis] - 1e-6:
                        for sec in g.section(axis, value):
                            self.caps.append((part, [tuple(p) for p in self.uv(lift(sec.outer, axis, value))],
                                              [[tuple(p) for p in self.uv(lift(h, axis, value))] for h in sec.holes]))
                    g = clip(g, axis, value, keep)
                if g is None or is_ground:
                    continue
                faces = self._faces2d(g, internal)
                if not faces:
                    continue
                lines = []
                nrm = np.array([0, -getattr(g, "slope", 0.0), 1.0])
                if float(nrm @ self.d) < 0:
                    for a, b in g.lines():
                        uv = self.uv([a, b])
                        lines.append((tuple(uv[0]), tuple(uv[1])))
                lo, hi = g.bbox()
                corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
                gd = self.depth(corners)
                best, best_area = None, -1.0
                for poly, holes, plane, mask, pts3, tube in faces:
                    it = Item(part)
                    it.polys = [(poly, holes, plane, mask)]
                    it.tube = tube
                    if pts3 is not None and plane is not None:
                        dp = self.depth(pts3)
                    else:
                        dp = gd
                    it.dmin, it.dmax, it.dmean = float(dp.min()), float(dp.max()), float(dp.mean())
                    a = np.array(poly)
                    it.bb = (*a.min(0), *a.max(0))
                    area = (it.bb[2] - it.bb[0]) * (it.bb[3] - it.bb[1])
                    if area > best_area:
                        best, best_area = it, area
                    self.items.append(it)
                if lines and best is not None:
                    best.lines = lines

    # ------------------------------------------------------------ ordering
    def _depth_at(self, it: Item, pts):
        """Depth of item surface at each 2-D point (nan where not covered)."""
        out = np.full(len(pts), np.nan)
        if it.tube is not None:
            c, cd, r = it.tube
            inside = np.zeros(len(pts), bool)
            for poly, *_ in it.polys:
                inside |= pip(pts, poly)
            out[inside] = cd - np.sqrt(np.maximum(r * r - (pts[inside, 0] - c) ** 2, 0.0))
            return out
        for poly, holes, plane, _ in it.polys:
            m = pip(pts, poly)
            for h in holes:
                m ^= pip(pts, h)
            if not m.any():
                continue
            if plane is None:
                dd = np.full(m.sum(), it.dmean)
            else:
                n, p0 = plane
                nd = float(n @ self.d)
                dd = (float(n @ p0) - pts[m, 0] * float(n @ self.ru) - pts[m, 1] * float(n @ self.rv)) / nd
            cur = out[m]
            out[m] = np.where(np.isnan(cur), dd, np.minimum(cur, dd))   # nearest surface wins
        return out

    def _order(self):
        items = self.items
        N = len(items)
        if N < 2:
            return
        base = sorted(range(N), key=lambda i: (-items[i].dmean, -items[i].dmin))
        rank = {i: r for r, i in enumerate(base)}
        bb = np.array([it.bb for it in items])
        dmin = np.array([it.dmin for it in items])
        dmax = np.array([it.dmax for it in items])
        e = 1e-6
        ov = ((bb[:, None, 0] < bb[None, :, 2] - e) & (bb[None, :, 0] < bb[:, None, 2] - e) &
              (bb[:, None, 1] < bb[None, :, 3] - e) & (bb[None, :, 1] < bb[:, None, 3] - e))
        ov = np.triu(ov, 1)
        succ = defaultdict(set)
        indeg = np.zeros(N, int)

        def edge(a, b):   # a drawn before b
            if b not in succ[a]:
                succ[a].add(b)
                indeg[b] += 1

        for i, j in np.argwhere(ov):
            if dmax[i] <= dmin[j] + 1e-6:          # i entirely nearer
                edge(j, i)
                continue
            if dmax[j] <= dmin[i] + 1e-6:
                edge(i, j)
                continue
            u0, v0 = max(bb[i, 0], bb[j, 0]), max(bb[i, 1], bb[j, 1])
            u1, v1 = min(bb[i, 2], bb[j, 2]), min(bb[i, 3], bb[j, 3])
            g = (np.arange(7) + 0.5) / 7
            pts = np.array([(u0 + a * (u1 - u0), v0 + b * (v1 - v0)) for a in g for b in g])
            di, dj = self._depth_at(items[i], pts), self._depth_at(items[j], pts)
            both = ~np.isnan(di) & ~np.isnan(dj)
            if not both.any():
                continue
            diff = float(np.median(di[both] - dj[both]))
            if diff < -1e-4:
                edge(j, i)
            elif diff > 1e-4:
                edge(i, j)
        heap = [(rank[i], i) for i in range(N) if indeg[i] == 0]
        heapq.heapify(heap)
        done = np.zeros(N, bool)
        order = []
        while len(order) < N:
            if not heap:   # cycle: release the lowest-ranked remaining node
                i = min((rank[i], i) for i in range(N) if not done[i])[1]
                indeg[i] = 0
                heap.append((rank[i], i))
            _, i = heapq.heappop(heap)
            if done[i]:
                continue
            done[i] = True
            order.append(i)
            for j in succ[i]:
                indeg[j] -= 1
                if indeg[j] == 0 and not done[j]:
                    heapq.heappush(heap, (rank[j], j))
        self.items = [items[i] for i in order]

    def _hidden(self, g):
        if isinstance(g, Tube) and self.va == 2:
            c = self.uv([[g.cx, g.cy, 0]])[0]
            self.extra_hidden_circles.append((c[0], c[1], g.ro))
            if g.ri > 0:
                self.extra_hidden_circles.append((c[0], c[1], g.ri))
            return
        if self.above_grade:
            lo, hi = g.bbox()
            if lo[2] < 0:
                g = clip(g, 2, 0.0, 1)
                if g is None:
                    return
        if isinstance(g, Tube) and abs(self.ru[2]) < 1e-9:
            c = self.uv([[g.cx, g.cy, 0]])[0][0]
            self.hidden.append([(c - g.ro, g.z0), (c + g.ro, g.z0), (c + g.ro, g.z1), (c - g.ro, g.z1)])
            if g.ri > 0:
                self.hidden.append([(c - g.ri, g.z0), (c + g.ri, g.z0), (c + g.ri, g.z1), (c - g.ri, g.z1)])
            return
        t, _ = g.triangles()
        if len(t):
            self.hidden.append([tuple(p) for p in convex_hull(self.uv(t.reshape(-1, 3)))])

    # ------------------------------------------------------------ extents & placement
    def extents(self):
        if self.crop:
            return self.crop
        pts = []
        for it in self.items:
            for p, *_ in it.polys:
                pts.extend(p)
        for part, p, _ in self.caps:
            if "ground" not in part.tags:
                pts.extend(p)
        for p in self.hidden:
            pts.extend(p)
        a = np.array(pts)
        u0, v0 = a.min(0)
        u1, v1 = a.max(0)
        if self.grade and self.cut is not None:
            v0 = min(v0, self.m.key["pipe_bot"] - 8)
        return (float(u0), float(v0), float(u1), float(v1))

    def place(self, x0, y0, scale, ext=None):
        """Map model extents so that (u0, v1) lands at paper (x0, y0)."""
        self.scale = scale
        self.ext = ext or self.extents()
        self.x0, self.y0 = x0, y0
        return self

    def P(self, u, v):
        return (self.x0 + (u - self.ext[0]) * self.scale, self.y0 + (self.ext[3] - v) * self.scale)

    def Pm(self, pt3):
        u, v = self.uv([pt3])[0]
        return self.P(u, v)

    def U(self, pt3):
        return tuple(self.uv([pt3])[0])

    def size(self, scale=None):
        e = self.extents()
        s = scale or getattr(self, "scale", 1.0)
        return (e[2] - e[0]) * s, (e[3] - e[1]) * s

    def paper_box(self):
        e = self.ext
        (x0, y0), (x1, y1) = self.P(e[0], e[3]), self.P(e[2], e[1])
        return x0, y0, x1, y1

    # ------------------------------------------------------------ drawing
    def _d(self, poly, holes=()):
        s = S.pts_to_d([self.P(*p) for p in poly])
        for h in holes:
            s += " " + S.pts_to_d([self.P(*p) for p in h])
        return s

    def _fill(self, svg, part):
        dr = self.cfg["materials"].get(part.material, {}).get("drawing", {})
        fill = dr.get("fill", "#ffffff")
        if svg.theme != "white":
            h_ = fill.lstrip("#")
            lum = sum(int(h_[i:i + 2], 16) for i in (0, 2, 4)) / 765 if len(h_) == 6 else 1.0
            fill = svg.t["fill"] if lum > 0.8 else svg.t["poche"]
        if self.face_fill:
            fill = self.face_fill(part) or fill
        return fill

    def draw(self, svg: S.SVG, lw_face="med"):
        t = svg.t
        x0, y0, x1, y1 = self.paper_box()
        svg.begin_clip(x0 - 0.03, y0 - 0.03, x1 - x0 + 0.06, y1 - y0 + 0.06)
        for it in self.items:
            part = it.part
            fill = self._fill(svg, part)
            lw = "light" if (part.tags & {"siding", "roofing", "hardware", "door"}) else lw_face
            for poly, holes, _, mask in it.polys:
                if mask is None:
                    svg.path(self._d(poly, holes), fill=fill, lw=lw, rule="evenodd")
                else:
                    svg.path(self._d(poly, holes), fill=fill, rule="evenodd", lw="fine", color=fill)
                    pp = [self.P(*p) for p in poly]
                    n = len(pp)
                    for i in range(n):
                        if mask[i]:
                            svg.line(*pp[i], *pp[(i + 1) % n], lw=lw)
            for a, b in it.lines:
                svg.line(*self.P(*a), *self.P(*b), lw="fine", color=t["light"])
        if self.cut is not None:            # earth cut face lies on the section plane: over everything behind it
            e = self.ext
            crop = (e[0], e[1], e[2], min(e[3], 0.0))
            for part, poly, holes in self.caps:
                if "ground" in part.tags:
                    pc = clip_poly_rect(poly, *crop)
                    if len(pc) >= 3:
                        svg.hatch([[self.P(*p) for p in pc]], "earth")
        for part, poly, holes in self.caps:
            if "ground" not in part.tags:
                self._cap(svg, part, poly, holes)
        svg.end_clip()

    def draw_ground(self, svg: S.SVG):
        """Grade line (sections: earth caps are drawn by draw()); elevations also get a hatch band."""
        t = svg.t
        e = self.ext
        a, b = self.P(e[0], 0), self.P(e[2], 0)
        if self.cut is None and self.grade:
            svg.hatch([[(a[0], a[1]), (b[0], a[1]), (b[0], a[1] + 0.09), (a[0], a[1] + 0.09)]], "earth")
        if self.cut is not None or self.grade:
            self._grade = (a, b)
            svg.line(*a, *b, lw="grade", color=t["grade"])

    def draw_hidden(self, svg: S.SVG, dash=(0.05, 0.03), lw="light"):
        for poly in self.hidden:
            svg.path(self._d(poly), fill="none", lw=lw, dash=dash)
        for cu, cv, r in self.extra_hidden_circles:
            x, y = self.P(cu, cv)
            svg.circle(x, y, r * self.scale, lw=lw, dash=dash)

    def _cap(self, svg, part, poly, holes):
        dr = self.cfg["materials"].get(part.material, {}).get("drawing", {})
        kind = dr.get("cut", "wood")
        pp = [self.P(*p) for p in poly]
        hh = [[self.P(*p) for p in h] for h in holes]
        a = np.array(pp)
        w, h = a.max(0) - a.min(0)
        thin = min(w, h) < 0.045
        d = self._d(poly, holes)
        if kind in ("steel", "solid", "hdpe") or thin:
            fill = svg.t["poche"] if kind != "hdpe" else ("#505050" if svg.theme == "white" else svg.t["poche"])
            svg.path(d, fill=fill, lw="cut" if not thin else "light", rule="evenodd")
            return
        if kind == "wood":
            svg.path(d, fill=svg.t["fill"], lw="cut", rule="evenodd")
            if len(poly) == 4:
                x0, y0 = a.min(0)
                x1, y1 = a.max(0)
                svg.line(x0, y0, x1, y1, lw="fine", color=svg.t["hatch"])
                svg.line(x0, y1, x1, y0, lw="fine", color=svg.t["hatch"])
            return
        pat = {"board": "diag", "plywood": "diag_dense", "earth": "earth", "glass": "glass"}.get(kind, "diag")
        svg.hatch([pp] + hh, pat, outline="cut")

    # ------------------------------------------------------------ annotation in model (u, v) coords
    def dim_h(self, svg, u0, u1, vf0, vf1, off, text=None):
        """Horizontal dim; line `off` paper-inches above (off>0) the higher / below (off<0) the lower feature."""
        x0, ya = self.P(u0, vf0)
        x1, yb = self.P(u1, vf1)
        y = (min(ya, yb) - off) if off > 0 else (max(ya, yb) - off)
        S.dim_h(svg, x0, x1, y, text or fmt_ftin(abs(u1 - u0)), ext_from=(ya, yb))
        return y

    def dim_chain_h(self, svg, us, vfs, off):
        vfs = vfs if isinstance(vfs, (list, tuple)) else [vfs] * len(us)
        xs = [self.P(u, 0)[0] for u in us]
        ys = [self.P(0, v)[1] for v in vfs]
        y = (min(ys) - off) if off > 0 else (max(ys) - off)
        for i in range(len(us) - 1):
            if abs(us[i + 1] - us[i]) > 1e-6:
                S.dim_h(svg, xs[i], xs[i + 1], y, fmt_ftin(abs(us[i + 1] - us[i])), ext_from=(ys[i], ys[i + 1]))
        return y

    def dim_v(self, svg, v0, v1, uf0, uf1, off, text=None):
        """Vertical dim; off<0 left of the leftmost feature, off>0 right of the rightmost."""
        xa, y0 = self.P(uf0, v0)
        xb, y1 = self.P(uf1, v1)
        x = (min(xa, xb) + off) if off < 0 else (max(xa, xb) + off)
        S.dim_v(svg, y0, y1, x, text or fmt_ftin(abs(v1 - v0)), ext_from=(xa, xb))
        return x

    def dim_chain_v(self, svg, vs, ufs, off):
        ufs = ufs if isinstance(ufs, (list, tuple)) else [ufs] * len(vs)
        xs = [self.P(u, 0)[0] for u in ufs]
        ys = [self.P(0, v)[1] for v in vs]
        x = (min(xs) + off) if off < 0 else (max(xs) + off)
        for i in range(len(vs) - 1):
            if abs(vs[i + 1] - vs[i]) > 1e-6:
                S.dim_v(svg, ys[i], ys[i + 1], x, fmt_ftin(abs(vs[i + 1] - vs[i])), ext_from=(xs[i], xs[i + 1]))
        return x

    def note(self, svg, u, v, dx, dy, lines, size=0.08):
        ax, ay = self.P(u, v)
        S.leader(svg, ax, ay, ax + dx, ay + dy, lines, size=size)

    def level(self, svg, v, label, u_at, left=False, value=None):
        x, y = self.P(u_at, v)
        S.level_mark(svg, x, y, label, value or fmt_elev(v), left=left)
