"""Solid primitives shared by the drawing and rendering back ends.

Coordinates are inches, z up.  Model axes:
  x : back wall (bench / removable panel) -> front (door / deck)
  y : low (eave) side -> high side
  z : up, z = 0 at grade

Every primitive can
  * ``faces()``            planar polygon faces with outward normals (drawings)
  * ``triangles()``        triangle soup + per-triangle normals (rendering)
  * ``section(axis, v)``   2-D cut polygons in the plane ``axis = v``
  * ``bbox()``

Section polygons are expressed in ``PLANE[axis]`` coordinates, i.e. the two
remaining axes in increasing order (x-cut -> (y, z), y-cut -> (x, z),
z-cut -> (x, y)).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

AXES = "xyz"
PLANE = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
EPS = 1e-7


def ax(a) -> int:
    return a if isinstance(a, int) else AXES.index(a)


@dataclass
class Face:
    pts: np.ndarray                      # (k, 3) outer loop
    normal: np.ndarray                   # (3,) outward unit normal
    holes: list = field(default_factory=list)   # list of (m, 3) loops


@dataclass
class Section:
    outer: np.ndarray                    # (k, 2) polygon in PLANE[axis] coords
    holes: list = field(default_factory=list)


# --------------------------------------------------------------------------- 2-D helpers

def poly_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def ccw(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    return p if poly_area(p) >= 0 else p[::-1].copy()


def convex_hull(pts: np.ndarray) -> np.ndarray:
    pts = np.unique(np.round(np.asarray(pts, float), 6), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    return np.array(lower[:-1] + upper[:-1])


def ear_clip(poly: np.ndarray) -> list[tuple[int, int, int]]:
    """Triangulate a simple (possibly concave) CCW polygon; returns index triples."""
    n = len(poly)
    if n < 3:
        return []
    idx = list(range(n))
    if poly_area(poly) < 0:
        idx.reverse()
    tris = []

    def is_convex(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 1e-12

    def inside(p, a, b, c):
        d1 = (p[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (p[1] - b[1])
        d2 = (p[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (p[1] - c[1])
        d3 = (p[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (p[1] - a[1])
        neg = d1 < -1e-12 or d2 < -1e-12 or d3 < -1e-12
        pos = d1 > 1e-12 or d2 > 1e-12 or d3 > 1e-12
        return not (neg and pos)

    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        m = len(idx)
        for i in range(m):
            i0, i1, i2 = idx[(i - 1) % m], idx[i], idx[(i + 1) % m]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if not is_convex(a, b, c):
                continue
            if any(inside(poly[j], a, b, c) for j in idx if j not in (i0, i1, i2)
                   and not (np.allclose(poly[j], a) or np.allclose(poly[j], b) or np.allclose(poly[j], c))):
                continue
            tris.append((i0, i1, i2))
            idx.pop(i)
            break
        else:  # degenerate remainder: fan it
            for k in range(1, len(idx) - 1):
                tris.append((idx[0], idx[k], idx[k + 1]))
            return tris
    if len(idx) == 3:
        tris.append(tuple(idx))
    return tris


def line_intervals(poly: np.ndarray, k: int, value: float) -> list[tuple[float, float]]:
    """Intervals of coordinate (1-k) where the line coord[k] == value is inside poly."""
    j = 1 - k
    xs = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        da, db = a[k] - value, b[k] - value
        da = 0.0 if abs(da) < EPS else da
        db = 0.0 if abs(db) < EPS else db
        # half-open rule: a vertex on the line counts as being above it
        if (da < 0) != (db < 0):
            if abs(b[k] - a[k]) < EPS:
                continue
            t = (value - a[k]) / (b[k] - a[k])
            xs.append(a[j] + t * (b[j] - a[j]))
    xs.sort()
    return [(xs[i], xs[i + 1]) for i in range(0, len(xs) - 1, 2) if xs[i + 1] - xs[i] > EPS]


def clip_poly3(pts: np.ndarray, axis: int, value: float, keep: int) -> np.ndarray:
    """Sutherland-Hodgman clip of a planar 3-D polygon to keep*(p[axis]-value) <= 0."""
    out = []
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        da, db = keep * (a[axis] - value), keep * (b[axis] - value)
        if da <= EPS:
            out.append(a)
        if (da < -EPS and db > EPS) or (da > EPS and db < -EPS):
            t = da / (da - db)
            out.append(a + t * (b - a))
    return np.array(out) if len(out) >= 3 else np.zeros((0, 3))


def clip_triangles(tris: np.ndarray, normals: np.ndarray, axis: int, value: float, keep: int):
    """Clip triangle soup to keep*(p[axis]-value) <= 0."""
    out_t, out_n = [], []
    for t, nrm in zip(tris, normals):
        d = keep * (t[:, axis] - value)
        if (d <= EPS).all():
            out_t.append(t)
            out_n.append(nrm)
            continue
        if (d > -EPS).all():
            continue
        poly = clip_poly3(t, axis, value, keep)
        for k in range(1, len(poly) - 1):
            out_t.append(np.array([poly[0], poly[k], poly[k + 1]]))
            out_n.append(nrm)
    if not out_t:
        return np.zeros((0, 3, 3)), np.zeros((0, 3))
    return np.array(out_t), np.array(out_n)


def lift(p2: np.ndarray, axis: int, value: float) -> np.ndarray:
    """2-D points in PLANE[axis] coordinates -> 3-D points on plane axis=value."""
    p2 = np.asarray(p2, float)
    out = np.zeros((len(p2), 3))
    a1, a2 = PLANE[axis]
    out[:, a1] = p2[:, 0]
    out[:, a2] = p2[:, 1]
    out[:, axis] = value
    return out


def unit(i: int, s: float = 1.0) -> np.ndarray:
    v = np.zeros(3)
    v[i] = s
    return v


def faces_to_triangles(faces: list[Face]):
    tris, nrms = [], []
    for f in faces:
        if f.holes:
            raise ValueError("face with holes needs a custom triangulation")
        n = f.normal
        # project to the plane best aligned with the normal
        k = int(np.argmax(np.abs(n)))
        a1, a2 = PLANE[k]
        p2 = f.pts[:, [a1, a2]]
        for i0, i1, i2 in ear_clip(p2):
            t = np.array([f.pts[i0], f.pts[i1], f.pts[i2]])
            if np.dot(np.cross(t[1] - t[0], t[2] - t[0]), n) < 0:
                t = t[[0, 2, 1]]
            tris.append(t)
            nrms.append(n)
    if not tris:
        return np.zeros((0, 3, 3)), np.zeros((0, 3))
    return np.array(tris), np.array(nrms)


# --------------------------------------------------------------------------- primitives

class Geometry:
    #: axis along which faces() are clean for drawing; other views use a hull silhouette
    smooth_axis: int | None = None

    def faces(self) -> list[Face]:
        raise NotImplementedError

    def triangles(self):
        return faces_to_triangles(self.faces())

    def section(self, axis: int, value: float) -> list[Section]:
        return []

    def lines(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Extra detail line segments (3-D) drawn over the faces (e.g. corrugations)."""
        return []

    def bbox(self):
        pts = np.vstack([f.pts for f in self.faces()])
        return pts.min(0), pts.max(0)


class Prism(Geometry):
    """Simple polygon (in PLANE[axis] coords) extruded along ``axis`` from lo to hi."""

    def __init__(self, axis, poly, lo: float, hi: float):
        self.axis = ax(axis)
        self.poly = ccw(np.asarray(poly, float))
        self.lo, self.hi = (lo, hi) if lo <= hi else (hi, lo)

    def _pt(self, p2, a):
        return lift(np.atleast_2d(p2), self.axis, a)

    def faces(self):
        a = self.axis
        a1, a2 = PLANE[a]
        bot = self._pt(self.poly, self.lo)
        top = self._pt(self.poly, self.hi)
        fs = [Face(top, unit(a, 1.0)), Face(bot[::-1].copy(), unit(a, -1.0))]
        n = len(self.poly)
        for i in range(n):
            p, q = self.poly[i], self.poly[(i + 1) % n]
            d = q - p
            L = math.hypot(*d)
            if L < EPS:
                continue
            n2 = np.array([d[1], -d[0]]) / L       # outward for CCW polygon
            nrm = np.zeros(3)
            nrm[a1], nrm[a2] = n2
            quad = np.array([bot[i], bot[(i + 1) % n], top[(i + 1) % n], top[i]])
            fs.append(Face(quad, nrm))
        return fs

    def bbox(self):
        lo = np.zeros(3)
        hi = np.zeros(3)
        a1, a2 = PLANE[self.axis]
        lo[a1], lo[a2] = self.poly.min(0)
        hi[a1], hi[a2] = self.poly.max(0)
        lo[self.axis], hi[self.axis] = self.lo, self.hi
        return lo, hi

    def section(self, axis, value):
        axis = ax(axis)
        if axis == self.axis:
            if self.lo + EPS < value < self.hi - EPS:
                return [Section(self.poly.copy())]
            return []
        a1, a2 = PLANE[self.axis]
        k = (a1, a2).index(axis)
        other = (a1, a2)[1 - k]
        out = []
        c1, c2 = PLANE[axis]
        for t0, t1 in line_intervals(self.poly, k, value):
            rng = {other: (t0, t1), self.axis: (self.lo, self.hi)}
            (u0, u1), (v0, v1) = rng[c1], rng[c2]
            out.append(Section(np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]])))
        return out


def Box(x0, y0, z0, x1, y1, z1) -> Prism:
    return Prism(2, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], z0, z1)


def rect(u0, v0, u1, v1):
    return [[u0, v0], [u1, v0], [u1, v1], [u0, v1]]


class CorrugatedSheet(Prism):
    """Corrugated roofing sheet lying on a plane sloped along y.

    Drawn as a thin slab (Prism along x); rendered as a sinusoidal surface with
    ridges running down the slope.  ``z0`` is the underside elevation at y = y0.
    """

    def __init__(self, x0, x1, y0, y1, z0, slope, pitch, depth, thickness=0.02):
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
        self.z0, self.slope, self.pitch, self.depth, self.t = z0, slope, pitch, depth, thickness
        zlo = lambda y: z0 + (y - y0) * slope  # noqa: E731
        poly = [[y0, zlo(y0)], [y1, zlo(y1)], [y1, zlo(y1) + depth], [y0, zlo(y0) + depth]]
        super().__init__(0, poly, x0, x1)

    def lines(self):
        segs = []
        n = int((self.x1 - self.x0) / self.pitch)
        zt = lambda y: self.z0 + (y - self.y0) * self.slope + self.depth  # noqa: E731
        for i in range(n + 1):
            x = self.x0 + (i + 0.25) * self.pitch
            if x > self.x1:
                break
            segs.append((np.array([x, self.y0, zt(self.y0)]), np.array([x, self.y1, zt(self.y1)])))
        return segs

    def triangles(self):
        nx = max(8, int((self.x1 - self.x0) / self.pitch * 12))
        xs = np.linspace(self.x0, self.x1, nx + 1)
        w = 0.5 * self.depth * (1 - np.cos(2 * np.pi * (xs - self.x0) / self.pitch))
        c = math.sqrt(1 + self.slope ** 2)
        tris, nrms = [], []
        for off in (self.t, 0.0):   # top surface then underside
            sgn = 1.0 if off else -1.0
            for i in range(nx):
                p = []
                for xi, wi in ((xs[i], w[i]), (xs[i + 1], w[i + 1])):
                    for y in (self.y0, self.y1):
                        p.append([xi, y, self.z0 + (y - self.y0) * self.slope + wi + off * c])
                p00, p01, p10, p11 = map(np.array, p)
                dzdx = (w[i + 1] - w[i]) / (xs[i + 1] - xs[i])
                n = np.array([-dzdx, -self.slope, 1.0])
                n = sgn * n / np.linalg.norm(n)
                for t in ((p00, p10, p11), (p00, p11, p01)):
                    t = np.array(t)
                    if np.dot(np.cross(t[1] - t[0], t[2] - t[0]), n) < 0:
                        t = t[[0, 2, 1]]
                    tris.append(t)
                    nrms.append(n)
        return np.array(tris), np.array(nrms)


class Tube(Geometry):
    """Vertical hollow cylinder (r_in = 0 -> solid).  Optional outer corrugation."""

    smooth_axis = 2

    def __init__(self, cx, cy, r_out, r_in, z0, z1, segments=64, corrugation=None):
        self.cx, self.cy, self.ro, self.ri = cx, cy, r_out, r_in
        self.z0, self.z1 = min(z0, z1), max(z0, z1)
        self.n = segments
        self.corr = corrugation   # (pitch, depth) or None

    def _ring(self, r, z):
        t = np.linspace(0, 2 * np.pi, self.n, endpoint=False)
        return np.stack([self.cx + r * np.cos(t), self.cy + r * np.sin(t), np.full_like(t, z)], 1)

    def faces(self):
        fs = []
        for z, s in ((self.z1, 1.0), (self.z0, -1.0)):
            outer = self._ring(self.ro, z)
            holes = [self._ring(self.ri, z)[::-1]] if self.ri > 0 else []
            fs.append(Face(outer if s > 0 else outer[::-1].copy(), unit(2, s), holes))
        ob, ot = self._ring(self.ro, self.z0), self._ring(self.ro, self.z1)
        for i in range(self.n):
            j = (i + 1) % self.n
            mid = 0.5 * (ob[i] + ob[j])
            nrm = np.array([mid[0] - self.cx, mid[1] - self.cy, 0.0])
            fs.append(Face(np.array([ob[i], ob[j], ot[j], ot[i]]), nrm / np.linalg.norm(nrm)))
        return fs

    def bbox(self):
        return (np.array([self.cx - self.ro, self.cy - self.ro, self.z0]),
                np.array([self.cx + self.ro, self.cy + self.ro, self.z1]))

    def _r_out(self, z):
        if not self.corr:
            return np.full_like(np.asarray(z, float), self.ro)
        p, d = self.corr
        return self.ro - d * 0.5 * (1 - np.cos(2 * np.pi * (np.asarray(z, float) - self.z1) / p))

    def triangles(self):
        t = np.linspace(0, 2 * np.pi, self.n + 1)
        ct, st = np.cos(t), np.sin(t)
        if self.corr:
            nz = max(2, int((self.z1 - self.z0) / self.corr[0] * 8))
        else:
            nz = 1
        zs = np.linspace(self.z0, self.z1, nz + 1)
        rs = self._r_out(zs)
        tris, nrms = [], []

        def add(a, b, c, n):
            tr = np.array([a, b, c])
            if np.dot(np.cross(tr[1] - tr[0], tr[2] - tr[0]), n) < 0:
                tr = tr[[0, 2, 1]]
            tris.append(tr)
            nrms.append(n)

        for k in range(nz):
            dr = (rs[k + 1] - rs[k]) / (zs[k + 1] - zs[k])
            for i in range(self.n):
                P = lambda r, z, ii: np.array([self.cx + r * ct[ii], self.cy + r * st[ii], z])  # noqa: E731
                a, b = P(rs[k], zs[k], i), P(rs[k], zs[k], i + 1)
                c, d = P(rs[k + 1], zs[k + 1], i + 1), P(rs[k + 1], zs[k + 1], i)
                tm = 0.5 * (t[i] + t[i + 1])
                n = np.array([math.cos(tm), math.sin(tm), -dr])
                n /= np.linalg.norm(n)
                add(a, b, c, n)
                add(a, c, d, n)
        if self.ri > 0:
            for i in range(self.n):
                P = lambda r, z, ii: np.array([self.cx + r * ct[ii], self.cy + r * st[ii], z])  # noqa: E731
                a, b = P(self.ri, self.z0, i), P(self.ri, self.z0, i + 1)
                c, d = P(self.ri, self.z1, i + 1), P(self.ri, self.z1, i)
                tm = 0.5 * (t[i] + t[i + 1])
                n = -np.array([math.cos(tm), math.sin(tm), 0.0])
                add(a, b, c, n)
                add(a, c, d, n)
        for z, s, r_o in ((self.z1, 1.0, rs[-1]), (self.z0, -1.0, rs[0])):
            n = unit(2, s)
            for i in range(self.n):
                P = lambda r, ii: np.array([self.cx + r * ct[ii], self.cy + r * st[ii], z])  # noqa: E731
                if self.ri > 0:
                    add(P(self.ri, i), P(r_o, i), P(r_o, i + 1), n)
                    add(P(self.ri, i), P(r_o, i + 1), P(self.ri, i + 1), n)
                else:
                    add(np.array([self.cx, self.cy, z]), P(r_o, i), P(r_o, i + 1), n)
        return np.array(tris), np.array(nrms)

    def section(self, axis, value):
        axis = ax(axis)
        if axis == 2:
            if self.z0 + EPS < value < self.z1 - EPS:
                outer = self._ring(float(self._r_out(value)), value)[:, :2]
                holes = [self._ring(self.ri, value)[:, :2]] if self.ri > 0 else []
                return [Section(outer, holes)]
            return []
        c = self.cx if axis == 0 else self.cy
        o = self.cy if axis == 0 else self.cx
        d = value - c
        if abs(d) >= self.ro:
            return []
        zs = np.linspace(self.z0, self.z1, max(2, int((self.z1 - self.z0) / (self.corr[0] / 6))) + 1) \
            if self.corr else np.array([self.z0, self.z1])
        ho = np.sqrt(np.maximum(self._r_out(zs) ** 2 - d * d, 0.0))
        hi = math.sqrt(self.ri ** 2 - d * d) if abs(d) < self.ri else None
        out = []
        if hi is None:
            left = np.stack([o - ho, zs], 1)
            right = np.stack([o + ho, zs], 1)[::-1]
            out.append(Section(ccw(np.vstack([left, right]))))
        else:
            for s in (-1, 1):
                outer = np.stack([o + s * ho, zs], 1)
                inner = np.array([[o + s * hi, self.z1], [o + s * hi, self.z0]])
                out.append(Section(ccw(np.vstack([outer, inner]))))
        return out


class Plate(Geometry):
    """Horizontal plate: rectangular or elliptical outline, optional elliptical hole."""

    smooth_axis = 2

    def __init__(self, outer, z0, z1, hole=None, segments=64):
        # outer: ('rect', x0, y0, x1, y1) | ('ellipse', cx, cy, rx, ry); hole: ('ellipse', cx, cy, rx, ry)
        self.outer, self.hole = outer, hole
        self.z0, self.z1 = min(z0, z1), max(z0, z1)
        self.n = segments

    # -- sampled boundaries at shared angles about the hole (or outline) centre
    def _center(self):
        if self.hole:
            return self.hole[1], self.hole[2]
        if self.outer[0] == "ellipse":
            return self.outer[1], self.outer[2]
        _, x0, y0, x1, y1 = self.outer
        return 0.5 * (x0 + x1), 0.5 * (y0 + y1)

    def _angles(self):
        t = list(np.linspace(0, 2 * np.pi, self.n, endpoint=False))
        if self.outer[0] == "rect":
            cx, cy = self._center()
            _, x0, y0, x1, y1 = self.outer
            for px, py in ((x1, y1), (x0, y1), (x0, y0), (x1, y0)):
                t.append(math.atan2(py - cy, px - cx) % (2 * np.pi))
        return np.array(sorted(set(np.round(t, 12))))

    def _outer_pt(self, th):
        cx, cy = self._center()
        c, s = math.cos(th), math.sin(th)
        if self.outer[0] == "ellipse":
            _, ex, ey, rx, ry = self.outer
            if (ex, ey) == (cx, cy):
                r = 1.0 / math.sqrt((c / rx) ** 2 + (s / ry) ** 2)
                return np.array([cx + r * c, cy + r * s])
            # ray/ellipse intersection from an off-centre point
            dx, dy = cx - ex, cy - ey
            A = (c / rx) ** 2 + (s / ry) ** 2
            B = 2 * (dx * c / rx ** 2 + dy * s / ry ** 2)
            C = (dx / rx) ** 2 + (dy / ry) ** 2 - 1
            r = (-B + math.sqrt(max(B * B - 4 * A * C, 0))) / (2 * A)
            return np.array([cx + r * c, cy + r * s])
        _, x0, y0, x1, y1 = self.outer
        rs = []
        if abs(c) > 1e-12:
            rs.append(((x1 if c > 0 else x0) - cx) / c)
        if abs(s) > 1e-12:
            rs.append(((y1 if s > 0 else y0) - cy) / s)
        r = min(r for r in rs if r > 0)
        return np.array([cx + r * c, cy + r * s])

    def _hole_pt(self, th):
        _, cx, cy, rx, ry = self.hole
        return np.array([cx + rx * math.cos(th), cy + ry * math.sin(th)])

    def outline(self):
        if self.outer[0] == "rect":
            _, x0, y0, x1, y1 = self.outer
            return np.array(rect(x0, y0, x1, y1), float)
        return np.array([self._outer_pt(t) for t in np.linspace(0, 2 * np.pi, self.n, endpoint=False)])

    def hole_outline(self):
        if not self.hole:
            return None
        return np.array([self._hole_pt(t) for t in np.linspace(0, 2 * np.pi, self.n, endpoint=False)])

    def faces(self):
        o2 = self.outline()
        h2 = self.hole_outline()
        fs = []
        for z, s in ((self.z1, 1.0), (self.z0, -1.0)):
            o3 = lift(o2, 2, z)
            holes = [lift(h2, 2, z)[::-1]] if h2 is not None else []
            fs.append(Face(o3 if s > 0 else o3[::-1].copy(), unit(2, s), holes))
        n = len(o2)
        for i in range(n):
            p, q = o2[i], o2[(i + 1) % n]
            d = q - p
            L = math.hypot(*d)
            if L < EPS:
                continue
            nrm = np.array([d[1] / L, -d[0] / L, 0.0])
            fs.append(Face(np.array([[*p, self.z0], [*q, self.z0], [*q, self.z1], [*p, self.z1]]), nrm))
        return fs

    def bbox(self):
        o = self.outline()
        return (np.array([*o.min(0), self.z0]), np.array([*o.max(0), self.z1]))

    def triangles(self):
        th = self._angles()
        th = np.append(th, th[0] + 2 * np.pi)
        tris, nrms = [], []

        def add(a, b, c, n):
            tr = np.array([a, b, c], float)
            if np.dot(np.cross(tr[1] - tr[0], tr[2] - tr[0]), n) < 0:
                tr = tr[[0, 2, 1]]
            tris.append(tr)
            nrms.append(n)

        cx, cy = self._center()
        for z, s in ((self.z1, 1.0), (self.z0, -1.0)):
            n = unit(2, s)
            for i in range(len(th) - 1):
                o0, o1 = self._outer_pt(th[i]), self._outer_pt(th[i + 1])
                if self.hole:
                    h0, h1 = self._hole_pt(th[i]), self._hole_pt(th[i + 1])
                    add([*h0, z], [*o0, z], [*o1, z], n)
                    add([*h0, z], [*o1, z], [*h1, z], n)
                else:
                    add([cx, cy, z], [*o0, z], [*o1, z], n)
        # outer wall
        for i in range(len(th) - 1):
            o0, o1 = self._outer_pt(th[i]), self._outer_pt(th[i + 1])
            d = o1 - o0
            L = math.hypot(*d)
            if L < EPS:
                continue
            n = np.array([d[1] / L, -d[0] / L, 0.0])
            add([*o0, self.z0], [*o1, self.z0], [*o1, self.z1], n)
            add([*o0, self.z0], [*o1, self.z1], [*o0, self.z1], n)
            if self.hole:
                h0, h1 = self._hole_pt(th[i]), self._hole_pt(th[i + 1])
                d = h1 - h0
                L = math.hypot(*d)
                if L > EPS:
                    n = -np.array([d[1] / L, -d[0] / L, 0.0])
                    add([*h0, self.z0], [*h1, self.z0], [*h1, self.z1], n)
                    add([*h0, self.z0], [*h1, self.z1], [*h0, self.z1], n)
        return np.array(tris), np.array(nrms)

    def section(self, axis, value):
        axis = ax(axis)
        if axis == 2:
            if self.z0 + EPS < value < self.z1 - EPS:
                h = self.hole_outline()
                return [Section(self.outline(), [h] if h is not None else [])]
            return []
        k = axis  # coordinate index within (x, y)
        ivs = line_intervals(self.outline(), k, value)
        if self.hole:
            hiv = line_intervals(self.hole_outline(), k, value)
            res = []
            for a, b in ivs:
                segs = [(a, b)]
                for c, d in hiv:
                    nxt = []
                    for s0, s1 in segs:
                        if d <= s0 or c >= s1:
                            nxt.append((s0, s1))
                        else:
                            if c > s0:
                                nxt.append((s0, c))
                            if d < s1:
                                nxt.append((d, s1))
                    segs = nxt
                res += segs
            ivs = res
        return [Section(np.array(rect(a, self.z0, b, self.z1), float)) for a, b in ivs if b - a > EPS]


class Clipped(Geometry):
    """A geometry restricted to the half-space keep*(p[axis]-value) <= 0."""

    def __init__(self, base: Geometry, axis: int, value: float, keep: int):
        self.base, self.axis, self.value, self.keep = base, ax(axis), value, keep
        self.smooth_axis = base.smooth_axis

    def faces(self):
        out = []
        for f in self.base.faces():
            p = clip_poly3(f.pts, self.axis, self.value, self.keep)
            if len(p) < 3:
                continue
            holes = [h2 for h in f.holes if len(h2 := clip_poly3(h, self.axis, self.value, self.keep)) >= 3]
            out.append(Face(p, f.normal, holes))
        return out

    def triangles(self):
        t, n = self.base.triangles()
        return clip_triangles(t, n, self.axis, self.value, self.keep)

    def lines(self):
        out = []
        for a, b in self.base.lines():
            da, db = self.keep * (a[self.axis] - self.value), self.keep * (b[self.axis] - self.value)
            if da <= 0 and db <= 0:
                out.append((a, b))
            elif da * db < 0:
                t = da / (da - db)
                m = a + t * (b - a)
                out.append((a, m) if da <= 0 else (m, b))
        return out

    def bbox(self):
        lo, hi = self.base.bbox()
        lo, hi = lo.copy(), hi.copy()
        if self.keep > 0:
            hi[self.axis] = min(hi[self.axis], self.value)
        else:
            lo[self.axis] = max(lo[self.axis], self.value)
        return lo, hi


def clip(geom: Geometry, axis: int, value: float, keep: int):
    """Return geom clipped to keep*(p[axis]-value) <= 0, or None if nothing remains."""
    lo, hi = geom.bbox()
    if keep * (lo[axis] - value) > -EPS and keep * (hi[axis] - value) > -EPS:
        return None
    if keep * (lo[axis] - value) <= EPS and keep * (hi[axis] - value) <= EPS:
        return geom
    # exact fast paths
    if type(geom) is Tube and axis == 2:
        z0, z1 = (geom.z0, value) if keep > 0 else (value, geom.z1)
        return Tube(geom.cx, geom.cy, geom.ro, geom.ri, z0, z1, geom.n, geom.corr)
    if type(geom) is Prism:
        if axis == geom.axis:
            a, b = (geom.lo, value) if keep > 0 else (value, geom.hi)
            return Prism(axis, geom.poly, a, b)
        k = PLANE[geom.axis].index(axis)
        p3 = np.column_stack([geom.poly, np.zeros(len(geom.poly))])
        p = clip_poly3(p3, k, value, keep)
        return Prism(geom.axis, p[:, :2], geom.lo, geom.hi) if len(p) >= 3 else None
    return Clipped(geom, axis, value, keep)


class Translated(Geometry):
    """A geometry shifted by (dx, dy, dz) - used to show the privy at staged positions."""

    def __init__(self, base: Geometry, d):
        self.base = base
        self.d = np.asarray(d, float)
        self.smooth_axis = base.smooth_axis

    def faces(self):
        return [Face(f.pts + self.d, f.normal, [h + self.d for h in f.holes]) for f in self.base.faces()]

    def triangles(self):
        t, n = self.base.triangles()
        return t + self.d, n

    def lines(self):
        return [(a + self.d, b + self.d) for a, b in self.base.lines()]

    def bbox(self):
        lo, hi = self.base.bbox()
        return lo + self.d, hi + self.d

    def section(self, axis, value):
        axis = ax(axis)
        out = []
        a1, a2 = PLANE[axis]
        for s in self.base.section(axis, value - self.d[axis]):
            off = np.array([self.d[a1], self.d[a2]])
            out.append(Section(s.outer + off, [h + off for h in s.holes]))
        return out
