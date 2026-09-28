"""Mitsuba 3 scene assembly, rendering and tone mapping for the photoreal views.

Conventions
  * Model units (inches, z up) are used directly as Mitsuba world units.
  * ``sunsky`` in Mitsuba 3.9 is already +Z-up (radiance below the xy plane is
    zero and ``sun_direction`` points toward the sun), so no ``to_world``
    rotation is applied.  The sun direction is
    (cos el cos az, cos el sin az, sin el) with azimuth measured in plan from
    +x counter-clockwise toward +y.
  * UVs are box-mapped per triangle: the projection plane is the one most
    facing the triangle normal; u runs along the part's grain axis when that
    axis lies in the plane (otherwise along the first plane axis), v along the
    other.  u = coord / tile[0] + offset, v = coord / tile[1] + offset.
  * A material's optional ``side_texture`` replaces ``texture`` on faces whose
    normal is not upward (n_z <= 0.5), e.g. litter on top of the pad, clay in section.
"""
from __future__ import annotations

import math
import os
import tempfile
import time

import numpy as np
from PIL import Image

from ..geometry import EPS, PLANE, Clipped, CorrugatedSheet, ccw, clip_triangles, ear_clip, lift
from . import textures as T
from .trees import place_trees, tree_geometry

N_VARIANTS = 4
MAX_LANES = 1 << 24          # samples per render pass (bounds JIT wavefront memory)
MIN_PASSES = 4               # passes are also used to reject single-pass firefly outliers
SURROUND_LIFT = 0.05         # forest floor sits just above the pad top so its ragged edge can overlap it
HORIZON_SCALE = 200          # grass skirt beyond the textured square, so the horizon is not black
CUT_NUDGE = 1e-3             # cut planes through exact vertices (e.g. hole centres) are degenerate
EXPOSURE_BASE_EV = 1.75      # sunsky radiance -> display calibration: exposure_ev 0 ~ "sunny 16"
LUMA = np.array([0.2126, 0.7152, 0.0722])

_mi = None


def mitsuba():
    """Import Mitsuba once, preferring the vectorised LLVM variant."""
    global _mi
    if _mi is None:
        import mitsuba as mi
        for v in ("llvm_ad_rgb", "scalar_rgb"):
            try:
                mi.set_variant(v)
                break
            except (ImportError, AttributeError, ValueError, RuntimeError):
                continue
        mi.set_log_level(mi.LogLevel.Error)
        _mi = mi
    return _mi


# --------------------------------------------------------------------------- materials

class Materials:
    """Material definitions -> Mitsuba BSDF dicts (cached per key/variant/mode).

    mode "" = as defined, "side" = with ``side_texture``, "detile" = two differently scaled
    texture copies blended by a large-scale mask (for the unbounded surround plane).
    """

    def __init__(self, defs: dict, tmp: str, log=print):
        self.defs, self.tmp, self.log = defs, tmp, log
        self._cache: dict = {}
        self._warned: set = set()

    def get(self, key) -> dict:
        m = self.defs.get(key)
        if m is None and key not in self._warned:
            self._warned.add(key)
            self.log(f"render: warning: material {key!r} not defined, using neutral grey")
        return m or {}

    def variant(self, key, ident: str) -> int:
        m = self.defs.get(key) or {}
        return T.stable_hash(ident) % N_VARIANTS if float(m.get("variation", 0)) > 0 else 0

    def tile(self, key, mode=""):
        m = self.defs.get(key) or {}
        return T.tex_tile(T.side_material(m) if mode == "side" else m)

    def bsdf(self, key, variant=0, mode="") -> dict:
        ck = (key, variant, mode)
        if ck not in self._cache:
            m = self.get(key)
            name = f"{key}_{variant}{mode}"
            if mode == "side":
                self._cache[ck] = self._make(T.side_material(m), name, variant)
            elif mode == "detile" and m.get("texture"):
                mi = mitsuba()
                path = self._save(T.blend_mask(int(m["texture"].get("seed", 0))), f"mask_{key}")
                weight = {"type": "bitmap", "filename": path, "raw": True,
                          "to_uv": mi.ScalarTransform3f().rotate(-21).scale([0.23, 0.23])}
                alt = mi.ScalarTransform3f().rotate(37).scale([0.618, 0.618])
                self._cache[ck] = {"type": "blendbsdf", "weight": weight, "bsdf_0": self._make(m, name, variant),
                                   "bsdf_1": self._make(m, name + "b", variant + 1, alt)}
            else:
                self._cache[ck] = self._make(m, name, variant)
        return self._cache[ck]

    def _save(self, img, name):
        path = os.path.join(self.tmp, f"{name}.png")
        if not os.path.exists(path):
            Image.fromarray(img).save(path)
        return path

    def _make(self, m, name, variant, to_uv=None):
        if not m:
            return {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.4, 0.4, 0.4]}}
        if m.get("type") == "glass":
            tint = T.srgb_to_linear(T.hex_srgb(m.get("tint", "#ffffff")))
            return {"type": "thindielectric", "int_ior": float(m.get("ior", 1.5)), "ext_ior": 1.0,
                    "specular_transmittance": {"type": "rgb", "value": tint.tolist()}}
        img = T.generate(m, variant)
        if img is not None:
            color = {"type": "bitmap", "filename": self._save(img, f"tex_{name}"), "raw": False,
                     "filter_type": "bilinear"}
            if to_uv is not None:
                color["to_uv"] = to_uv
        else:
            seed = int((m.get("texture") or {}).get("seed", T.stable_hash(name) % 1000))
            lin = T.srgb_to_linear(T.hex_srgb(m.get("base_color", "#808080")))
            lin = np.clip(lin * T.variant_tint(float(m.get("variation", 0)), variant, seed), 0, 1)
            color = {"type": "rgb", "value": lin.tolist()}
        b = {"type": "principled", "base_color": color,
             "roughness": float(m.get("roughness", 0.5)),
             "metallic": float(m.get("metallic", 0.0)),
             "specular": float(m.get("specular", 0.5))}
        if m.get("two_sided"):
            b = {"type": "twosided", "bsdf": b}
        return b


# --------------------------------------------------------------------------- geometry helpers

def box_uv(tris, normals, grain, tile, offset=(0.0, 0.0)):
    """Per-vertex (n, 3, 2) box-mapped UVs; u follows ``grain`` when it lies in the projection plane."""
    k = np.argmax(np.abs(normals), axis=1)
    a1, a2 = np.array([1, 0, 0])[k], np.array([2, 2, 1])[k]
    ua = np.where(a2 == grain, a2, a1)
    va = np.where(ua == a1, a2, a1)
    pick = lambda a: np.take_along_axis(tris, np.repeat(a[:, None, None], 3, 1), 2)[..., 0]  # noqa: E731
    return np.stack([pick(ua) / tile[0] + offset[0], pick(va) / tile[1] + offset[1]], -1)


def part_offset(pid: str):
    r = np.random.default_rng(T.stable_hash(pid))
    return tuple(r.uniform(0, 1, 2))


def _segments_cross(a, b, P, Q):
    """True where segment a-b properly crosses any of segments P[i]-Q[i]."""
    def orient(p, q, r):
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])
    d1, d2 = orient(a, b, P), orient(a, b, Q)
    d3, d4 = orient(P, Q, a), orient(P, Q, b)
    return bool(np.any((d1 * d2 < -1e-12) & (d3 * d4 < -1e-12)))


def bridge_holes(outer, holes):
    """Merge holes into the outer loop with keyhole bridges so ear_clip can triangulate it."""
    poly = ccw(outer)
    holes = [ccw(h)[::-1] for h in holes]                    # holes clockwise
    for h in sorted(holes, key=lambda h: -h[:, 0].max()):
        i = int(np.argmax(h[:, 0]))
        loops = [poly] + holes
        P = np.vstack(loops)
        Q = np.vstack([np.roll(lp, -1, 0) for lp in loops])
        for j in np.argsort(np.hypot(*(poly - h[i]).T)):
            if not _segments_cross(h[i], poly[j], P, Q):
                break
        hr = np.roll(h, -i, 0)
        poly = np.vstack([poly[:j + 1], hr, hr[:1], poly[j:]])
    return poly


def cap_triangles(geom, axis, value, keep):
    """Triangulated cut faces of ``geom`` on plane axis=value, facing the removed side."""
    nrm = np.zeros(3)
    nrm[axis] = keep
    out = []
    for s in geom.section(axis, value):
        poly = bridge_holes(s.outer, s.holes) if s.holes else ccw(s.outer)
        p3 = lift(poly, axis, value)
        for tri in ear_clip(poly):
            t = p3[list(tri)]
            if np.dot(np.cross(t[1] - t[0], t[2] - t[0]), nrm) < 0:
                t = t[[0, 2, 1]]
            out.append(t)
    if not out:
        return np.zeros((0, 3, 3)), np.zeros((0, 3))
    return np.array(out), np.tile(nrm, (len(out), 1))


def is_sheet(geom) -> bool:
    """Open surfaces (no volume): never capped in a cutaway."""
    while isinstance(geom, Clipped):
        geom = geom.base
    return isinstance(geom, CorrugatedSheet)


def cut_part(geom, tris, normals, cut):
    """Clip a part's triangles to the kept half-space and add cap faces; cut = (axis, value, keep)."""
    axis, value, keep = cut
    d = keep * (tris[:, :, axis] - value)
    inside, outside = (d <= EPS).all(1), (d > -EPS).all(1)
    straddle = ~inside & ~outside
    ts, ns = [tris[inside]], [normals[inside]]
    if straddle.any():
        ct, cn = clip_triangles(tris[straddle], normals[straddle], axis, value, keep)
        ts.append(ct)
        ns.append(cn)
    if d.min() < -EPS and d.max() > EPS and not is_sheet(geom):
        ct, cn = cap_triangles(geom, axis, value, keep)
        ts.append(ct)
        ns.append(cn)
    return np.concatenate(ts), np.concatenate(ns)


def rect_minus(outer, inner):
    """Rectangle ``outer`` minus rectangle ``inner`` (u0, v0, u1, v1) as up to 4 rectangles."""
    if inner is None:
        return [outer]
    x0, y0, x1, y1 = inner
    X0, Y0, X1, Y1 = outer
    parts = [(X0, Y0, X1, y0), (X0, y1, X1, Y1), (X0, y0, x0, y1), (x1, y0, X1, y1)]
    return [(a, b, max(a, min(c, X1)), max(b, min(e, Y1))) for a, b, c, e in parts]


def plane_rects(rects, axis, value, normal_sign):
    """Triangles for 2-D rectangles (PLANE[axis] coords) on plane axis=value, facing normal_sign."""
    nrm = np.zeros(3)
    nrm[axis] = normal_sign
    tris = []
    for a, b, c, e in rects:
        if c - a > EPS and e - b > EPS:
            p = lift(np.array([[a, b], [c, b], [c, e], [a, e]]), axis, value)
            for t in (p[[0, 1, 2]], p[[0, 2, 3]]):
                tris.append(t if np.dot(np.cross(t[1] - t[0], t[2] - t[0]), nrm) > 0 else t[[0, 2, 1]])
    return np.array(tris).reshape(-1, 3, 3), np.tile(nrm, (len(tris), 1))


def ragged_ring(rect, inset, outset, z, seed, n=160):
    """Upward band from a wavy loop ~``inset`` inside ``rect`` out to ``rect`` grown by ``outset``."""
    x0, y0, x1, y1 = rect
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    X0, Y0, X1, Y1 = x0 - outset, y0 - outset, x1 + outset, y1 + outset
    corners = [math.atan2(py - cy, px - cx) % (2 * np.pi) for px, py in ((X0, Y0), (X1, Y0), (X1, Y1), (X0, Y1))]
    th = np.unique(np.concatenate([np.linspace(0, 2 * np.pi, n, endpoint=False), corners]))
    rng = np.random.default_rng(seed)
    k = np.arange(2, 16)
    wav = ((rng.uniform(0.4, 1, len(k)) / k)[:, None] * np.sin(k[:, None] * th + rng.uniform(0, 7, len(k))[:, None])).sum(0)
    wav /= np.abs(wav).max()
    c, s_ = np.cos(th), np.sin(th)

    def ray(a0, b0, a1, b1):          # distance from the centre to the rectangle boundary along th
        with np.errstate(divide="ignore"):
            tx = np.where(c > 1e-12, (a1 - cx) / c, np.where(c < -1e-12, (a0 - cx) / c, np.inf))
            ty = np.where(s_ > 1e-12, (b1 - cy) / s_, np.where(s_ < -1e-12, (b0 - cy) / s_, np.inf))
        return np.minimum(tx, ty)

    ro = ray(X0, Y0, X1, Y1)
    ri = np.maximum(ray(x0, y0, x1, y1) - inset * (1 + 0.6 * wav), 0.3 * ro)
    outer = np.stack([cx + ro * c, cy + ro * s_, np.full_like(th, z)], 1)
    inner = np.stack([cx + ri * c, cy + ri * s_, np.full_like(th, z)], 1)
    o1, i1 = np.roll(outer, -1, 0), np.roll(inner, -1, 0)
    tris = np.concatenate([np.stack([inner, outer, o1], 1), np.stack([inner, o1, i1], 1)])
    up = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])[:, 2] > 0
    tris[~up] = tris[~up][:, [0, 2, 1]]
    return tris, np.tile([0.0, 0.0, 1.0], (len(tris), 1)), (X0, Y0, X1, Y1)


def write_ply(path, tris, normals, uv):
    """Binary PLY with unshared vertices: position, normal (per triangle or per vertex), texcoord."""
    n = len(tris)
    names = "x y z nx ny nz u v".split()
    v = np.zeros(3 * n, dtype=[(c, "<f4") for c in names])
    vn = normals.reshape(-1, 3) if normals.ndim == 3 else np.repeat(normals, 3, 0)
    cols = np.hstack([tris.reshape(-1, 3), vn, uv.reshape(-1, 2)])
    for i, c in enumerate(names):
        v[c] = cols[:, i]
    f = np.zeros(n, dtype=[("n", "u1"), ("i", "<i4", (3,))])
    f["n"], f["i"] = 3, np.arange(3 * n).reshape(n, 3)
    head = ("ply\nformat binary_little_endian 1.0\n"
            f"element vertex {3 * n}\n" + "".join(f"property float {c}\n" for c in names)
            + f"element face {n}\nproperty list uchar int vertex_indices\nend_header\n")
    with open(path, "wb") as fh:
        fh.write(head.encode())
        fh.write(v.tobytes())
        fh.write(f.tobytes())


# --------------------------------------------------------------------------- scene

def resolve_cut(view, model):
    c = view.get("cutaway")
    if not c:
        return None
    axis = "xyz".index(c["axis"]) if isinstance(c["axis"], str) else int(c["axis"])
    value = c["value"]
    if value == "pipe_center":
        value = model.key[{0: "pipe_cx", 1: "pipe_cy"}[axis]]
    keep = -1 if c.get("keep", "high") == "high" else 1      # clip keeps keep*(p-value) <= 0
    return axis, float(value) + CUT_NUDGE, keep


def camera(view, model, defaults):
    cam = view.get("camera", {})
    fov = float(cam.get("fov_deg", defaults.get("fov_deg", 36)))
    if "orbit" in cam:
        o = cam["orbit"]
        lo, hi = model.bbox()
        target = (lo + hi) / 2 + np.asarray(cam.get("target_offset", [0, 0, 0]), float)
        d = 0.5 * np.linalg.norm(hi - lo) / math.sin(math.radians(fov) / 2) * float(o.get("distance_scale", 1.0))
        az, el = math.radians(o.get("azimuth_deg", 45)), math.radians(o.get("elevation_deg", 15))
        pos = target + d * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    else:
        pos, target = np.asarray(cam["position"], float), np.asarray(cam["target"], float)
    return pos, target, fov


def sun_direction(lighting, cut=None):
    """Unit vector toward the sun.  For a vertical cutaway whose cut face would be in shade,
    the sun is mirrored across the cut plane so the section is lit (same elevation)."""
    s = lighting.get("sun", {})
    az, el = math.radians(s.get("azimuth_deg", 45)), math.radians(s.get("elevation_deg", 40))
    d = [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
    if cut and cut[0] != 2 and d[cut[0]] * cut[2] < 0:     # cap normal is keep * e_axis
        d[cut[0]] = -d[cut[0]]
    return d


def build_scene(model, view, mats: Materials, tmp: str, size, spp, trees=()):
    """Mitsuba scene dict for one view; meshes are written as PLYs into ``tmp``."""
    mi = mitsuba()
    cfg = model.cfg
    rcfg = cfg.get("renders", {})
    defaults = rcfg.get("defaults", {})
    env = rcfg.get("environment", {})
    hide = set(view.get("hide_tags", []))
    cut = resolve_cut(view, model)
    groups: dict = {}

    def add(key, variant, tris, normals, uv=None, grain=0, offset=(0.0, 0.0), mode=""):
        """Queue triangles for material key/variant; plain faces split into top / side textures."""
        if not len(tris):
            return
        if mode == "" and mats.get(key).get("side_texture"):
            side = normals[:, 2] <= 0.5
            if side.any():
                add(key, variant, tris[side], normals[side], None, grain, offset, "side")
            tris, normals = tris[~side], normals[~side]
            if not len(tris):
                return
        if uv is None:
            uv = box_uv(tris, normals, grain, mats.tile(key, mode), offset)
        groups.setdefault((key, variant, mode), []).append((tris, normals, uv))

    ground = []
    for p in model.parts:
        if p.tags & hide:
            continue
        try:
            tris, nrm = p.geom.triangles()
        except Exception as e:  # noqa: BLE001 - a bad part must not kill the whole render
            mats.log(f"render: warning: skipping part {p.id}: {e}")
            continue
        if not len(tris):
            continue
        if "ground" in p.tags:
            ground.append(p.bbox())
        if cut:
            tris, nrm = cut_part(p.geom, tris, nrm, cut)
        add(p.material, mats.variant(p.material, p.id), tris, nrm, grain=p.grain(),
            offset=(0.0, 0.0) if "ground" in p.tags else part_offset(p.id))

    # surround (forest floor / grass): ragged edge over the pad, textured square, far skirt to the horizon
    skey, size_in = env.get("surround", {}).get("material", "grass"), float(env.get("surround", {}).get("size", 6000))
    lo, hi = model.bbox(exclude_tags=())
    c = (lo + hi) / 2
    sq = lambda s, i=0, j=1: (c[i] - s / 2, c[j] - s / 2, c[i] + s / 2, c[j] + s / 2)  # noqa: E731
    glo = ghi = hole = None
    z_s = SURROUND_LIFT
    parts = []
    if ground:
        glo, ghi = np.min([b[0] for b in ground], 0), np.max([b[1] for b in ground], 0)
        z_s = ghi[2] + SURROUND_LIFT
        margin = float(env.get("dirt_pad", {}).get("margin", 48))
        t, n, hole = ragged_ring((glo[0], glo[1], ghi[0], ghi[1]), 0.55 * margin, 1.0, z_s, 11)
        parts.append((t, n))
    parts.append(plane_rects(rect_minus(sq(size_in), hole) + rect_minus(sq(size_in * HORIZON_SCALE), sq(size_in)),
                             2, z_s, 1))
    tris, nrm = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    if cut:
        tris, nrm = clip_triangles(tris, nrm, *cut)
    add(skey, 0, tris, nrm, mode="detile")
    if cut and cut[0] != 2:
        # earth below the surround, seen in section: everything outside the ground parts' own caps
        axis, value, keep = cut
        h = 1 - axis
        H0, _, H1, _ = sq(size_in * HORIZON_SCALE, h, h)
        box = (glo[h], glo[2], ghi[h], z_s) if ground else None
        tris, nrm = plane_rects(rect_minus((H0, -size_in / 2, H1, z_s), box), axis, value, keep)
        add(env.get("dirt_pad", {}).get("material", "soil"), 0, tris, nrm)

    scene = {"type": "scene"}

    def bsdf_ref(key, variant, mode):
        bid = f"bsdf_{key}_{variant}_{mode or 'plain'}"
        if bid not in scene:
            scene[bid] = mats.bsdf(key, variant, mode)
        return {"type": "ref", "id": bid}

    # trees: smooth-shaded mesh groups per bark / foliage material variant
    for t in trees:
        if cut and (cut[0] == 2 or cut[2] * ((t.x, t.y)[cut[0]] - cut[1]) > 0):
            continue                                       # trunk on the removed side of the cut
        bkey = t.sp.get("bark_material", "pine_bark")
        fkey = t.sp.get("foliage_material", "pine_needles")
        bark, foliage = tree_geometry(t, mats.tile(bkey))
        for key, meshes in ((bkey, bark), (fkey, foliage)):
            for tr, nr, uv in meshes:
                add(key, mats.variant(key, f"tree{t.seed}"), tr, nr, uv, mode="smooth")

    for i, ((key, variant, mode), chunks) in enumerate(sorted(groups.items())):
        path = os.path.join(tmp, f"mesh_{i:03d}.ply")
        write_ply(path, *(np.concatenate(c) for c in zip(*chunks)))
        scene[f"mesh_{i:03d}"] = {"type": "ply", "filename": path, "face_normals": mode != "smooth",
                                  "bsdf": bsdf_ref(key, variant, "" if mode == "smooth" else mode)}

    light = cfg.get("lighting", {})
    sky = light.get("sky", {})
    scene["sky"] = {"type": "sunsky", "sun_direction": sun_direction(light, cut),
                    "turbidity": float(sky.get("turbidity", 3.0)),
                    "albedo": float(sky.get("ground_albedo", 0.3)),
                    "sun_scale": float(light.get("sun", {}).get("scale", 1.0)),
                    "sky_scale": float(sky.get("scale", 1.0))}
    scene["integrator"] = {"type": "path", "max_depth": int(defaults.get("max_depth", 8))}
    pos, target, fov = camera(view, model, defaults)
    w, h = size
    scene["camera"] = {
        "type": "perspective", "fov": fov, "fov_axis": "x", "near_clip": 1.0, "far_clip": 1e8,
        "to_world": mi.ScalarTransform4f().look_at(origin=pos.tolist(), target=target.tolist(), up=[0, 0, 1]),
        "sampler": {"type": "multijitter", "sample_count": spp},
        "film": {"type": "hdrfilm", "width": w, "height": h, "pixel_format": "rgb",
                 "rfilter": {"type": "gaussian"}},
    }
    return scene


def view_size(view, defaults, quality=1.0):
    w = max(16, int(round(view.get("width", defaults.get("width", 1800)) * quality)))
    h = max(16, int(round(view.get("height", defaults.get("height", 1200)) * quality)))
    return w, h


def site_trees(model, log=print):
    """Trees for the site, placed once for all configured views (see trees.py)."""
    rcfg = model.cfg.get("renders", {})
    tcfg = rcfg.get("environment", {}).get("trees")
    if not tcfg:
        return []
    defaults = rcfg.get("defaults", {})
    sights = []
    for v in rcfg.get("views", []):
        pos, tgt, fov = camera(v, model, defaults)
        w, h = view_size(v, defaults)
        tan_h = math.tan(math.radians(fov) / 2)
        sights.append((pos, tgt, tan_h * h / w, tan_h))
    lo, hi = model.bbox()
    trees = place_trees(tcfg, (lo + hi) / 2, sights)
    log(f"render: placed {len(trees)} trees")
    return trees


# --------------------------------------------------------------------------- output

ACES_IN = np.array([[0.59719, 0.35458, 0.04823], [0.07600, 0.90834, 0.01566], [0.02840, 0.13383, 0.83777]])
ACES_OUT = np.array([[1.60475, -0.53108, -0.07367], [-0.10208, 1.10813, -0.00605], [-0.00327, -0.07276, 1.07602]])


def aces(x):
    """ACES filmic (Stephen Hill's RRT+ODT fit with the sRGB <-> ACES matrices); linear sRGB in/out."""
    v = x @ ACES_IN.T
    v = (v * (v + 0.0245786) - 0.000090537) / (v * (0.983729 * v + 0.4329510) + 0.238081)
    return np.clip(v @ ACES_OUT.T, 0, 1)


def tonemap(img, lighting):
    x = np.asarray(img, float)[..., :3] * 2.0 ** (EXPOSURE_BASE_EV + float(lighting.get("exposure_ev", 0.0)))
    x = aces(x) if lighting.get("tonemap", "aces") == "aces" else np.clip(x, 0, 1)
    return (T.linear_to_srgb(x) * 255 + 0.5).astype(np.uint8)


def render_image(scene_dict, spp, size):
    """Render in passes (bounded wavefront size) and average them.

    The brightest pass of each pixel is dropped when it is a clear outlier against
    the mean of the other passes (sun caustics through glass etc. are sampled only by
    rare BSDF paths and would otherwise leave white specks).  Returns (linear image, spp).
    """
    mi = mitsuba()
    scene = mi.load_dict(scene_dict)
    n = max(MIN_PASSES, math.ceil(spp * size[0] * size[1] / MAX_LANES))
    k = max(1, math.ceil(spp / n))
    n = max(2, math.ceil(spp / k))
    total, top, top_l = 0.0, None, None
    for seed in range(n):
        img = np.array(mi.render(scene, spp=k, seed=seed), dtype=np.float64)[..., :3]
        lum = img @ LUMA
        total = total + img
        if top is None:
            top, top_l = img, lum
        else:
            sel = lum > top_l
            top[sel], top_l[sel] = img[sel], lum[sel]
    rest = (total - top) / (n - 1)
    rest_l = rest @ LUMA
    fly = top_l - rest_l > np.maximum(3.0 * rest_l, 0.25)
    return np.where(fly[..., None], rest, total / n), n * k


def render_views(model, out_dir, names=None, quality=1.0, log=print) -> list[dict]:
    """Render the views in ``cfg['renders']['views']`` (optionally filtered by name) to PNGs.

    ``quality`` scales width, height and samples (samples never below 16).
    Returns ``[{"name", "title", "path"}]`` with paths under ``out_dir/renders/``.
    """
    rcfg = model.cfg.get("renders", {})
    defaults = rcfg.get("defaults", {})
    views = [v for v in rcfg.get("views", []) if names is None or v["name"] in names]
    os.makedirs(os.path.join(out_dir, "renders"), exist_ok=True)
    mi = mitsuba()
    results = []
    trees = site_trees(model, log)
    with tempfile.TemporaryDirectory(prefix="privy_render_") as tmp:
        mats = Materials(model.cfg.get("materials", {}), tmp, log)
        for view in views:
            t0 = time.time()
            w, h = view_size(view, defaults, quality)
            spp = max(16, int(round(view.get("samples", defaults.get("samples", 160)) * quality)))
            vdir = os.path.join(tmp, view["name"])
            os.makedirs(vdir, exist_ok=True)
            scene = build_scene(model, view, mats, vdir, (w, h), spp, trees)
            t1 = time.time()
            img, spp = render_image(scene, spp, (w, h))
            path = os.path.join(out_dir, "renders", f"{view['name']}.png")
            Image.fromarray(tonemap(img, model.cfg.get("lighting", {}))).save(path)
            log(f"render {view['name']}: {w}x{h} @ {spp} spp [{mi.variant()}] "
                f"scene {t1 - t0:.1f}s, render {time.time() - t1:.1f}s -> {path}")
            results.append({"name": view["name"], "title": view.get("title", view["name"]), "path": path})
    return results
