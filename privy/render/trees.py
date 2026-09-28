"""Procedural trees for the wooded site (``cfg['renders']['environment']['trees']``).

Placement (deterministic from ``trees.seed``):
  * ``count`` trunks per species, area-uniform in the annulus clear_radius..outer_radius
    around the model bbox centre, at least ~half a crown radius apart;
  * no trunk within ``camera_corridor`` (+ trunk radius) of any view's plan-view
    camera->target segment, and no crown within corridor + crown radius of it
    unless the crown bottom is above that view's upper frame edge there;
  * per view, one "framing" tree just outside each side of the frame (same rules
    for all other views), so every view reads as a clearing in the woods;
  * optional ``understory: {count, height, foliage_material, leaf_cluster}``: shrubs and
    saplings (leaf clumps only) between ~1.6 x clear_radius and outer_radius;
  * optional ``backdrop: {count, outer_radius}``: extra trees (species mixed by
    count) between outer_radius and backdrop.outer_radius, so the horizon reads as woods.
The same trees exist in every view (shadows stay consistent).

Geometry: trunks/limbs are tapered frusta with smooth normals and cylindrical
UVs (u along the trunk); crowns are 6-12 ellipsoid clumps, each filled on two
layers with opaque leaf-cluster blobs (optional species keys ``leaf_cluster``: blob
radius in inches, default pine 9 / oak 7, and ``crown_density``: fraction of the clump
surface covered, default 0.55).  Real gaps give dappled shade with the plain ``path``
integrator; opacity-masked shells would need ``volpath`` and render ~6x slower.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Tree:
    x: float
    y: float
    height: float
    trunk_r: float
    crown_r: float
    crown_z0: float          # crown bottom elevation
    sp: dict                 # species config
    seed: int
    lod: float = 1.0         # leaf-cluster size multiplier (backdrop trees use bigger, fewer clusters)


def _species_tree(rng, sp, x, y) -> Tree:
    t = rng.random()
    h = float(np.interp(t, [0, 1], sp.get("height", [300, 600])))
    d = float(np.interp(np.clip(t + rng.normal(0, 0.2), 0, 1), [0, 1], sp.get("trunk_diameter", [6, 12])))
    cf = float(sp.get("crown_fraction", 0.4))
    cr = {"pine": 0.4 * cf * h, "shrub": 0.55 * h}.get(sp.get("form"), 0.5 * h)
    return Tree(x, y, h, d / 2, cr, h * (1 - cf), sp, int(rng.integers(1 << 30)))


def _clear_of_views(t: Tree, sights, corridor) -> bool:
    p = np.array([t.x, t.y])
    for cam, tgt, tan_v, _ in sights:
        a, ab = cam[:2], tgt[:2] - cam[:2]
        s = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1))
        d = float(np.linalg.norm(p - (a + s * ab)))
        if d < corridor + t.trunk_r:
            return False
        z_top = cam[2] + s * (tgt[2] - cam[2]) + s * np.linalg.norm(ab) * tan_v
        if d < corridor + t.crown_r and t.crown_z0 < z_top:
            return False
    return True


def place_trees(tcfg: dict, center, sights) -> list[Tree]:
    """sights: [(camera_pos, target, tan(vertical half fov), tan(horizontal half fov))], all views."""
    rng = np.random.default_rng(int(tcfg.get("seed", 0)))
    r0, r1 = float(tcfg.get("clear_radius", 150)), float(tcfg.get("outer_radius", 1100))
    corridor = float(tcfg.get("camera_corridor", 70))
    species = tcfg.get("species", [])
    cx, cy = float(center[0]), float(center[1])
    trees: list[Tree] = []

    def ok(t):
        if math.hypot(t.x - cx, t.y - cy) < r0 or not _clear_of_views(t, sights, corridor):
            return False
        return all(math.hypot(t.x - o.x, t.y - o.y) > 0.5 * max(t.crown_r, o.crown_r) + 12 for o in trees)

    def scatter(count, ra, rb, pick):
        n = 0
        for _ in range(300 * count):
            if n >= count:
                break
            r = math.sqrt(rng.uniform(ra * ra, rb * rb))
            a = rng.uniform(0, 2 * math.pi)
            t = _species_tree(rng, pick(), cx + r * math.cos(a), cy + r * math.sin(a))
            t.lod = 1.0 if r <= r1 else 2.0
            if ok(t):
                trees.append(t)
                n += 1

    weights = np.array([max(int(sp.get("count", 0)), 0) for sp in species], float)
    for sp in species:
        scatter(int(sp.get("count", 0)), r0, r1, lambda sp=sp: sp)
    under = tcfg.get("understory")
    if under:                                                  # saplings / shrubs, away from the privy
        shrub = {"form": "shrub", "height": under.get("height", [24, 72]), "trunk_diameter": [0.5, 1.0],
                 "crown_fraction": 1.0, "foliage_material": under.get("foliage_material", "oak_leaves"),
                 "leaf_cluster": under.get("leaf_cluster", 4.0), "crown_density": 0.7}
        scatter(int(under.get("count", 0)), max(1.6 * r0, r0 + 120), r1, lambda: shrub)
    back = tcfg.get("backdrop")
    if back and species and weights.sum() > 0:                 # thinner woods out toward the horizon
        scatter(int(back.get("count", 0)), r1, float(back.get("outer_radius", 3 * r1)),
                lambda: species[rng.choice(len(species), p=weights / weights.sum())])
    for cam, tgt, _, tan_h in sights:
        if not species or weights.sum() <= 0:
            break
        v = tgt[:2] - cam[:2]
        dist, phi = float(np.linalg.norm(v)), math.atan2(v[1], v[0])
        for side in (-1, 1):
            for _ in range(40):
                sp = species[rng.choice(len(species), p=weights / weights.sum())]
                d = dist * rng.uniform(0.7, 1.5)
                t = _species_tree(rng, sp, 0, 0)
                off = math.atan(tan_h) + math.asin(min(1.0, (t.trunk_r + 4) / d)) + math.radians(rng.uniform(1, 4))
                t.x, t.y = cam[0] + d * math.cos(phi + side * off), cam[1] + d * math.sin(phi + side * off)
                if ok(t):
                    trees.append(t)
                    break
    return trees


# --------------------------------------------------------------------------- geometry

def grid_mesh(P, N, UV):
    """(m+1, n+1) grids of points / vertex normals / uvs -> outward-wound triangles."""
    m, n = P.shape[0] - 1, P.shape[1] - 1
    i, j = (a.ravel() for a in np.meshgrid(np.arange(m), np.arange(n), indexing="ij"))
    q = [(i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)]
    out = []
    for arr in (P, N, UV):
        out.append(np.concatenate([np.stack([arr[q[a]], arr[q[b]], arr[q[c]]], 1) for a, b, c in ((0, 1, 2), (0, 2, 3))]))
    tris, nrms, uvs = out
    cr = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    keep = np.linalg.norm(cr, axis=1) > 1e-9                       # drop degenerate pole triangles
    flip = np.einsum("ij,ij->i", cr, nrms.mean(1)) < 0
    for arr in (tris, nrms, uvs):
        arr[flip] = arr[flip][:, [0, 2, 1]]
    return tris[keep], nrms[keep], uvs[keep]


def frustum(p0, p1, r0, r1, tile, n=20, rings=None):
    """Open tapered tube p0->p1: (tris, per-vertex normals, uv) with u along the axis."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    axis = p1 - p0
    L = float(np.linalg.norm(axis))
    a = axis / L
    e1 = np.cross(a, [0.0, 0.0, 1.0] if abs(a[2]) < 0.9 else [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    s = np.linspace(0, 1, (rings or max(1, int(L / 60))) + 1)
    th = np.linspace(0, 2 * np.pi, n + 1)
    rad = np.cos(th)[None, :, None] * e1 + np.sin(th)[None, :, None] * e2           # (1, n+1, 3)
    P = p0 + s[:, None, None] * axis + (r0 + (r1 - r0) * s)[:, None, None] * rad
    N = rad - ((r1 - r0) / L) * a
    N = np.broadcast_to(N / np.linalg.norm(N, axis=-1, keepdims=True), P.shape)
    k = max(1, round(np.pi * (r0 + r1) / tile[1]))                 # whole texture repeats around
    UV = np.stack(np.broadcast_arrays((s * L / tile[0])[:, None], (th / (2 * np.pi) * k)[None, :]), -1)
    return grid_mesh(P, N, UV)


_OCTA_V = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
_OCTA_F = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])


def leaf_blobs(clumps, rng, size, cover, layers=(1.0, 0.62)):
    """Foliage as opaque leaf-cluster blobs (smooth-shaded octahedra) scattered over each clump
    ellipsoid on two layers; the gaps between blobs let sun flecks through (dappled shade).
    Every blob gets one random uv so a mottled foliage texture varies its shade."""
    cen, rad = [], []
    for c, radii, rot in clumps:
        cr, sr = math.cos(math.radians(rot)), math.sin(math.radians(rot))
        Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
        for k in layers:
            a, b, e = np.asarray(radii, float) * k
            area = 4 * np.pi * (((a * b) ** 1.6 + (a * e) ** 1.6 + (b * e) ** 1.6) / 3) ** (1 / 1.6)
            n = max(4, int(cover * area / (np.pi * size * size)))
            d = rng.normal(size=(n, 3))
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            cen.append(np.asarray(c, float) + (d * [a, b, e] * rng.uniform(0.85, 1.05, (n, 1))) @ Rz.T)
            rad.append(size * rng.uniform(0.6, 1.3, (n, 3)))
    cen, rad = np.concatenate(cen), np.concatenate(rad)
    n = len(cen)
    ang = rng.uniform(0, 2 * np.pi, n)
    ca, sa = np.cos(ang), np.sin(ang)
    R = np.zeros((n, 3, 3))
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1], R[:, 2, 2] = ca, -sa, sa, ca, 1
    unit = np.einsum("nij,vj->nvi", R, _OCTA_V)                         # (n, 6, 3) rotated directions
    V = cen[:, None, :] + np.einsum("nij,nvj->nvi", R, _OCTA_V[None] * rad[:, None, :])
    tris = V[:, _OCTA_F].reshape(-1, 3, 3)
    nrms = unit[:, _OCTA_F].reshape(-1, 3, 3)
    uvs = np.repeat(rng.random((n, 1, 1, 2)), 8, 1).repeat(3, 2).reshape(-1, 3, 2)
    return tris, nrms, uvs


def tree_geometry(t: Tree, bark_tile):
    """-> (bark meshes, foliage meshes); each mesh is (tris, vertex normals, uv)."""
    rng = np.random.default_rng(t.seed)
    lean = rng.normal(0, math.radians(2.5)) * np.array([math.cos(a := rng.uniform(0, 2 * math.pi)), math.sin(a)])
    base = np.array([t.x, t.y, -3.0])
    up = lambda z: base + np.array([lean[0] * z, lean[1] * z, z + 3.0])  # noqa: E731 - point on the leaning axis
    meshes, clumps = [], []
    H, R, z0 = t.height, t.crown_r, t.crown_z0
    if t.sp.get("form") == "shrub":
        for k in range(int(rng.integers(2, 5))):
            a, rr = rng.uniform(0, 2 * math.pi), 0.4 * R * rng.random()
            rh = R * rng.uniform(0.45, 0.7)
            c = base + np.array([rr * math.cos(a), rr * math.sin(a), 3 + H * rng.uniform(0.35, 0.6)])
            clumps.append((c, (rh, rh * rng.uniform(0.7, 1.0), H * rng.uniform(0.3, 0.45)), rng.uniform(0, 180)))
    elif t.sp.get("form") == "pine":
        meshes.append(frustum(base, up(0.03 * H), 1.3 * t.trunk_r, t.trunk_r, bark_tile, rings=1))
        meshes.append(frustum(up(0.03 * H), up(0.97 * H), t.trunk_r, 0.12 * t.trunk_r, bark_tile))
        Lc = H - z0
        for k in range(int(rng.integers(6, 10))):
            z = z0 + Lc * (0.12 + 0.78 * rng.random())
            a, rr = rng.uniform(0, 2 * math.pi), R * 0.5 * math.sqrt(rng.random()) * (1.1 - (z - z0) / Lc)
            c = up(z) + rr * np.array([math.cos(a), math.sin(a), 0])
            rh = R * rng.uniform(0.35, 0.6) * (1.15 - 0.5 * (z - z0) / Lc)
            clumps.append((c, (rh, rh * rng.uniform(0.75, 1.0), Lc * rng.uniform(0.14, 0.24)), rng.uniform(0, 180)))
        clumps.append((up(0.96 * H), (0.3 * R, 0.28 * R, 0.14 * Lc), 0.0))
        for k in range(int(rng.integers(2, 6))):                    # short dead branch stubs on the bole
            z = rng.uniform(0.35, 0.95) * z0
            a = rng.uniform(0, 2 * math.pi)
            d = np.array([math.cos(a), math.sin(a), rng.uniform(-0.1, 0.25)])
            meshes.append(frustum(up(z), up(z) + d * rng.uniform(8, 24), 0.12 * t.trunk_r + 0.4, 0.3, bark_tile,
                                  n=6, rings=1))
    else:
        hf = z0 + (H - z0) * rng.uniform(0.0, 0.1)
        top = up(hf)
        meshes.append(frustum(base, top, 1.15 * t.trunk_r, 0.8 * t.trunk_r, bark_tile))
        n_limbs = int(rng.integers(2, 4))
        a0 = rng.uniform(0, 2 * math.pi)
        for k in range(n_limbs):
            a = a0 + 2 * math.pi * k / n_limbs + rng.normal(0, 0.3)
            tilt = math.radians(rng.uniform(25, 50))
            d = np.array([math.sin(tilt) * math.cos(a), math.sin(tilt) * math.sin(a), math.cos(tilt)])
            end = top + d * (H - hf) * rng.uniform(0.5, 0.7)
            meshes.append(frustum(top, end, 0.6 * t.trunk_r, 0.2 * t.trunk_r, bark_tile))
            clumps.append((end, (0.5 * R, 0.45 * R, 0.32 * (H - hf)), rng.uniform(0, 180)))
        for k in range(int(rng.integers(1, 3))):                    # a low, near-horizontal limb
            z = rng.uniform(0.45, 0.85) * hf
            a = rng.uniform(0, 2 * math.pi)
            d = np.array([math.cos(a), math.sin(a), rng.uniform(0.15, 0.5)])
            end = up(z) + d * R * rng.uniform(0.6, 0.9)
            meshes.append(frustum(up(z), end, 0.35 * t.trunk_r, 0.12 * t.trunk_r, bark_tile, n=10))
            clumps.append((end, (0.35 * R, 0.3 * R, 0.22 * R), rng.uniform(0, 180)))
        zc = hf + 0.5 * (H - hf)
        for k in range(int(rng.integers(8, 14))):
            a, rr = rng.uniform(0, 2 * math.pi), 0.6 * R * math.sqrt(rng.random())
            c = up(zc) + np.array([rr * math.cos(a), rr * math.sin(a), 0.3 * (H - hf) * rng.uniform(-1, 1)])
            rh = R * rng.uniform(0.35, 0.6)
            clumps.append((c, (rh, rh * rng.uniform(0.75, 1.0), rh * rng.uniform(0.6, 0.85)), rng.uniform(0, 180)))
    size = float(t.sp.get("leaf_cluster", 9.0 if t.sp.get("form") == "pine" else 7.0)) * t.lod
    return meshes, [leaf_blobs(clumps, rng, size, float(t.sp.get("crown_density", 0.55)))]
