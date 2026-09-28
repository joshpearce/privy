"""Procedural, seamlessly tileable sRGB textures for the photoreal renderer.

All generators return ``uint8`` arrays of shape (H, W, 3) in sRGB.  Image x
(columns) is texture u; for wood, u runs along the grain.  Noise is made by
filtering white noise in the Fourier domain, so every layer tiles exactly.
"""
from __future__ import annotations

import hashlib

import numpy as np

WOOD_SIZE = (256, 1024)      # (H, W): across grain x along grain
OTHER_SIZE = (512, 512)
LITTER_SIZE = 1024


def stable_hash(s: str) -> int:
    return int.from_bytes(hashlib.md5(s.encode()).digest()[:8], "little")


def hex_srgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def srgb_to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, float), 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def variant_tint(variation: float, variant: int, seed: int) -> np.ndarray:
    """Multiplicative linear-RGB tint for material variant ``variant`` (0 = unchanged)."""
    if variation <= 0 or variant == 0:
        return np.ones(3)
    r = np.random.default_rng([seed, variant, 7919])
    bright = 1.0 + variation * r.uniform(-1.6, 1.6)
    warm = variation * r.uniform(-0.6, 0.6)            # shift toward yellow/brown or grey
    return bright * np.array([1 + warm, 1 + 0.3 * warm, 1 - warm])


# --------------------------------------------------------------------------- noise helpers

def pnoise(rng, shape, sx, sy=None):
    """Tileable Gaussian-filtered noise with feature size (sigma) sx, sy pixels; zero mean, unit std."""
    sy = sx if sy is None else sy
    h, w = shape
    fy, fx = np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None, :]
    filt = np.exp(-2 * np.pi ** 2 * ((fx * sx) ** 2 + (fy * sy) ** 2))
    n = np.fft.ifft2(np.fft.fft2(rng.standard_normal(shape)) * filt).real
    return (n - n.mean()) / (n.std() + 1e-12)


def fbm(rng, shape, sx, sy=None, octaves=4, gain=0.5):
    sy = sx if sy is None else sy
    out, amp, tot = np.zeros(shape), 1.0, 0.0
    for _ in range(octaves):
        out += amp * pnoise(rng, shape, max(sx, 0.5), max(sy, 0.5))
        tot += amp
        amp *= gain
        sx, sy = sx / 2, sy / 2
    return out / tot


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def wrap_dist(a, b, period):
    d = np.abs(a - b) % period
    return np.minimum(d, period - d)


def _finish(col_lin):
    return (linear_to_srgb(col_lin) * 255 + 0.5).astype(np.uint8)


# --------------------------------------------------------------------------- generators

def wood(base, grain, tile, rings=6, contrast=0.35, saw_marks=0.0, seed=0, knots=None):
    """Board face: warped growth-ring stripes along x, fibre streaks, blotches, optional saw marks/knots."""
    H, W = WOOD_SIZE
    rng = np.random.default_rng(seed)
    Lu, Lv = tile
    X = (np.arange(W) + 0.5)[None, :] * (Lu / W)          # inches along grain
    Y = (np.arange(H) + 0.5)[:, None] * (Lv / H)          # inches across grain
    rings = max(1, int(round(rings)))
    warp = 0.55 * pnoise(rng, (H, W), W / 7, H / 4) + 0.18 * pnoise(rng, (H, W), W / 30, H / 10)
    warp += 0.22 * pnoise(rng, (H, W), W / 3, H / (1.6 * rings))       # uneven ring spacing
    phase = rings * Y / Lv + warp
    knots = (saw_marks > 0) if knots is None else knots
    knot_mask = np.zeros((H, W))
    if knots:
        for _ in range(int(rng.integers(1, 3))):
            kx, ky, r = rng.uniform(0, Lu), rng.uniform(0, Lv), rng.uniform(0.45, 0.8)
            dy = (Y - ky + Lv / 2) % Lv - Lv / 2
            d = np.hypot(wrap_dist(X, kx, Lu) / 2.2, dy)
            i, j = int(ky / Lv * H) % H, int(kx / Lu * W) % W
            # squeeze the ring field toward the knot's own phase: rings bulge around it ("eye")
            phase = phase - 0.92 * np.exp(-(d / (2.4 * r)) ** 2) * (phase - phase[i, j])
            dk = np.hypot(wrap_dist(X, kx, Lu) / 1.2, dy)
            core = smoothstep(r, 0.8 * r, dk) * (0.8 + 0.2 * np.cos(dk / r * 10))
            knot_mask = np.maximum(knot_mask, np.maximum(core, 0.45 * smoothstep(1.3 * r, r, dk)))
    f = phase % 1.0
    late = smoothstep(0.5, 0.8, f) * (1 - smoothstep(0.86, 1.0, f))       # latewood band
    fiber = pnoise(rng, (H, W), 40.0, 0.6) + 0.6 * pnoise(rng, (H, W), 8.0, 0.5)
    blotch = pnoise(rng, (H, W), W / 10, H / 3)
    m = np.clip(late * (0.5 + contrast) + 0.3 * contrast * fiber, 0, 1)
    m = np.maximum(m, knot_mask)
    b, g = srgb_to_linear(base), srgb_to_linear(grain)
    col = b * (1 - m[..., None]) + g * m[..., None]
    col = col * (1 - 0.5 * smoothstep(0.5, 0.9, knot_mask))[..., None]
    shade = 1 + 0.45 * contrast * blotch + 0.06 * pnoise(rng, (H, W), 0.7)
    if saw_marks > 0:
        n = max(1, int(round(Lu / 0.4)))                   # ~0.4" spacing, integer count for tiling
        skew = (Lu / n) / Lv                               # one mark spacing of drift across the tile
        pos = (X + skew * Y) * n / Lu + 0.12 * pnoise(rng, (H, W), W / 12, H / 2)
        s = pos % 1.0
        mark = np.exp(-((s - 0.5) / 0.09) ** 2) - 0.35 * np.exp(-((s - 0.3) / 0.12) ** 2)
        strength = saw_marks * (0.8 + 0.4 * pnoise(rng, (H, W), W / 6, H / 3))
        shade = shade * (1 - 0.35 * strength * mark)
        shade = shade * (1 + 0.12 * saw_marks * pnoise(rng, (H, W), 3.0, 0.5))   # fuzzy fibres
    return _finish(col * shade[..., None])


def mottled(base, tile, scale=10.0, contrast=0.1, seed=0):
    """Mottled colour: brightness + slight hue variation with feature size ``scale`` inches."""
    H, W = OTHER_SIZE
    rng = np.random.default_rng(seed)
    px = tile[0] / W
    s = scale / px
    n1 = fbm(rng, (H, W), s, octaves=5, gain=0.55)
    n2 = fbm(rng, (H, W), 0.7 * s, octaves=3)
    fine = pnoise(rng, (H, W), 0.8)
    col = srgb_to_linear(base)[None, None, :] * (1 + contrast * n1 + 0.5 * contrast * fine)[..., None]
    hue = np.array([1.0, 0.35, -0.8]) * (0.3 * contrast)
    col = col * (1 + hue[None, None, :] * n2[..., None])
    return _finish(np.clip(col, 0, None))


def _dirt(rng, shape, px, base, contrast):
    """Linear-RGB dirt: multi-octave mottling and small pebble speckles (px = inches per pixel)."""
    big = fbm(rng, shape, 8.0 / px, octaves=3)
    mid = fbm(rng, shape, 1.2 / px, octaves=3)
    grit = pnoise(rng, shape, 0.6)
    shade = 1 + contrast * (0.6 * big + 0.35 * mid + 0.3 * grit)
    col = srgb_to_linear(base)[None, None, :] * shade[..., None]
    col = col * (1 + np.array([0.08, 0.0, -0.1])[None, None, :] * big[..., None] * contrast)
    for size, thresh, tone in ((0.12, 2.3, 1.45), (0.07, 2.5, 0.6), (0.25, 2.6, 1.25)):
        p = pnoise(rng, shape, max(size / px, 0.5))       # isolated noise peaks -> pebbles
        a = smoothstep(thresh, thresh + 0.35, p)[..., None]
        t = tone * (1 + 0.25 * pnoise(rng, shape, max(0.3 / px, 0.5)))[..., None]
        grey = srgb_to_linear(hex_srgb("#8d8579")) * t * (1 + 0.15 * (p[..., None] - thresh))
        col = col * (1 - a) + grey * a
    return col


def soil(base, tile, contrast=0.35, seed=0):
    """Dirt / clay: multi-octave mottling, darker damp patches and small pebble speckles."""
    return _finish(_dirt(np.random.default_rng(seed), OTHER_SIZE, tile[0] / OTHER_SIZE[1], base, contrast))


def _leaf_outline(rng, length):
    """Lobed oak-leaf outline in inches, long axis along x, centred on the origin."""
    t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    lobes = rng.integers(3, 6)
    scallop = 0.7 + 0.3 * np.abs(np.cos(lobes * t))
    w = 0.5 * length * rng.uniform(0.45, 0.65)
    x = 0.5 * length * np.cos(t) * (0.9 + 0.1 * scallop)
    y = w * np.sin(t) * scallop * (1 + 0.25 * np.cos(t))       # widest toward the tip
    return np.stack([x, y], 1)


def _wrapped(pts, size):
    """Copies of a pixel-space polyline shifted by the tile size wherever it crosses an edge."""
    lo, hi = pts.min(0), pts.max(0)
    for dx in (-size, 0, size):
        for dy in (-size, 0, size):
            if hi[0] + dx >= 0 and lo[0] + dx <= size and hi[1] + dy >= 0 and lo[1] + dy <= size:
                yield [tuple(p) for p in pts + (dx, dy)]


def litter(dirt, tile, leaf_colors, needle_color, leaf_density=0.3, needle_density=0.3, contrast=0.3, seed=0):
    """Forest-floor litter: fallen oak leaves and pine needles scattered on noisy dirt (tileable).

    Colours are sRGB triples; densities are approximate area coverage fractions (0..1).
    """
    from PIL import Image, ImageDraw
    N, S = LITTER_SIZE, 2 * LITTER_SIZE                   # drawn 2x supersampled, then reduced
    rng = np.random.default_rng(seed)
    base = _dirt(rng, (N, N), tile[0] / N, dirt, contrast)
    canvas = Image.fromarray(_finish(base)).resize((S, S), Image.BILINEAR)
    draw = ImageDraw.Draw(canvas)
    ppi = S / tile[0]
    area = tile[0] * tile[1]
    kinds = np.array([0] * int(needle_density * area / 0.5) + [1] * int(leaf_density * area / 7.0))
    rng.shuffle(kinds)
    shadow = tuple(int(c * 150) for c in dirt)
    leaf_cols = [np.asarray(c, float) for c in leaf_colors] or [hex_srgb("#7a4a22")]
    rgb = lambda c: tuple(int(v) for v in np.clip(c * 255, 0, 255))  # noqa: E731
    for kind in kinds:
        c = rng.uniform(0, S, 2)
        a = rng.uniform(0, 2 * np.pi)
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        if kind:                                          # leaf, with a soft contact shadow
            poly = _leaf_outline(rng, rng.uniform(2.5, 5.0)) @ rot.T * ppi + c
            col = leaf_cols[rng.integers(len(leaf_cols))] * rng.uniform(0.75, 1.15)
            for q in _wrapped(poly + 0.12 * ppi, S):
                draw.polygon(q, fill=shadow)
            for q in _wrapped(poly, S):
                draw.polygon(q, fill=rgb(col))
            rib = np.array([[-0.45, 0], [0.5, 0]]) @ rot.T * (np.ptp(poly @ rot, 0)[0]) + c
            for q in _wrapped(rib, S):
                draw.line(q, fill=rgb(col * 0.72), width=max(1, int(0.05 * ppi)))
        else:                                             # slightly bowed pine needle
            L = rng.uniform(5.0, 9.0) * ppi
            d, n = rot[:, 0], rot[:, 1]
            pts = np.array([c - d * L / 2, c + n * rng.normal(0, 0.03) * L, c + d * L / 2])
            col = np.asarray(needle_color, float) * rng.uniform(0.65, 1.2) * (1 + np.array([0.05, 0, -0.08]) * rng.normal())
            for q in _wrapped(pts, S):
                draw.line(q, fill=rgb(col), width=max(1, int(round(0.06 * ppi))))
    img = srgb_to_linear(np.asarray(canvas.resize((N, N), Image.LANCZOS)) / 255.0)
    img *= (1 + 0.08 * pnoise(rng, (N, N), 1.0))[..., None]
    return _finish(img)


def bark(base, fissure, tile, plate=(6.0, 3.0), contrast=0.35, seed=0):
    """Bark: plates separated by a network of dark fissures running mostly along the trunk (x)."""
    H, W = 256, 512
    rng = np.random.default_rng(seed)
    su, sv = 0.3 * plate[0] * W / tile[0], 0.3 * plate[1] * H / tile[1]   # noise sigma ~ plate / 3 (px)
    ridge = np.abs(pnoise(rng, (H, W), su, sv) + 0.35 * pnoise(rng, (H, W), 0.35 * su, 0.35 * sv))
    fiss = (1 - smoothstep(0.06, 0.4, ridge))[..., None]
    tone = 0.5 * pnoise(rng, (H, W), 0.6 * su, 0.6 * sv) + 0.3 * pnoise(rng, (H, W), W / 5, H / 3) \
        + 0.2 * pnoise(rng, (H, W), 1.0)
    col = srgb_to_linear(base)[None, None, :] * (1 + contrast * tone)[..., None]
    return _finish(col * (1 - fiss) + srgb_to_linear(fissure)[None, None, :] * fiss)


def blend_mask(seed=0, size=256):
    """Soft large-scale 0..1 mask (grayscale) used to blend two texture copies against tiling."""
    rng = np.random.default_rng([seed, 99])
    n = fbm(rng, (size, size), size / 10, octaves=3)
    return (smoothstep(-0.6, 0.6, n) * 255 + 0.5).astype(np.uint8)


def side_material(mat: dict) -> dict:
    """Material as seen on non-upward faces: ``side_texture`` (and its base_color) replaces ``texture``."""
    side = mat.get("side_texture")
    if not side:
        return mat
    return {**mat, "texture": side, "base_color": side.get("base_color", mat.get("base_color", "#808080"))}


def generate(mat: dict, variant: int = 0) -> np.ndarray | None:
    """Texture image for material dict ``mat`` (cfg['materials'][key]) and variant index."""
    tex = mat.get("texture")
    if not tex:
        return None
    variation = float(mat.get("variation", 0.0))
    seed = int(tex.get("seed", 0))
    tint = variant_tint(variation, variant, seed)

    def color(h):
        return linear_to_srgb(np.clip(srgb_to_linear(hex_srgb(h)) * tint, 0, 1))

    base = color(mat.get("base_color", "#808080"))
    vseed = seed * 1000 + variant
    kind = tex.get("type", "noise")
    tile = tex_tile(mat)
    if kind == "wood":
        return wood(base, color(tex.get("grain_color", "#5a4a35")), tile, tex.get("rings", 6),
                    tex.get("contrast", 0.35), tex.get("saw_marks", 0.0), vseed, tex.get("knots"))
    if kind == "soil":
        return soil(base, tile, tex.get("contrast", 0.35), vseed)
    if kind == "litter":
        return litter(color(tex.get("dirt_color", mat.get("base_color", "#7a5a40"))), tile,
                      [color(c) for c in tex.get("leaf_colors", [])], color(tex.get("needle_color", "#8a5a30")),
                      tex.get("leaf_density", 0.3), tex.get("needle_density", 0.3), tex.get("contrast", 0.3), vseed)
    if kind == "bark":
        return bark(base, color(tex.get("fissure_color", "#2e241c")), tile, tex.get("plate", (6, 3)),
                    tex.get("contrast", 0.35), vseed)
    return mottled(base, tile, tex.get("scale", 10.0), tex.get("contrast", 0.1), vseed)


def tex_tile(mat: dict) -> tuple[float, float]:
    """Texture tile size [inches along u (grain), inches along v]."""
    tex = mat.get("texture") or {}
    if "tile" in tex:
        return float(tex["tile"][0]), float(tex["tile"][1])
    s = 8.0 * float(tex.get("scale", 10.0))
    return s, s
