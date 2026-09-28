"""Minimal SVG writer in paper inches, with architectural line weights and hatches."""
from __future__ import annotations

import base64
import html
import math

THEMES = {
    "white": dict(bg="#ffffff", ink="#1a1a1a", light="#555555", faint="#9a9a9a", hatch="#6a6a6a", fill="#ffffff",
                  poche="#3c3c3c", accent="#9c2b1f", note="#1a1a1a", table="#f1efe9", grade="#1a1a1a"),
    "blueprint": dict(bg="#18427a", ink="#f4f7fb", light="#d0dcec", faint="#8ea8cb", hatch="#b8cbe4", fill="#18427a",
                      poche="#dfe8f5", accent="#ffd36b", note="#f4f7fb", table="#21508f", grade="#f4f7fb"),
}

# stroke widths in inches (paper)
LW = dict(cut=0.019, heavy=0.013, med=0.0085, light=0.0055, fine=0.0038, dim=0.0042, grade=0.022, border=0.03)


def f(x: float) -> str:
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    return s if s not in ("-0", "") else "0"


def pts_to_d(pts, close=True) -> str:
    if len(pts) == 0:
        return ""
    s = "M" + " L".join(f"{f(x)} {f(y)}" for x, y in pts)
    return s + (" Z" if close else "")


class SVG:
    def __init__(self, w: float, h: float, theme: str = "white", font: str = "sans-serif"):
        self.w, self.h = w, h
        self.t = THEMES.get(theme, THEMES["white"])
        self.theme = theme
        self.font = font
        self.body: list[str] = []
        self.defs: list[str] = []
        self._pat: set[str] = set()
        self._clip = 0

    # ------------------------------------------------------------ primitives
    def raw(self, s: str):
        self.body.append(s)

    def _stroke(self, lw, color=None, dash=None, cap="round", join="round"):
        c = color or self.t["ink"]
        s = f'stroke="{c}" stroke-width="{f(LW.get(lw, lw) if isinstance(lw, str) else lw)}" ' \
            f'stroke-linecap="{cap}" stroke-linejoin="{join}"'
        if dash:
            s += f' stroke-dasharray="{" ".join(f(d) for d in dash)}"'
        return s

    def line(self, x1, y1, x2, y2, lw="med", color=None, dash=None, cap="round"):
        self.body.append(f'<line x1="{f(x1)}" y1="{f(y1)}" x2="{f(x2)}" y2="{f(y2)}" {self._stroke(lw, color, dash, cap)}/>')

    def polyline(self, pts, lw="med", color=None, dash=None, close=False, fill="none"):
        self.body.append(f'<path d="{pts_to_d(pts, close)}" fill="{fill}" {self._stroke(lw, color, dash)}/>')

    def path(self, d, fill="none", lw="med", color=None, dash=None, rule="nonzero", opacity=None, stroke=True):
        op = f' fill-opacity="{f(opacity)}"' if opacity is not None else ""
        st = self._stroke(lw, color, dash) if stroke else 'stroke="none"'
        self.body.append(f'<path d="{d}" fill="{fill}" fill-rule="{rule}"{op} {st}/>')

    def rect(self, x, y, w, h, fill="none", lw="med", color=None, stroke=True, rx=0, dash=None):
        st = self._stroke(lw, color, dash) if stroke else 'stroke="none"'
        r = f' rx="{f(rx)}"' if rx else ""
        self.body.append(f'<rect x="{f(x)}" y="{f(y)}" width="{f(w)}" height="{f(h)}"{r} fill="{fill}" {st}/>')

    def circle(self, cx, cy, r, fill="none", lw="med", color=None, dash=None, stroke=True):
        st = self._stroke(lw, color, dash) if stroke else 'stroke="none"'
        self.body.append(f'<circle cx="{f(cx)}" cy="{f(cy)}" r="{f(r)}" fill="{fill}" {st}/>')

    def text(self, x, y, s, size=0.09, anchor="start", weight="normal", color=None, rotate=0, italic=False,
             family=None, spacing=None):
        c = color or self.t["note"]
        tr = f' transform="rotate({f(rotate)} {f(x)} {f(y)})"' if rotate else ""
        st = ' font-style="italic"' if italic else ""
        ls = f' letter-spacing="{f(spacing)}"' if spacing else ""
        fam = html.escape(family or self.font, quote=True)
        self.body.append(
            f'<text x="{f(x)}" y="{f(y)}" font-family="{fam}" font-size="{f(size)}" font-weight="{weight}"{st}{ls} '
            f'fill="{c}" text-anchor="{anchor}"{tr}>{html.escape(str(s))}</text>')

    def mtext(self, x, y, lines, size=0.09, lead=1.35, **kw):
        for i, ln in enumerate(lines):
            self.text(x, y + i * size * lead, ln, size, **kw)
        return y + len(lines) * size * lead

    def image(self, x, y, w, h, data: bytes, mime="image/jpeg", clip=None):
        b64 = base64.b64encode(data).decode()
        self.body.append(f'<image x="{f(x)}" y="{f(y)}" width="{f(w)}" height="{f(h)}" '
                         f'preserveAspectRatio="xMidYMid slice" href="data:{mime};base64,{b64}"/>')

    def begin_clip(self, x, y, w, h):
        self._clip += 1
        cid = f"clip{self._clip}"
        self.defs.append(f'<clipPath id="{cid}"><rect x="{f(x)}" y="{f(y)}" width="{f(w)}" height="{f(h)}"/></clipPath>')
        self.body.append(f'<g clip-path="url(#{cid})">')

    def end_clip(self):
        self.body.append("</g>")

    # ------------------------------------------------------------ hatches (explicit line work)
    HATCH = {
        "diag": [(45, 0.045, None)],
        "diag_dense": [(45, 0.028, None)],
        "cross": [(45, 0.05, None), (-45, 0.05, None)],
        "earth": [(45, 0.075, None), (45, 0.075, (0.012, 0.022), 0.0375)],
        "ply": [(0, 0.03, None)],
        "glass": [(60, 0.06, (0.02, 0.04))],
    }

    def hatch(self, loops, kind="diag", fill=None, outline=None):
        """Fill paper-space loops (outer + holes, even-odd) with hatch `kind`."""
        d = " ".join(pts_to_d(lp) for lp in loops if len(lp) >= 3)
        if not d:
            return
        self.path(d, fill=fill or self.t["fill"], rule="evenodd", stroke=False)
        for spec in self.HATCH.get(kind, self.HATCH["diag"]):
            ang, sp, dash = spec[0], spec[1], spec[2]
            off = spec[3] if len(spec) > 3 else 0.0
            segs = hatch_segments(loops, ang, sp, off)
            if segs:
                dd = " ".join(f"M{f(a[0])} {f(a[1])} L{f(b[0])} {f(b[1])}" for a, b in segs)
                self.path(dd, fill="none", lw="fine", color=self.t["hatch"], dash=dash)
        if outline:
            self.path(d, fill="none", lw=outline, rule="evenodd")

    # ------------------------------------------------------------ output
    def tostring(self) -> str:
        return (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
                f'width="{f(self.w)}in" height="{f(self.h)}in" viewBox="0 0 {f(self.w)} {f(self.h)}">'
                f'<defs>{"".join(self.defs)}</defs>'
                f'<rect x="0" y="0" width="{f(self.w)}" height="{f(self.h)}" fill="{self.t["bg"]}"/>'
                + "".join(self.body) + "</svg>")


def hatch_segments(loops, angle_deg, spacing, offset=0.0):
    """Parallel hatch segments clipped (even-odd) to the given loops."""
    a = math.radians(angle_deg)
    dv = (math.cos(a), math.sin(a))
    nv = (-math.sin(a), math.cos(a))
    edges = []
    lo, hi = 1e18, -1e18
    for lp in loops:
        n = len(lp)
        for i in range(n):
            p, q = lp[i], lp[(i + 1) % n]
            edges.append((p, q))
            c = p[0] * nv[0] + p[1] * nv[1]
            lo, hi = min(lo, c), max(hi, c)
    segs = []
    c = math.ceil((lo - offset) / spacing) * spacing + offset
    while c < hi:
        ts = []
        for p, q in edges:
            dp = p[0] * nv[0] + p[1] * nv[1] - c
            dq = q[0] * nv[0] + q[1] * nv[1] - c
            if (dp < 0) != (dq < 0):
                t = dp / (dp - dq)
                x, y = p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])
                ts.append(x * dv[0] + y * dv[1])
        ts.sort()
        for i in range(0, len(ts) - 1, 2):
            t0, t1 = ts[i], ts[i + 1]
            if t1 - t0 > 1e-4:
                segs.append(((t0 * dv[0] + c * nv[0], t0 * dv[1] + c * nv[1]),
                             (t1 * dv[0] + c * nv[0], t1 * dv[1] + c * nv[1])))
        c += spacing
    return segs


# ---------------------------------------------------------------- annotation helpers (paper coords)

def tick(svg: SVG, x, y, size=0.045):
    svg.line(x - size, y + size, x + size, y - size, lw=LW["heavy"] * 0.9)


def dim_h(svg: SVG, x0, x1, y, text, ext_from=None, size=0.085, text_above=True, gap=0.03, over=0.045):
    """Horizontal dimension; ext_from=(ya, yb) are feature y's the extension lines start at."""
    if x1 < x0:
        x0, x1 = x1, x0
    svg.line(x0 - over * 0.6, y, x1 + over * 0.6, y, lw="dim")
    tick(svg, x0, y)
    tick(svg, x1, y)
    if ext_from:
        for x, ya in ((x0, ext_from[0]), (x1, ext_from[1])):
            if ya is None:
                continue
            s = 1 if y > ya else -1
            svg.line(x, ya + s * gap, x, y + s * over, lw="dim")
    tw = len(text) * size * 0.52
    mid = 0.5 * (x0 + x1)
    if tw > (x1 - x0) - 0.06:          # doesn't fit: put text beside
        svg.text(x1 + 0.08, y + size * 0.35, text, size, "start")
    else:
        svg.text(mid, y - 0.035 if text_above else y + size + 0.02, text, size, "middle")


def dim_v(svg: SVG, y0, y1, x, text, ext_from=None, size=0.085, gap=0.03, over=0.045, side=-1):
    """Vertical dimension (text reads bottom-to-top, left of the line)."""
    if y1 < y0:
        y0, y1 = y1, y0
    svg.line(x, y0 - over * 0.6, x, y1 + over * 0.6, lw="dim")
    tick(svg, x, y0)
    tick(svg, x, y1)
    if ext_from:
        for y, xa in ((y0, ext_from[0]), (y1, ext_from[1])):
            if xa is None:
                continue
            s = 1 if x > xa else -1
            svg.line(xa + s * gap, y, x + s * over, y, lw="dim")
    tw = len(text) * size * 0.52
    mid = 0.5 * (y0 + y1)
    if tw > (y1 - y0) - 0.06:
        svg.text(x, y0 - 0.08, text, size, "middle")
    else:
        svg.text(x - 0.035 if side < 0 else x + size + 0.01, mid, text, size, "middle", rotate=-90)


def leader(svg: SVG, ax, ay, tx, ty, lines, size=0.085, dot=True):
    """Leader from anchor (ax, ay) to a note at (tx, ty); note aligns away from the anchor."""
    right = tx >= ax
    sh = 0.12 if right else -0.12
    svg.polyline([(ax, ay), (tx, ty), (tx + sh, ty)], lw="dim", close=False)
    if dot:
        svg.circle(ax, ay, 0.012, fill=svg.t["ink"], stroke=False)
    anchor = "start" if right else "end"
    x = tx + sh + (0.03 if right else -0.03)
    lines = [lines] if isinstance(lines, str) else lines
    y0 = ty + size * 0.35 - (len(lines) - 1) * size * 1.25 / 2
    for i, ln in enumerate(lines):
        svg.text(x, y0 + i * size * 1.25, ln, size, anchor)


def view_title(svg: SVG, x, y, num, title, scale_text, width=None):
    r = 0.13
    svg.circle(x + r, y, r, lw="med")
    svg.text(x + r, y + 0.045, str(num), 0.12, "middle", weight="bold")
    svg.text(x + 2 * r + 0.08, y + 0.02, title.upper(), 0.14, "start", weight="bold")
    w = width if width else max(1.6, len(title) * 0.14 * 0.62 + 0.3)
    svg.line(x + 2 * r + 0.06, y + 0.07, x + 2 * r + 0.06 + w, y + 0.07, lw="heavy")
    if scale_text:
        svg.text(x + 2 * r + 0.08, y + 0.2, f"SCALE: {scale_text}", 0.08, "start")


def level_mark(svg: SVG, x, y, label, value, left=False, length=0.0):
    """Datum/level marker at paper (x, y)."""
    r = 0.045
    svg.circle(x, y, r, lw="light")
    svg.path(f"M{f(x)} {f(y)} L{f(x + r)} {f(y)} A{f(r)} {f(r)} 0 0 0 {f(x)} {f(y - r)} Z", fill=svg.t["ink"], lw="fine")
    svg.path(f"M{f(x)} {f(y)} L{f(x - r)} {f(y)} A{f(r)} {f(r)} 0 0 0 {f(x)} {f(y + r)} Z", fill=svg.t["ink"], lw="fine")
    tx = x - 0.08 if left else x + 0.08
    a = "end" if left else "start"
    svg.text(tx, y - 0.025, label.upper(), 0.075, a, weight="bold")
    svg.text(tx, y + 0.085, value, 0.075, a)


def jpeg_bytes(png_path: str, max_w=2000, quality=88) -> tuple[bytes, float]:
    from PIL import Image
    import io
    im = Image.open(png_path).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue(), im.width / im.height


def scale_label(s: float) -> str:
    """Paper inches per model foot -> architectural scale text."""
    per_ft = s * 12
    table = {3: '3"', 1.5: '1-1/2"', 1: '1"', 0.75: '3/4"', 0.5: '1/2"', 0.375: '3/8"', 0.25: '1/4"',
             0.1875: '3/16"', 0.125: '1/8"', 0.0625: '1/16"'}
    for k, v in table.items():
        if math.isclose(per_ft, k, rel_tol=1e-6):
            return f"{v} = 1'-0\""
    return f"1:{round(1 / s)}"


def parse_scale(txt: str) -> float:
    """'1/2\"=1\\'-0\"' -> paper inches per model inch."""
    a = txt.split("=")[0].replace('"', "").strip()
    if "-" in a:
        whole, frac = a.split("-")
        n, d = frac.split("/")
        v = float(whole) + float(n) / float(d)
    elif "/" in a:
        n, d = a.split("/")
        v = float(n) / float(d)
    else:
        v = float(a)
    return v / 12.0
