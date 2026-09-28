"""Part / Model containers and lumber stock sizing."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .geometry import Geometry

# S4S (surfaced four sides) actual sizes, inches
_S4S_THICK = {1: 0.75, 2: 1.5, 3: 2.5, 4: 3.5, 6: 5.5, 8: 7.25}
_S4S_WIDTH = {2: 1.5, 3: 2.5, 4: 3.5, 6: 5.5, 8: 7.25, 10: 9.25, 12: 11.25}


def parse_nominal(nom: str) -> tuple[float, float]:
    m = re.match(r"^\s*(\d+(?:/\d+)?)\s*x\s*(\d+)\s*$", nom)
    if not m:
        raise ValueError(f"bad nominal size {nom!r}")
    a = m.group(1)
    t = eval(a) if "/" in a else float(a)  # noqa: S307  ("5/4" style)
    return t, float(m.group(2))


@dataclass
class Stock:
    key: str
    nominal: str
    t: float                 # actual thickness (small dimension)
    w: float                 # actual width (large dimension)
    material: str
    species: str = ""
    treatment: str = ""
    surfacing: str = "S4S"
    sheet: bool = False
    sheet_size: tuple = (48, 96)
    lengths: tuple = (96, 120, 144, 168, 192)
    description: str = ""

    @property
    def label(self) -> str:
        if self.sheet:
            return self.description or f'{self.t}" sheet'
        bits = [self.nominal]
        if self.surfacing == "rough":
            bits.append("rough-sawn")
        if self.treatment:
            bits.append("PT")
        return " ".join(bits)


def make_stock(key: str, d: dict) -> Stock:
    if d.get("sheet"):
        return Stock(key, d.get("description", key), d["thickness"], d["size"][0], d["material"],
                     sheet=True, sheet_size=tuple(d.get("size", (48, 96))), description=d.get("description", ""),
                     treatment=d.get("treatment", ""))
    nom = d["nominal"]
    surf = d.get("surfacing", "S4S")
    if "actual" in d:
        t, w = d["actual"]
    else:
        nt, nw = parse_nominal(nom)
        if surf == "rough":
            t, w = nt, nw
        else:
            t = _S4S_THICK.get(int(nt), nt - 0.5) if nt >= 1 else nt
            w = _S4S_WIDTH.get(int(nw), nw - 0.5)
    t, w = min(t, w), max(t, w)
    return Stock(key, nom, t, w, d["material"], d.get("species", ""), d.get("treatment", ""), surf,
                 lengths=tuple(d.get("lengths", (96, 120, 144, 168, 192))), description=d.get("description", ""))


@dataclass
class Part:
    id: str                       # unique id, e.g. "wall.right.stud.03"
    name: str                     # human readable, e.g. "Stud"
    group: str                    # e.g. "wall.right", "roof", "deck", "pit"
    geom: Geometry
    material: str                 # key into cfg["materials"]
    stock: str | None = None      # key into cfg["stock"] (for cut list)
    length: float | None = None   # cut length along grain (cut list)
    note: str = ""                # cut note, e.g. "3:12 top cut"
    qty_factor: float = 1.0       # e.g. sheet goods area fraction
    tags: set = field(default_factory=set)
    grain_axis: int | None = None  # 0/1/2 model axis the wood grain runs along (None = auto)

    def bbox(self):
        return self.geom.bbox()

    def grain(self) -> int:
        if self.grain_axis is not None:
            return self.grain_axis
        lo, hi = self.bbox()
        return int(np.argmax(hi - lo))


@dataclass
class Model:
    cfg: dict
    stock: dict                    # key -> Stock
    parts: list = field(default_factory=list)
    key: dict = field(default_factory=dict)       # named coordinates / derived values
    checks: list = field(default_factory=list)    # design checks (dicts)
    notes: list = field(default_factory=list)

    def add(self, part: Part) -> Part:
        self.parts.append(part)
        return part

    def select(self, *, groups=None, tags=None, exclude_tags=None, kinds=None):
        out = []
        for p in self.parts:
            if groups and not any(p.group == g or p.group.startswith(g + ".") for g in groups):
                continue
            if tags and not (p.tags & set(tags)):
                continue
            if exclude_tags and (p.tags & set(exclude_tags)):
                continue
            out.append(p)
        return out

    def bbox(self, parts=None, exclude_tags=("ground", "site")):
        parts = parts if parts is not None else [p for p in self.parts if not (p.tags & set(exclude_tags))]
        los, his = zip(*(p.bbox() for p in parts))
        return np.min(los, 0), np.max(his, 0)
