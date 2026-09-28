"""Length parsing and architectural (feet-inch) formatting.

All internal lengths are inches.  JSON values may be plain numbers (inches) or
strings such as ``"6'-2 1/2\""``, ``"74.5"``, ``"3/4\""`` or ``"6'"``.
"""
from __future__ import annotations

import re
from fractions import Fraction

_FT_IN = re.compile(
    r"""^\s*(?:(?P<ft>-?\d+(?:\.\d+)?)\s*(?:'|ft)\s*-?\s*)?
         (?:(?P<in_whole>\d+(?:\.\d+)?)?\s*(?P<in_frac>\d+/\d+)?\s*(?:"|in)?)?\s*$""",
    re.X,
)


def parse_len(v) -> float:
    """Parse a length (number = inches, or ft-in string) to float inches."""
    if isinstance(v, (int, float)):
        return float(v)
    if not isinstance(v, str):
        raise TypeError(f"cannot parse length from {v!r}")
    s = v.strip().replace("′", "'").replace("″", '"')
    m = _FT_IN.match(s)
    if not m or not any(m.group(g) for g in ("ft", "in_whole", "in_frac")):
        raise ValueError(f"cannot parse length {v!r}")
    total = 0.0
    if m.group("ft"):
        total += float(m.group("ft")) * 12.0
    if m.group("in_whole"):
        total += float(m.group("in_whole"))
    if m.group("in_frac"):
        total += float(Fraction(m.group("in_frac")))
    return total


def _frac(x: float, denom: int = 16) -> str:
    """Format a non-negative inch value as whole + fraction, e.g. 5 1/2."""
    n = round(x * denom)
    whole, rem = divmod(n, denom)
    if rem == 0:
        return f"{whole}"
    f = Fraction(rem, denom)
    return f"{f.numerator}/{f.denominator}" if whole == 0 else f"{whole} {f.numerator}/{f.denominator}"


def fmt_in(x: float, denom: int = 16) -> str:
    """Inches only: 11 3/4\"."""
    sign = "-" if x < 0 else ""
    return f'{sign}{_frac(abs(x), denom)}"'


def fmt_ftin(x: float, denom: int = 16, inch_only_below: float = 12.0) -> str:
    """Architectural feet-inch format: 6'-2 1/2\"; values under 1' as inches."""
    sign = "-" if x < -1e-9 else ""
    x = abs(x)
    n = round(x * denom)
    if n < inch_only_below * denom:
        return sign + fmt_in(n / denom, denom)
    ft, rem = divmod(n, 12 * denom)
    return f"{sign}{ft}'-{_frac(rem / denom, denom)}\""


def fmt_elev(z: float) -> str:
    """Elevation relative to grade: +7'-0\" / -3'-0\"."""
    return ("+" if z >= 0 else "-") + fmt_ftin(abs(z))
