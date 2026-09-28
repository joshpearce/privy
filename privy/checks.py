"""Design / fit checks and screening structural checks.

Structural checks are simplified NDS allowable-stress checks meant to catch a
bad parameter choice (e.g. a 2x4 rafter at 48" o.c. under heavy snow).  They are
not a substitute for an engineered design.
"""
from __future__ import annotations

import math

import numpy as np

from .units import fmt_ftin, fmt_in, parse_len

L = parse_len

# Reference design values (psi) for No. 2 grade, NDS Supplement (rounded).
SPECIES = {
    "SPF": dict(Fb=875, E=1.4e6, sized=True),
    "DF-L": dict(Fb=900, E=1.6e6, sized=True),
    "SYP": dict(Fb={3.5: 1100, 5.5: 1000, 7.25: 925, 9.25: 800, 11.25: 750}, E=1.4e6, sized=False),
    "HF": dict(Fb=850, E=1.3e6, sized=True),
}
CF = {3.5: 1.5, 5.5: 1.3, 7.25: 1.2, 9.25: 1.1, 11.25: 1.0}
DENSITY_PCF = {
    "treated_lumber": 40, "framing_lumber": 28, "rough_siding": 26, "rough_trim": 26, "decking": 40,
    "treated_plywood": 36, "bench_plywood": 34, "window_frame": 30, "glass": 156, "steel": 490,
    "seat": 40, "epdm": 80, "galvanized": 490, "vent_pipe": 88,
}


def _species(stock):
    sp = (stock.species or "SPF").upper()
    for k in SPECIES:
        if sp.startswith(k.upper()):
            return k, SPECIES[k]
    return "SPF", SPECIES["SPF"]


def allowable_fb(stock, CD=1.0, Cr=1.0, flat=False):
    name, sp = _species(stock)
    w = round(stock.w * 4) / 4
    if isinstance(sp["Fb"], dict):
        Fb = sp["Fb"].get(w, min(sp["Fb"].values()))
        cf = 1.0
    else:
        Fb = sp["Fb"]
        cf = CF.get(w, 1.0)
    cfu = 1.1 if flat and stock.t <= 1.5 else 1.0
    return Fb * CD * cf * Cr * cfu, sp["E"]


def mesh_volume(geom) -> float:
    t, _ = geom.triangles()
    if len(t) == 0:
        return 0.0
    return float(abs(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum()) / 6.0)


def estimate_weight(model) -> tuple[float, dict]:
    by = {}
    for p in model.parts:
        if p.group.split(".")[0] in ("pit", "ground") or "void" in p.tags:
            continue
        if "roofing" in p.tags:
            lo, hi = p.bbox()
            wt = (hi[0] - lo[0]) * (hi[1] - lo[1]) * model.key["c"] / 144.0 * 0.6    # 29 ga ~0.6 psf
        else:
            dens = DENSITY_PCF.get(p.material, 30)
            wt = mesh_volume(p.geom) / 1728.0 * dens
        g = p.group.split(".")[0]
        by[g] = by.get(g, 0.0) + wt
    return sum(by.values()), by


def run_checks(m):
    k, cfg, S = m.key, m.cfg, m.stock
    out = m.checks

    def chk(cat, title, value, limit, ok, note=""):
        out.append(dict(cat=cat, title=title, value=value, limit=limit, ok=ok, note=note))

    W, D, OD, ID = k["W"], k["D"], k["OD"], k["ID"]
    # ------------------------------------------------------------ fit / clearance
    (a0, a1), (b0, b1) = k["skid_ys"]
    side = ((b0 - a1) - OD) / 2
    chk("Fit", "Pipe passes between skids (clear each side)", fmt_in(side), '>= 1"', side >= 1.0)
    side2 = ((W - 2 * k["st_d"]) - OD) / 2
    chk("Fit", "Pipe passes between side-wall framing (each side)", fmt_in(side2), '>= 1"', side2 >= 1.0)
    hc = k["header_z"][0] - k["pipe_top"]
    chk("Fit", "Pipe top clears bench header while sliding", fmt_in(hc), '>= 1/2"', hc >= 0.5)
    jc = k["x_bj"] - k["pipe_front"]
    chk("Fit", "Pipe clears bench-front floor joist", fmt_in(jc), '>= 1/4"', jc >= 0.25)
    hx, hy, hrx, hry = k["hole"]
    t = np.linspace(0, 2 * np.pi, 180)
    dmax = np.max(np.hypot(hx + hrx * np.cos(t) - k["pipe_cx"], hy + hry * np.sin(t) - k["pipe_cy"]))
    mg = ID / 2 - dmax
    chk("Fit", "Seat hole lies inside pipe ID (margin)", fmt_in(mg), '>= 1/2"', mg >= 0.5)
    cf = (D - k["st_d"]) - k["x_bf"]
    chk("Use", "Clear floor in front of bench", fmt_ftin(cf), '>= 24" (IPC min 21")', cf >= 24,
        "knee room for a seated adult")
    col = k["flange_z"][0] - k["collar_z0"]
    chk("Fit", "Flange collar height (plate to pipe top less gap)", fmt_in(col), '>= 0', col >= 0)
    z_rb, pv = k["z_rb"], S[cfg["building"]["walls"]["plate"]].t * k["c"]
    dro = k["door"]
    worst = min(z_rb(u) - pv for u in (dro["u0"] - 3, dro["u1"] + 3))
    hdr_top = dro["v1"] + S[cfg["door"]["header"]["stock"]].w
    chk("Fit", "Door header fits under sloped top plate", fmt_in(worst - hdr_top), ">= 0", worst - hdr_top >= 0)
    dl = k["door_leaf"]
    chk("Use", "Door leaf clear height", fmt_ftin(dl[3] - dl[2]), ">= 6'-0\"", dl[3] - dl[2] >= 72)
    wo = k["window"]
    stud_top = k["stud_top_high"] if wo["wall"] == "right" else k["stud_top_low"]
    chk("Fit", "Window header fits under top plate", fmt_in(stud_top - wo["header_top"]), ">= 0",
        stud_top - wo["header_top"] >= 0)
    chk("Durability", "Untreated siding clearance to grade (IRC R317.1)", fmt_in(k["side_bot"]), '>= 6"',
        k["side_bot"] >= 6.0 - 1e-6, "PT skids, rims, plates and panel kick board are exempt")
    sill = wo["v0"] - k["FF"]
    chk("Use", "Window sill above floor (privacy)", fmt_ftin(sill), ">= 4'-6\"", sill >= 54)
    hr = k["posts"][0][3] - k["deck_top"]
    chk("Use", "Deck headroom under low beam", fmt_ftin(hr), ">= 6'-6\"", hr >= 78)
    kb = cfg["deck"].get("knee_brace")
    if kb:
        hk = hr - L(kb["leg"])
        chk("Use", "Headroom at knee-brace foot (low side)", fmt_ftin(hk), "info", None, "braces are at the deck corners")
    plen = max(p[3] - p[2] for p in k["posts"])
    chk("Stock", "Deck post length (from 8' stock)", fmt_ftin(plen), "<= 8'-0\"", plen <= 96)
    studs = [p.length for p in m.parts if p.name in ("Stud", "King stud") and p.length]
    chk("Stock", "Longest stud (92-5/8\" precut or 8')", fmt_ftin(max(studs)), "<= 8'-0\"", max(studs) <= 96)
    skl = k["skid_x1"] - k["skid_x0"]
    chk("Stock", "Skid length", fmt_ftin(skl), "<= 16'-0\"", skl <= 192)
    bm = k["beam_w"] * k["slope"]
    rd = S[cfg["roof"]["rafter"]].w
    chk("Framing", "Rafter birdsmouth depth", fmt_in(bm), f"<= d/3 = {fmt_in(rd / 3)}", bm <= rd / 3)
    chk("Framing", "Vent located inside pit ID, clear of rafters/purlins/seat", "found" if k.get("vent") else "none",
        "found", bool(k.get("vent")) or not cfg["pit"]["vent"].get("enabled", True))
    for wname in ("left", "right", "front", "back"):
        n = k.get(f"straps_{wname}", 0)
        chk("Framing", f"Let-in strap braces, {wname} wall", str(n), ">= 1", n >= 1 if wname in ("left", "right") else None,
            "" if n or wname in ("left", "right") else "wall too narrow for a 45-62 deg strap; racking taken by bench box, "
            "roof diaphragm and back wall")

    # ------------------------------------------------------------ loads
    ld = cfg["loads"]
    pg = ld["ground_snow_psf"]
    Is, Ct, Ce = 0.8, 1.2, 1.0       # Risk Cat. I, unheated, partially exposed
    pf = max(0.7 * Ce * Ct * Is * pg, (Is * pg if pg <= 20 else 20 * Is))
    Dr = ld["roof_dead_psf"]
    k["snow_pf"] = pf
    roof = cfg["roof"]

    def bend(title, stock_key, span_in, w_plf=0.0, P_lb=0.0, CD=1.0, Cr=1.0, defl_lim=240, w_defl=None, flat=False,
             plies=1, note=""):
        st = S[stock_key]
        b, d = (st.w, st.t) if flat else (st.t, st.w)
        b *= plies
        Sx, Ix = b * d * d / 6, b * d ** 3 / 12
        Lft = span_in / 12
        M = (w_plf * Lft ** 2 / 8 + P_lb * Lft / 4) * 12          # in-lb
        fb = M / Sx
        Fb, E = allowable_fb(st, CD, Cr, flat)
        wd = (w_defl if w_defl is not None else w_plf) / 12        # lb/in
        dl_ = 5 * wd * span_in ** 4 / (384 * E * Ix) + P_lb * span_in ** 3 / (48 * E * Ix)
        ratio = fb / Fb
        lim = span_in / defl_lim
        chk("Structure", title, f"fb/Fb' = {ratio:.2f};  defl {dl_:.3f}\" (L/{span_in / max(dl_, 1e-9):.0f})",
            f"<= 1.0; <= L/{defl_lim}", ratio <= 1.0 and dl_ <= lim,
            note or f"{st.label} {st.species}, span {fmt_ftin(span_in)}")

    rs = S[roof["rafter"]]
    sp = L(roof["rafter_spacing"])
    span = W - k["beam_w"]
    bend("Rafter, snow + dead", roof["rafter"], span, (pf + Dr) * sp / 12, CD=1.15, Cr=1.15, defl_lim=180,
         w_defl=pf * sp / 12)
    psp = L(roof["purlin_spacing"])
    bend("Purlin (flat) between rafters", roof["purlin"], max(np.diff(k["rafter_x"])),
         (pf + Dr) * psp / 12, CD=1.15, Cr=1.15, defl_lim=180, flat=True)
    trib = (W - k["beam_w"]) / 2 + max(L(roof["overhang_low"]), L(roof["overhang_high"]))
    bl = k["DD"] - S[cfg["deck"]["post"]].t / 2
    bend("Roof beam over deck", roof["beam"], bl, (pf + Dr) * trib / 12, CD=1.15, defl_lim=240, w_defl=pf * trib / 12)
    fl = cfg["building"]["floor"]
    js = L(fl["joist_spacing"])
    fspan = W - 2 * S[fl["rim"]].t * int(fl.get("rim_plies", 2))
    bend("Floor joist", fl["joist"], fspan, (ld["floor_live_psf"] + ld["floor_dead_psf"]) * js / 12, Cr=1.15,
         defl_lim=360, w_defl=ld["floor_live_psf"] * js / 12)
    dk = cfg["deck"]
    dspan = W - 2 * S[dk["rim"]].t * int(dk.get("rim_plies", 2))
    bend("Deck joist", dk["joist"], dspan, (ld["deck_live_psf"] + ld["deck_dead_psf"]) * L(dk["joist_spacing"]) / 12,
         Cr=1.15, defl_lim=360, w_defl=ld["deck_live_psf"] * L(dk["joist_spacing"]) / 12)
    occ = ld["occupant_point_lb"]
    b = cfg["bench"]
    bend("Bench front rail, occupant at midspan", b["framing"], W - 2 * k["st_d"], P_lb=occ, defl_lim=360)
    bend("Bench joist, occupant at midspan", b["framing"], k["x_fp0"] - k["st_d"] - S[b["framing"]].t, P_lb=occ,
         defl_lim=360)
    bend("Bench header (back wall), occupant at midspan", b["header"], W - 2 * k["st_d"], P_lb=occ, defl_lim=360)

    # ------------------------------------------------------------ pit capacity
    use = cfg["pit"].get("usage", {})
    fb = L(use.get("freeboard", 18))
    depth = -k["pipe_bot"] - fb
    vol_in3 = math.pi * (ID / 2) ** 2 * max(depth, 0)
    gal, liters = vol_in3 / 231.0, vol_in3 * 0.0163871
    k["pit_volume_gal"] = gal
    rate = use.get("accumulation_liters_per_person_year", 60) * use.get("persons", 2) * use.get("fraction_of_year", 0.25)
    yrs = liters / rate if rate > 0 else None
    k["pit_years"] = yrs
    chk("Pit", f"Usable pit volume (to {fmt_in(fb)} below grade)", f"{gal:.0f} gal ({liters:.0f} L)", "info", None,
        "move the privy when solids reach the freeboard line")
    if yrs:
        chk("Pit", "Estimated service life before relocation", f"{yrs:.1f} yr", "info", None,
            f"{use.get('persons', 2)} persons, {use.get('fraction_of_year', 0.25):.0%} of the year, "
            f"{use.get('accumulation_liters_per_person_year', 60)} L/person-yr (dry pit, WHO range 40-90)")

    # ------------------------------------------------------------ weight, bearing, towing
    wt, by = estimate_weight(m)
    k["weight"], k["weight_by_group"] = wt, by
    sk = S[cfg["foundation"]["skids"]["stock"]]
    area_ft2 = 2 * k["skid_w"] * (k["skid_x1"] - k["skid_x0"]) / 144
    bearing = (wt + (ld["floor_live_psf"] * W * D + ld["deck_live_psf"] * W * k["DD"]) / 144) / area_ft2
    chk("Structure", "Skid soil bearing (dead + live)", f"{bearing:.0f} psf", "<= 1500 psf (IBC presumptive)",
        bearing <= 1500)
    chk("Moving", "Estimated shipping weight (empty, no pit parts)", f"{wt:,.0f} lb", "info", None,
        ", ".join(f"{g} {v:,.0f}" for g, v in sorted(by.items(), key=lambda x: -x[1])))
    chk("Moving", "Towing force to drag on dirt (mu 0.5-0.7)", f"{0.5 * wt:,.0f}-{0.7 * wt:,.0f} lb", "info", None,
        "pull from both skid tow holes with a bridle")
    chk("Loads", "Roof snow load pf (ASCE 7: 0.7 Ce Ct Is pg)", f"{pf:.1f} psf", "info", None,
        f"pg={pg} psf, Ce={Ce}, Ct={Ct}, Is={Is}")
    return out
