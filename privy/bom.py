"""Cut list, purchase list (first-fit-decreasing on stock lengths), sheet goods and hardware."""
from __future__ import annotations

import math
from collections import OrderedDict, defaultdict

import numpy as np

from .units import fmt_ftin, fmt_in, parse_len

L = parse_len
KERF = 0.125
ORDER = ["foundation", "floor", "deck", "bench", "wall", "roof", "siding", "removable_panel", "door", "window", "pit"]


def _grp(p):
    g = p.group.split(".")[0]
    return g if g in ORDER else "other"


def cut_list(model):
    """Rows: dict(group, stock, label, species, qty, length, note)."""
    rows = OrderedDict()
    for p in model.parts:
        s = model.stock.get(p.stock) if p.stock else None
        if s is None or s.sheet or not p.length:
            continue
        ln = round(p.length * 8) / 8
        key = (_grp(p), p.stock, p.name, ln, p.note)
        if key not in rows:
            rows[key] = dict(group=_grp(p), stock=p.stock, label=s.label, species=s.species, name=p.name,
                             qty=0, length=ln, note=p.note)
        rows[key]["qty"] += 1
    out = list(rows.values())
    out.sort(key=lambda r: (ORDER.index(r["group"]) if r["group"] in ORDER else 99, r["label"], r["name"], -r["length"]))
    return out


def purchase(model, rows):
    """Pack cut pieces into standard lengths per lumber type (FFD). Returns rows + board feet."""
    by = defaultdict(list)
    stocks = defaultdict(list)
    for r in rows:
        s = model.stock[r["stock"]]
        key = (s.label, s.species)
        by[key] += [r["length"]] * r["qty"]
        stocks[key].append(s)
    out = []
    for key, pieces in by.items():
        s = stocks[key][0]
        lengths = sorted({x for st in stocks[key] for x in st.lengths})
        bins = []   # [stock_len, remaining]
        for pc in sorted(pieces, reverse=True):
            need = pc + KERF
            for b in bins:
                if b[1] >= need:
                    b[1] -= need
                    break
            else:
                fit = [x for x in lengths if x >= pc]
                if fit:
                    bins.append([fit[0], fit[0] - need])
                else:
                    n_join = math.ceil(pc / lengths[-1])
                    bins += [[lengths[-1], 0.0]] * n_join
        cnt = defaultdict(int)
        for sl, _ in bins:
            cnt[sl] += 1
        nt, nw = _nominal(s)
        for sl, n in sorted(cnt.items()):
            bf = n * nt * nw * sl / 144.0
            treat = sorted({st.treatment for st in stocks[key]}, key=lambda t: ("UC4B" in t, "ground" in t, t))[-1]
            out.append(dict(stock=s.key, label=s.label, species=s.species, treatment=treat, length=sl, qty=n, bf=bf))
    out.sort(key=lambda r: (r["label"], r["length"]))
    return out


def _nominal(s):
    try:
        a, b = s.nominal.lower().split("x")
        return (eval(a) if "/" in a else float(a)), float(b)   # noqa: S307
    except Exception:
        return s.t, s.w


def sheet_goods(model):
    by = defaultdict(list)
    for p in model.parts:
        s = model.stock.get(p.stock) if p.stock else None
        if s is None or not s.sheet:
            continue
        lo, hi = p.bbox()
        ext = sorted(hi - lo)[::-1]
        by[p.stock].append((p.name, ext[0], ext[1]))
    # flange collar rings come from the flange sheet stock (2 plies)
    k = model.key
    out = []
    for sk, pcs in by.items():
        s = model.stock[sk]
        if sk == model.cfg["pit"]["flange"]["plate"]:
            n_rings = max(1, round((k["flange_z"][0] - k["collar_z0"]) / s.t))
            pcs = pcs + [("Collar ring (from 2 half-rings)", k["OD"], k["OD"])] * n_rings
        area = sum(a * b for _, a, b in pcs)
        sw, sh = s.sheet_size
        n = max(1, math.ceil(area / (sw * sh) * 1.2 - 1e-9))
        # pieces larger than a sheet in either direction still need at least as many sheets
        n = max(n, max(math.ceil(a / sh) * math.ceil(b / sw) for _, a, b in pcs))
        desc = "; ".join(f"{nm} {fmt_ftin(a)} x {fmt_ftin(b)}" for nm, a, b in pcs)
        out.append(dict(stock=sk, label=s.description or s.label, qty=n, pieces=desc, area=area / 144))
    return out


def hardware(model):
    cfg, k = model.cfg, model.key
    pit = cfg["pit"]
    rows = []

    def add(item, qty, note=""):
        rows.append(dict(item=item, qty=qty, note=note))

    pipe_len = k["pipe_top"] - k["pipe_bot"]
    add(pit["pipe"]["description"], fmt_ftin(pipe_len), f"cut square; top at {fmt_in(k['pipe_top'])} above grade")
    add("EPDM rubber strip, 45 mil", f"{fmt_ftin(math.pi * k['OD'] + 6)} x {fmt_in(L(pit['flange']['skirt_width']))}",
        "skirt around pipe / collar joint")
    add("Stainless worm-gear band clamp, 36\"+ (or strapping)", 2, "over skirt, above and below joint")
    add("Closed-cell foam sill gasket", fmt_ftin(math.pi * k["OD"]), "under flange collar")
    if k.get("vent"):
        vx, vy, vr, vtop = k["vent"]
        add(pit["vent"]["description"], fmt_ftin(vtop - k["flange_z"][0] + 2), "cut to fit; paint black")
        add("Roof pipe boot (EPDM, corrugated-metal type) for 4\" pipe", 1)
        add("Vent cap / tee with stainless fly screen", 1)
    rf = cfg["roof"]["roofing"]
    area_ft2 = (k["roof_x"][1] - k["roof_x"][0]) * (k["roof_y"][1] - k["roof_y"][0]) * k["c"] / 144
    L_x = k["roof_x"][1] - k["roof_x"][0]
    if k.get("roof_profile") == "standing_seam":
        add(rf["description"], f"{k['roof_sheets']} @ {fmt_ftin(math.ceil(k['roof_sheet_len']))}",
            f"~{area_ft2:.0f} sq ft; order cut to length, one piece eave to eave")
        add("Synthetic high-temp roof underlayment", f"{math.ceil(area_ft2 * 1.15 / 100)} sq", "over the plywood deck")
        add("Pancake-head #10 x 1\" screws for panel nail strips", int(math.ceil(k["roof_sheets"] * 7 / 50) * 50),
            "12\" o.c. along each panel")
        add("Eave drip trim + low-eave closure", fmt_ftin(L_x), "")
        add("High-side (shed ridge) trim w/ Z-closure", fmt_ftin(L_x), "")
        add("Rake trim", fmt_ftin(2 * k["roof_sheet_len"]), "both rakes")
        add("H-clips for 5/8\" plywood", int(math.ceil(L_x / 24) * 3), "between rafters at unsupported edges")
    else:
        add(rf["description"], f"{k['roof_sheets']} @ {fmt_ftin(math.ceil(k['roof_sheet_len']))}",
            f"~{area_ft2:.0f} sq ft; order cut-to-length")
        add("#9 x 1-1/2\" roofing screws w/ EPDM washers", int(math.ceil(area_ft2 * 0.9 / 50) * 50), "~0.9 per sq ft")
        add("Foam closure strips, inside + outside", fmt_ftin(2 * L_x), "low eave and high edge")
        add("Metal rake trim / peak flashing", fmt_ftin(2 * k["roof_sheet_len"] + L_x), "optional")
    add("Z-flashing, galvanized", fmt_ftin(k["W"] + 2), "head of removable panel")
    add(cfg["window"]["description"], 1, f"RO {fmt_in(k['window']['u1'] - k['window']['u0'])} x "
        f"{fmt_in(k['window']['v1'] - k['window']['v0'])}")
    if k.get("door_prehung"):
        add(cfg["door"]["description"], 1, f"RO {fmt_in(k['door']['u1'] - k['door']['u0'])} x "
            f"{fmt_in(k['door']['v1'] - k['door']['v0'])}; " + cfg["door"]["hardware"])
        add("Sill pan flashing + flashing tape", 1, "door rough opening")
    else:
        add("Door hardware: " + cfg["door"]["hardware"], 1)
    if cfg.get("room_vent", {}).get("enabled"):
        add(cfg["room_vent"].get("description", "Louver vent"), 1)
    add("Toilet seat, elongated, with lid", 1, "mount with stainless hinge bolts through bench top")
    add("Face-mount hangers for 4x6 (bench header to corner studs)", 2)
    add("Face-mount hangers for 2x4 (bench joists)", 4)
    add("Galv. 4x4 post base / standoff", 2, "fasten to skid")
    add("1/2\" x 7\" HDG carriage bolts w/ nuts & washers", 4, "posts to deck rim")
    add("1/2\" x 8\" HDG carriage bolts", 4, "beam to post (or post cap)")
    add("Let-in steel wall-bracing strap, 10'", sum(1 for p in model.parts if "strap" in p.tags), "")
    add(cfg["removable_panel"]["fasteners"], 1, "panel to corner studs, header, rims")
    add("3/4\" galv. shackle or clevis", 4, "tow points, both skid ends")
    el = cfg.get("electrical", {})
    if el.get("enabled"):
        fd = el.get("feed", {})
        awg = k.get("feed_awg", 12)
        bury = parse_len(el.get("burial_depth", 24))
        ds_z = next(d["z"] for d in k["electrical"]["devices"] if d["tag"] == "DS")
        svc = k.get("service")
        if svc:
            sv = el["service"]
            add(sv.get("description", "Meter-main combination"), 1, "MP; utility-approved socket")
            add("20 A 1-pole GFCI breaker (listed for the MP panel)", 1, "privy branch circuit")
            add("5/8\" x 8' copper-clad ground rods + acorn clamps", 2, "at MP, 6' min apart")
            add("#6 bare Cu grounding electrode conductor", "20 ft", "MP to rods")
            add("Service entrance conduit, riser and fittings (per utility spec)", "1 lot", "lateral into MP")
            add(f"{awg}-2 w/ ground UF-B cable (feed)", f"{math.ceil(svc['run_ft'] + 2 * (bury + 60) / 12 + 10)} ft",
                "MP to DS, direct buried")
            add("3/4\" Sch 80 PVC conduit + sweeps, straps", fmt_ftin(bury + 66 + bury + ds_z + 24),
                "feed risers at MP and DS")
        else:
            add(f"{awg}-2 w/ ground UF-B cable (feed)", f"{fd.get('length_ft', 100) + 15} ft", "source panel to DS")
            add("20 A 1-pole GFCI breaker (match source panel)", 1, "")
            add("3/4\" Sch 80 PVC conduit + LB/fittings", fmt_ftin(bury + ds_z + 24), "feed riser")
        add("12-2 w/ ground UF-B cable (in building)", "50 ft", "DS to devices and lights")
        add("Weatherproof disconnect switch, 20 A, with box", 1, "DS")
        add("Weatherproof boxes + covers (switch / in-use receptacle)", 2, "S2, R2")
        add("Old-work / surface boxes + plates", 2, "S1, R1")
        add("20 A switches", 2, "S1, S2")
        add("20 A GFCI receptacles (TR; WR for exterior)", 2, "R1, R2")
        add("Damp-rated LED ceiling fixtures", 2, "L1 interior, L2 deck")
        add("Round fixture boxes (pancake) + cable staples / connectors", "1 lot", "")
    nb = sum(1 for p in model.parts if p.name == "Siding board")
    add("8d ring-shank HDG siding nails", f"{math.ceil(nb * 12 / 100) * 100}", "~2 per board per nailer")
    add("10d ring-shank HDG batten nails", f"{math.ceil(sum(1 for p in model.parts if p.name == 'Batten') * 6 / 50) * 50}",
        "through the gap, 16\" o.c.")
    add("16d HDG framing nails / 3\" structural screws", "10 lb", "")
    add("Exterior deck screws 2-1/2\"", "5 lb", "decking and trim")
    return rows


def totals(purchase_rows):
    return sum(r["bf"] for r in purchase_rows)
