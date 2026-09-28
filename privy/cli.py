"""Command line: python -m privy build design/privy.json [-o build]"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

from . import bom
from .model import build as build_model, load
from .units import fmt_ftin


def validate(cfg, schema_path):
    try:
        import jsonschema
    except ImportError:
        return
    if os.path.exists(schema_path):
        with open(schema_path) as f:
            jsonschema.validate(cfg, json.load(f))


def cmd_build(a):
    t0 = time.time()
    cfg = load(a.design)
    here = os.path.dirname(os.path.abspath(a.design))
    validate(cfg, os.path.join(here, "..", "schema", "privy.schema.json"))
    print(f"model: {a.design}")
    m = build_model(cfg)
    print(f"  {len(m.parts)} parts, est. weight {m.key['weight']:,.0f} lb")
    fails = [c for c in m.checks if c["ok"] is False]
    for c in fails:
        print(f"  CHECK FAILED: {c['title']}: {c['value']} (limit {c['limit']})")
    os.makedirs(a.out, exist_ok=True)

    renders = []
    if not a.no_render:
        try:
            from .render import render_views
            print("rendering...")
            renders = render_views(m, a.out, quality=a.quality, log=lambda s: print("  " + s))
        except Exception as e:  # keep drawings usable if the renderer is unavailable
            print(f"  rendering skipped: {type(e).__name__}: {e}")
    else:
        rd = os.path.join(a.out, "renders")
        for v in cfg.get("renders", {}).get("views", []):
            p = os.path.join(rd, f"{v['name']}.png")
            if os.path.exists(p):
                renders.append(dict(name=v["name"], title=v.get("title", v["name"]), path=p))
        if renders:
            print(f"  reusing {len(renders)} existing renders from {rd}")

    from .drawing.sheets import build_sheets
    print("drawing sheets...")
    sheets = build_sheets(m, renders, source_name=os.path.relpath(a.design), only=a.only)
    sd = os.path.join(a.out, "svg")
    os.makedirs(sd, exist_ok=True)
    for n, t, svg in sheets:
        with open(os.path.join(sd, f"{n}.svg"), "w") as f:
            f.write(svg)
    print(f"  {len(sheets)} SVG sheets -> {sd}")

    # cut list CSV + checks JSON
    cl = bom.cut_list(m)
    with open(os.path.join(a.out, "cut_list.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["area", "stock", "piece", "qty", "length_in", "length", "note"])
        for r in cl:
            wr.writerow([r["group"], r["label"], r["name"], r["qty"], r["length"], fmt_ftin(r["length"]), r["note"]])
    with open(os.path.join(a.out, "checks.json"), "w") as f:
        json.dump(m.checks, f, indent=1, default=str)

    if not a.no_pdf:
        from .pdf import svgs_to_pdf
        name = os.path.splitext(os.path.basename(a.design))[0]
        pdf = os.path.join(a.out, f"{name}.pdf")
        svgs_to_pdf(sheets, pdf, meta=dict(Title=cfg["project"]["name"], Subject=cfg["project"].get("subtitle", ""),
                                           Creator="privy (parametric JSON -> SVG -> PDF)"))
        print(f"  PDF -> {pdf}")
    print(f"done in {time.time() - t0:.1f}s")
    return 1 if (fails and a.strict) else 0


def fmt_json(v, ind=0, width=112):
    """Compact, hand-editable JSON: short objects/arrays stay on one line."""
    sp = "  " * ind
    one = json.dumps(v, ensure_ascii=False)
    if not isinstance(v, (dict, list)) or len(one) + len(sp) <= width:
        return one
    if isinstance(v, list):
        return "[\n" + ",\n".join("  " * (ind + 1) + fmt_json(x, ind + 1) for x in v) + "\n" + sp + "]"
    items = []
    for k, x in v.items():
        key = json.dumps(k) + ": "
        items.append("  " * (ind + 1) + key + fmt_json(x, ind + 1, width - len(key)))
    return "{\n" + ",\n".join(items) + "\n" + sp + "}"


def cmd_fmt(a):
    cfg = load(a.design)
    with open(a.design, "w") as f:
        f.write(fmt_json(cfg) + "\n")
    print(f"formatted {a.design}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="privy", description="Parametric privy: JSON -> SVG blueprints -> PDF")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build drawings, renders and PDF")
    b.add_argument("design", nargs="?", default="design/privy.json")
    b.add_argument("-o", "--out", default="build")
    b.add_argument("--no-render", action="store_true", help="skip path tracing (reuse existing renders if present)")
    b.add_argument("--no-pdf", action="store_true")
    b.add_argument("--quality", type=float, default=1.0, help="render quality scale (0.3 = fast draft)")
    b.add_argument("--only", nargs="*", help="only these sheet numbers, e.g. A-101 A-301")
    b.add_argument("--strict", action="store_true", help="exit non-zero if any design check fails")
    f_ = sub.add_parser("fmt", help="re-format the design JSON (compact, hand-editable)")
    f_.add_argument("design", nargs="?", default="design/privy.json")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        sys.exit(cmd_build(a))
    if a.cmd == "fmt":
        sys.exit(cmd_fmt(a))


if __name__ == "__main__":
    main()
