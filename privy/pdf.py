"""SVG sheets -> single PDF (cairosvg per page, merged with pypdf, with bookmarks)."""
from __future__ import annotations

import io

import cairosvg
from pypdf import PdfReader, PdfWriter


def svgs_to_pdf(sheets, path, meta=None):
    """sheets: [(number, title, svg_string)]"""
    w = PdfWriter()
    for number, title, svg in sheets:
        buf = io.BytesIO()
        cairosvg.svg2pdf(bytestring=svg.encode("utf-8"), write_to=buf)
        buf.seek(0)
        r = PdfReader(buf)
        for pg in r.pages:
            w.add_page(pg)
        w.add_outline_item(f"{number}  {title}", len(w.pages) - 1)
    if meta:
        w.add_metadata({f"/{k}": str(v) for k, v in meta.items()})
    w.page_mode = "/UseOutlines"
    with open(path, "wb") as f:
        w.write(f)
    return path
