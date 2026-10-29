#!/usr/bin/env python3
# pdf-tool: read a PDF's text runs, and rewrite some of them in place. AGPL-3.0 (built on PyMuPDF), see LICENSE.
#
#   pdf_tool.py extract in.pdf                  -> JSON on stdout: every horizontal run of text
#                                                  {page, line, bbox, origin, text, font, size, color, hidden}
#   pdf_tool.py apply in.pdf edits.json out.pdf -> each edit's box erased (its text only) and its new text written
#
# An edit: {"page", "bbox", "origin", "text", "font", "size", "color", "was"?, "align"?: "right", "mark"?: true}.
# An empty bbox erases nothing; an empty text writes nothing. The new text keeps the PDF's own font when it
# has every glyph, else the closest standard one (Helvetica, Times, Courier), bold and italic kept, made as
# dark as the text it replaces (`was`, the box's text before).
# Exit 2: the PDF is protected; exit 3: a character no font here can draw.
import json
import sys

import pymupdf


def runs(page, line_no):
    """The page's horizontal text, in runs: neighbouring spans of a line in the same font, size and colour."""
    out = []
    for b in page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)["blocks"]:
        for line in b.get("lines", []):
            # shortcut: rotated or vertical text is left out (never changed), until a customer's file needs it.
            if line["wmode"] or abs(line["dir"][1]) > 1e-3 or line["dir"][0] < 0:
                continue
            line_no += 1
            cur = None
            for s in line["spans"]:
                key = (s["font"], round(s["size"], 2), s["color"], s["alpha"] == 0)
                if cur and cur["key"] == key:
                    cur["text"] += s["text"]
                    cur["bbox"] = list(pymupdf.Rect(cur["bbox"]) | s["bbox"])
                else:
                    cur = {"key": key, "page": page.number, "line": line_no, "text": s["text"], "bbox": list(s["bbox"]), "origin": list(s["origin"]), "font": s["font"], "size": s["size"], "color": s["color"], "hidden": s["alpha"] == 0}
                    out.append(cur)
    for r in out:
        del r["key"]
    return out, line_no


def extract(path):
    doc = pymupdf.open(path)
    if doc.needs_pass:
        sys.exit(2)
    out, line_no = [], 0
    for page in doc:
        found, line_no = runs(page, line_no)
        out += found
    return out


def fallback(name):
    """The standard font closest to `name` (MuPDF's built-in Nimbus fonts), bold and italic kept."""
    n = name.lower()
    family = "co" if any(k in n for k in ("mono", "courier", "cmtt", "code", "consol", "typewriter")) else "ti" if any(k in n for k in ("times", "serif", "roman", "georgia", "garamond", "cmr", "feymr")) and "sans" not in n else "he"
    bold, italic = any(k in n for k in ("bold", "black", "heavy", "semibold", "demi", "feybr")), any(k in n for k in ("italic", "oblique"))
    return pymupdf.Font({(0, 0): {"he": "helv", "ti": "tiro", "co": "cour"}[family], (1, 0): family + "bo", (0, 1): family + "it", (1, 1): family + "bi"}[(bold, italic)])


def draws(font, text):
    return all(font.has_glyph(ord(c)) for c in text if not c.isspace())


def own_font(doc, page, name, text):
    """The PDF's font named `name`, if it can draw every character of `text` (a subset often lacks some)."""
    for xref, _, _, base, *_ in page.get_fonts(full=True):
        if base.split("+")[-1] != name:
            continue
        try:
            buf = doc.extract_font(xref)[3]
            font = pymupdf.Font(fontbuffer=buf) if buf else None
        except Exception:
            return None
        return font if font and draws(font, text) else None
    return None


def ink(page, rect):
    """How much of `rect` is dark: the weight of the text in it."""
    pix = page.get_pixmap(clip=rect, dpi=300, colorspace=pymupdf.csGRAY)
    return sum(1 for b in pix.samples if b < 128) / max(1, len(pix.samples))


def write(page, font, x, y, text, size, color, bold=0.0):
    """`text` at (x, y); `bold`: drawn again that much (a share of its size) to the right, a heavier stroke."""
    for dx in (0, bold * size) if bold else (0,):
        tw = pymupdf.TextWriter(page.rect, color=color)
        tw.append((x + dx, y), text, font=font, fontsize=size)
        tw.write_text(page)


def heavier(page, e, font, color):
    """How much heavier a stand-in font is drawn to look as dark as the text it replaces (its box, before)."""
    r, was = pymupdf.Rect(e["bbox"]), e.get("was", "").strip()
    if r.is_empty or not was:
        return 0.0
    target = ink(page, r)
    # A dark box (white text on a fill) says nothing of the text's weight.
    if not target or target > 0.6:
        return 0.0
    best, gap = 0.0, None
    for bold in (0.0, 0.01, 0.02, 0.03, 0.04, 0.05):
        sheet = pymupdf.open().new_page(width=r.x1 + 20, height=r.y1 + 20)
        write(sheet, font, r.x0, e["origin"][1], was, e["size"], color, bold)
        width = font.text_length(was, fontsize=e["size"]) + bold * e["size"]
        d = abs(ink(sheet, pymupdf.Rect(r.x0, r.y0, r.x0 + width, r.y1)) - target)
        if gap is None or d < gap:
            best, gap = bold, d
    return best


def apply(src, edits, dst):
    doc = pymupdf.open(src)
    if doc.needs_pass:
        sys.exit(2)
    by_page = {}
    for e in edits:
        by_page.setdefault(e["page"], []).append(e)
    for n, todo in by_page.items():
        page = doc[n]
        own = [own_font(doc, page, e["font"], e["text"]) for e in todo]
        fonts = [o or fallback(e["font"]) for o, e in zip(own, todo)]
        if any(not draws(f, e["text"]) for f, e in zip(fonts, todo)):
            sys.exit(3)
        colors = [tuple(((e["color"] >> s) & 255) / 255 for s in (16, 8, 0)) for e in todo]
        # Measured before the boxes are erased: a stand-in font as dark as the text it replaces.
        bolds = [0.0 if o else heavier(page, e, f, c) for o, e, f, c in zip(own, todo, fonts, colors)]
        # Erase only the text in each box, its middle (the lines above and below stay): images and drawings stay.
        erase = [pymupdf.Rect(e["bbox"]) for e in todo]
        for r in erase:
            if not r.is_empty:
                page.add_redact_annot(pymupdf.Rect(r.x0 + 0.5, r.y0 + r.height * 0.2, r.x1 - 0.5, r.y1 - r.height * 0.2))
        if any(not r.is_empty for r in erase):
            page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE, graphics=pymupdf.PDF_REDACT_LINE_ART_NONE, text=pymupdf.PDF_REDACT_TEXT_REMOVE)
        for font, color, bold, e in zip(fonts, colors, bolds, todo):
            if not e["text"].strip():
                continue
            width = font.text_length(e["text"], fontsize=e["size"])
            # "align": "right" keeps the run's right edge (an amount in a table's column).
            x, y = (e["bbox"][2] - width, e["origin"][1]) if e.get("align") == "right" else e["origin"]
            if e.get("mark"):
                page.draw_rect(pymupdf.Rect(x - 1.5, y - e["size"] * 0.85, x + width + 1.5, y + e["size"] * 0.3), color=None, fill=(1, 0.72, 0.38), fill_opacity=0.45)
            write(page, font, x, y, e["text"], e["size"], color, bold)
    # The fonts added, reduced to the glyphs used.
    doc.subset_fonts()
    doc.save(dst, garbage=3, deflate=True)


if __name__ == "__main__":
    cmd, *args = sys.argv[1:] or ["help"]
    if cmd == "extract" and len(args) == 1:
        json.dump(extract(args[0]), sys.stdout, ensure_ascii=False)
    elif cmd == "apply" and len(args) == 3:
        with open(args[1]) as f:
            apply(args[0], json.load(f), args[2])
    else:
        sys.exit("usage: pdf_tool.py extract in.pdf | apply in.pdf edits.json out.pdf")
