#!/usr/bin/env python3
# pdf-tool: read a PDF's text runs, and rewrite some of them in place. AGPL-3.0 (built on PyMuPDF), see LICENSE.
#
#   pdf_tool.py extract in.pdf                  -> JSON on stdout: every horizontal run of text
#                                                  {page, line, bbox, origin, text, font, size, color, hidden, ocr?}
#   pdf_tool.py apply in.pdf edits.json out.pdf -> each edit's box erased (its text only) and its new text written
#   pdf_tool.py stamp in.pdf marks.json out.pdf -> a signature, a line of text, a drawing placed on its pages, its fields filled, all written into them
#
# `in.pdf` may be an image too, a photo or a scan: it becomes a page of its size, and `out.pdf` a PDF. A page with no
# text of its own is read by OCR (Tesseract, in PyMuPDF's wheel; its language data in tessdata/, setup.sh): its runs
# come with `ocr`, `font` being "helv" or "hebo" by the weight of their strokes.
#
# An edit: {"page", "bbox", "origin", "text", "font", "size", "color", "was"?, "align"?: "right", "mark"?: true, "ocr"?: true}.
# An empty bbox erases nothing; an empty text writes nothing. The new text keeps the PDF's own font when it
# has every glyph, else the closest standard one (Helvetica, Times, Courier), bold and italic kept, made as
# dark as the text it replaces (`was`, the box's text before). A page with `ocr` edits is drawn again as a picture:
# each box's pixels replaced by the paper around them (nothing of what was there is left in the file), its new text
# in `font`, in the shade of the text it replaces.
#
# A mark: {"page", "kind": "image" | "text" | "path" | "field", "x", "y", "w", "h", "image"? (a PNG, base64), "text"?, "size"?,
# "path"?, "ref"?, "value"?}, in points from the top left of the page as it shows (turned, cropped); a text in Helvetica, its
# baseline 0.8 of its size under "y"; a path's strokes, black lines "size" thick, their points from 0 to 1 across and down its
# box. A field: "ref" the object number of one of its widgets, "value" its text (its option's words), or "On" or
# "Off" for that checkbox or radio button; every widget of the field follows, a list shows the option chosen alone.
# Form fields and annotations are drawn into the pages first: nothing covers the marks, nothing stays to change (links do).
# Exit 2: the PDF is protected (it opens only with a password); exit 3: a character no font here can draw; exit 4: no such page;
# exit 5: no such field, or it doesn't take that value.
import base64
import io
import json
import os
import re
import sys

import numpy as np
import pillow_heif
import pymupdf
from PIL import Image, ImageOps, ImageSequence

pillow_heif.register_heif_opener()

TESSDATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tessdata")
LANGUAGES = "fra+eng+spa"
# Below this many characters of its own, a page that is mostly a picture is a scan (a scanner's footer on it).
SCAN_CHARS = 40
# How wide a page is read and drawn again, in pixels: A4 at 300 dpi.
PAGE_PIXELS = 2480
# A stroke this much thicker than the page's usual one is bold (measured on scans and photos: regular up to 1.35, bold from 1.8).
BOLD_STROKE = 1.5
# This share of a page dark: maybe light text on a fill (a table's head), read in a second pass.
DARK_SHARE = 0.03
# What the OCR reads on a photo with no text is gibberish: this share of its characters at least must be in real words or numbers (measured: gibberish 0.45, documents 0.75).
WORDY_SHARE = 0.6
# Pages of a scan read at most: each takes a second or two, and pdf-tool has a minute (pdf-tool.ts).
OCR_PAGES = 15


def load(path):
    """The file as a PDF: an image becomes a page of its size (one per frame of a TIFF), upright (a phone's says how in its EXIF)."""
    with open(path, "rb") as f:
        data = f.read()
    try:
        doc = pymupdf.open(stream=data)
        if doc.is_pdf:
            return doc
    except pymupdf.FileDataError:
        pass
    pages = pymupdf.open()
    with Image.open(io.BytesIO(data)) as image:
        dpi = image.info.get("dpi", (0, 0))[0] or 150
        for frame in ImageSequence.Iterator(image):
            upright = ImageOps.exif_transpose(frame).convert("RGB")
            jpeg = io.BytesIO()
            upright.save(jpeg, "JPEG", quality=92)
            page = pages.new_page(width=upright.width * 72 / dpi, height=upright.height * 72 / dpi)
            page.insert_image(page.rect, stream=jpeg.getvalue())
    return pymupdf.open("pdf", pages.tobytes())


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


def scanned(page, found):
    """Nothing to read on the page but its picture: no text of its own, or a few words on a page that is mostly a picture."""
    chars = sum(len(r["text"].strip()) for r in found if not r["hidden"])
    if not chars:
        return True
    area = sum(abs(pymupdf.Rect(i["bbox"]) & page.rect) for i in page.get_image_info())
    return chars < SCAN_CHARS and area > abs(page.rect) / 2


def gray_of(page, zoom):
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csGRAY)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.stride)[:, :pix.width]


def evened(a):
    """A photo's light evened out: each pixel over the paper around it (the bright end of a coarse grid, smoothed)."""
    h, w = a.shape
    k = max(w // 12, 64)
    rows, cols = h // k + 2, w // k + 2
    grid = np.empty((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            cell = a[max(0, (i - 1) * k + k // 2):(i + 1) * k + k // 2, max(0, (j - 1) * k + k // 2):(j + 1) * k + k // 2]
            grid[i, j] = np.percentile(cell, 90) if cell.size else 255
    ys, xs = np.linspace(0, rows - 1, h), np.linspace(0, cols - 1, w)
    y0, x0 = ys.astype(int), xs.astype(int)
    y1, x1 = np.minimum(y0 + 1, rows - 1), np.minimum(x0 + 1, cols - 1)
    fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
    paper = (grid[y0][:, x0] * (1 - fx) + grid[y0][:, x1] * fx) * (1 - fy) + (grid[y1][:, x0] * (1 - fx) + grid[y1][:, x1] * fx) * fy
    return np.clip(a / np.maximum(paper, 1) * 255, 0, 255).astype(np.uint8)


def ocr_lines(a, rect, dpi):
    """What Tesseract reads on a picture `a`: [(text, box in the page's points)]."""
    pix = pymupdf.Pixmap(pymupdf.csGRAY, a.shape[1], a.shape[0], a.tobytes(), False)
    page = pymupdf.open().new_page(width=rect.width, height=rect.height)
    page.insert_image(page.rect, pixmap=pix)
    textpage = page.get_textpage_ocr(language=LANGUAGES, dpi=dpi, full=True, tessdata=TESSDATA)
    out = []
    for b in page.get_text("dict", textpage=textpage, flags=pymupdf.TEXTFLAGS_TEXT)["blocks"]:
        for line in b.get("lines", []):
            text = "".join(s["text"] for s in line["spans"]).strip(" |")
            if re.search(r"\w", text):
                out.append((text, pymupdf.Rect(line["bbox"])))
    return out


def measure(a, box, zoom):
    """A line in the picture: its box close around its ink (the OCR's is loose on a photo), its foot, its strokes in pixels."""
    x0, y0, x1, y1 = (round(v * zoom) for v in box)
    crop = a[max(0, y0):y1, max(0, x0):x1]
    if not crop.size:
        return box, box.y1, 0
    paper = float(np.median(crop))
    far = float(crop.min() if paper >= 128 else crop.max())
    if abs(paper - far) < 24:
        return box, box.y1, 0
    ink = crop < (paper + far) / 2 if paper >= 128 else crop > (paper + far) / 2
    rows = ink.sum(1)
    peak = int(rows.argmax())
    top, bottom = peak, peak
    while top > 0 and rows[top - 1]:
        top -= 1
    while bottom < len(rows) - 1 and rows[bottom + 1]:
        bottom += 1
    well_inked = np.nonzero(rows[top:bottom + 1] >= rows[peak] * 0.3)[0]
    foot = top + int(well_inked[-1]) + 1
    edges = np.diff(np.pad(ink[top:bottom + 1].astype(np.int8), ((0, 0), (1, 1))), axis=1)
    runs_of_ink = np.nonzero(edges == -1)[1] - np.nonzero(edges == 1)[1]
    origin = max(0, y0)
    return pymupdf.Rect(box.x0, (origin + top) / zoom, box.x1, (origin + bottom + 1) / zoom), (origin + foot) / zoom, float(np.median(runs_of_ink)) if runs_of_ink.size else 0


def read_picture(page, line_no):
    """A page read by OCR, a run per line: dark text on its paper, then light text on a fill; each sized to fill its box in Helvetica, bold when its strokes are thick."""
    zoom = PAGE_PIXELS / page.rect.width
    a = evened(gray_of(page, zoom))
    dpi = round(zoom * 72)
    lines = ocr_lines(a, page.rect, dpi)
    if (a < 170).mean() > DARK_SHARE:
        fill = lambda b: np.median(a[round(b.y0 * zoom):round(b.y1 * zoom), round(b.x0 * zoom):round(b.x1 * zoom)]) < 170
        lines += [(t, b) for t, b in ocr_lines(255 - a, page.rect, dpi) if fill(b) and not any(abs(b & o) > abs(b) * 0.3 for _, o in lines)]
    # A box a tenth of the page high is the OCR reading a shadow.
    lines = [(t, b) for t, b in lines if b.height < page.rect.height / 10]
    wordy = sum(len(w) for t, _ in lines for w in re.findall(r"[^\W\d_]{4,}|\d[\d .,]*\d", t))
    if wordy < sum(len(t) for t, _ in lines) * WORDY_SHARE:
        return [], line_no
    helv, hebo = pymupdf.Font("helv"), pymupdf.Font("hebo")
    found = []
    for text, box in lines:
        tight, foot, stroke = measure(a, box, zoom)
        size = box.width / helv.text_length(text, fontsize=1)
        found.append((text, box if tight.is_empty else tight, foot, stroke / zoom / max(size, 1)))
    usual = float(np.median([f[3] for f in found])) if found else 0
    out = []
    for text, box, foot, stroke in sorted(found, key=lambda f: (f[1].y1, f[1].x0)):
        line_no += 1
        font = hebo if usual and len(text) > 3 and stroke >= usual * BOLD_STROKE else helv
        size = box.width / font.text_length(text, fontsize=1) if len(text) > 3 else box.height / 0.72
        out.append({"page": page.number, "line": line_no, "text": text, "bbox": list(box), "origin": [box.x0, foot], "font": "hebo" if font is hebo else "helv", "size": round(size, 2), "color": 0, "hidden": False, "ocr": True})
    return out, line_no


def extract(path):
    doc = load(path)
    if doc.needs_pass:
        sys.exit(2)
    out, line_no = [], 0
    read = 0
    for page in doc:
        found, line_no = runs(page, line_no)
        # shortcut: a scan's pages after the 15th have no text to click; upgrade when a client has such a file.
        if read < OCR_PAGES and scanned(page, found) and all(os.path.isfile(os.path.join(TESSDATA, f"{l}.traineddata")) for l in LANGUAGES.split("+")):
            read += 1
            found, line_no = read_picture(page, line_no)
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
    """`text` at (x, y); `bold`: its outline drawn too, that thick (a share of its size), a heavier stroke at any zoom."""
    tw = pymupdf.TextWriter(page.rect, color=color)
    tw.append((x, y), text, font=font, fontsize=size)
    tw.write_text(page, render_mode=2 if bold else 0)
    if bold:
        # TextWriter sets its own outline width: this one instead.
        xref = page.get_contents()[-1]
        page.parent.update_stream(xref, re.sub(rb"\n2 Tr\n[\d.]+ w", b"\n2 Tr\n%.3f w" % (bold * size), page.parent.xref_stream(xref), count=1))


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
    for bold in (0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04):
        sheet = pymupdf.open().new_page(width=r.x1 + 20, height=r.y1 + 20)
        write(sheet, font, r.x0, e["origin"][1], was, e["size"], color, bold)
        width = font.text_length(was, fontsize=e["size"])
        d = abs(ink(sheet, pymupdf.Rect(r.x0, r.y0, r.x0 + width, r.y1)) - target)
        if gap is None or d < gap:
            best, gap = bold, d
    return best


def redraw(doc, n, todo):
    """A page's picture drawn again without what `todo` covers: each box takes the paper just above and below it. Back: the page, its text shades on the edits."""
    page = doc[n]
    # Its own resolution when it has a single picture, at most an A4's at 300 dpi.
    pictures = page.get_image_info()
    zoom = min(PAGE_PIXELS / page.rect.width, max((i["width"] / abs(pymupdf.Rect(i["bbox"]).width) for i in pictures), default=PAGE_PIXELS / page.rect.width))
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csRGB)
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.stride // 3, 3)[:, :pix.width].astype(np.float32)
    for e in todo:
        r = pymupdf.Rect(e["bbox"])
        if r.is_empty:
            continue
        x0, y0 = max(0, round(r.x0 * zoom) - 2), max(1, round(r.y0 * zoom) - 3)
        x1, y1 = min(pix.width, round(r.x1 * zoom) + 2), min(pix.height - 1, round(r.y1 * zoom) + 3)
        crop = a[y0:y1, x0:x1].reshape(-1, 3)
        gray = crop.mean(1)
        # Its paper, the box's commonest shade; its text, the furthest from it.
        paper = np.median(crop, 0)
        far = np.abs(gray - paper.mean())
        ink = crop[far >= np.percentile(far, 95)].mean(0)
        e["color"] = sum(int(round(c)) << s for c, s in zip(ink, (16, 8, 0)))
        # Each column from the rows just outside the box, smoothed along the line so a stray tail leaves no streak.
        size = 9
        pad = lambda row: np.stack([np.convolve(np.pad(row[:, c], size // 2, mode="edge"), np.ones(size) / size, "valid") for c in range(3)], 1)
        top, bottom = pad(np.median(a[max(0, y0 - 2):y0, x0:x1], 0)), pad(np.median(a[y1:y1 + 2, x0:x1], 0))
        fade = np.linspace(0, 1, y1 - y0)[:, None, None]
        a[y0:y1, x0:x1] = top[None] * (1 - fade) + bottom[None] * fade
    out = pymupdf.Pixmap(pymupdf.csRGB, pix.width, pix.height, np.clip(a, 0, 255).astype(np.uint8).tobytes(), False)
    picture = out.tobytes("jpeg", jpg_quality=88)
    rect = page.rect
    doc.delete_page(n)
    page = doc.new_page(pno=n, width=rect.width, height=rect.height)
    page.insert_image(page.rect, stream=picture)
    return page


def apply(src, edits, dst):
    doc = load(src)
    if doc.needs_pass:
        sys.exit(2)
    by_page = {}
    for e in edits:
        by_page.setdefault(e["page"], []).append(e)
    for n, todo in by_page.items():
        if any(e.get("ocr") for e in todo):
            page = redraw(doc, n, todo)
            own = [None] * len(todo)
            fonts = [pymupdf.Font(e["font"]) for e in todo]
            bolds = [0.0] * len(todo)
            colors = [tuple(((e["color"] >> s) & 255) / 255 for s in (16, 8, 0)) for e in todo]
            if any(not draws(f, e["text"]) for f, e in zip(fonts, todo)):
                sys.exit(3)
        else:
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


def state_on(mupdf, pdf, xref):
    """The name of the state a checkbox or a radio button takes when on: the one of its looks that isn't "Off"; None, it has none."""
    looks = mupdf.pdf_dict_get(mupdf.pdf_dict_get(mupdf.pdf_new_indirect(pdf, xref, 0), mupdf.PDF_ENUM_NAME_AP), mupdf.PDF_ENUM_NAME_N)
    names = [mupdf.pdf_to_name(mupdf.pdf_dict_get_key(looks, i)) for i in range(mupdf.pdf_dict_len(looks))] if mupdf.pdf_is_dict(looks) and not mupdf.pdf_is_stream(looks) else []
    return next((n for n in names if n != "Off"), None)


def fill(doc, marks):
    """The form's fields with their values: each widget's look made again by MuPDF, a box's or a button's state chosen among its own."""
    if not marks:
        return
    mupdf = pymupdf.mupdf
    pdf = mupdf.pdf_document_from_fz_document(doc)
    # (its number, its field's name, its type, its flags, its options, its longest text, its state when on), by widget.
    widgets = [(w.xref, w.field_name, w.field_type, w.field_flags, w.choice_values, w.text_maxlen, state_on(mupdf, pdf, w.xref)) for page in doc for w in page.widgets()]
    for m in marks:
        of = next((w for w in widgets if w[0] == m["ref"]), None)
        if not of:
            sys.exit(5)
        same = [w for w in widgets if w[0] == m["ref"] or (of[1] and w[1] == of[1] and w[2] == of[2])]
        text, kind, flags = m["value"], of[2], of[3]
        if kind in (pymupdf.PDF_WIDGET_TYPE_CHECKBOX, pymupdf.PDF_WIDGET_TYPE_RADIOBUTTON):
            # Its other widgets go off, but those of the same state (a button drawn twice).
            state = {w[0]: w[6] for w in same}
            on = state[m["ref"]] if text == "On" else None
            if text not in ("On", "Off") or (text == "On" and not on):
                sys.exit(5)
            for w in same:
                obj = mupdf.pdf_new_indirect(pdf, w[0], 0)
                mupdf.pdf_dict_put_name(obj, mupdf.PDF_ENUM_NAME_AS, on if on and w[6] == on else "Off")
            continue
        if kind not in (pymupdf.PDF_WIDGET_TYPE_TEXT, pymupdf.PDF_WIDGET_TYPE_COMBOBOX, pymupdf.PDF_WIDGET_TYPE_LISTBOX):
            sys.exit(5)
        if not draws(pymupdf.Font("helv"), text):
            sys.exit(3)
        if kind == pymupdf.PDF_WIDGET_TYPE_TEXT:
            text = text[: of[5]] if of[5] else text
        else:
            options = [o[1] if isinstance(o, (list, tuple)) else o for o in of[4] or []]
            if text and text not in options and not flags & pymupdf.PDF_CH_FIELD_IS_EDIT:
                sys.exit(5)
        for w in same:
            obj = mupdf.pdf_new_indirect(pdf, w[0], 0)
            if kind == pymupdf.PDF_WIDGET_TYPE_LISTBOX:
                # MuPDF draws a list without its choice: as a menu closed on it.
                mupdf.pdf_dict_put_int(obj, mupdf.PDF_ENUM_NAME_Ff, flags | pymupdf.PDF_CH_FIELD_IS_COMBO)
            if not mupdf.pdf_set_field_value(pdf, obj, text, 1):
                sys.exit(5)


def stamp(src, marks, dst):
    doc = load(src)
    if doc.needs_pass:
        sys.exit(2)
    if any(not 0 <= m["page"] < doc.page_count for m in marks):
        sys.exit(4)
    helv = pymupdf.Font("helv")
    if any(m["kind"] == "text" and not draws(helv, m["text"]) for m in marks):
        sys.exit(3)
    fill(doc, [m for m in marks if m["kind"] == "field"])
    marks = [m for m in marks if m["kind"] != "field"]
    doc.bake()
    doc.xref_set_key(doc.pdf_catalog(), "AcroForm", "null")
    for n in sorted({m["page"] for m in marks}):
        page = doc[n]
        # The marks on a sheet of their own, as the page shows, then that sheet laid on the page as the page is turned.
        sheet = pymupdf.open()
        layer = sheet.new_page(width=page.rect.width, height=page.rect.height)
        for m in marks:
            if m["page"] != n:
                continue
            if m["kind"] == "image":
                layer.insert_image(pymupdf.Rect(m["x"], m["y"], m["x"] + m["w"], m["y"] + m["h"]), stream=base64.b64decode(m["image"]), keep_proportion=False)
            elif m["kind"] == "path":
                shape = layer.new_shape()
                for stroke in m["path"]:
                    shape.draw_polyline([(m["x"] + x * m["w"], m["y"] + y * m["h"]) for x, y in [*stroke, stroke[-1]]])
                shape.finish(width=m["size"], color=(0, 0, 0), lineCap=1, lineJoin=1, closePath=False)
                shape.commit()
            else:
                write(layer, helv, m["x"], m["y"] + m["size"] * 0.8, m["text"], m["size"], (0, 0, 0))
        sheet.subset_fonts()
        if not page.is_wrapped:
            page.wrap_contents()
        turn = page.rotation
        page.set_rotation(0)
        page.show_pdf_page(page.rect, sheet, 0, rotate=turn)
        page.set_rotation(turn)
    # An owner's protection stays on the copy.
    doc.save(dst, garbage=3, deflate=True, encryption=pymupdf.PDF_ENCRYPT_KEEP)


if __name__ == "__main__":
    cmd, *args = sys.argv[1:] or ["help"]
    if cmd == "extract" and len(args) == 1:
        json.dump(extract(args[0]), sys.stdout, ensure_ascii=False)
    elif cmd == "apply" and len(args) == 3:
        with open(args[1]) as f:
            apply(args[0], json.load(f), args[2])
    elif cmd == "stamp" and len(args) == 3:
        with open(args[1]) as f:
            stamp(args[0], json.load(f), args[2])
    else:
        sys.exit("usage: pdf_tool.py extract in.pdf | apply in.pdf edits.json out.pdf | stamp in.pdf marks.json out.pdf")
