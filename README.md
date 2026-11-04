# pdf-tool

Read a PDF's text runs, and rewrite some of them in place: only the edited text is erased and
written again, everything else (images, drawings, other text) stays as it was. Built on
[PyMuPDF](https://pymupdf.readthedocs.io), licensed AGPL-3.0.

```sh
pip install -r requirements.txt
python3 pdf_tool.py extract in.pdf                  # JSON: every horizontal run of text
python3 pdf_tool.py apply in.pdf edits.json out.pdf # each edit's box erased, its new text written
```

An edit: `{"page", "bbox", "origin", "text", "font", "size", "color", "align"?: "right", "mark"?: true}`,
as `extract` gives them with a new `text`. The new text keeps the PDF's own font when it has every
glyph, else the closest standard one (Helvetica, Times, Courier), bold and italic kept, drawn as dark as
the text it replaces (`"was"`: that text, in the edit).
Exit 2: the PDF is protected; exit 3: a character no font here can draw.

## Scans and photos

`in.pdf` may be an image too (JPG, PNG, WEBP, HEIC, a TIFF's pages…): it becomes a page of its size,
upright, and `out.pdf` a PDF. A page with no text of its own is read by OCR (Tesseract, built into
PyMuPDF's wheel), its light evened out first; its runs come with `"ocr": true`, and `"font"` `helv`
or `hebo` by the weight of their strokes. An edit on such a page draws the page again as a picture:
each box's pixels replaced by the paper around it, nothing of the old text left, the new text in
Helvetica, in the shade of the text it replaces. Tesseract's language data goes in `tessdata/`,
next to the script (French, English and Spanish are read); without it, a scan has no text:

```sh
mkdir -p tessdata
for l in fra eng spa; do
  curl -fsSL -o tessdata/$l.traineddata https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/87416418657359cb625c412a48b6e1d6d41c29bd/$l.traineddata
done
```
