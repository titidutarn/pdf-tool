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
