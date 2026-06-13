"""Render docs/FIELD_GUIDE.md into a light, accessible standalone PDF.

A non-expert companion to the full compendium: bigger type, generous spacing,
plain-language "explanatory insert" call-out boxes, explicit step-to-step
transitions, and the load-bearing figures. Reuses the KaTeX + WeasyPrint pipeline
from build_book.py.

    python docs/book/build_field_guide.py     # -> docs/Field-Guide.pdf
"""
from __future__ import annotations

import re

from weasyprint import HTML

from build_book import DOCS, IMG, katex_css, md_to_html

OUT = DOCS / "Field-Guide.pdf"
DATE = "13 June 2026"

STYLE = """
@page { size: A4; margin: 22mm 20mm 20mm 20mm;
  @top-left { content: "A Plain-Language Field Guide"; font: 8pt Georgia; color: #9aa3b2; }
  @bottom-center { content: counter(page); font: 8.5pt Georgia; color: #6b7382; } }
@page :first { margin: 0; @top-left { content: ""; } @bottom-center { content: ""; } }
@page cover { margin: 0; }
html { font-size: 11.6pt; }
body { font-family: Georgia, 'Times New Roman', serif; color: #1f2733; line-height: 1.62; }
h2 { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #16263a; font-size: 15pt;
  margin: 1.6em 0 0.4em; padding-bottom: 3px; border-bottom: 2px solid #d8a93a;
  page-break-before: always; page-break-after: avoid; }
h2:first-of-type { page-break-before: avoid; }
p { margin: 0.55em 0; }
em { color: #3a4655; }
strong { color: #16263a; }
hr { border: none; margin: 6px 0; }
.lead { font-size: 11pt; color: #44505f; font-style: italic; border-left: 3px solid #b8860b;
  padding: 2px 14px; margin: 4px 0 16px; background: #fbf8f0; }
.transition { color: #6a4a00; }
aside.insert { background: #eef4fb; border: 1px solid #cfe0f2; border-left: 4px solid #2e6e9e;
  border-radius: 6px; padding: 9px 14px; margin: 13px 0; font-size: 10.4pt; line-height: 1.55;
  page-break-inside: avoid; }
aside.insert p { margin: 0; }
aside.insert strong:first-child { color: #1a5f9e; }
figure { margin: 16px auto; text-align: center; page-break-inside: avoid; }
figure img { max-width: 92%; border: 1px solid #e2e8f0; border-radius: 5px; }
figcaption { font-family: Helvetica, Arial, sans-serif; font-size: 8.8pt; color: #55606f;
  margin-top: 6px; text-align: center; line-height: 1.4; }
.eq { margin: 11px 0; text-align: center; }
.eq .katex-display { margin: 0; }
.cover { page: cover; height: 297mm; box-sizing: border-box; padding: 50mm 28mm 24mm;
  background: linear-gradient(160deg, #1a5f9e 0%, #2e6e9e 55%, #3f86b8 100%); color: #fff;
  page-break-after: always; }
.cover .rule { width: 70px; height: 4px; background: #ffd373; margin: 0 0 26px; }
.cover h1 { font-size: 33pt; margin: 0 0 12px; line-height: 1.1; border: none; }
.cover .sub { font-size: 14pt; color: #e6f0fa; font-family: Helvetica, Arial, sans-serif;
  line-height: 1.4; margin-bottom: 30px; }
.cover .note { font-size: 10.5pt; color: #d3e4f3; font-family: Helvetica, Arial, sans-serif; }
.cover .by { position: absolute; bottom: 24mm; font-size: 10pt; color: #cfe2f3;
  font-family: Helvetica, Arial, sans-serif; }
"""


def build():
    text = (DOCS / "FIELD_GUIDE.md").read_text()
    text = re.sub(r"^#\s+.*\n", "", text, count=1)          # title -> cover
    text = re.sub(r"^###\s+.*\n", "", text, count=1)        # subtitle -> cover
    text = text.replace("](book/img/", f"](file://{IMG}/")  # absolute figure paths
    body = md_to_html(text)

    # mark the "Where this leads" transition sentences
    body = body.replace("<em>Where this leads:</em>", '<em class="transition">Where this leads:</em>')
    body = body.replace("<em>This is the keystone", '<em class="transition">This is the keystone')

    # markdown image -> captioned figure
    body = re.sub(
        r'<p>\s*<img\b[^>]*?alt="(.*?)"[^>]*?src="(.*?)"[^>]*?/?>\s*</p>',
        r'<figure><img src="\2"/><figcaption>\1</figcaption></figure>', body)
    # first paragraph (the italic intro) -> styled lead
    body = body.replace("<p><em>The full compendium proves", '<p class="lead"><em>The full compendium proves', 1)

    cover = f"""
    <section class="cover">
      <div class="rule"></div>
      <h1>A Plain-Language<br/>Field Guide</h1>
      <div class="sub">The short, guided road through<br/>
        <i>Asset Pricing, Time Series, and the Limits of Prediction</i></div>
      <div class="note">Six stepping-stones · plain-language inserts · the proofs are in the full compendium</div>
      <div class="by">Companion to the Session Compendium &nbsp;·&nbsp; {DATE} &nbsp;·&nbsp; not financial advice</div>
    </section>"""

    css = STYLE + "\n" + katex_css()
    doc = f'<!doctype html><html><head><meta charset="utf-8"><style>{css}</style></head><body>{cover}{body}</body></html>'
    HTML(string=doc, base_url=str(DOCS)).write_pdf(str(OUT))
    print("wrote", OUT, OUT.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    build()
