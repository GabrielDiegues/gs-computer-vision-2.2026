"""Gera docs/relatorio.pdf a partir de docs/relatorio.md e confere os limites de palavras."""
import re

import markdown
from weasyprint import HTML

from common import ROOT

DOCS = ROOT / "docs"
CSS = """
@page { size: A4; margin: 2.2cm 2cm; @bottom-center { content: counter(page); font-size: 9pt; color: #666; } }
body { font-family: 'DejaVu Sans', sans-serif; font-size: 10.5pt; line-height: 1.5; color: #1a1a1a; }
h1 { font-size: 20pt; margin: 0 0 4pt; color: #0b3d5c; }
h2 { font-size: 13.5pt; margin: 18pt 0 6pt; color: #0b3d5c; border-bottom: 1px solid #c9d6df; padding-bottom: 2pt; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; }
p, li { text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 9.5pt; }
th, td { border: 1px solid #c9d6df; padding: 4pt 6pt; text-align: left; }
th { background: #eef3f6; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.8pt; background: #f3f5f7; padding: 0 2pt; }
pre { background: #f3f5f7; padding: 6pt 8pt; font-size: 8.5pt; white-space: pre-wrap; }
pre code { background: none; padding: 0; }
img { max-width: 100%; max-height: 12.5cm; display: block; margin: 6pt auto; }
figure, .fig { page-break-inside: avoid; }
em.legenda { display: block; text-align: center; font-size: 9pt; color: #555; }
"""


def section_words(md: str, title: str) -> int:
    match = re.search(rf"^## {title}\s*\n(.*?)(?=^## |\Z)", md, re.S | re.M)
    return len(re.findall(r"\b[\wÀ-ÿ%-]+\b", match.group(1))) if match else 0


def main() -> None:
    md = (DOCS / "relatorio.md").read_text()
    for title, limit in [("Resumo", 250), ("Objetivo", 50)]:
        n = section_words(md, title)
        print(f"{title}: {n} palavras (limite {limit}){'  <-- ACIMA DO LIMITE' if n > limit else ''}")
    html = markdown.markdown(md, extensions=["tables", "fenced_code", "attr_list", "md_in_html"])
    doc = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html}</body></html>"
    HTML(string=doc, base_url=str(DOCS)).write_pdf(DOCS / "relatorio.pdf")
    print(f"PDF gerado: {DOCS / 'relatorio.pdf'}")


if __name__ == "__main__":
    main()
