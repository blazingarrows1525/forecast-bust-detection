"""Render a project Markdown document to a print-ready PDF via headless Chrome.

Why Chrome and not weasyprint/pandoc: neither is installed, weasyprint needs
GTK on Windows, and Chrome's print engine handles our tables, code blocks and
page breaks well with no extra system dependencies.

Two document-specific fixes matter here:

1. ``<details>/<summary>`` blocks collapse by default. In a PDF that would hide
   every self-check answer -- exactly the content a learner needs. They are
   expanded into visible, labelled answer boxes instead.
2. Long code blocks and tables are kept off page boundaries with
   ``break-inside: avoid``, so an exercise command is never split across pages.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

CSS = """
@page { size: A4; margin: 16mm 14mm 18mm 14mm; }
* { box-sizing: border-box; }
body {
  font: 10.5pt/1.55 "Segoe UI", -apple-system, system-ui, sans-serif;
  color: #1a1f26; max-width: 100%; margin: 0; padding: 0;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
h1 { font-size: 22pt; color: #0b3d6b; border-bottom: 3px solid #0b3d6b;
     padding-bottom: 6px; margin: 0 0 4px; break-after: avoid; }
h2 { font-size: 15pt; color: #0b3d6b; margin: 22px 0 8px;
     border-bottom: 1px solid #c6d4e1; padding-bottom: 4px; break-after: avoid; }
h3 { font-size: 12pt; color: #23517d; margin: 16px 0 6px; break-after: avoid; }
h4 { font-size: 10.5pt; color: #23517d; margin: 12px 0 4px; break-after: avoid; }
p, li { orphans: 3; widows: 3; }
ul, ol { padding-left: 22px; }
li { margin: 2px 0; }
a { color: #1364a8; text-decoration: none; }
code {
  font: 9pt/1.4 "Cascadia Mono", Consolas, monospace;
  background: #eef2f6; padding: 1px 4px; border-radius: 3px; color: #b3236a;
}
pre {
  background: #f6f8fa; border: 1px solid #d6dee6; border-left: 3px solid #0b3d6b;
  border-radius: 4px; padding: 9px 11px; overflow-x: auto;
  break-inside: avoid; margin: 8px 0;
}
pre code { background: none; padding: 0; color: #1a1f26; font-size: 8.6pt; }
table {
  border-collapse: collapse; width: 100%; margin: 10px 0;
  font-size: 8.8pt; break-inside: avoid;
}
th { background: #0b3d6b; color: #fff; text-align: left; padding: 5px 7px;
     font-weight: 600; }
td { border: 1px solid #d6dee6; padding: 4px 7px; vertical-align: top; }
tr:nth-child(even) td { background: #f6f8fa; }
blockquote {
  border-left: 4px solid #f0a500; background: #fffaf0; margin: 10px 0;
  padding: 7px 13px; color: #4a3a1a; break-inside: avoid;
}
hr { border: none; border-top: 1px solid #c6d4e1; margin: 20px 0; }
strong { color: #0b2c4d; }

/* Expanded <details> -> visible answer box (collapsed content is invisible in PDF) */
.answer {
  background: #f0f7f0; border: 1px solid #b9d9b9; border-left: 4px solid #2e7d32;
  border-radius: 4px; padding: 8px 12px; margin: 8px 0; break-inside: avoid;
}
.answer .answer-label {
  font-weight: 700; color: #2e7d32; font-size: 8.5pt;
  text-transform: uppercase; letter-spacing: .5px; display: block; margin-bottom: 4px;
}
.answer p, .answer li { margin: 3px 0; }

/* Cover block */
.cover-meta { color: #5a6b7c; font-size: 9.5pt; margin: 2px 0 18px; }
"""


def expand_details(md_text: str) -> str:
    """Turn collapsible <details> blocks into always-visible answer boxes."""
    pattern = re.compile(
        r"<details>\s*\n?\s*<summary>(.*?)</summary>(.*?)</details>",
        re.DOTALL | re.IGNORECASE,
    )

    def repl(m: re.Match) -> str:
        label = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        body = m.group(2).strip()
        return (
            f'\n<div class="answer">\n'
            f'<span class="answer-label">{label}</span>\n\n'
            f"{body}\n\n</div>\n"
        )

    return pattern.sub(repl, md_text)


def build_html(md_path: Path, title: str | None) -> str:
    raw = md_path.read_text(encoding="utf-8")
    raw = expand_details(raw)

    body = markdown.markdown(
        raw,
        extensions=["tables", "fenced_code", "toc", "sane_lists", "md_in_html"],
    )
    doc_title = title or md_path.stem.replace("_", " ").title()
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{doc_title}</title><style>{CSS}</style></head>"
        f"<body>{body}</body></html>"
    )


def to_pdf(md_path: Path, out_pdf: Path, title: str | None = None) -> bool:
    html = build_html(md_path, title)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".html", delete=False, encoding="utf-8"
    ) as fh:
        tmp_html = Path(fh.name)
        fh.write(html)

    if not Path(CHROME).exists():
        print(f"[FAIL] Chrome not found at {CHROME}")
        return False

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    # Chrome resolves --print-to-pdf against its own working directory, not
    # ours, so a relative path silently fails with "cannot find the path".
    out_abs = out_pdf.resolve()
    cmd = [
        CHROME,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        f"--print-to-pdf={out_abs}",
        tmp_html.as_uri(),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    tmp_html.unlink(missing_ok=True)

    if not out_pdf.exists() or out_pdf.stat().st_size < 5000:
        print(f"[FAIL] {out_pdf.name} not produced")
        print((res.stderr or "")[-600:])
        return False
    print(f"[ok]   {out_pdf.name}  ({out_pdf.stat().st_size/1e6:.2f} MB)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("markdown", nargs="+", help="markdown file(s) to convert")
    ap.add_argument("--outdir", default="docs/pdf")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ok = True
    for md in args.markdown:
        p = Path(md)
        if not p.exists():
            print(f"[FAIL] {md} does not exist")
            ok = False
            continue
        ok &= to_pdf(p, outdir / f"{p.stem}.pdf", args.title)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
