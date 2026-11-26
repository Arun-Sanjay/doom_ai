"""Render docs/brief/brief.md to docs/brief/brief.pdf and check that it fits on two pages.

    python docs/brief/build_brief.py            # writes brief.pdf, prints the page count
    python docs/brief/build_brief.py --html     # also keeps brief.html next to the PDF
    python docs/brief/build_brief.py --force    # writes the PDF even when it is too long, to look at it

Standard library only. The Markdown subset used by the brief (headings, paragraphs, bullet and
numbered lists, pipe tables, fenced blocks, `code`, **bold**, *italic*, links) is converted to
HTML here, and a headless Chrome or Chromium prints the HTML to PDF. The browser is looked up in
the environment variable CHROME, then in the usual install places, then on PATH.

Exit status: 0 when the PDF has at most MAX_PAGES pages, 1 when it has more or when the page count
cannot be read (brief.pdf is then left as it was), 2 when no browser was found. The script also prints how many `[fill: ...]`
placeholders the text still contains and an estimate of how full the last page is.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "brief.md"
TARGET = HERE / "brief.pdf"
MAX_PAGES = 2

# Page geometry (millimetres). The fill estimate below uses the same numbers as the style sheet.
PAGE_W, PAGE_H = 210.0, 297.0
MARGIN_X, MARGIN_Y = 13.0, 11.0
PX_PER_MM = 96.0 / 25.4

CHROME_PLACES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
)
CHROME_NAMES = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome")

STYLE = f"""
@page {{ size: A4; margin: {MARGIN_Y}mm {MARGIN_X}mm; }}
* {{ box-sizing: border-box; }}
html {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
body {{ margin: 0; font: 8.45pt/1.3 "Helvetica Neue", Helvetica, Arial, sans-serif; color: #111; }}
h1 {{ font-size: 14pt; margin: 0 0 1pt; }}
h2 {{ font-size: 10.4pt; margin: 6.5pt 0 2pt; padding-bottom: 1pt; border-bottom: 0.6pt solid #888; break-after: avoid; }}
h3 {{ font-size: 8.8pt; margin: 4pt 0 1pt; break-after: avoid; }}
p {{ margin: 0 0 3pt; text-align: justify; hyphens: auto; }}
p.lead {{ color: #444; margin-bottom: 3pt; text-align: left; }}
ul, ol {{ margin: 0 0 3pt; padding-left: 12pt; }}
li {{ margin: 0 0 1.5pt; text-align: justify; hyphens: auto; }}
code {{ font: 7.5pt Menlo, Consolas, "DejaVu Sans Mono", monospace; }}
pre {{ font: 7pt/1.16 Menlo, Consolas, "DejaVu Sans Mono", monospace; margin: 2pt 0 4pt; padding: 4pt 5pt;
      background: #f4f4f1; border: 0.5pt solid #bbb; white-space: pre; overflow: hidden; break-inside: avoid; }}
table {{ border-collapse: collapse; width: 100%; margin: 2pt 0 4pt; font-size: 7.7pt; break-inside: avoid; }}
th, td {{ border: 0.5pt solid #999; padding: 1.1pt 3pt; text-align: left; vertical-align: top; }}
th {{ background: #e9e9e4; }}
a {{ color: inherit; text-decoration: none; }}
.fill {{ background: #fff3a8; }}
"""

# Measures the laid-out height at the printable width, for the fill estimate (screen layout, so
# it is an estimate: the printer also keeps tables and the diagram from splitting across pages).
MEASURE = """
<script>
  document.documentElement.setAttribute("data-content-height", String(document.body.scrollHeight));
</script>
"""

FILL = re.compile(r"\[fill:[^\]]*\]")


# ---------------------------------------------------------------------------------------------
# Markdown subset -> HTML
# ---------------------------------------------------------------------------------------------
def inline(text: str) -> str:
    """Inline markup of one piece of text: `code`, **bold**, *italic*, [text](url), [fill: ...]."""
    spans: list[str] = []

    def keep(match: re.Match) -> str:              # code spans are set aside so that bold may contain them
        spans.append("<code>" + html.escape(match.group(1), quote=False) + "</code>")
        return f"\x00{len(spans) - 1}\x00"

    s = html.escape(re.sub(r"`([^`]*)`", keep, text), quote=False)
    s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![*\w])\*([^*\s][^*]*)\*(?![*\w])", r"<em>\1</em>", s)
    s = FILL.sub(lambda m: '<span class="fill">' + m.group(0) + "</span>", s)
    return re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], s)


def table(rows: list[str]) -> str:
    """Pipe table: first row is the header, the second (dashes) is dropped."""
    cells = [[c.strip() for c in row.strip().strip("|").split("|")] for row in rows]
    body = [r for k, r in enumerate(cells) if not (k == 1 and all(re.fullmatch(r":?-+:?", c) for c in r))]
    out = ["<table>"]
    for k, row in enumerate(body):
        tag = "th" if k == 0 else "td"
        out.append("<tr>" + "".join(f"<{tag}>{inline(c)}</{tag}>" for c in row) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def convert(markdown: str) -> str:
    """The subset of Markdown the brief uses -> an HTML fragment."""
    lines = markdown.splitlines()
    out: list[str] = []
    para: list[str] = []
    items: list[str] = []
    kind = ""                                  # "ul", "ol" or "" while no list is open
    first_para = True

    def end_para() -> None:
        nonlocal first_para
        if para:
            cls = ' class="lead"' if first_para and out and out[-1].startswith("<h1") else ""
            out.append(f"<p{cls}>" + inline(" ".join(para)) + "</p>")
            first_para = False
            para.clear()

    def end_list() -> None:
        nonlocal kind
        if items:
            out.append(f"<{kind}>" + "".join("<li>" + inline(i) + "</li>" for i in items) + f"</{kind}>")
            items.clear()
        kind = ""

    k = 0
    while k < len(lines):
        line = lines[k]
        stripped = line.strip()
        if stripped.startswith("```"):
            end_para(), end_list()
            block = []
            k += 1
            while k < len(lines) and not lines[k].strip().startswith("```"):
                block.append(lines[k])
                k += 1
            out.append("<pre>" + html.escape("\n".join(block), quote=False) + "</pre>")
        elif stripped.startswith("|"):
            end_para(), end_list()
            rows = []
            while k < len(lines) and lines[k].strip().startswith("|"):
                rows.append(lines[k])
                k += 1
            out.append(table(rows))
            continue
        elif not stripped:
            end_para(), end_list()
        elif m := re.match(r"(#{1,3})\s+(.*)", stripped):
            end_para(), end_list()
            level = len(m.group(1))
            out.append(f"<h{level}>" + inline(m.group(2)) + f"</h{level}>")
        elif m := re.match(r"(-|\d+\.)\s+(.*)", stripped):
            end_para()
            wanted = "ul" if m.group(1) == "-" else "ol"
            if kind and kind != wanted:
                end_list()
            kind = wanted
            items.append(m.group(2))
        elif items and line.startswith(" "):   # continuation line of a list item
            items[-1] += " " + stripped
        else:
            end_list()
            para.append(stripped)
        k += 1
    end_para(), end_list()
    return "\n".join(out)


def page(markdown: str, measure: bool = False) -> str:
    title = next((line[2:].strip() for line in markdown.splitlines() if line.startswith("# ")), "Project brief")
    width = f"<style>body {{ width: {PAGE_W - 2 * MARGIN_X}mm; }}</style>" if measure else ""
    return ("<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\"><title>" + html.escape(title)
            + "</title><style>" + STYLE + "</style>" + width + "</head>\n<body>\n" + convert(markdown)
            + "\n" + (MEASURE if measure else "") + "</body></html>\n")


# ---------------------------------------------------------------------------------------------
# Browser, PDF, page count
# ---------------------------------------------------------------------------------------------
def find_chrome() -> str | None:
    named = os.environ.get("CHROME")
    if named and Path(named).exists():
        return named
    for place in CHROME_PLACES:
        if Path(place).exists():
            return place
    for name in CHROME_NAMES:
        found = shutil.which(name)
        if found:
            return found
    return None


def run_chrome(chrome: str, profile: Path, *args: str) -> subprocess.CompletedProcess:
    base = [chrome, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
            "--disable-extensions", f"--user-data-dir={profile}"]
    return subprocess.run(base + list(args), capture_output=True, text=True, timeout=120, check=False)


def count_pages(pdf: bytes) -> int | None:
    """Pages of a PDF: its page objects, or the /Count of the page tree when none are in plain view."""
    pages = len(re.findall(rb"/Type\s*/Page(?![s\w])", pdf))
    if pages:
        return pages
    counts = [int(c) for c in re.findall(rb"/Type\s*/Pages\b[^>]*?/Count\s+(\d+)", pdf, re.S)]
    counts += [int(c) for c in re.findall(rb"/Count\s+(\d+)[^>]*?/Type\s*/Pages\b", pdf, re.S)]
    return max(counts) if counts else None


def fill_estimate(chrome: str, profile: Path, markdown: str, work: Path) -> float | None:
    """Laid-out height of the text in printable pages (2.0 = two pages exactly full), or None."""
    probe = work / "measure.html"
    probe.write_text(page(markdown, measure=True), encoding="utf-8")
    done = run_chrome(chrome, profile, "--dump-dom", probe.as_uri())
    found = re.search(r'data-content-height="(\d+(?:\.\d+)?)"', done.stdout or "")
    if not found:
        return None
    return float(found.group(1)) / ((PAGE_H - 2 * MARGIN_Y) * PX_PER_MM)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default=str(SOURCE))
    ap.add_argument("--out", default=str(TARGET))
    ap.add_argument("--html", action="store_true", help="keep the intermediate HTML next to the PDF")
    ap.add_argument("--force", action="store_true", help="write the PDF even when it is over the page limit")
    args = ap.parse_args()

    source, target = Path(args.source), Path(args.out)
    markdown = source.read_text(encoding="utf-8")
    chrome = find_chrome()
    if chrome is None:
        print("no Chrome or Chromium found; set CHROME to the browser binary", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="brief_") as tmp:
        work = Path(tmp)
        profile = work / "profile"
        page_file = work / "brief.html"
        page_file.write_text(page(markdown), encoding="utf-8")
        pdf_file = work / "brief.pdf"
        done = run_chrome(chrome, profile, "--no-pdf-header-footer", "--print-to-pdf-no-header",
                          f"--print-to-pdf={pdf_file}", page_file.as_uri())
        if not pdf_file.exists():
            print("the browser wrote no PDF:\n" + (done.stderr or done.stdout)[-800:], file=sys.stderr)
            return 1
        data = pdf_file.read_bytes()
        pages = count_pages(data)
        fits = pages is not None and pages <= MAX_PAGES
        if fits or args.force:               # a PDF over the limit never replaces a good one unasked
            target.write_bytes(data)
        if args.html:
            target.with_suffix(".html").write_text(page(markdown), encoding="utf-8")
        fill = fill_estimate(chrome, profile, markdown, work)

    placeholders = FILL.findall(markdown)
    written = "" if fits or args.force else " (not written; --force writes it anyway)"
    print(f"{target}: {pages if pages is not None else 'unknown number of'} page(s), {len(data) / 1024:.0f} KB{written}")
    if fill is not None:
        print(f"text height at the printable width: {fill:.2f} pages of {MAX_PAGES} (estimate)")
    print(f"placeholders left: {len(placeholders)}")
    if pages is None:
        print("FAILED: could not read the page count from the PDF", file=sys.stderr)
        return 1
    if pages > MAX_PAGES:
        print(f"FAILED: {pages} pages, the limit is {MAX_PAGES}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
