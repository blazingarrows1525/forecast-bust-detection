"""Turn a fetched page into plain, bounded text for the evidence packet.

Standard library only (html.parser): no new dependency reaches the serving
image for this. What is kept is what a reader would read -- headings, paragraphs,
list items and table rows -- and what is dropped is everything that runs or
decorates: script, style, nav, footer, forms, iframes, SVG.

A publication date is taken only from metadata the page itself declares
(article:published_time, datePublished, <time datetime>, and the like). When a
page declares none, the date is None and the interface says "no date given";
inferring one from the URL or the text would be inventing it.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser

_SKIP = {"script", "style", "noscript", "nav", "footer", "form", "iframe", "svg", "button", "template", "aside"}
_BLOCK = {"p", "div", "section", "article", "li", "tr", "br", "h1", "h2", "h3", "h4", "h5", "h6",
          "header", "main", "table", "ul", "ol", "dd", "dt", "blockquote", "pre"}
_DATE_META = {"article:published_time", "og:published_time", "datepublished", "date", "dc.date",
              "dc.date.issued", "publish-date", "pubdate", "article:modified_time", "datemodified"}
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ][0-9:.+\-Z]*)?$")


class _Extractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.skip = 0
        self.title = ""
        self._in_title = False
        self.published: str | None = None
        self.site_name: str | None = None

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag in _SKIP:
            self.skip += 1
            return
        if tag == "title":
            self._in_title = True
        if tag == "meta":
            key = (a.get("property") or a.get("name") or a.get("itemprop") or "").lower()
            content = a.get("content", "").strip()
            if key in _DATE_META and content and _DATE_RE.match(content) and not self.published:
                self.published = content[:25]
            if key == "og:site_name" and content:
                self.site_name = content[:80]
        if tag == "time" and not self.published:
            dt = a.get("datetime", "").strip()
            if _DATE_RE.match(dt):
                self.published = dt[:25]
        if tag in _BLOCK:
            self.out.append("\n")
        if tag in ("h1", "h2", "h3"):
            self.out.append("## ")

    def handle_endtag(self, tag):
        if tag in _SKIP:
            self.skip = max(0, self.skip - 1)
            return
        if tag == "title":
            self._in_title = False
        if tag in ("td", "th"):
            self.out.append(" | ")
        if tag in _BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
            return
        if self.skip:
            return
        self.out.append(data)


def extract(html: str, content_type: str = "text/html", max_chars: int = 5000) -> dict:
    """Return {title, text, published_at, site_name}. Text is whitespace-normalised
    and cut to ``max_chars`` at a line boundary where possible."""
    if content_type == "text/plain":
        text, title, published, site = html, "", None, None
    else:
        p = _Extractor()
        try:
            p.feed(html)
            p.close()
        except Exception:  # noqa: BLE001 - malformed markup degrades, never raises
            pass
        text, title, published, site = "".join(p.out), p.title.strip(), p.published, p.site_name
    lines = [re.sub(r"[ \t ]+", " ", ln).strip(" |") for ln in text.splitlines()]
    lines = [ln for ln in lines if len(ln) > 1]
    body = "\n".join(lines)
    if len(body) > max_chars:
        cut = body.rfind("\n", 0, max_chars)
        body = body[: cut if cut > max_chars * 0.6 else max_chars]
    return {"title": re.sub(r"\s+", " ", title)[:200], "text": body,
            "published_at": published, "site_name": site}
