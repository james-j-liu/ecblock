"""Timely euro-area NCB speeches from the BIS central bankers' speeches site.

The BIS bulk files (ecb_ncb) lag by months, so daily updates use the live site:
  - RSS  https://www.bis.org/doclist/cbspeeches.rss   latest ~50 speeches (~1 week)
  - list https://www.bis.org/cbspeeches/index.htm?page=N   10 per page, newest first
The listing is only walked back to `since` (normally the newest BIS speech already
in the corpus), so a missed day - or a months-long gap - fills itself in.

Each candidate's affiliation comes from its description ("Address by Mr X, Governor
of the Bank of Italy, at ..."), so only euro-area NCB speakers are kept; the full
text comes from the speech PDF (the HTML page carries only a summary).
"""
from __future__ import annotations

import datetime as dt
import html
import io
import re
import xml.etree.ElementTree as ET

import requests

from ..schema import ST_INTERVIEW, ST_SPEECH, Speech
from .ecb_ncb import _institution

HOST = "https://www.bis.org"
RSS = f"{HOST}/doclist/cbspeeches.rss"
LISTING = f"{HOST}/cbspeeches/index.htm?page={{}}"
UA = {"User-Agent": "Mozilla/5.0 (compatible; ECBLock/1.0; +https://github.com/james-j-liu/ecblock)"}
_INTERVIEW = re.compile(r"\binterview\b", re.I)


def _get(sess, url, timeout=60):
    try:
        r = sess.get(url, timeout=timeout)
        return r if r.status_code == 200 else None
    except requests.RequestException:
        return None


def _rss(sess) -> list[dict]:
    r = _get(sess, RSS)
    if not r:
        return []
    ns = {"dc": "http://purl.org/dc/elements/1.1/", "rss": "http://purl.org/rss/1.0/"}
    out = []
    for it in ET.fromstring(r.content).iter("{http://purl.org/rss/1.0/}item"):
        g = lambda p: (it.findtext(p, default="", namespaces=ns) or "").strip()
        out.append({"url": g("rss:link"), "title": g("rss:title"), "desc": g("rss:description"),
                    "speaker": g("dc:creator"), "date": g("dc:date")[:10]})
    return out


_CARD = re.compile(r'<a href="(/speeches/\d{8}-[a-z0-9-]+)" class="card-link">.*?'
                   r'card-date[^>]*>([^<]+)<.*?card-heading">(.*?)</h5>.*?'
                   r'card-description">(.*?)</div>', re.S)
_BY = re.compile(r"\bby\s+(?:(?:Mr|Ms|Mrs|Dr|Prof|Professor)\.?\s+)?([^,]{2,60}),")


def _listing(sess, since: str, max_pages: int = 80) -> list[dict]:
    """Listing cards dated >= since, newest first. Each card already carries the
    date, title and 'Speech by Mr X, Governor of ...' description, so no per-item
    page fetch is needed to decide relevance."""
    out = []
    for p in range(max_pages):
        r = _get(sess, LISTING.format(p))
        cards = _CARD.findall(r.text) if r else []
        if not cards:
            break
        oldest = "9999"
        for path, date, title, desc in cards:
            try:
                d = dt.datetime.strptime(date.strip(), "%d %b %Y").date().isoformat()
            except ValueError:
                d = f"{path[10:14]}-{path[14:16]}-{path[16:18]}"
            oldest = min(oldest, d)
            desc = html.unescape(re.sub(r"<[^>]+>", "", desc)).strip()
            who = _BY.search(desc)
            if d >= since:
                out.append({"url": HOST + path, "date": d, "desc": desc,
                            "title": html.unescape(re.sub(r"<[^>]+>", "", title)).strip(),
                            "speaker": who.group(1).strip() if who else ""})
        if oldest < since:
            break
    return out



def _pdf_text(sess, url) -> str:
    r = _get(sess, url + ".pdf", timeout=120)
    if not r or not r.content.startswith(b"%PDF"):
        return ""
    from pypdf import PdfReader
    try:
        pages = PdfReader(io.BytesIO(r.content)).pages
        return re.sub(r"[ \t]+", " ", "\n".join(p.extract_text() or "" for p in pages)).strip()
    except Exception:  # noqa: BLE001 - a malformed PDF must not stop the run
        return ""


def _page_text(sess, url) -> str:
    r = _get(sess, url)
    if not r:
        return ""
    import trafilatura
    return (trafilatura.extract(r.text, favor_recall=True) or "").strip()


def load(since: str, known_urls: set[str] = frozenset(), min_chars: int = 400) -> list[Speech]:
    sess = requests.Session()
    sess.headers.update(UA)
    cands = {c["url"]: c for c in _listing(sess, since)}
    cands.update({c["url"]: c for c in _rss(sess) if c["date"] >= since})  # RSS: creator field

    out = []
    for url, c in cands.items():
        if url in known_urls:
            continue
        inst = _institution(c["desc"])
        if not inst or not c["speaker"]:
            continue
        # most speeches have a PDF; the rest carry the full text on the page itself
        text = _pdf_text(sess, url) or _page_text(sess, url)
        if len(text) < min_chars:
            continue
        out.append(Speech(
            date=c["date"], speaker=c["speaker"], title=c["title"], text=text,
            source_type=ST_INTERVIEW if _INTERVIEW.search(c["title"]) else ST_SPEECH,
            institution=inst, source_url=url, orig_language="en"))
    return out


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    since = sys.argv[1] if len(sys.argv) > 1 else (dt.date.today() - dt.timedelta(days=10)).isoformat()
    for s in load(since):
        print(f"{s.date}  {s.speaker:26} {s.institution:32} {len(s.text):6}  {s.title[:50]}")
