"""Direct scraping of euro-area NCB websites for governor speeches & interviews.

The BIS bulk speech database under-represents the smaller national central banks,
so several Governing Council governors (Latvia, Cyprus, Malta, ...) have almost no
records. This module scrapes the NCBs' own sites to fill that gap.

It is a general engine driven by a per-NCB config: a listing page is crawled
(with pagination) to enumerate article URLs, those are filtered to the governor's
speeches/interviews, and each article's text + date are extracted generically with
trafilatura. Output is a list of Speech records, ready to merge into the corpus.

Some NCB sites hard-block automated access (Cloudflare/403: NL, PT, GR, LT) - those
are handled separately / flagged, not here.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import requests
import trafilatura

from ..schema import ST_INTERVIEW, ST_SPEECH, Speech

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept-Language": "en"}

_INTERVIEW_RE = re.compile(r"interview|q&a|q-a|talks-to|spoke-to", re.I)


@dataclass
class NCBConfig:
    key: str                      # short id, e.g. "LV"
    institution: str              # canonical institution name (matches corpus)
    base: str                     # https://www.bank.lv
    listing_path: str             # /en/news-and-events/news-and-articles/news
    link_re: str                  # regex capturing article hrefs on the listing
    governors: dict[str, str]     # {surname_regex: canonical full name}
    pages: int = 60               # max listing pages to crawl
    start_step: int = 10          # pagination increment
    start_param: str = "start"    # ?start=N (Joomla); "" => path-based / none
    lang: str = "en"


def fetch(url: str, tries: int = 3, timeout: int = 30) -> str | None:
    for i in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            if r.status_code == 200 and r.text:
                # some NCBs omit the charset header; requests then defaults to
                # ISO-8859-1 and mojibakes UTF-8 pages. Trust the detected encoding.
                if not r.encoding or r.encoding.lower() in ("iso-8859-1", "latin-1", "ascii"):
                    r.encoding = r.apparent_encoding or r.encoding
                return r.text
            if r.status_code in (403, 404):
                return None
        except requests.RequestException:
            pass
        time.sleep(1.5 * (i + 1))
    return None


def _page_url(cfg: NCBConfig, p: int) -> str:
    url = cfg.base + cfg.listing_path
    if p == 0 or not cfg.start_param:
        return url
    sep = "&" if "?" in cfg.listing_path else "?"
    return f"{url}{sep}{cfg.start_param}={p * cfg.start_step}"


def enumerate_articles(cfg: NCBConfig) -> list[str]:
    """Crawl the listing pages and return absolute article URLs (deduped, in order)."""
    seen: dict[str, None] = {}
    link_re = re.compile(cfg.link_re)
    for p in range(cfg.pages):
        html = fetch(_page_url(cfg, p))
        if not html:
            break
        found = [m if m.startswith("http") else cfg.base + m for m in link_re.findall(html)]
        new = [u for u in found if u not in seen]
        for u in new:
            seen[u] = None
        if p > 0 and not new:        # pagination exhausted
            break
    return list(seen)


def wayback_url(url: str) -> str | None:
    """Nearest Wayback Machine snapshot, used to reach Cloudflare-blocked NCB
    sites (NL, PT, GR, LT) that 403 direct requests but are archived."""
    try:
        r = requests.get("http://archive.org/wayback/available",
                         params={"url": url}, headers=UA, timeout=30)
        snap = r.json().get("archived_snapshots", {}).get("closest", {})
        return snap.get("url") if snap.get("available") else None
    except (requests.RequestException, ValueError):
        return None


_MON = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}


def guess_date(url: str, title: str) -> str | None:
    """Best-effort date from the URL slug or title when no metadata date exists."""
    s = (url + " " + title).lower()
    m = re.search(r"(20\d{2})[-/](\d{2})[-/](\d{2})", url)        # YYYY-MM-DD
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.search(r"(\d{2})[-/](\d{2})[-/](20\d{2})", url)        # DD-MM-YYYY
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    m = re.search(r"(20\d{2})(\d{2})(\d{2})", url)                # YYYYMMDD
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*-?(\d{2})\b", s)
    if m:                                                         # sep22 / mar-24
        return f"20{m.group(2)}-{_MON[m.group(1)]:02d}-01"
    m = re.search(r"\b(20[12]\d)\b", s)                           # bare year
    if m:
        return f"{m.group(1)}-07-01"
    return None


def extract_article(url: str) -> dict | None:
    """Generic article extraction: title, ISO date, full text. Falls back to the
    Wayback Machine when the live site blocks automated access, and to a date
    parsed from the URL/title when no metadata date is present."""
    html = fetch(url)
    if not html:
        wb = wayback_url(url)
        html = fetch(wb) if wb else None
    if not html:
        return None
    raw = trafilatura.extract(html, output_format="json", with_metadata=True,
                              favor_recall=True)
    if not raw:
        return None
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        return None
    text = (d.get("text") or "").strip()
    title = (d.get("title") or "").strip()
    meta = (d.get("date") or "")[:10]
    meta = meta if re.match(r"\d{4}-\d{2}-\d{2}", meta) else ""
    slug = guess_date(url, title) or ""
    # trafilatura often grabs a footer/copyright date; trust the URL/title date
    # when they disagree on the year, otherwise keep the more specific metadata.
    if meta and slug and meta[:4] != slug[:4]:
        date = slug
    else:
        date = meta or slug
    if not text or not re.match(r"\d{4}-\d{2}-\d{2}", date):
        return None
    return {"title": title, "date": date, "text": text}


def scrape_ncb(cfg: NCBConfig, min_chars: int = 600,
               since: str = "2010-01-01", verbose: bool = True) -> list[Speech]:
    urls = enumerate_articles(cfg)
    gov_pat = re.compile("|".join(cfg.governors), re.I)
    # cheap pre-filter: keep URLs whose slug names the governor
    cand = [u for u in urls if gov_pat.search(u)]
    if verbose:
        print(f"[{cfg.key}] {len(urls)} listed, {len(cand)} match a governor slug")
    out: list[Speech] = []
    for u in cand:
        art = extract_article(u)
        if not art or len(art["text"]) < min_chars or art["date"] < since:
            continue
        speaker = _attribute(cfg, u + " " + art["title"])
        if not speaker:
            continue
        stype = ST_INTERVIEW if _INTERVIEW_RE.search(u + " " + art["title"]) else ST_SPEECH
        out.append(Speech(date=art["date"], speaker=speaker, title=art["title"][:200],
                          text=art["text"], source_type=stype, institution=cfg.institution,
                          source_url=u, orig_language=cfg.lang))
    if verbose:
        print(f"[{cfg.key}] -> {len(out)} governor speeches/interviews extracted")
    return out


def _attribute(cfg: NCBConfig, hay: str) -> str | None:
    for pat, name in cfg.governors.items():
        if re.search(pat, hay, re.I):
            return name
    return None


def collect_urls(urls: list[str], speaker: str, institution: str,
                 lang: str = "en", min_chars: int = 600,
                 since: str = "2010-01-01") -> list[Speech]:
    """Search-discovery path: extract a list of known article URLs (found via web
    search) into Speech records. Robust across sites - no per-site structure needed.
    """
    out, seen = [], set()
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        art = extract_article(u)
        if not art or len(art["text"]) < min_chars or art["date"] < since:
            continue
        stype = ST_INTERVIEW if _INTERVIEW_RE.search(u + " " + art["title"]) else ST_SPEECH
        out.append(Speech(date=art["date"], speaker=speaker, title=art["title"][:200],
                          text=art["text"], source_type=stype, institution=institution,
                          source_url=u, orig_language=lang))
    return out


# --- per-NCB configs (expanding) -------------------------------------------
CONFIGS: dict[str, NCBConfig] = {
    "LV": NCBConfig(
        key="LV", institution="Latvijas Banka", base="https://www.bank.lv",
        listing_path="/en/news-and-events/news-and-articles/news",
        link_re=r'href="(/en/[^"]*?news/\d{4,6}-[^"?#]+)"',
        governors={r"kazaks": "Mārtiņš Kazāks", r"rimsevics": "Ilmārs Rimšēvičs"},
        pages=60, start_step=10, start_param="start"),
}


if __name__ == "__main__":
    import sys
    key = sys.argv[1] if len(sys.argv) > 1 else "LV"
    speeches = scrape_ncb(CONFIGS[key])
    for s in speeches[:15]:
        print(f"  {s.date}  {s.source_type:9}  {s.speaker:18}  {s.title[:60]}")
