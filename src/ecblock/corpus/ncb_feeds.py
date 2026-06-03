"""Daily polling of NCB feeds/listings for new governor speeches & interviews.

A lightweight, automatable complement to the BIS bulk file (which lags a few days):
each NCB's RSS feed or speeches-listing page is polled for RECENT items, filtered to
the governor, and the full text is pulled with the shared trafilatura engine
(`ncb_scrape.extract_article`, which also falls back to the Wayback Machine).

Config-driven: add an `NCBFeed` to `FEEDS` to cover more national central banks as
their feeds/listings are identified. Most euro-area NCB sites are JS-rendered with no
RSS, so direct daily coverage is partial - the rest arrive via BIS with a short lag.
"""
from __future__ import annotations

import datetime
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from ..schema import ST_INTERVIEW, ST_SPEECH, Speech
from .ncb_scrape import _INTERVIEW_RE, extract_article, fetch


@dataclass
class NCBFeed:
    key: str
    institution: str
    kind: str                          # "rss" | "listing"
    url: str
    governors: dict[str, str] = None   # {surname_regex: canonical name}; filter by name
    link_re: str = ""                  # article-href regex (kind == "listing")
    single: str = ""                   # per-governor page: attribute ALL items to this name
    lang: str = "en"

    def __post_init__(self):
        if self.governors is None:
            self.governors = {}


def _rss_items(xml_text: str) -> list[tuple[str, str]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    out = []
    for el in root.iter():
        if el.tag.lower().split("}")[-1] not in ("item", "entry"):
            continue
        link, title = "", ""
        for c in el:
            t = c.tag.lower().split("}")[-1]
            if t == "link":
                link = c.get("href") or (c.text or "").strip()
            elif t == "title":
                title = (c.text or "").strip()
        if link:
            out.append((link, title))
    return out


def _listing_items(html: str, base: str, link_re: str) -> list[tuple[str, str]]:
    seen: dict[str, None] = {}
    for m in re.findall(link_re, html or ""):
        seen[m if m.startswith("http") else base + m] = None
    return [(u, "") for u in seen]


def poll(cfg: NCBFeed, since: str | None = None, max_items: int = 40) -> list[Speech]:
    since = since or (datetime.date.today() - datetime.timedelta(days=60)).isoformat()
    html = fetch(cfg.url)
    if not html:
        return []
    base = re.match(r"https?://[^/]+", cfg.url).group(0)
    if cfg.kind == "rss":
        cand = [(u if u.startswith("http") else base + (u if u.startswith("/") else "/" + u), t)
                for u, t in _rss_items(html)]
    else:
        cand = _listing_items(html, base, cfg.link_re)
    gov = re.compile("|".join(cfg.governors), re.I) if cfg.governors else None
    out = []
    for link, title in cand[:max_items]:
        # name-filter for general feeds; per-governor pages take everything
        if not cfg.single and not gov.search(link + " " + title):
            continue
        art = extract_article(link)
        if not art or art["date"] < since:
            continue
        hay = link + " " + title + " " + art["title"]
        if cfg.single:
            speaker = cfg.single
        else:
            speaker = next((n for p, n in cfg.governors.items() if re.search(p, hay, re.I)), None)
        if not speaker:
            continue
        st = ST_INTERVIEW if _INTERVIEW_RE.search(hay) else ST_SPEECH
        # prefer the feed-provided title (the article <title> is sometimes just the
        # bank name); fall back to the extracted title
        final_title = (title.strip() or art["title"])[:200]
        out.append(Speech(date=art["date"], speaker=speaker, title=final_title,
                          text=art["text"], source_type=st, institution=cfg.institution,
                          source_url=link, orig_language=cfg.lang))
    return out


def poll_all(since: str | None = None) -> list[Speech]:
    out: list[Speech] = []
    for cfg in FEEDS:
        try:
            got = poll(cfg, since)
            out.extend(got)
            print(f"[feed {cfg.key}] {len(got)} recent governor items")
        except Exception as e:  # noqa: BLE001 - never let one feed break the run
            print(f"[feed {cfg.key}] {type(e).__name__}: {e}")
    return out


FEEDS: list[NCBFeed] = [
    NCBFeed("SK", "Národná banka Slovenska", "rss", "https://nbs.sk/en/rss",
            {r"kazimir": "Peter Kažimír", r"makuch": "Jozef Makúch"}),
    NCBFeed("LV", "Latvijas Banka", "listing",
            "https://www.bank.lv/en/news-and-events/news-and-articles/news",
            {r"kazaks": "Mārtiņš Kazāks", r"rimsevics": "Ilmārs Rimšēvičs"},
            link_re=r'href="(/en/[^"]*?news/\d{4,6}-[^"?#]+)"'),
    # governor "keyword" page lists all his items; his speeches/interviews carry his
    # name in the slug (admin press releases don't), so the name filter keeps the right ones
    NCBFeed("EE", "Eesti Pank", "listing", "https://www.eestipank.ee/en/teemad/madis-muller",
            {r"muller": "Madis Müller"}, link_re=r'href="(/en/press/[^"?#]+)"'),
    NCBFeed("BG", "Bulgarian National Bank", "rss",
            "https://www.bnb.bg/AboutUs/PressOffice/PORSS/index.htm?getRSS=1&lang=EN&cat=2",
            {r"radev": "Dimitar Radev"}),
]


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    for s in poll_all():
        print(f"  {s.date}  {s.source_type:9}  {s.speaker:18}  {s.title[:55]}")
