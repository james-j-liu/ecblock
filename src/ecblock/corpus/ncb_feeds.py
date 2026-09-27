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
    name_in_text: bool = False         # names not in link/title: check the article's opening
    title_re: str = ""                 # also accept items whose title matches (e.g. "Governor's address")
    title_since: str = ""              # ...unnamed ones only from here (current governor's tenure)
    min_chars: int = 1500              # drop stubs that just link out to a news site

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


def _listing_items(html: str, page_url: str, link_re: str) -> list[tuple[str, str]]:
    """(absolute url, link text) per match; a second regex group, if present,
    captures the link text/title (often the only place a PDF's date appears)."""
    from urllib.parse import urljoin
    seen: dict[str, str] = {}
    for m in re.findall(link_re, html or "", re.S):
        href, text = (m, "") if isinstance(m, str) else (m[0], m[1])
        u = urljoin(page_url, href.replace("&amp;", "&"))
        if u not in seen:
            seen[u] = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()
    return list(seen.items())


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
        cand = _listing_items(html, cfg.url, cfg.link_re)
    # bilingual listings: English first, so it wins the (speaker, date) de-dup
    cand.sort(key=lambda it: "/en/" not in it[0])
    gov = re.compile("|".join(cfg.governors), re.I) if cfg.governors else None
    title_ok = re.compile(cfg.title_re, re.I) if cfg.title_re else None
    out = []
    for link, title in cand[:max_items]:
        # name-filter for general feeds; per-governor pages take everything
        named = cfg.single or gov.search(link + " " + title)
        if not (named or cfg.name_in_text or (title_ok and title_ok.search(title))):
            continue
        art = extract_article(link, title_hint=title)
        if not art or art["date"] < since or len(art["text"]) < cfg.min_chars:
            continue
        hay = link + " " + title + " " + art["title"]
        if cfg.name_in_text:
            hay += " " + art.get("description", "") + " " + art["text"][:1500]
        if cfg.single:
            speaker = cfg.single
        else:
            speaker = next((n for p, n in cfg.governors.items() if re.search(p, hay, re.I)), None)
            if (not speaker and title_ok and title_ok.search(title)
                    and art["date"] >= cfg.title_since):
                speaker = next(iter(cfg.governors.values()))   # unnamed "Governor's address"
            if speaker and cfg.title_since and art["date"] < cfg.title_since \
                    and speaker == next(iter(cfg.governors.values())) and not gov.search(hay):
                speaker = None
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
    NCBFeed("EE", "Eesti Pank", "listing", "https://www.eestipank.ee/en/teemad/ulo-kaasik",
            {r"kaasik": "Ülo Kaasik"}, link_re=r'href="(/en/press/[^"?#]+)"'),
    NCBFeed("BG", "Bulgarian National Bank", "rss",
            "https://www.bnb.bg/AboutUs/PressOffice/PORSS/index.htm?getRSS=1&lang=EN&cat=2",
            {r"radev": "Dimitar Radev"}),
    # Bundesbank hosts full interview transcripts (en + de; English listed first so
    # it wins the (speaker, date) de-dup). Slugs rarely name the speaker, so the
    # name is checked in the transcript's opening.
    NCBFeed("DE-int", "Deutsche Bundesbank", "listing", "https://www.bundesbank.de/en/press/interviews",
            {r"\bnagel\b": "Joachim Nagel"}, name_in_text=True,
            link_re=r'href="((?:https://www\.bundesbank\.de)?/(?:en/press|de/presse)/interviews/[^"?#]+-\d+)"'),
    # BCL hosts the governor's interviews as HTML or PDF (fr/de/lb/en; translated
    # downstream). The link title carries the outlet and date, e.g. "... donnée au
    # Luxemburger Wort, édition du 28 novembre 2017".
    NCBFeed("LU-int", "Banque centrale du Luxembourg", "listing",
            "https://www.bcl.lu/fr/media_actualites/Interviews/index.html",
            {r"reinesch": "Gaston Reinesch"}, lang="fr",
            link_re=r'<a href="([^"]+)"[^>]*title="([^"]*Reinesch[^"]*)"'),
    # Central Bank of Malta posts the governor's addresses as press releases
    # (a summary of the address, not a verbatim text); interviews there only link
    # out (Econostream), so the length floor drops them.
    NCBFeed("MT", "Central Bank of Malta", "rss", "https://www.centralbankmalta.org/rssnews.ashx",
            {r"demarco": "Alexander Demarco", r"scicluna": "Edward Scicluna"},
            title_re=r"governor.{0,3}s address|governor addresses|address by the governor",
            title_since="2026-01-01"),
]


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    for s in poll_all():
        print(f"  {s.date}  {s.source_type:9}  {s.speaker:18}  {s.title[:55]}")
