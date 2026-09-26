"""Recent ECB Executive Board speeches from the FoeDB index (/press/key/).

The official speeches CSV (ecb_speeches) is the full-history source but is only
refreshed every few weeks. FoeDB lists a speech the day it is published, so this
loader covers the recent window; once the CSV catches up, daily_update's
(speaker, date) / (speaker, title) de-dup keeps the two from double counting.
"""
from __future__ import annotations

import datetime as dt

from ..config import RAW
from ..schema import ST_SPEECH, Speech
from . import ecb_foedb

TEXT_CACHE = RAW / "key"
_INST = "European Central Bank"


def load(use_cache: bool = True, days: int = 120, min_chars: int = 400,
         known_urls: set[str] = frozenset(), concurrency: int = 8) -> list[Speech]:
    since = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    recs = [r for r in ecb_foedb.filter_by_path(ecb_foedb.fetch_records(use_cache), "/press/key/")
            if r.boardmember and r.date >= since and r.url.endswith(".en.html")
            and r.url not in known_urls]
    out = []
    for rec, txt in ecb_foedb.fetch_texts(recs, TEXT_CACHE, use_cache, concurrency, "key"):
        if txt and len(txt) >= min_chars:
            out.append(Speech(date=rec.date, speaker=rec.boardmember, title=rec.title or "Speech",
                              text=txt, source_type=ST_SPEECH, institution=_INST,
                              source_url=rec.url, orig_language="en"))
    return out


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    for s in sorted(load(use_cache=False, days=45), key=lambda s: s.date):
        print(f"{s.date}  {s.speaker:20} {len(s.text):6}  {s.title[:60]}")
