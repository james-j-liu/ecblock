"""ECB media interviews -> Speech records.

Interviews are media Q&As (FT, Die Zeit, Reuters, Corriere, ...) conducted with
ECB Executive Board members and republished on ecb.europa.eu. They are a separate
publication type from speeches, so they don't appear on the speeches list.

Source: the FoeDB publications index, filtered to /press/inter/. Each transcript's
text is fetched once and cached to data/raw/interviews/{id}.txt.
"""
from __future__ import annotations

from ..config import RAW
from ..schema import ST_INTERVIEW, Speech
from . import ecb_foedb

TEXT_CACHE = RAW / "interviews"
_INST = "European Central Bank"


def load(use_cache: bool = True, concurrency: int = 8, min_chars: int = 400,
         known_urls: set[str] = frozenset()) -> list[Speech]:
    # known_urls: pages already in the corpus are skipped rather than re-downloaded
    recs = [r for r in ecb_foedb.filter_by_path(ecb_foedb.fetch_records(use_cache), "/press/inter/")
            if r.boardmember and r.url not in known_urls]
    out: list[Speech] = []
    for rec, txt in ecb_foedb.fetch_texts(recs, TEXT_CACHE, use_cache, concurrency, "interviews"):
        if not txt or len(txt) < min_chars:
            continue
        out.append(Speech(
            date=rec.date,
            speaker=rec.boardmember,
            title=rec.title or "Interview",
            text=txt,
            source_type=ST_INTERVIEW,
            institution=_INST,
            source_url=rec.url,
            orig_language="en",
        ))
    return out


if __name__ == "__main__":
    sp = load()
    print(f"Loaded {len(sp)} ECB interviews with text")
    if sp:
        print("Date range:", min(s.date for s in sp), "..", max(s.date for s in sp))
        import collections
        c = collections.Counter(s.speaker for s in sp)
        for name, n in c.most_common(8):
            print(f"  {n:3d}  {name}")
        s = sp[0]
        print("\nExample:", s.date, s.speaker, "|", s.title[:70])
        print(s.text[:300])
