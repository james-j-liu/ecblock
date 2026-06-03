"""ECB media interviews -> Speech records.

Interviews are media Q&As (FT, Die Zeit, Reuters, Corriere, ...) conducted with
ECB Executive Board members and republished on ecb.europa.eu. They are a separate
publication type from speeches, so they don't appear on the speeches list.

Source: the FoeDB publications index, filtered to /press/inter/. Each transcript's
text is fetched once and cached to data/raw/interviews/{id}.txt.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from ..config import RAW
from ..schema import ST_INTERVIEW, Speech
from . import ecb_foedb

TEXT_CACHE = RAW / "interviews"
_INST = "European Central Bank"


def load(use_cache: bool = True, concurrency: int = 8,
         min_chars: int = 400) -> list[Speech]:
    TEXT_CACHE.mkdir(parents=True, exist_ok=True)
    recs = ecb_foedb.filter_by_path(ecb_foedb.fetch_records(use_cache), "/press/inter/")
    sess = ecb_foedb._session()

    def get_text(rec: ecb_foedb.PubRecord) -> tuple[ecb_foedb.PubRecord, str]:
        cache_f = TEXT_CACHE / f"{rec.id}.txt"
        if use_cache and cache_f.exists():
            return rec, cache_f.read_text(encoding="utf-8")
        txt = ecb_foedb.fetch_text(rec.url, sess)
        if txt:
            cache_f.write_text(txt, encoding="utf-8")
        return rec, txt

    out: list[Speech] = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for rec, txt in tqdm(ex.map(get_text, recs), total=len(recs), desc="interviews"):
            if not txt or len(txt) < min_chars or not rec.boardmember:
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
