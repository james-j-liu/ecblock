"""ECB Governing Council communications -> Speech records (composite 'ECB council').

Three council document types, all sourced from the FoeDB publications index:
  /press/press_conference/monetary-policy-statement/  introductory statement + Q&A
                                                       (one page; split into two records)
  /press/accounts/                                     monetary policy accounts (minutes)

Every record gets speaker == ECB_COUNCIL so they aggregate together in the rankings,
while each document is still judged individually in the tournament. Council records
are policy by construction and bypass the classifier (see schema.COUNCIL_TYPES).

A press-conference page concatenates the prepared statement and the journalist Q&A.
On the modern format (2006+) the two are separated by a "* * *" delimiter; the
earliest pages (pre-2006) carry only the statement.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from ..config import RAW
from ..schema import ECB_COUNCIL, ST_ACCOUNT, ST_QA, ST_STATEMENT, Speech
from . import ecb_foedb

TEXT_CACHE = RAW / "council"
_INST = "European Central Bank"
_PC_PATH = "/press/press_conference/monetary-policy-statement/"
_ACC_PATH = "/press/accounts/"
_QA_SEP = "* * *"
_NAV = "Jump to the transcript of the questions and answers"


def _split_pc(text: str) -> tuple[str, str]:
    """Return (statement, qa); qa == '' when the page has no Q&A transcript."""
    body = text.replace(_NAV, " ").strip()
    i = body.find(_QA_SEP, 2000)
    if i == -1:
        return body, ""
    return body[:i].strip(), body[i + len(_QA_SEP):].strip()


def load(use_cache: bool = True, concurrency: int = 8,
         min_chars: int = 400) -> list[Speech]:
    TEXT_CACHE.mkdir(parents=True, exist_ok=True)
    recs = ecb_foedb.fetch_records(use_cache)
    pc = [r for r in ecb_foedb.filter_by_path(recs, _PC_PATH) if r.url.endswith(".en.html")]
    acc = [r for r in ecb_foedb.filter_by_path(recs, _ACC_PATH) if r.url.endswith(".en.html")]
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
        for rec, txt in tqdm(ex.map(get_text, pc), total=len(pc), desc="pressconf"):
            if not txt:
                continue
            base = rec.title or "Monetary policy statement"
            stmt, qa = _split_pc(txt)
            if len(stmt) >= min_chars:
                out.append(Speech(
                    date=rec.date, speaker=ECB_COUNCIL,
                    title=f"{base} — Statement",
                    text=stmt, source_type=ST_STATEMENT, institution=_INST,
                    source_url=rec.url, orig_language="en",
                ))
            if len(qa) >= min_chars:
                out.append(Speech(
                    date=rec.date, speaker=ECB_COUNCIL,
                    title=f"{base} — Q&A",
                    text=qa, source_type=ST_QA, institution=_INST,
                    source_url=rec.url, orig_language="en",
                ))

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for rec, txt in tqdm(ex.map(get_text, acc), total=len(acc), desc="accounts"):
            if not txt or len(txt) < min_chars:
                continue
            out.append(Speech(
                date=rec.date, speaker=ECB_COUNCIL,
                title=f"Account: {rec.title or rec.date}",
                text=txt, source_type=ST_ACCOUNT, institution=_INST,
                source_url=rec.url, orig_language="en",
            ))
    return out


if __name__ == "__main__":
    sp = load()
    import collections
    c = collections.Counter(s.source_type for s in sp)
    print(f"Loaded {len(sp)} ECB-council records:", dict(c))
    if sp:
        print("Date range:", min(s.date for s in sp), "..", max(s.date for s in sp))
        for st in (ST_STATEMENT, ST_QA, ST_ACCOUNT):
            ex = next((s for s in sp if s.source_type == st), None)
            if ex:
                print(f"\n[{st}] {ex.date} | {ex.title[:60]} | {ex.word_count} words")
                print(ex.text[:240])
