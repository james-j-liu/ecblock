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


def load(use_cache: bool = True, concurrency: int = 8, min_chars: int = 400,
         known_urls: set[str] = frozenset()) -> list[Speech]:
    # known_urls: pages already in the corpus are skipped rather than re-downloaded
    recs = ecb_foedb.fetch_records(use_cache)
    pick = lambda p: [r for r in ecb_foedb.filter_by_path(recs, p)
                      if r.url.endswith(".en.html") and r.url not in known_urls]
    pc, acc = pick(_PC_PATH), pick(_ACC_PATH)

    out: list[Speech] = []
    for rec, txt in ecb_foedb.fetch_texts(pc, TEXT_CACHE, use_cache, concurrency, "pressconf"):
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

    for rec, txt in ecb_foedb.fetch_texts(acc, TEXT_CACHE, use_cache, concurrency, "accounts"):
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
