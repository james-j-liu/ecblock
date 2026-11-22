"""Text-level duplicate detection: the same speech held twice.

An ECB speech is often republished by a national central bank or by BIS, under a
different URL, a title with the speaker prefixed ("Philip R Lane: …"), or a date a few
days off. Matching on ids, dates or titles misses those; this compares the texts.
Two records are the same speech when they are by the same person (after alias
folding), dated within 10 days, and share at least 85% of the shorter text's
8-word shingles (sampled 1 in 8 for speed).
"""
from __future__ import annotations

import datetime as dt
import re
import zlib

_W = re.compile(r"[a-z]+")
WINDOW_DAYS = 10
THRESHOLD = 0.85


def shingles(text: str) -> set[int]:
    w = _W.findall((text or "").lower())
    out = set()
    for i in range(max(0, len(w) - 8)):
        h = zlib.crc32(" ".join(w[i:i + 8]).encode())
        if h % 8 == 0:
            out.add(h)
    return out


def containment(a: set, b: set) -> float:
    return len(a & b) / max(1, min(len(a), len(b))) if a and b else 0.0


def _days(a: str, b: str) -> int:
    return abs((dt.date.fromisoformat(a) - dt.date.fromisoformat(b)).days)


def find_duplicates(records, canon) -> list[tuple]:
    """[(keep, drop, share)] for every pair that is the same speech. Keeps the ECB's own
    copy over a republished one, else the earlier-dated one, else the longer text."""
    by_person: dict[str, list] = {}
    for s in records:
        if (s.text or "") and len(s.text) > 1500 and s.date:
            by_person.setdefault(canon(s.speaker), []).append(s)
    pairs, dropped = [], set()
    for person, recs in by_person.items():
        recs.sort(key=lambda s: s.date)
        sh = {s.id: shingles(s.text) for s in recs}
        for i, a in enumerate(recs):
            for b in recs[i + 1:]:
                if _days(a.date, b.date) > WINDOW_DAYS:
                    break
                if a.id in dropped or b.id in dropped or a.source_type != b.source_type:
                    continue
                share = containment(sh[a.id], sh[b.id])
                if share >= THRESHOLD:
                    keep, drop = sorted((a, b), key=lambda s: (
                        s.institution != "European Central Bank", s.date, -len(s.text)))
                    pairs.append((keep, drop, share))
                    dropped.add(drop.id)
    return pairs


def is_duplicate_of(new, existing_same_person) -> object | None:
    """The record in `existing_same_person` that `new` duplicates, if any."""
    sn = shingles(new.text)
    for s in existing_same_person:
        if s.date and new.date and _days(s.date, new.date) <= WINDOW_DAYS and \
                s.source_type == new.source_type and containment(sn, shingles(s.text)) >= THRESHOLD:
            return s
    return None
