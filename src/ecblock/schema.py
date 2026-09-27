"""Core data model for the corpus.

A Speech is one judged unit. Individual speakers (Exec Board members, NCB
governors) get one record per speech. The "ECB council" composite speaker
(press-conference statement + Q&A, monetary policy accounts) is modelled by
setting speaker == ECB_COUNCIL on those records, so they aggregate together in
the rankings while still being scored per-document in the tournament.
"""
from __future__ import annotations

import base64
import hashlib
import json
import zlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Optional

ECB_COUNCIL = "ECB council"

# Document types
ST_SPEECH = "speech"            # individual member speech / lecture / remarks
ST_INTERVIEW = "interview"      # media interview (incl. from news sources)
ST_STATEMENT = "mp_statement"   # ECB press-conference monetary policy statement
ST_QA = "mp_qa"                 # ECB press-conference Q&A
ST_ACCOUNT = "mp_account"       # ECB monetary policy account (minutes)

COUNCIL_TYPES = {ST_STATEMENT, ST_QA, ST_ACCOUNT}


@dataclass
class Speech:
    date: str                       # ISO YYYY-MM-DD
    speaker: str                    # canonical speaker name, or ECB_COUNCIL
    title: str
    text: str                       # full text, English (post-translation)
    source_type: str                # one of the ST_* constants
    institution: str = ""           # e.g. "European Central Bank", "Deutsche Bundesbank"
    source_url: str = ""
    orig_language: str = "en"
    translated: bool = False
    word_count: int = 0
    is_policy: Optional[bool] = None      # set by classifier
    text_anon: str = ""                   # in-memory cache only; never saved (see save_corpus)

    # pairwise-tournament outputs (filled later)
    mu: Optional[float] = None
    mu_adj: Optional[float] = None
    sigma: Optional[float] = None
    n_comparisons: int = 0

    # direct-scoring outputs (alternative method: one LLM 0-100 rating per speech)
    direct_score: Optional[float] = None
    direct_adj: Optional[float] = None

    id: str = ""

    def __post_init__(self):
        if not self.word_count and self.text:
            self.word_count = len(self.text.split())
        if not self.id:
            self.id = self.make_id()

    def make_id(self) -> str:
        h = hashlib.sha1(
            f"{self.date}|{self.speaker}|{self.title}|{self.source_url}".encode("utf-8")
        ).hexdigest()[:16]
        return h

    @property
    def is_council(self) -> bool:
        return self.speaker == ECB_COUNCIL or self.source_type in COUNCIL_TYPES


# Long texts are stored compressed (zlib, then base64 so the file stays JSONL),
# keeping the daily-committed file well inside GitHub's 100 MB per-file limit.
# zlib is deterministic, so a record that has not changed is byte-identical on
# every save and git's daily deltas stay as small as they were with plain text.
_COMPRESS_OVER = 2000


def _record(s: "Speech") -> dict:
    d = asdict(s)
    # text_anon is derived (Anonymizer.text_of recomputes it on demand) and would
    # double the file, so it is dropped on save
    d["text_anon"] = ""
    if len(d["text"]) > _COMPRESS_OVER:
        d["text_z"] = base64.b64encode(zlib.compress(d["text"].encode("utf-8"), 9)).decode("ascii")
        d["text"] = ""
    return d


def _from_json(line: str) -> "Speech":
    rec = json.loads(line)
    packed = rec.pop("text_z", None)
    if packed:
        rec["text"] = zlib.decompress(base64.b64decode(packed)).decode("utf-8")
    return Speech(**rec)


def save_corpus(speeches: list[Speech], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in speeches:
            f.write(json.dumps(_record(s), ensure_ascii=False) + "\n")


def load_corpus(path: str | Path) -> list[Speech]:
    path = Path(path)
    out: list[Speech] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(_from_json(line))
    return out


def iter_corpus(path: str | Path) -> Iterator[Speech]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield _from_json(line)
