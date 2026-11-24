"""(Re-)read core members' and the Council's documents for the next-meeting framework.

    python scripts/code_guidance.py      # needs OPENROUTER_API_KEY
Reads only documents not yet read by the live model (policy/guidance.LIVE_MODEL).
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecblock.config import PROCESSED
from ecblock.policy import calendar as cal
from ecblock.policy import guidance as gd
from ecblock.process.anonymize import Anonymizer
from ecblock.process.roster import build_roster
from ecblock.schema import load_corpus


def main():
    today = dt.date.today().isoformat()
    corpus = load_corpus(PROCESSED / "corpus.jsonl")
    meetings = cal.build(corpus)
    anon = Anonymizer(build_roster([s.speaker for s in corpus]))
    core = [s for s in corpus if gd.is_core(s) and s.is_policy and s.text and "2010-05-28" <= s.date <= today]
    print(f"core documents: {len(core)}")
    gd.classify(core, anon)
    council_docs = [s for s in corpus if gd.is_council(s) and s.text and "2010-01-01" <= s.date <= today]
    print(f"council documents: {len(council_docs)}")
    gd.classify(council_docs, anon, decisions={m.date: m.bp for m in meetings if m.bp is not None})


if __name__ == "__main__":
    main()
