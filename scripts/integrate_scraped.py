"""Integrate scraped NCB governor speeches into the corpus.

Dedups the scraped records against the existing corpus (by URL and by
speaker+date+title), appends the new ones, classifies ONLY the new records for
monetary-policy relevance, and reports the counts that will enter the tournament.
Cheap step (classification only); the scoring spend happens in score_increment.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from ecblock.config import PROCESSED
from ecblock.process.classify import Classifier
from ecblock.roster_gc import canon, is_gc
from ecblock.schema import load_corpus, save_corpus

CORPUS = PROCESSED / "corpus.jsonl"
SCRAPED = PROCESSED / "ncb_scraped.jsonl"
SINCE, UNTIL = "2010-05-28", "2026-05-28"


def main():
    corpus = load_corpus(CORPUS)
    scraped = load_corpus(SCRAPED)

    existing_urls = {s.source_url for s in corpus if s.source_url}
    existing_kd = {(canon(s.speaker), s.date, s.title.strip().lower()[:40]) for s in corpus}
    new = []
    for s in scraped:
        if s.source_url in existing_urls:
            continue
        if (canon(s.speaker), s.date, s.title.strip().lower()[:40]) in existing_kd:
            continue
        new.append(s)
    print(f"scraped={len(scraped)}  new-after-dedup={len(new)}")

    in_window = [s for s in new if SINCE <= s.date <= UNTIL and is_gc(s.speaker)]
    print(f"new in-window & GC: {len(in_window)}")

    clf = Classifier()
    print(f"Classifying {len(in_window)} new records (model {clf.model})...")
    clf.classify_all(in_window, concurrency=6)
    policy = [s for s in in_window if s.is_policy]
    print(f"policy-relevant: {len(policy)}/{len(in_window)}")

    # append all new (classified) to the corpus
    corpus.extend(new)
    save_corpus(corpus, CORPUS)
    print(f"corpus now {len(corpus)} records (was {len(corpus) - len(new)})")

    from collections import Counter
    c = Counter(canon(s.speaker) for s in policy)
    print("\nnew policy records per governor:")
    for g, n in c.most_common():
        print(f"  +{n}  {g}")
    print(f"\nNew speeches entering the tournament: {len(policy)}")
    print(f"Incremental pairwise cost est: ~{len(policy)*30:,} comparisons "
          f"~${len(policy)*30*0.00096:.2f}")


if __name__ == "__main__":
    main()
