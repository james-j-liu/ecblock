"""Classify the full corpus for monetary-policy relevance, in place.

Loads data/processed/corpus.jsonl, runs the two-stage classifier (council records
auto-pass), writes is_policy back to the same file, and reports how many survive
plus the resulting tournament cost estimate.

Usage:
  python scripts/classify_corpus.py
  python scripts/classify_corpus.py --model google/gemini-3.1-flash-lite --concurrency 8
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecblock.config import PROCESSED
from ecblock.process.classify import Classifier
from ecblock.schema import COUNCIL_TYPES, load_corpus, save_corpus

CORPUS = PROCESSED / "corpus.jsonl"
COST_PER_COMPARISON = 0.00096  # measured 2026-05-28 on gemini-3.1-flash-lite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--appearances", type=int, default=30)
    ap.add_argument("--since", default=None,
                    help="only classify records with date >= this ISO date (e.g. 2016-05-28)")
    ap.add_argument("--path", default=str(CORPUS))
    args = ap.parse_args()

    corpus = load_corpus(args.path)
    print(f"Loaded {len(corpus)} records")

    window = corpus
    if args.since:
        window = [s for s in corpus if s.date >= args.since]
        print(f"Classifying {len(window)} records in window (date >= {args.since})")

    clf = Classifier(model=args.model)
    print(f"Classifier model: {clf.model}")
    clf.classify_all(window, concurrency=args.concurrency)

    save_corpus(corpus, args.path)

    policy = [s for s in window if s.is_policy]
    council = [s for s in window if s.source_type in COUNCIL_TYPES]
    by_type = collections.Counter(s.source_type for s in policy)
    n = len(policy)
    comparisons = args.appearances // 2 * n  # 15*N at 30 appearances
    print(f"\nPolicy-relevant: {n}/{len(window)} ({100*n/len(window):.1f}%)")
    print(f"  council (auto-pass): {len(council)}")
    print(f"  by source type: {dict(by_type)}")
    print(f"\nTournament at {args.appearances} appearances -> ~{comparisons:,} comparisons")
    print(f"Estimated tournament cost: ~${comparisons * COST_PER_COMPARISON:,.0f}")


if __name__ == "__main__":
    main()
