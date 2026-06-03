"""Proof-of-concept driver: ECB Exec Board speeches -> anonymize -> tournament -> data.json.

Usage:
  python scripts/run_poc.py --n 50                 # real judge (needs OPENROUTER_API_KEY)
  python scripts/run_poc.py --n 50 --dry-run       # mock judge, no API, validates wiring
  python scripts/run_poc.py --n 50 --model anthropic/claude-haiku-4.5
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecblock.corpus import ecb_speeches
from ecblock.macro.euro_macro import MacroContext
from ecblock.output.build_data import era_adjust, write_data_json
from ecblock.process.anonymize import Anonymizer
from ecblock.process.roster import build_roster
from ecblock.tournament.runner import run_tournament


class MockJudge:
    """Deterministic heuristic judge for --dry-run wiring tests (no API).

    Crude hawkish lexicon vs dovish lexicon; the more net-hawkish excerpt wins.
    This is NOT a real measurement, only a plumbing check.
    """
    model = "mock"
    HAWK = ["inflation", "price stability", "tighten", "restrictive", "vigilant",
            "overheating", "hike", "raise rates", "upside risk", "anchor"]
    DOVE = ["growth", "unemployment", "accommodation", "stimulus", "downside",
            "support", "cut", "ease", "recovery", "slack", "purchase"]

    def _score(self, t: str) -> int:
        tl = t.lower()
        return sum(tl.count(w) for w in self.HAWK) - sum(tl.count(w) for w in self.DOVE)

    def compare(self, a_text, a_macro, b_text, b_macro):
        sa, sb = self._score(a_text), self._score(b_text)
        if sa == sb:
            w = random.choice(["A", "B"])
        else:
            w = "A" if sa > sb else "B"
        return {"winner": w, "confidence": 0.5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--appearances", type=int, default=15)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="site/data.json")
    args = ap.parse_args()

    print("Loading ECB Executive Board corpus...")
    speeches = ecb_speeches.load()
    rng = random.Random(args.seed)
    # sample across the whole period so era adjustment has multiple quarters
    sample = rng.sample(speeches, min(args.n, len(speeches)))
    print(f"Sampled {len(sample)} speeches")

    roster = build_roster([s.speaker for s in speeches])
    anon = Anonymizer(roster)
    for s in sample:
        s.text_anon = anon(s.text)

    macro = MacroContext()

    if args.dry_run:
        judge = MockJudge()
        print("DRY RUN: using MockJudge (no API calls)")
    else:
        from ecblock.judge.openrouter import Judge
        judge = Judge(model=args.model)
        print(f"Judge model: {judge.model}")

    run_tournament(sample, judge, appearances_per_speech=args.appearances,
                   macro=macro, seed=args.seed)
    era_adjust(sample)
    meta = write_data_json(sample, args.out)
    print("Wrote", args.out, meta)

    ranked = sorted(sample, key=lambda s: -(s.mu_adj or 0))
    print("\nMost hawkish (era-adjusted):")
    for s in ranked[:8]:
        print(f"  {s.mu_adj:5.1f}  {s.date}  {s.speaker[:22]:22}  {s.title[:50]}")
    print("Most dovish:")
    for s in ranked[-8:]:
        print(f"  {s.mu_adj:5.1f}  {s.date}  {s.speaker[:22]:22}  {s.title[:50]}")


if __name__ == "__main__":
    main()
