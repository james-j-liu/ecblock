"""Generate a FAITHFUL placeholder data.json for previewing the finished site.

Uses the real corpus (actual speakers, dates, titles, institutions) but assigns
*synthetic* scores so the layout/density/rankings/toggle render as they will with
real data. Scores follow a plausible euro-area hawk->dove->hawk regime curve plus
per-speaker dispositions. The pairwise score is continuous; the direct score is the
same signal snapped toward round numbers, to mimic the clustering the real direct
method exhibits. NOT a measurement -- a visual stand-in only.
"""
from __future__ import annotations

import random
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecblock.config import PROCESSED
from ecblock.output.build_data import era_adjust, write_data_json
from ecblock.roster_gc import is_gc
from ecblock.schema import load_corpus

SINCE, UNTIL = "2010-05-28", "2026-05-28"
rng = random.Random(7)


def regime(d: str) -> float:
    """Baseline hawkishness over time (0-100), echoing the real ECB cycle."""
    y = date.fromisoformat(d)
    t = y.year + (y.month - 1) / 12
    if t < 2011.6:   base = 58          # 2010-11 tightening run-up to the July-2011 hike
    elif t < 2012.5: base = 60 - (t - 2011.6) * 18   # post-hike pivot to easing
    elif t < 2021.8: base = 42 + 3 * (t - 2016)       # long easing / low-inflation era, drifting
    elif t < 2023.6: base = 42 + (t - 2021.8) * 16    # 2022-23 hiking cycle, sharply hawkish
    else:            base = 70 - (t - 2023.6) * 12     # 2024+ easing
    return max(30.0, min(72.0, base))


def main():
    corpus = load_corpus(PROCESSED / "corpus.jsonl")
    pool = [s for s in corpus if s.is_policy and SINCE <= s.date <= UNTIL]
    pool = [s for s in pool if is_gc(s.speaker)]  # GC members only (match the real run)
    print(f"{len(pool)} GC policy records in window")

    # per-speaker disposition (hawkish/dovish lean), stable across their speeches
    lean = {}
    for s in pool:
        lean.setdefault(s.speaker, rng.gauss(0, 5))

    for s in pool:
        mu = regime(s.date) + lean[s.speaker] + rng.gauss(0, 4)
        mu = max(20.0, min(80.0, mu))
        s.mu = round(mu, 2)
        s.sigma = round(rng.uniform(1.6, 2.2), 2)
        s.n_comparisons = 30
        # direct: same signal, snapped toward nearest 5 (clustering), with mild noise
        d = mu + rng.gauss(0, 3)
        s.direct_score = round(max(0.0, min(100.0, round(d / 5) * 5 + rng.gauss(0, 1))), 2)

    era_adjust(pool)
    meta = write_data_json(pool, "site/data.json")
    print("Wrote site/data.json (PLACEHOLDER):", meta)


if __name__ == "__main__":
    main()
