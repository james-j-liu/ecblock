"""Finalize the incremental score after the pairwise comparisons are already in
the log: replay the full log for ratings, reuse existing direct scores from the
current data.json, direct-score only the NEW speeches, then write data.json.

Avoids re-running direct scoring on the whole pool (slow, and the long task kept
getting killed) - only the handful of new speeches are scored.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from ecblock.config import PROCESSED, cfg
from ecblock.macro.euro_macro import MacroContext
from ecblock.output.build_data import _jitter, era_adjust, write_data_json
from ecblock.process.anonymize import Anonymizer
from ecblock.process.roster import build_roster
from ecblock.roster_gc import is_gc
from ecblock.schema import load_corpus, save_corpus
from ecblock.tournament.engine import Tournament

CORPUS = PROCESSED / "corpus.jsonl"
LOG = PROCESSED / "tournament_log.jsonl"
SINCE, UNTIL = "2010-05-28", "2026-05-28"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site/data.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    corpus = load_corpus(CORPUS)
    pool = [s for s in corpus if s.is_policy and is_gc(s.speaker) and SINCE <= s.date <= UNTIL]
    by_id = {s.id: s for s in pool}
    print(f"Pool: {len(pool)} GC policy records")

    roster = build_roster([s.speaker for s in corpus])
    anon = Anonymizer(roster)
    for s in pool:
        if not s.text_anon:
            s.text_anon = anon(s.text)

    # ---- pairwise ratings from the full log (free) ----
    tcfg = cfg()["tournament"]
    tour = Tournament(list(by_id), initial_mu=tcfg["initial_mu"],
                      initial_sigma=tcfg["initial_sigma"], seed=args.seed)
    replayed = 0
    with LOG.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line); a, b, w = r["a"], r["b"], r.get("winner")
            except (json.JSONDecodeError, KeyError):
                continue
            if a in by_id and b in by_id:
                if w == "A":
                    tour.record(a, b); replayed += 1
                elif w == "B":
                    tour.record(b, a); replayed += 1
    print(f"Replayed {replayed} comparisons")
    for sid, s in by_id.items():
        r = tour.rating(sid)
        s.mu, s.sigma, s.n_comparisons = round(r.mu, 3), round(r.sigma, 3), tour.n_comp[sid]

    # ---- direct scores: corpus is the canonical (discrete, un-centered) source;
    # only score speeches that don't have one yet ----
    new_to_score = [s for s in pool if s.direct_score is None]
    print(f"Direct: have {len(pool) - len(new_to_score)} from corpus, "
          f"scoring {len(new_to_score)} new")

    macro = MacroContext()
    if new_to_score:
        if args.dry_run:
            from run_full import MockDirectScorer
            MockDirectScorer().score_all(new_to_score, macro)
        else:
            from ecblock.judge.direct import DirectScorer
            DirectScorer().score_all(new_to_score, macro, concurrency=6)
            save_corpus(corpus, CORPUS)   # persist new direct scores

    era_adjust(pool)
    meta = write_data_json(pool, args.out)
    print(f"Wrote {args.out}: {meta['n_speeches']} speeches, {meta['n_speakers']} speakers, "
          f"pairwise={meta['n_pairwise']} direct={meta['n_direct']}")


if __name__ == "__main__":
    main()
