"""Finalize the incremental score after the pairwise comparisons are already in
the log: replay the full log for ratings, reuse existing direct scores from the
current data.json, direct-score only the NEW speeches, then write data.json.

Avoids re-running direct scoring on the whole pool (slow, and the long task kept
getting killed) - only the handful of new speeches are scored.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from ecblock.config import PROCESSED
from ecblock.macro.euro_macro import MacroContext
from ecblock.output.build_data import era_adjust, write_data_json
from ecblock.process.anonymize import Anonymizer
from ecblock.process.roster import build_roster
from ecblock.roster_gc import is_gc
from ecblock.schema import load_corpus, save_corpus
from ecblock.tournament.runner import run_tournament

import datetime
CORPUS = PROCESSED / "corpus.jsonl"
LOG = PROCESSED / "tournament_log.jsonl"
SINCE, UNTIL = "2010-05-28", datetime.date.today().isoformat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site/data.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    corpus = load_corpus(CORPUS)
    pool = [s for s in corpus if s.is_policy and is_gc(s.speaker) and SINCE <= s.date <= UNTIL]
    print(f"Pool: {len(pool)} GC policy records", flush=True)

    # ---- pairwise ratings from the full log (free; no API calls) ----
    # replay through the shared runner so draw handling etc. match the daily job
    replay_only = type("Replay", (), {"model": "replay"})()
    run_tournament(pool, replay_only, macro=MacroContext(), log_path=LOG,
                   resume=True, seed=args.seed, max_new=0)

    # ---- direct scores: corpus is the canonical (discrete, un-centered) source;
    # only score speeches that don't have one yet ----
    new_to_score = [s for s in pool if s.direct_score is None]
    print(f"Direct: have {len(pool) - len(new_to_score)} from corpus, "
          f"scoring {len(new_to_score)} new", flush=True)

    macro = MacroContext()
    if new_to_score:
        # the scorer anonymises lazily - only the records it scores
        anon = Anonymizer(build_roster([s.speaker for s in corpus]))
        if args.dry_run:
            from run_full import MockDirectScorer
            for s in new_to_score:
                anon.text_of(s)
            MockDirectScorer().score_all(new_to_score, macro)
        else:
            from ecblock.judge.factory import make_direct_scorer
            make_direct_scorer().score_all(new_to_score, macro, concurrency=6, anonymizer=anon)
            save_corpus(corpus, CORPUS)   # persist new direct scores

    era_adjust(pool)
    meta = write_data_json(pool, args.out)
    print(f"Wrote {args.out}: {meta['n_speeches']} speeches, {meta['n_speakers']} speakers, "
          f"pairwise={meta['n_pairwise']} direct={meta['n_direct']}")


if __name__ == "__main__":
    main()
