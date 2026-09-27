"""Full re-score of the pool with the configured scoring models (e.g. a judge switch).

Resumable in chunks, so each call fits a time limit:
  1. pairwise  -> a fresh log (tournament_log.next.jsonl) until ~appearances*N/2
  2. direct    -> scores kept in direct.next.json until every pool speech is done
  3. --finalize swaps the new log in (old one kept as tournament_log.prev.jsonl),
     writes the new ratings + direct scores into the corpus, rebuilds data.json.
Nothing touches the live log/corpus scores until --finalize.

    python scripts/rescore_full.py --chunk 5000      # repeat until it says "done"
    python scripts/rescore_full.py --finalize
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from ecblock.config import PROCESSED
from ecblock.judge.factory import make_direct_scorer, make_pairwise_judge
from ecblock.macro.euro_macro import MacroContext
from ecblock.output.build_data import era_adjust, write_data_json
from ecblock.process.anonymize import Anonymizer
from ecblock.process.roster import build_roster
from ecblock.roster_gc import is_gc
from ecblock.schema import load_corpus, save_corpus
from ecblock.tournament.runner import run_tournament

CORPUS = PROCESSED / "corpus.jsonl"
LOG, NEXT_LOG, PREV_LOG = (PROCESSED / f for f in
                           ("tournament_log.jsonl", "tournament_log.next.jsonl", "tournament_log.prev.jsonl"))
NEXT_DIRECT = PROCESSED / "direct.next.json"
SINCE = "2010-05-28"


def _n_logged() -> int:
    return sum(1 for _ in NEXT_LOG.open(encoding="utf-8")) if NEXT_LOG.exists() else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--appearances", type=int, default=30)
    ap.add_argument("--chunk", type=int, default=5000, help="max new API calls this run")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()

    corpus = load_corpus(CORPUS)
    today = datetime.date.today().isoformat()
    pool = [s for s in corpus if s.is_policy and is_gc(s.speaker) and SINCE <= s.date <= today]
    target = args.appearances * len(pool) // 2
    anon = Anonymizer(build_roster([s.speaker for s in corpus]))
    macro = MacroContext()
    direct = json.loads(NEXT_DIRECT.read_text()) if NEXT_DIRECT.exists() else {}

    if not args.finalize:
        have = _n_logged()
        print(f"pool {len(pool)} | pairwise {have}/{target} | direct {len(direct)}/{len(pool)}", flush=True)
        if have < target:
            judge = make_pairwise_judge()
            run_tournament(pool, judge, appearances_per_speech=args.appearances, macro=macro,
                           log_path=NEXT_LOG, resume=True, anonymizer=anon,
                           concurrency=args.concurrency, min_total=target, max_new=args.chunk)
            print(f"pairwise now {_n_logged()}/{target}  (${getattr(judge, 'cost', 0):.3f} this chunk)")
            return
        todo = [s for s in pool if s.id not in direct][:args.chunk]
        if todo:
            scorer = make_direct_scorer()
            for s in todo:
                s.direct_score = None
            scorer.score_all(todo, macro, concurrency=args.concurrency, anonymizer=anon)
            direct.update({s.id: s.direct_score for s in todo if s.direct_score is not None})
            NEXT_DIRECT.write_text(json.dumps(direct))
            print(f"direct now {len(direct)}/{len(pool)}  (${getattr(scorer, 'cost', 0):.3f} this chunk)")
            if len(direct) < len(pool):
                return
        print("done - run with --finalize")
        return

    # ---- finalize: swap in the new log + scores ----
    missing = [s.id for s in pool if s.id not in direct]
    if _n_logged() < target * 0.95 or len(missing) > len(pool) * 0.01:
        sys.exit(f"not complete: pairwise {_n_logged()}/{target}, direct missing {len(missing)}")
    shutil.copyfile(LOG, PREV_LOG)
    shutil.move(NEXT_LOG, LOG)
    # replay the new log for final ratings (max_new=0: no API calls)
    replay_only = type("Replay", (), {"model": "replay"})()
    run_tournament(pool, replay_only, appearances_per_speech=args.appearances, macro=macro,
                   log_path=LOG, resume=True, anonymizer=anon, max_new=0)
    for s in pool:
        s.direct_score = direct.get(s.id)
    save_corpus(corpus, CORPUS)
    NEXT_DIRECT.unlink()
    era_adjust(pool)
    meta = write_data_json(pool, ROOT / "site" / "data.json")
    print(f"finalized: {meta['n_speeches']} speeches, pairwise={meta['n_pairwise']} direct={meta['n_direct']}")


if __name__ == "__main__":
    main()
