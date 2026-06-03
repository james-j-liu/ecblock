"""Daily incremental update — ingest new speeches and score only the new ones.

Pulls the latest from the automated sources (ECB FoeDB speeches/interviews/press
conferences + BIS bulk NCB speeches), finds records not already in the corpus,
translates + classifies them, then scores ONLY the new ones:
  - pairwise: resume the TrueSkill tournament; the new high-uncertainty speeches
    draw the comparisons while existing ratings are replayed from the log,
  - direct: score only speeches that don't have a direct score yet.
Finally rebuilds site/data.json and site/macro.json.

State (data/processed/corpus.jsonl with classifications + scores, and
tournament_log.jsonl) is the persistent memory between runs, so in CI it should
be committed back to the repo after each run. Cost is a few cents/day.
"""
from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from ecblock.config import PROCESSED
from ecblock.corpus import assemble, ncb_feeds
from ecblock.macro.euro_macro import MacroContext
from ecblock.output.build_data import era_adjust, write_data_json
from ecblock.process.anonymize import Anonymizer
from ecblock.process.classify import Classifier
from ecblock.process.roster import build_roster
from ecblock.process.translate import Translator
from ecblock.roster_gc import canon, is_gc
from ecblock.schema import load_corpus, save_corpus
from ecblock.tournament.runner import run_tournament

CORPUS = PROCESSED / "corpus.jsonl"
SINCE = "2010-05-28"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--appearances", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true", help="use mock scorers (no API spend)")
    ap.add_argument("--feeds-only", action="store_true",
                    help="skip the ECB/BIS loaders; only poll the NCB feeds")
    ap.add_argument("--feed-since", default=None,
                    help="ISO date lower bound for feed items (default: last 60 days)")
    ap.add_argument("--out", default="site/data.json")
    args = ap.parse_args()

    today = datetime.date.today().isoformat()
    existing = load_corpus(CORPUS) if CORPUS.exists() else []
    ids = {s.id for s in existing}

    # 1) fetch latest from all automated sources + NCB feeds (RSS/listings), keep
    #    genuinely new records. Dedup by id and by (speaker, date, source_type) so a
    #    speech picked up early from an NCB feed isn't re-added when BIS catches up.
    fresh = [] if args.feeds_only else assemble.load_all(use_cache=False)
    feeds = ncb_feeds.poll_all(since=args.feed_since)
    seen_keys = {(canon(s.speaker), s.date, s.source_type) for s in existing}
    new = []
    for s in fresh + feeds:
        if s.id in ids:
            continue
        k = (canon(s.speaker), s.date, s.source_type)
        if k in seen_keys:
            continue
        ids.add(s.id); seen_keys.add(k)
        new.append(s)
    print(f"sources: {len(fresh)} loaders + {len(feeds)} feed items | {len(new)} new since last run")

    def make_pool(c):
        return [s for s in c if s.is_policy and is_gc(s.speaker) and SINCE <= s.date <= today]

    # 2-4) ingest + score the new speeches. Wrapped so that an API failure (e.g.
    #      OpenRouter out of credits -> 402, or a network blip) does NOT fail the
    #      whole job/deploy: we log it and fall back to redeploying the existing
    #      scored data; the new speeches are retried on the next run.
    corpus, scored_ok = existing, True
    try:
        if new:
            Translator().translate_all(new)        # non-English -> English
            Classifier().classify_all(new)          # is_policy (council types auto-pass)
            corpus = existing + new
            save_corpus(corpus, CORPUS)

        pool = make_pool(corpus)
        anon = Anonymizer(build_roster([s.speaker for s in corpus]))
        for s in pool:
            if not s.text_anon:
                s.text_anon = anon(s.text)
        macro = MacroContext()
        new_ids = {s.id for s in new}
        n_new_pool = sum(1 for s in pool if s.id in new_ids)
        print(f"pool {len(pool)} GC policy records | {n_new_pool} new to score")

        if args.dry_run:
            from run_full import MockDirectScorer, MockJudge
            judge, scorer = MockJudge(), MockDirectScorer()
        else:
            from ecblock.judge.direct import DirectScorer
            from ecblock.judge.openrouter import Judge
            judge, scorer = Judge(), DirectScorer()

        if n_new_pool or any(s.n_comparisons < args.appearances for s in pool):
            run_tournament(pool, judge, appearances_per_speech=args.appearances,
                           macro=macro, resume=True)
        to_direct = [s for s in pool if s.direct_score is None]
        if to_direct:
            scorer.score_all(to_direct, macro, concurrency=6)
        save_corpus(corpus, CORPUS)   # persist classifications, ratings, direct scores
    except Exception as e:  # noqa: BLE001
        scored_ok = False
        print(f"[warn] update/scoring failed: {type(e).__name__}: {e}")
        print("[warn] redeploying existing scored data; new speeches retried next run")
        corpus = load_corpus(CORPUS)

    # 5) always rebuild outputs so the site redeploys (even on a degraded run)
    pool = make_pool(corpus)
    era_adjust(pool)
    meta = write_data_json(pool, args.out)
    print(f"wrote {args.out}: {meta['n_speeches']} speeches (scored_ok={scored_ok})")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_macro.py")], check=False)
    print("daily update complete")


if __name__ == "__main__":
    main()
