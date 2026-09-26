"""Daily incremental update — ingest new speeches and score only the new ones.

Pulls the latest from every automated source, finds records not already in the
corpus, translates + classifies them, then scores ONLY the new ones:
  - pairwise: resume the TrueSkill tournament; the new high-uncertainty speeches
    draw the comparisons while existing ratings are replayed from the log,
  - direct: score only speeches that don't have a direct score yet.
Finally rebuilds site/data.json and site/macro.json.

Sources (each fails independently; one bad site never stops the run):
  ECB   speeches CSV (full history) + FoeDB /press/key/ (recent, no CSV lag),
        FoeDB interviews, press conferences (statement + Q&A), accounts
  NCBs  BIS live site (RSS + paged listing, back-filled from the last BIS speech
        we hold), plus direct NCB feeds (ncb_feeds)
  Roster  Governing Council membership re-synced from the ECB website

State (data/processed/corpus.jsonl, tournament_log.jsonl, roster_state.json) is
the persistent memory between runs, so CI commits it back after each run.

    python scripts/daily_update.py                  # normal run
    python scripts/daily_update.py --check-sources  # list what's new; no API, no writes
"""
from __future__ import annotations

import argparse
import collections
import datetime
import re
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

from ecblock import roster_gc
from ecblock.config import PROCESSED
from ecblock.corpus import assemble, bis_live, ncb_feeds
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
ECB_INST = "European Central Bank"


def _title_key(s):
    return (canon(s.speaker), re.sub(r"[^a-z0-9]+", " ", s.title.lower()).strip()[:60])


def _bis_since(existing, today: datetime.date) -> str:
    """Walk BIS back to a week before the newest NCB speech we hold (max 6 months)."""
    held = [s.date for s in existing if "bis.org" in s.source_url and s.date <= today.isoformat()]
    start = datetime.date.fromisoformat(max(held)) - datetime.timedelta(days=7) if held else today
    return max(start, today - datetime.timedelta(days=183)).isoformat()


def fetch_new(existing, args, today: datetime.date):
    known_urls = {s.source_url for s in existing if s.source_url}
    fresh = [] if args.feeds_only else assemble.load_all(
        use_cache=False, skip=("bis",), known_urls=known_urls)
    bis = []
    if not args.feeds_only:
        since = _bis_since(existing, today)
        print(f"[bis live] since {since}")
        try:
            bis = bis_live.load(since, known_urls)
        except Exception as e:  # noqa: BLE001
            print(f"    [warn] BIS live failed: {type(e).__name__}: {e}")
    feeds = ncb_feeds.poll_all(since=args.feed_since)

    # de-dup against the corpus and within this batch: by id, by (speaker, date,
    # type), and by (speaker, title) - the same speech often reaches us from two
    # sources (e.g. FoeDB today, the ECB CSV weeks later) with different URLs/dates.
    ids = {s.id for s in existing}
    keys = {(canon(s.speaker), s.date, s.source_type) for s in existing}
    tkeys = {_title_key(s) for s in existing}
    horizon = (today + datetime.timedelta(days=1)).isoformat()
    new, per_source = [], collections.Counter()
    for label, batch in (("ecb", fresh), ("bis", bis), ("feeds", feeds)):
        for s in batch:
            k, tk = (canon(s.speaker), s.date, s.source_type), _title_key(s)
            if s.id in ids or k in keys or tk in tkeys or not s.date or s.date > horizon:
                continue
            ids.add(s.id); keys.add(k); tkeys.add(tk)
            new.append(s)
            per_source[label] += 1
    print(f"sources: {len(fresh)} ECB + {len(bis)} BIS + {len(feeds)} feed items "
          f"| {len(new)} new {dict(per_source)}")
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--appearances", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true", help="use mock scorers (no API spend)")
    ap.add_argument("--check-sources", action="store_true",
                    help="fetch + de-dup only, print what's new, write nothing")
    ap.add_argument("--feeds-only", action="store_true",
                    help="skip the ECB/BIS loaders; only poll the NCB feeds")
    ap.add_argument("--feed-since", default=None,
                    help="ISO date lower bound for feed items (default: last 60 days)")
    ap.add_argument("--out", default="site/data.json")
    args = ap.parse_args()

    today = datetime.date.today()
    try:
        for change in roster_gc.sync_from_ecb():
            print(f"[roster] {change}")
    except Exception as e:  # noqa: BLE001 - fall back to the stored/static roster
        print(f"[roster] sync failed, using stored roster: {type(e).__name__}: {e}")

    existing = load_corpus(CORPUS) if CORPUS.exists() else []
    new = fetch_new(existing, args, today)

    if args.check_sources:
        for s in sorted(new, key=lambda s: s.date):
            tag = "GC " if is_gc(s.speaker) else "   "
            print(f"  {tag}{s.date}  {s.source_type:12} {canon(s.speaker)[:26]:26} {s.title[:60]}")
        return

    def make_pool(c):
        return [s for s in c if s.is_policy and is_gc(s.speaker) and SINCE <= s.date <= today.isoformat()]

    # ingest + score. Wrapped so an API failure (e.g. OpenRouter out of credits ->
    # 402, or a network blip) does NOT fail the job/deploy: we log it and redeploy
    # the existing scored data; new speeches are retried on the next run.
    corpus, scored_ok = existing, True
    try:
        if new:
            Translator().translate_all(new)
        corpus = existing + new
        # classify new records, plus any older GC record never classified (e.g. a
        # governor who only just joined the roster, or a newly matched spelling)
        todo = [s for s in corpus if s.is_policy is None and is_gc(s.speaker)
                and SINCE <= s.date <= today.isoformat()]
        todo_ids = {s.id for s in todo}
        todo += [s for s in new if s.id not in todo_ids]
        if todo:
            Classifier().classify_all(todo)
        save_corpus(corpus, CORPUS)

        pool = make_pool(corpus)
        new_ids = {s.id for s in new}
        n_new_pool = sum(1 for s in pool if s.id in new_ids or s.mu is None)
        print(f"pool {len(pool)} GC policy records | {n_new_pool} new to score")

        if args.dry_run:
            from run_full import MockDirectScorer, MockJudge
            judge, scorer = MockJudge(), MockDirectScorer()
        else:
            from ecblock.judge.direct import DirectScorer
            from ecblock.judge.openrouter import Judge
            judge, scorer = Judge(), DirectScorer()

        # one anonymiser for both judges; texts are anonymised lazily, only for the
        # speeches actually sent to a model
        anon = Anonymizer(build_roster([s.speaker for s in corpus]))
        macro = MacroContext()
        if n_new_pool:
            run_tournament(pool, judge, appearances_per_speech=args.appearances,
                           macro=macro, resume=True, anonymizer=anon)
        to_direct = [s for s in pool if s.direct_score is None]
        if to_direct:
            scorer.score_all(to_direct, macro, concurrency=6, anonymizer=anon)
        save_corpus(corpus, CORPUS)   # persist classifications, ratings, direct scores
    except Exception as e:  # noqa: BLE001
        scored_ok = False
        print(f"[warn] update/scoring failed: {type(e).__name__}: {e}")
        print("[warn] redeploying existing scored data; new speeches retried next run")
        corpus = load_corpus(CORPUS)

    # always rebuild outputs so the site redeploys (even on a degraded run)
    pool = make_pool(corpus)
    era_adjust(pool)
    meta = write_data_json(pool, args.out)
    print(f"wrote {args.out}: {meta['n_speeches']} speeches (scored_ok={scored_ok})")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_macro.py")], check=False)
    print("daily update complete")


if __name__ == "__main__":
    main()
