"""Incremental scoring: add newly-collected speeches to the existing tournament.

Rebuilds the existing TrueSkill ratings by replaying the tournament log (free),
then runs ~30 fresh pairwise comparisons per NEW speech against the existing
pool (so each new speech is properly calibrated), updates ratings, re-runs the
era adjustment, direct-scores, and rewrites data.json. Existing speeches keep
their ratings - only the new speeches cost new comparisons.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from tqdm import tqdm

from ecblock.config import PROCESSED, cfg
from ecblock.macro.euro_macro import MacroContext
from ecblock.output.build_data import era_adjust, write_data_json
from ecblock.process.anonymize import Anonymizer
from ecblock.process.roster import build_roster
from ecblock.roster_gc import canon, is_gc
from ecblock.schema import load_corpus
from ecblock.tournament.engine import Tournament

CORPUS = PROCESSED / "corpus.jsonl"
LOG = PROCESSED / "tournament_log.jsonl"
SINCE, UNTIL = "2010-05-28", "2026-05-28"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--appearances", type=int, default=30)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-direct", action="store_true")
    ap.add_argument("--out", default="site/data.json")
    args = ap.parse_args()

    corpus = load_corpus(CORPUS)
    pool = [s for s in corpus if s.is_policy and is_gc(s.speaker)
            and SINCE <= s.date <= UNTIL]
    by_id = {s.id: s for s in pool}
    print(f"Pool: {len(pool)} GC policy records")

    # anonymise any speeches lacking it (the new ones)
    roster = build_roster([s.speaker for s in corpus])
    anon = Anonymizer(roster)
    for s in pool:
        if not s.text_anon:
            s.text_anon = anon(s.text)

    macro = MacroContext()
    macro_str = {sid: macro.string(by_id[sid].date) for sid in by_id}
    tcfg = cfg()["tournament"]
    tour = Tournament(list(by_id), initial_mu=tcfg["initial_mu"],
                      initial_sigma=tcfg["initial_sigma"], seed=args.seed)

    # 1) replay existing log to reconstruct ratings (free)
    replayed = 0
    if LOG.exists():
        with LOG.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    a, b, w = r["a"], r["b"], r.get("winner")
                except (json.JSONDecodeError, KeyError):
                    continue
                if a in by_id and b in by_id:
                    if w == "A":
                        tour.record(a, b); replayed += 1
                    elif w == "B":
                        tour.record(b, a); replayed += 1
    print(f"Replayed {replayed} existing comparisons")

    new_ids = [sid for sid in by_id if tour.n_comp[sid] == 0]
    existing_ids = [sid for sid in by_id if tour.n_comp[sid] > 0]
    print(f"New speeches to score: {len(new_ids)} (vs {len(existing_ids)} existing)")
    if not new_ids:
        sys.exit("No new speeches to score.")

    # 2) judge
    if args.dry_run:
        from run_poc import MockJudge
        judge = MockJudge()
        print("DRY RUN: MockJudge")
    else:
        from ecblock.judge.openrouter import Judge
        judge = Judge(model=args.model)
        print(f"Judge: {judge.model}")

    rng = random.Random(args.seed)
    # build the comparison schedule: each new speech vs N random existing partners
    jobs = []
    for sid in new_ids:
        partners = (rng.sample(existing_ids, args.appearances)
                    if len(existing_ids) >= args.appearances
                    else [rng.choice(existing_ids) for _ in range(args.appearances)])
        for pid in partners:
            jobs.append((sid, pid))
    print(f"Scheduling {len(jobs)} new comparisons "
          f"(~${len(jobs) * 0.00096:.2f} pairwise)")

    def judge_pair(job):
        i, j = job
        a, b = (i, j) if rng.random() < 0.5 else (j, i)
        res = judge.compare(by_id[a].text_anon or by_id[a].text, macro_str[a],
                            by_id[b].text_anon or by_id[b].text, macro_str[b])
        return a, b, res

    # never write mock outcomes to the real log during a dry run
    logf = LOG.open("a", encoding="utf-8") if not args.dry_run else None
    done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(judge_pair, jb) for jb in jobs]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="increment"):
            try:
                a, b, res = fut.result()
            except Exception:
                continue
            w = res.get("winner")
            if w == "A":
                tour.record(a, b)
            elif w == "B":
                tour.record(b, a)
            else:
                continue
            if logf:
                logf.write(json.dumps({"a": a, "b": b, **res}) + "\n")
                logf.flush()
            done += 1
    if logf:
        logf.close()
    print(f"Recorded {done} new comparisons")

    # 3) write ratings back onto the pool
    for sid, s in by_id.items():
        r = tour.rating(sid)
        s.mu, s.sigma, s.n_comparisons = round(r.mu, 3), round(r.sigma, 3), tour.n_comp[sid]

    # 4) direct scoring (full pool keeps both methods consistent)
    if not args.no_direct:
        if args.dry_run:
            from run_full import MockDirectScorer
            scorer = MockDirectScorer()
        else:
            from ecblock.judge.direct import DirectScorer
            scorer = DirectScorer(model=args.model)
            print(f"Direct scorer: {scorer.model}")
        scorer.score_all(pool, macro, concurrency=8)

    era_adjust(pool)
    meta = write_data_json(pool, args.out)
    print(f"Wrote {args.out}: {meta['n_speeches']} speeches, {meta['n_speakers']} speakers")

    # show where the new speakers land
    newset = set(new_ids)
    print("\nNew speeches (era-adjusted pairwise):")
    for s in sorted([by_id[i] for i in new_ids], key=lambda s: -(s.mu_adj or 0)):
        print(f"  {s.mu_adj:5.1f}  {s.date}  {s.speaker[:22]:22}  {s.title[:42]}")


if __name__ == "__main__":
    main()
