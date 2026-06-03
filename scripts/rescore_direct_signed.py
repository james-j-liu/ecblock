"""Re-score the direct method on the signed -100..+100 scale (finer resolution),
mapped back to 0-100 internal units (= pairwise units) inside DirectScorer.score.

Fully resumable: every result is appended to a side file as it completes, so a
killed run loses nothing and re-running continues. When all pool speeches are
scored, the scores are written into corpus.jsonl (the canonical source) and
finalize_increment can rebuild data.json.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from ecblock.config import PROCESSED
from ecblock.judge.direct import DirectScorer
from ecblock.macro.euro_macro import MacroContext
from ecblock.process.anonymize import Anonymizer
from ecblock.process.roster import build_roster
from ecblock.roster_gc import is_gc
from ecblock.schema import load_corpus, save_corpus

CORPUS = PROCESSED / "corpus.jsonl"
SIDE = PROCESSED / "direct_signed.jsonl"
SINCE, UNTIL = "2010-05-28", "2026-05-28"


def main():
    corpus = load_corpus(CORPUS)
    pool = [s for s in corpus if s.is_policy and is_gc(s.speaker) and SINCE <= s.date <= UNTIL]
    anon = Anonymizer(build_roster([s.speaker for s in corpus]))
    macro = MacroContext()

    done = {}
    if SIDE.exists():
        for line in SIDE.open(encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                done[r["id"]] = r["ds"]
    todo = [s for s in pool if s.id not in done]
    print(f"Pool {len(pool)} | already scored {len(done)} | to score {len(todo)}")

    scorer = DirectScorer()

    def work(s):
        return s.id, scorer.score(anon(s.text), macro.string(s.date))

    if todo:
        with SIDE.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=8) as ex:
            for j, (sid, val) in enumerate(ex.map(work, todo), 1):
                if val is not None:
                    done[sid] = val
                    f.write(json.dumps({"id": sid, "ds": val}) + "\n")
                    f.flush()
                if j % 100 == 0:
                    print(f"  scored {j}/{len(todo)}")

    n = 0
    for s in pool:
        if s.id in done:
            s.direct_score = done[s.id]
            n += 1
    save_corpus(corpus, CORPUS)
    print(f"Wrote {n}/{len(pool)} signed direct scores into corpus.jsonl")


if __name__ == "__main__":
    main()
