"""One-time: recover the natural (un-centered, un-jittered) direct scores from the
current data.json and persist them into corpus.jsonl, so direct scores have a
clean canonical source (no centering, no jitter compounding).

data.json ds = natural_discrete + centering_shift + jitter(id).
We de-jitter exactly (deterministic per id), then remove the global centering
shift S. Because the LLM's raw scores are multiples of 5, the de-jittered values
are all congruent to S (mod 5), which pins S down exactly.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from ecblock.config import PROCESSED
from ecblock.output.build_data import _jitter
from ecblock.roster_gc import is_gc
from ecblock.schema import load_corpus, save_corpus

CORPUS = PROCESSED / "corpus.jsonl"
DATA = Path("site/data.json")
SINCE, UNTIL = "2010-05-28", "2026-05-28"


def main():
    corpus = load_corpus(CORPUS)
    pool = [s for s in corpus if s.is_policy and is_gc(s.speaker) and SINCE <= s.date <= UNTIL]
    key2id = {(s.date, s.speaker, s.title[:40]): s.id for s in pool}
    byid = {s.id: s for s in pool}

    data = json.load(DATA.open(encoding="utf-8"))["speeches"]
    dej = {}            # id -> de-jittered (centered-discrete) score
    for r in data:
        if r.get("ds") is None:
            continue
        sid = key2id.get((r["d"], r["a"], (r.get("tt") or "")[:40]))
        if sid:
            dej[sid] = r["ds"] - _jitter(sid)

    # the de-jittered values are natural_discrete + S, with natural_discrete a
    # multiple of 5, so (value mod 5) is constant == S mod 5. Recover S in [6,13].
    frac = statistics.median(round(v % 5, 2) for v in dej.values())
    S = next((c for c in (frac, frac + 5, frac + 10) if 6 <= c <= 13), frac + 5)
    natural_mean = statistics.mean(dej.values()) - S
    print(f"matched={len(dej)}  frac(mod5)={frac}  centering_shift S={S}")
    print(f"natural direct mean = {natural_mean:.2f} (norm {(natural_mean-50)/5:+.2f})")
    if not (33 <= natural_mean <= 47):
        sys.exit(f"ABORT: recovered natural mean {natural_mean:.1f} looks wrong")

    for sid, cd in dej.items():
        byid[sid].direct_score = round(cd - S, 3)
    have = sum(1 for s in pool if s.direct_score is not None)
    print(f"persisted natural direct_score for {have}/{len(pool)} pool records")
    save_corpus(corpus, CORPUS)
    print("saved corpus.jsonl")


if __name__ == "__main__":
    main()
