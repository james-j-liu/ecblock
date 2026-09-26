"""Bake-off: does Jev judge hawkishness like the current tournament?

Samples logged comparisons, re-judges them with several judges, and reports
agreement against (a) the original logged verdict and (b) the consensus TrueSkill
ranking (does the chosen winner have the higher existing rating?). A re-run of the
ORIGINAL judge gives the self-consistency ceiling any new judge should be held to.

Results append to data/processed/bakeoff_jev.jsonl, so a killed run resumes.
    python scripts/bakeoff_jev.py --arm jev56k --n 500
    python scripts/bakeoff_jev.py --report
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from ecblock.config import PROCESSED
from ecblock.macro.euro_macro import MacroContext
from ecblock.process.anonymize import Anonymizer
from ecblock.process.roster import build_roster
from ecblock.roster_gc import is_gc
from ecblock.schema import load_corpus

OUT = PROCESSED / "bakeoff_jev.jsonl"
ARMS = {  # name -> (judge factory, excerpt cap)
    "jev9k":      (lambda: _jev(9000), 9000),
    "jev56k":     (lambda: _jev(56000), 56000),
    "orig9k":     (lambda: _chat("google/gemini-3.1-flash-lite"), 9000),
    "current9k":  (lambda: _chat("google/gemini-2.5-flash-lite"), 9000),
    "haiku9k":    (lambda: _chat("anthropic/claude-haiku-4.5"), 9000),
    "router9k":   (lambda: _Router(), 9000),
    "router56k":  (lambda: _Router(), 56000),
}


class _Router:
    """Chat judge on typesafe/jev-router that records which model served each call."""
    def __init__(self):
        import collections
        import threading
        from ecblock.config import openrouter_key
        self._key, self.cost = openrouter_key(), 0.0
        self.models, self._lock = collections.Counter(), threading.Lock()

    def compare(self, a_text, a_macro, b_text, b_macro):
        import requests
        from ecblock.judge import prompts
        from ecblock.judge.openrouter import Judge
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", timeout=180,
                          headers={"Authorization": f"Bearer {self._key}"},
                          json={"model": "typesafe/jev-router", "temperature": 0,
                                "messages": [{"role": "system", "content": prompts.SYSTEM},
                                             {"role": "user", "content": prompts.build_user_prompt(
                                                 a_text, a_macro, b_text, b_macro)}]})
        r.raise_for_status()
        j = r.json()
        with self._lock:
            self.cost += float(j.get("usage", {}).get("cost") or 0)
            self.models[f'{j.get("model")} @ {j.get("provider")}'] += 1
        return Judge._parse(j["choices"][0]["message"]["content"])


def _jev(cap):
    from ecblock.judge.jev import JevJudge
    return JevJudge(max_excerpt_chars=cap)


def _chat(model):
    from ecblock.judge.openrouter import Judge
    return Judge(model=model)


def sample_pairs(pool_ids, n, seed=7):
    rows = []
    with (PROCESSED / "tournament_log.jsonl").open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("winner") in ("A", "B") and r["a"] in pool_ids and r["b"] in pool_ids:
                rows.append(r)
    random.Random(seed).shuffle(rows)
    seen, out = set(), []
    for r in rows:
        k = tuple(sorted((r["a"], r["b"])))
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out[:n]


def run_arm(arm, n, workers):
    corpus = load_corpus(PROCESSED / "corpus.jsonl")
    pool = {s.id: s for s in corpus if s.is_policy and s.text and is_gc(s.speaker)}
    pairs = sample_pairs(set(pool), n)
    done = set()
    if OUT.exists():
        for line in OUT.open(encoding="utf-8"):
            r = json.loads(line)
            if r["arm"] == arm:
                done.add((r["a"], r["b"]))
    todo = [p for p in pairs if (p["a"], p["b"]) not in done]
    print(f"[{arm}] {len(pairs)} sampled, {len(done)} done, {len(todo)} to judge", flush=True)
    if not todo:
        return

    factory, cap = ARMS[arm]
    judge = factory()
    anon = Anonymizer(build_roster([s.speaker for s in corpus]))
    macro = MacroContext()
    cache = {}

    def excerpt_anon(sid):
        # excerpt first (head 2/3 + tail 1/3, as the judges do), then anonymise
        if sid not in cache:
            t = pool[sid].text
            if len(t) > cap:
                h = cap * 2 // 3
                t = t[:h] + "\n[...]\n" + t[-(cap - h):]
            cache[sid] = anon(t)
        return cache[sid]

    rng = random.Random(hash(arm) & 0xFFFF)

    def job(p):
        a, b = (p["a"], p["b"]) if rng.random() < 0.5 else (p["b"], p["a"])  # fresh slot draw
        res = judge.compare(excerpt_anon(a), macro.string(pool[a].date),
                            excerpt_anon(b), macro.string(pool[b].date))
        w = res.get("winner")
        win_id = a if w == "A" else b if w == "B" else None
        orig_win = p["a"] if p["winner"] == "A" else p["b"]
        return {"arm": arm, "a": p["a"], "b": p["b"], "slot_a": a, "said": w,
                "winner_id": win_id, "orig_winner_id": orig_win,
                "conf": res.get("confidence"), "p_a": res.get("p_a")}

    with ThreadPoolExecutor(max_workers=workers) as ex, OUT.open("a", encoding="utf-8") as f:
        futs = [ex.submit(job, p) for p in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                f.write(json.dumps(fut.result()) + "\n")
                f.flush()
            except Exception as e:  # noqa: BLE001
                print(f"  [skip] {type(e).__name__}: {str(e)[:120]}", flush=True)
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    if hasattr(judge, "cost"):
        print(f"[{arm}] API-reported cost this run: ${judge.cost:.4f} "
              f"(${judge.cost / len(todo) * 1000:.3f} per 1k comparisons)", flush=True)
    if hasattr(judge, "models"):
        print(f"[{arm}] served by: {dict(judge.models)}", flush=True)


def report():
    corpus = load_corpus(PROCESSED / "corpus.jsonl")
    mu = {s.id: s.mu for s in corpus if s.mu is not None}
    by = {}
    for line in OUT.open(encoding="utf-8"):
        r = json.loads(line)
        by.setdefault(r["arm"], {})[tuple(sorted((r["a"], r["b"])))] = r

    print(f"{'arm':<11}{'n':>5}{'vs log':>9}{'vs rating':>11}{'said A':>8}   (rating = consensus TrueSkill order)")
    for arm, rs in by.items():
        v = [r for r in rs.values() if r["winner_id"]]
        log = sum(r["winner_id"] == r["orig_winner_id"] for r in v) / len(v)
        rated = [r for r in v if r["a"] in mu and r["b"] in mu and mu[r["a"]] != mu[r["b"]]]
        cons = sum(r["winner_id"] == max((r["a"], r["b"]), key=mu.get) for r in rated) / len(rated)
        sa = sum(r["said"] == "A" for r in v) / len(v)
        print(f"{arm:<11}{len(v):>5}{log:>9.1%}{cons:>11.1%}{sa:>8.1%}")

    arms = list(by)
    print("\npairwise agreement between judges (same pairs):")
    for i, x in enumerate(arms):
        for y in arms[i + 1:]:
            common = [k for k in by[x] if k in by[y] and by[x][k]["winner_id"] and by[y][k]["winner_id"]]
            if len(common) >= 30:
                ag = sum(by[x][k]["winner_id"] == by[y][k]["winner_id"] for k in common) / len(common)
                print(f"  {x:<10} vs {y:<10} n={len(common):<4} {ag:.1%}")

    # does confidence mean anything? agreement with rating by confidence band
    for arm in [a for a in arms if a.startswith("jev")]:
        v = [r for r in by[arm].values() if r["winner_id"] and r["a"] in mu and r["b"] in mu]
        for lo, hi in [(0, .7), (.7, .9), (.9, 1.01)]:
            band = [r for r in v if lo <= (r["conf"] or 0) < hi]
            if band:
                ok = sum(r["winner_id"] == max((r["a"], r["b"]), key=mu.get) for r in band) / len(band)
                print(f"  {arm} conf [{lo:.1f},{min(hi,1):.1f}): n={len(band):<4} vs rating {ok:.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.arm:
        run_arm(args.arm, args.n, args.workers)
    if args.report:
        report()
