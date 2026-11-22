"""Remove speeches held twice (same text from the ECB and a republisher), dropping each
removed copy's tournament comparisons and guidance codes so it stops counting double.

    python scripts/dedupe_text.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecblock.process.textdup import find_duplicates
from ecblock.roster_gc import canon
from ecblock.schema import load_corpus, save_corpus

P = ROOT / "data/processed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    corpus = load_corpus(P / "corpus.jsonl")
    pairs = find_duplicates(corpus, canon)
    for keep, drop, share in pairs:
        print(f"  {share:.0%}  keep {keep.date} {keep.speaker} [{keep.institution}]  "
              f"drop {drop.date} {drop.speaker} [{drop.institution}]  {drop.title[:50]!r}")
    print(f"{len(pairs)} duplicate copies")
    if args.dry_run or not pairs:
        return
    gone = {d.id for _, d, _ in pairs}
    save_corpus([s for s in corpus if s.id not in gone], P / "corpus.jsonl")
    log = P / "tournament_log.jsonl"
    lines = log.read_text(encoding="utf-8").splitlines()
    keep = [ln for ln in lines if json.loads(ln).get("a") not in gone and json.loads(ln).get("b") not in gone]
    log.write_text("\n".join(keep) + "\n", encoding="utf-8")
    g = P / "guidance.jsonl"
    if g.exists():
        gl = g.read_text(encoding="utf-8").splitlines()
        gk = [ln for ln in gl if json.loads(ln).get("id") not in gone]
        g.write_text("\n".join(gk) + ("\n" if gk else ""), encoding="utf-8")
        print(f"dropped {len(gl) - len(gk)} guidance codes")
    print(f"removed {len(gone)} copies; dropped {len(lines) - len(keep)} comparisons involving them")


if __name__ == "__main__":
    main()
