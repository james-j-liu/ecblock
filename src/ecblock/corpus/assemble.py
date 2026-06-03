"""Assemble the full ECBLock corpus from all four sources.

Sources:
  ecb_speeches  -> ECB Executive Board speeches (official CSV)
  ecb_interviews-> media interviews republished by the ECB (FoeDB /press/inter/)
  ecb_council   -> ECB council composite: statements, Q&A, accounts (FoeDB)
  ecb_ncb       -> euro-area NCB speeches (BIS bulk extract)

Records are de-duplicated by Speech.id (date|speaker|title|url hash). The combined
corpus is written as JSONL to data/processed/corpus.jsonl. Translation to English
is applied here (most sources are already English; see process.translate).
"""
from __future__ import annotations

import collections

from ..config import PROCESSED
from ..schema import Speech, save_corpus
from . import ecb_council, ecb_interviews, ecb_ncb, ecb_speeches

CORPUS_PATH = PROCESSED / "corpus.jsonl"


def load_all(use_cache: bool = True) -> list[Speech]:
    speeches: list[Speech] = []
    print("[1/4] ECB Executive Board speeches...")
    speeches += ecb_speeches.load()
    print("[2/4] ECB media interviews...")
    speeches += ecb_interviews.load(use_cache=use_cache)
    print("[3/4] ECB council (statements, Q&A, accounts)...")
    speeches += ecb_council.load(use_cache=use_cache)
    print("[4/4] Euro-area NCB speeches (BIS)...")
    speeches += ecb_ncb.load(use_cache=use_cache)

    seen: dict[str, Speech] = {}
    for s in speeches:
        seen.setdefault(s.id, s)
    deduped = list(seen.values())
    print(f"Combined {len(speeches)} -> {len(deduped)} after de-dup")
    return deduped


def build(use_cache: bool = True, translate: bool = True,
          out_path=CORPUS_PATH) -> list[Speech]:
    corpus = load_all(use_cache=use_cache)
    if translate:
        from ..process.translate import Translator
        Translator().translate_all(corpus)
    save_corpus(corpus, out_path)
    print(f"Saved {len(corpus)} records to {out_path}")
    return corpus


def _summary(corpus: list[Speech]) -> None:
    by_type = collections.Counter(s.source_type for s in corpus)
    by_lang = collections.Counter(s.orig_language for s in corpus)
    print("\nBy source type:", dict(by_type))
    print("By orig language:", dict(by_lang.most_common(10)))
    print("Date range:", min(s.date for s in corpus), "..", max(s.date for s in corpus))
    print("Distinct speakers:", len({s.speaker for s in corpus}))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-translate", action="store_true",
                    help="skip the LLM translation pass (detection only)")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    corpus = build(use_cache=not args.no_cache, translate=not args.no_translate)
    _summary(corpus)
