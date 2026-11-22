"""Explicit forward guidance from core Governing Council members, read by Jev.

"Core" = the Executive Board plus the governors of the five largest NCBs
(Bundesbank, Banque de France, Banca d'Italia, Banco de España, De Nederlandsche
Bank). For each of their speeches/interviews, Jev answers two typed questions:
  next_decision : larger_hike | hike | hold | cut | larger_cut | no_signal
  pace          : accelerate | maintain | slow_or_pause | end_or_reverse | no_signal
and returns a probability for each option, so a vague remark contributes a soft
vote rather than a forced label.

The text is anonymised and carries no date, which limits (but cannot remove) the
chance that the model answers from memory of what the ECB later did. Results are
cached by document id in data/processed/guidance.jsonl, so each document is read
once; the daily job only classifies new ones.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config import PROCESSED
from ..roster_gc import canon, is_gc
from ..schema import Speech

CACHE = PROCESSED / "guidance.jsonl"
BIG5 = {"Deutsche Bundesbank", "Banque de France", "Banca d'Italia", "Banco de España",
        "De Nederlandsche Bank"}
MAX_CHARS = 60000

INSTR_NEXT = (
    "This is an anonymised speech or interview by a member of the ECB Governing Council. "
    "What does the speaker signal about the ECB's NEXT interest-rate decision (the next "
    "monetary-policy meeting, or the immediate rate path)? Only count explicit or clearly "
    "implied guidance about policy rates. A description of inflation or the economy with no "
    "stated policy implication, or a generic 'data-dependent, meeting-by-meeting' line with "
    "no lean, is no_signal."
)
CRIT_NEXT = {
    "larger_hike": "Signals a hike of 50 basis points or more, or a faster pace of tightening.",
    "hike": "Signals a rate increase at the next meeting or imminently.",
    "hold": "Signals keeping rates unchanged next: a pause, patience, no need to move yet.",
    "cut": "Signals a rate cut at the next meeting or imminently.",
    "larger_cut": "Signals a cut of 50 basis points or more, or a faster pace of easing.",
    "no_signal": "No specific guidance about the next rate decision.",
}
INSTR_PACE = (
    "Relative to the ECB's current direction and pace of rate changes (whether it is "
    "currently raising, cutting or holding rates), what does the speaker signal?"
)
CRIT_PACE = {
    "accelerate": "Move faster than the current pace (larger or more frequent steps).",
    "maintain": "Continue at the current pace / keep doing what we are doing.",
    "slow_or_pause": "Slow down, skip a meeting or pause, without ending the cycle.",
    "end_or_reverse": "The cycle is over, or the next move could be in the other direction.",
    "no_signal": "No view on the pace of the cycle.",
}


def is_core(s: Speech) -> bool:
    return (is_gc(s.speaker) and s.source_type in ("speech", "interview")
            and (s.institution == "European Central Bank" or s.institution in BIG5))


def _role(s: Speech) -> str:
    return ("a member of the ECB Executive Board" if s.institution == "European Central Bank"
            else f"the governor of {s.institution}")


def load_cache() -> dict[str, dict]:
    if not CACHE.exists():
        return {}
    out = {}
    for line in CACHE.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r
    return out


def classify(docs: list[Speech], anonymizer, concurrency: int = 12, limit: int | None = None) -> int:
    """Classify core docs not yet cached; append results. Returns # newly classified."""
    from ..judge.jev import JevJudge
    jev = JevJudge()
    done = load_cache()
    todo = [s for s in docs if s.id not in done][:limit]
    if not todo:
        return 0

    def one(s: Speech) -> dict:
        text = anonymizer.text_of(s)
        if len(text) > MAX_CHARS:
            h = MAX_CHARS * 2 // 3
            text = text[:h] + "\n[...]\n" + text[-(MAX_CHARS - h):]
        res = jev._post({
            "model": jev.model,
            "state": {"speaker_role": _role(s), "document_type": s.source_type, "text": text},
            "questions": {
                "next_decision": {"type": "choice", "instructions": INSTR_NEXT, "criteria": CRIT_NEXT},
                "pace": {"type": "choice", "instructions": INSTR_PACE, "criteria": CRIT_PACE},
            }})
        jev.cost += float(res.get("usage", {}).get("cost") or 0)
        a = res["answers"]
        return {"id": s.id, "date": s.date, "speaker": canon(s.speaker), "inst": s.institution,
                "type": s.source_type,
                "next": {k: round(float(v), 4) for k, v in a["next_decision"]["probabilities"].items()},
                "pace": {k: round(float(v), 4) for k, v in a["pace"]["probabilities"].items()}}

    n = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex, CACHE.open("a", encoding="utf-8") as f:
        futs = [ex.submit(one, s) for s in todo]
        for fut in as_completed(futs):
            try:
                f.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
                f.flush()
                n += 1
            except Exception as e:  # noqa: BLE001 - skip, retried next run
                print(f"  [guidance skip] {type(e).__name__}: {str(e)[:120]}")
    print(f"[guidance] classified {n} documents (${jev.cost:.3f})")
    return n
