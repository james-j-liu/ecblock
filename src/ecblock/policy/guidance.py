"""Explicit forward guidance from core Governing Council members, read by a decision
model (Drex Max Finance; Jev 1.13 before November 2026).

"Core" = the Executive Board plus the governors of the five largest NCBs
(Bundesbank, Banque de France, Banca d'Italia, Banco de España, De Nederlandsche
Bank). For each of their speeches/interviews, Jev answers two typed questions:
  next_decision : larger_hike | hike | hold | cut | larger_cut | no_signal
  pace          : accelerate | maintain | slow_or_pause | end_or_reverse | no_signal
and returns a probability for each option, so a vague remark contributes a soft
vote rather than a forced label.

The text is anonymised, with years masked, and carries no date, which limits (but cannot remove) the
chance that the model answers from memory of what the ECB later did. Results are
cached by document id in data/processed/guidance.jsonl, so each document is read
once; the daily job only classifies new ones.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config import PROCESSED
from ..roster_gc import canon, is_gc
from ..schema import Speech

CACHE = PROCESSED / "guidance.jsonl"
COUNCIL_CACHE = PROCESSED / "guidance_council.jsonl"
BIG5 = {"Deutsche Bundesbank", "Banque de France", "Banca d'Italia", "Banco de España",
        "De Nederlandsche Bank"}
MAX_CHARS = 60000
# Live reader: Drex Max Finance. On the MPC project it was the only decision model
# significantly better than Jev 1.13 through the forecast model (read blind). Texts here
# were already anonymised and undated; years are now masked too.
LIVE_MODEL = "nace-ai/drex-max-finance"
_YEAR = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")

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
# The Council's own documents are released AT a meeting and spend most of their
# length explaining the decision just taken, so the question must separate that
# decision (given to the model as `decision_taken`) from what comes next.
INSTR_NEXT_COUNCIL = (
    "This anonymised document comes from the ECB Governing Council itself: the monetary "
    "policy statement or press-conference Q&A released at a meeting, or the account of a "
    "meeting. The decision taken at that meeting is given in decision_taken and is ALREADY "
    "MADE - do not treat it as the answer. What does the document signal about the "
    "FOLLOWING meeting's interest-rate decision? Language that rates are now at a sufficient "
    "or appropriate level, or should be kept there for some time, signals hold; language that "
    "further moves in the same direction are likely or needed signals another move; a purely "
    "data-dependent, meeting-by-meeting line with no lean is no_signal."
)
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


def is_council(s: Speech) -> bool:
    """The Governing Council's own words: the monetary policy statement, the press
    conference Q&A, and the account of the meeting (published ~4 weeks later)."""
    return s.source_type in ("mp_statement", "mp_qa", "mp_account")


def _role(s: Speech) -> str:
    if is_council(s):
        return {"mp_statement": "the ECB Governing Council (official monetary policy statement)",
                "mp_qa": "the ECB President answering questions at the press conference",
                "mp_account": "the ECB Governing Council (account of its last meeting)"}[s.source_type]
    return ("a member of the ECB Executive Board" if s.institution == "European Central Bank"
            else f"the governor of {s.institution}")


def _decision_text(s: Speech, decisions: dict[str, int]) -> str:
    """The decision at the meeting this document belongs to (an account belongs to
    the latest meeting on or before its title date / publication)."""
    dates = sorted(d for d in decisions if d <= s.date)
    if not dates:
        return "unknown"
    bp = decisions[dates[-1]]
    what = f"raised the key interest rates by {bp}bp" if bp > 0 else (
        f"lowered the key interest rates by {-bp}bp" if bp < 0 else "kept the key interest rates unchanged")
    return f"At the meeting this document refers to, the Governing Council {what}."


def load_cache(path=CACHE) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r
    return out


def classify(docs: list[Speech], anonymizer, concurrency: int = 12, limit: int | None = None,
             decisions: dict[str, int] | None = None, model: str | None = None) -> int:
    """Classify docs not yet cached; append results. Returns # newly classified.
    Council documents (is_council) go to their own cache and prompt; `decisions`
    maps meeting date -> bp change so the model is told the decision just taken."""
    from ..judge.jev import JevJudge
    jev = JevJudge(model=model or LIVE_MODEL)
    council = bool(docs) and is_council(docs[0])
    path = COUNCIL_CACHE if council else CACHE
    done = load_cache(path)
    # re-read anything read by another model (older records carry none: Jev 1.13)
    todo = [s for s in docs if s.id not in done or done[s.id].get("model", "typesafe/jev-1.13") != jev.model][:limit]
    if not todo:
        return 0

    def one(s: Speech) -> dict:
        text = _YEAR.sub("[YEAR]", anonymizer.text_of(s))
        if len(text) > MAX_CHARS:
            h = MAX_CHARS * 2 // 3
            text = text[:h] + "\n[...]\n" + text[-(MAX_CHARS - h):]
        state = {"speaker_role": _role(s), "document_type": s.source_type, "text": text}
        if council:
            state["decision_taken"] = _decision_text(s, decisions or {})
        res = jev._post({
            "model": jev.model,
            "state": state,
            "questions": {
                "next_decision": {"type": "choice", "criteria": CRIT_NEXT,
                                  "instructions": INSTR_NEXT_COUNCIL if council else INSTR_NEXT},
                "pace": {"type": "choice", "instructions": INSTR_PACE, "criteria": CRIT_PACE},
            }})
        jev.cost += float(res.get("usage", {}).get("cost") or 0)
        a = res["answers"]
        return {"id": s.id, "date": s.date, "speaker": canon(s.speaker), "inst": s.institution,
                "type": s.source_type, "model": jev.model, "served_model": res.get("model"),
                "next": {k: round(float(v), 4) for k, v in a["next_decision"]["probabilities"].items()},
                "pace": {k: round(float(v), 4) for k, v in a["pace"]["probabilities"].items()}}

    n = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex, path.open("a", encoding="utf-8") as f:
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
