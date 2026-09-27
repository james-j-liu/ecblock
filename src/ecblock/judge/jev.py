"""Pairwise judge backed by TypeSafe's Jev decision model (via OpenRouter).

Jev is a structured-decision model: instead of generating text it answers a typed
"choice" question and returns a probability per option. It is a drop-in for
openrouter.Judge - same compare() signature and return shape - so the tournament
runner needs no changes. Output tokens are free; only input tokens are billed.

Jev's context window is 32k tokens, so each excerpt is capped at ~56k chars
(two excerpts + instructions stay under the limit). That still reads ~97% of the
corpus in full, versus ~16% at the chat judge's 9k-char cap.
"""
from __future__ import annotations

import random
import time

import requests

from ..config import openrouter_key

API_URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"

INSTRUCTIONS = (
    "Two anonymized excerpts from euro-area monetary policy communication (ECB Governing "
    "Council members or the Council itself) are given, each with the macroeconomic context "
    "at the time it was delivered. Decide which excerpt takes the MORE HAWKISH stance "
    "RELATIVE TO its own macro context. Hawkish = leaning toward tighter policy: concern "
    "about inflation or overheating, higher rates, faster or longer tightening, balance-"
    "sheet reduction, scepticism about accommodation. Dovish = concern about growth, "
    "employment or disinflation, cuts, prolonged accommodation, asset purchases. Judge "
    "relative to conditions: urging inflation vigilance at 2% HICP is meaningfully "
    "hawkish; the same words at 8.5% HICP merely state the obvious. Names are replaced "
    "by tokens like [OFFICIAL]; do not guess identities."
)
CRITERIA = {"A": "Excerpt A is more hawkish relative to its macro context.",
            "B": "Excerpt B is more hawkish relative to its macro context."}


class JevJudge:
    def __init__(self, model: str = MODEL, max_excerpt_chars: int = 56000,
                 timeout: int = 90, max_attempts: int = 8):
        self.model = model
        self.max_excerpt_chars = max_excerpt_chars
        self.timeout = timeout
        self.max_attempts = max_attempts
        self._key = openrouter_key()
        self.cost = 0.0   # running $ total, from the API's own usage.cost

    def _excerpt(self, text: str, cap: int) -> str:
        if len(text) <= cap:
            return text
        head = cap * 2 // 3
        return text[:head] + "\n[...]\n" + text[-(cap - head):]

    def _post(self, payload: dict) -> dict:
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        last_err: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                r = requests.post(API_URL, headers=headers, json=payload, timeout=self.timeout)
            except requests.RequestException as e:
                last_err = e
                time.sleep(min(2 ** attempt + random.random(), 60))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                last_err = requests.HTTPError(f"{r.status_code} {r.reason}")
                time.sleep(min(2 ** attempt + random.random(), 60))
                continue
            if r.status_code >= 400:
                raise requests.HTTPError(f"{r.status_code}: {r.text[:300]}")
            return r.json()
        raise last_err or RuntimeError("Jev request exhausted retries")

    def compare(self, a_text: str, a_macro: str, b_text: str, b_macro: str) -> dict:
        """Return {'winner': 'A'|'B', 'confidence': float, 'p_a': float}."""
        cap = self.max_excerpt_chars
        while True:
            payload = {
                "model": self.model,
                "state": {"excerpt_A": {"macro_context": a_macro, "text": self._excerpt(a_text, cap)},
                          "excerpt_B": {"macro_context": b_macro, "text": self._excerpt(b_text, cap)}},
                "questions": {"more_hawkish": {"type": "choice", "instructions": INSTRUCTIONS,
                                               "criteria": CRITERIA}},
            }
            try:
                res = self._post(payload)
                break
            except requests.HTTPError as e:
                # over the 32k context (e.g. two very long accounts): shrink and retry
                if str(e).startswith("400") and cap > 8000:
                    cap = int(cap * 0.7)
                    continue
                raise
        self.cost += float(res.get("usage", {}).get("cost") or 0)
        ans = res["answers"]["more_hawkish"]
        w = ans.get("choice")
        p_a = float(ans.get("probabilities", {}).get("A", 0.5))
        if w not in ("A", "B"):
            return {"winner": None, "confidence": 0.0, "p_a": p_a}
        return {"winner": w, "confidence": float(ans.get("confidence", max(p_a, 1 - p_a))),
                "p_a": p_a}


# ---- direct scoring via Jev's "score" primitive ---------------------------------
from .direct import DirectScorer  # noqa: E402

LEVELS = ["Very dovish", "Dovish", "Slightly dovish", "Neutral",
          "Slightly hawkish", "Hawkish", "Very hawkish"]
SCORE_INSTRUCTIONS = (
    "Rate the monetary-policy stance of this anonymized excerpt from euro-area central "
    "bank communication (ECB Governing Council members or the Council itself) RELATIVE "
    "TO the macroeconomic context given. Hawkish = leaning toward tighter policy: concern "
    "about inflation or overheating, higher rates, faster or longer tightening, balance-"
    "sheet reduction. Dovish = concern about growth, employment or disinflation, cuts, "
    "prolonged accommodation, asset purchases. Urging inflation vigilance at 2% HICP is "
    "meaningfully hawkish; the same words at 8.5% HICP merely state the obvious."
)


class JevScorer(DirectScorer):
    """Direct score from Jev's ordered 7-level scale. Jev returns a probability-
    weighted position (0-6), i.e. a continuous score rather than the round-number
    clusters chat models produce; mapped to the same 0-100 units as pairwise."""

    def __init__(self, model: str = MODEL, max_excerpt_chars: int = 100000):
        self.model = model
        self.max_excerpt_chars = max_excerpt_chars   # one excerpt fits Jev's 32k window
        self._jev = JevJudge(model=model)

    def score(self, text: str, macro: str) -> float | None:
        cap = self.max_excerpt_chars
        while True:
            try:
                res = self._jev._post({
                    "model": self.model,
                    "state": {"macro_context": macro, "excerpt": self._jev._excerpt(text, cap)},
                    "questions": {"stance": {"type": "score", "instructions": SCORE_INSTRUCTIONS,
                                             "criteria": LEVELS}}})
                break
            except requests.HTTPError as e:
                if str(e).startswith("400") and cap > 8000:
                    cap = int(cap * 0.7)
                    continue
                return None
            except Exception:  # noqa: BLE001 - exhausted retries: skip, don't kill the run
                return None
        self._jev.cost += float(res.get("usage", {}).get("cost") or 0)
        s = res["answers"]["stance"].get("score")
        return None if s is None else round(float(s) / (len(LEVELS) - 1) * 100, 2)

    @property
    def cost(self) -> float:
        return self._jev.cost
