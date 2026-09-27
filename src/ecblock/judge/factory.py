"""Pick the model for each scoring job.

Pairwise judging and direct scoring can use a decision model (TypeSafe Jev) while
classification and translation - which need a chat model (translation generates
text) - keep using judge.model / JUDGE_MODEL. Override per job with the
PAIRWISE_MODEL / DIRECT_MODEL environment variables.
"""
from __future__ import annotations

import os

from ..config import cfg, judge_model


def _model(key: str, env: str) -> str:
    return os.environ.get(env) or cfg()["judge"].get(key) or judge_model()


def pairwise_model() -> str:
    return _model("pairwise_model", "PAIRWISE_MODEL")


def direct_model() -> str:
    return _model("direct_model", "DIRECT_MODEL")


def make_pairwise_judge():
    m = pairwise_model()
    if m.startswith("typesafe/jev"):
        from .jev import JevJudge
        return JevJudge(model=m)
    from .openrouter import Judge
    return Judge(model=m)


def make_direct_scorer():
    m = direct_model()
    if m.startswith("typesafe/jev"):
        from .jev import JevScorer
        return JevScorer(model=m)
    from .direct import DirectScorer
    return DirectScorer(model=m)
