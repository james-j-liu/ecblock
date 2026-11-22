"""Next-meeting probability framework + backtest.

For every meeting, features are built only from information available before it:
the window runs from the previous meeting to the start of the quiet period.

  pace      - the cycle-cadence dummy. In an established cycle (>=2 moves in one
              direction), a move is "due" when the meetings since the last move have
              reached the spacing between the last two moves; moving earlier is an
              acceleration, holding when due is a slow-down/end.
  guidance  - Jev's reading of core members' explicit guidance (soft votes).
  hawk      - change in core members' pairwise hawkishness vs the previous window.
  inflation - market-implied inflation (€i Bund breakeven) change since the last
              staff projections, and latest HICP minus the projection for that year.

Approaches are multinomial logistic models (cut / hold / hike) on subsets of those
features, fitted on all earlier meetings (expanding window) and scored on each
meeting from 2013 by log loss, Brier score and accuracy.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

import numpy as np

from . import breakeven as be_mod
from .calendar import Meeting
from .guidance import is_core

CLASSES = (-1, 0, 1)
N = lambda mu: (mu - 50) / 5            # site scale


@dataclass
class Row:
    meeting: Meeting
    feats: dict = field(default_factory=dict)
    docs: list = field(default_factory=list)       # for display


# ---------------------------------------------------------------- pace (cadence)
def pace_state(hist: list[Meeting]) -> dict:
    """Cycle state from decided meetings up to (not incl.) the target meeting."""
    moves = [(i, m.outcome) for i, m in enumerate(hist) if m.outcome]
    n = len(hist)
    if not moves:
        return dict(dir=0, n_cycle=0, gap=0, since=n, established=0, due=0, early=0)
    last_i, last_dir = moves[-1]
    since = n - 1 - last_i + 1            # meetings since the last move (1 = just moved)
    if since > 6:                          # cycle has gone quiet
        return dict(dir=0, n_cycle=0, gap=0, since=since, established=0, due=0, early=0)
    cyc = []
    for i, o in reversed(moves):
        if o != last_dir or (cyc and cyc[-1] - i > 4):
            break
        cyc.append(i)
    gap = cyc[0] - cyc[1] if len(cyc) >= 2 else 0
    est = int(len(cyc) >= 2)
    if est and since > gap:
        # the cadence date has passed without a move: the cycle has stalled/ended,
        # so later meetings are not "due" again (keeps post-cycle holds out of the test)
        est = 0
    return dict(dir=last_dir, n_cycle=len(cyc), gap=gap, since=since, established=est,
                due=int(est and since == gap), early=int(est and since < gap))


# ---------------------------------------------------------------- feature builder
def build_rows(meetings: list[Meeting], corpus, guidance: dict, be_data: dict,
               hicp: dict[str, float], today: str) -> list[Row]:
    core = sorted((s for s in corpus if is_core(s) and s.is_policy), key=lambda s: s.date)
    rows: list[Row] = []
    for k in range(1, len(meetings)):
        m, prev = meetings[k], meetings[k - 1]
        if m.bp is None and m.date != next(x.date for x in meetings if x.bp is None):
            break                                   # only the next undecided meeting
        hi = min(m.quiet_start, today) if m.bp is None else m.quiet_start
        win = [s for s in core if prev.date < s.date <= hi]
        prev_lo = meetings[k - 2].date if k >= 2 else "2000-01-01"
        prev_win = [s for s in core if prev_lo < s.date <= prev.quiet_start]
        f = {}
        # pace
        ps = pace_state(meetings[:k])
        f.update({f"pace_{a}": ps[a] for a in ("dir", "established", "due", "early", "since")})
        f["due_up"] = int(ps["due"] and ps["dir"] > 0)
        f["due_down"] = int(ps["due"] and ps["dir"] < 0)
        f["cyc_up"] = int(ps["dir"] > 0)
        f["cyc_down"] = int(ps["dir"] < 0)
        f["early_up"] = int(ps["early"] and ps["dir"] > 0)
        f["early_down"] = int(ps["early"] and ps["dir"] < 0)
        # the last decision, and how this cycle has been paced: was the last move the
        # first of a new cycle; did the Council move then pause; how long is the run
        outs = [x.outcome for x in meetings[:k] if x.outcome is not None]
        last = outs[-1] if outs else 0
        f["last_up"], f["last_down"] = int(last > 0), int(last < 0)
        run = 0
        for o in reversed(outs):
            if o and o == last:
                run += 1
            else:
                break
        prior_same = [o for o in outs[-9:-1] if o == last] if last else []
        f["first_up"] = int(last > 0 and run == 1 and not prior_same)
        f["first_down"] = int(last < 0 and run == 1 and not prior_same)
        f["paused_up"] = int(len(outs) >= 2 and outs[-1] == 0 and outs[-2] > 0)
        f["paused_down"] = int(len(outs) >= 2 and outs[-1] == 0 and outs[-2] < 0)
        f["run_len"] = float(np.sign(last) * min(run, 5))
        # guidance (soft votes over core docs in the window)
        g = [guidance[s.id] for s in win if s.id in guidance]
        if g:
            f["g_up"] = float(np.mean([x["next"].get("hike", 0) + x["next"].get("larger_hike", 0) for x in g]))
            f["g_down"] = float(np.mean([x["next"].get("cut", 0) + x["next"].get("larger_cut", 0) for x in g]))
            f["g_hold"] = float(np.mean([x["next"].get("hold", 0) for x in g]))
            f["g_large"] = float(np.mean([x["next"].get("larger_hike", 0) + x["next"].get("larger_cut", 0) for x in g]))
            f["p_acc"] = float(np.mean([x["pace"].get("accelerate", 0) for x in g]))
            f["p_slow"] = float(np.mean([x["pace"].get("slow_or_pause", 0) + x["pace"].get("end_or_reverse", 0) for x in g]))
            f["p_end"] = float(np.mean([x["pace"].get("end_or_reverse", 0) for x in g]))
            # the single clearest statement in the window (an average dilutes one
            # explicit "we should hike in December" among several no-signal speeches)
            f["g_up_max"] = float(max(x["next"].get("hike", 0) + x["next"].get("larger_hike", 0) for x in g))
            f["g_down_max"] = float(max(x["next"].get("cut", 0) + x["next"].get("larger_cut", 0) for x in g))
            f["g_hold_max"] = float(max(x["next"].get("hold", 0) for x in g))
        else:
            f.update(g_up=0.0, g_down=0.0, g_hold=0.0, g_large=0.0, p_acc=0.0, p_slow=0.0, p_end=0.0,
                     g_up_max=0.0, g_down_max=0.0, g_hold_max=0.0)
        f["g_n"] = len(g)
        # hawkishness level + change (pairwise, core members)
        lv = [N(s.mu) for s in win if s.mu is not None]
        lv0 = [N(s.mu) for s in prev_win if s.mu is not None]
        f["hawk"] = float(np.mean(lv)) if lv else 0.0
        f["d_hawk"] = float(np.mean(lv) - np.mean(lv0)) if lv and lv0 else 0.0
        # inflation paradigm: breakeven change since the last projections + HICP surprise
        lastp = next((x for x in reversed(meetings[:k]) if x.projection), None)
        if lastp and lastp.date >= "2012-03-22":
            dbe, _ = be_mod.change(be_data, lastp.date, hi)
            f["d_be"] = dbe if dbe is not None else 0.0
        else:
            f["d_be"] = 0.0
        projm = next((x for x in reversed(meetings[:k]) if x.projection and x.proj), None)
        months = sorted(mo for mo in hicp if _month_end(mo) < m.date)
        f["hicp_gap"] = 0.0
        if months and projm:
            mo = months[-1]
            p = projm.proj.get(int(mo[:4]))
            if p is not None:
                f["hicp_gap"] = round(hicp[mo] - p, 2)
        rows.append(Row(m, f, win))
    return rows


def _month_end(mo: str) -> str:
    y, mth = int(mo[:4]), int(mo[5:7])
    nxt = dt.date(y + (mth == 12), mth % 12 + 1, 1)
    return (nxt - dt.timedelta(days=1)).isoformat()


# ---------------------------------------------------------------- models
APPROACHES = {
    "Base rate": [],
    "Pace (cycle dummy)": ["due_up", "due_down", "cyc_up", "cyc_down", "early_up", "early_down"],
    "Member guidance": ["g_up", "g_down", "g_hold", "g_up_max", "g_down_max", "g_hold_max", "p_acc", "p_slow"],
    "Hawkishness shift": ["hawk", "d_hawk", "cyc_up", "cyc_down"],
    "Inflation vs projections": ["d_be", "hicp_gap", "cyc_up", "cyc_down"],
    "Combined": ["due_up", "due_down", "cyc_up", "cyc_down", "early_up", "early_down",
                 "g_up", "g_down", "g_hold", "g_up_max", "g_down_max", "g_hold_max", "p_acc", "p_slow",
                 "hawk", "d_hawk", "d_be", "hicp_gap"],
    "Combined without pace": ["cyc_up", "cyc_down", "g_up", "g_down", "g_hold", "g_up_max", "g_down_max",
                              "g_hold_max", "p_acc", "p_slow",
                              "hawk", "d_hawk", "d_be", "hicp_gap"],
}


def _fit_predict(train: list[Row], test: Row, cols: list[str], C: float = 0.5) -> np.ndarray:
    y = np.array([r.meeting.outcome for r in train])
    counts = np.array([(y == c).sum() for c in CLASSES], float)
    if not cols or len(set(y)) < 2:
        return (counts + 1) / (counts.sum() + 3)
    from sklearn.linear_model import LogisticRegression
    X = np.array([[r.feats[c] for c in cols] for r in train], float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    clf = LogisticRegression(C=C, max_iter=2000)
    clf.fit((X - mu) / sd, y)
    x = (np.array([[test.feats[c] for c in cols]], float) - mu) / sd
    p = dict(zip(clf.classes_, clf.predict_proba(x)[0]))
    out = np.array([p.get(c, 0.0) for c in CLASSES])
    out = 0.97 * out + 0.03 * (counts + 1) / (counts.sum() + 3)   # floor for unseen classes
    return out / out.sum()


def backtest(rows: list[Row], start: str = "2013-01-01") -> dict:
    decided = [r for r in rows if r.meeting.outcome is not None]
    res = {name: [] for name in APPROACHES}
    for i, r in enumerate(decided):
        if r.meeting.date < start:
            continue
        train = decided[:i]
        for name, cols in APPROACHES.items():
            p = _fit_predict(train, r, cols)
            res[name].append((r.meeting.date, r.meeting.outcome, p.tolist()))
    summary = {}
    for name, preds in res.items():
        ll = [-math.log(max(p[CLASSES.index(y)], 1e-6)) for _, y, p in preds]
        br = [sum((p[j] - (CLASSES[j] == y)) ** 2 for j in range(3)) for _, y, p in preds]
        acc = [CLASSES[int(np.argmax(p))] == y for _, y, p in preds]
        # "turning" meetings: decision differs from the previous meeting's
        turn = [k for k in range(1, len(preds)) if preds[k][1] != preds[k - 1][1]]
        ll_turn = [ll[k] for k in turn]
        summary[name] = dict(n=len(preds), log_loss=round(float(np.mean(ll)), 3),
                             brier=round(float(np.mean(br)), 3), accuracy=round(float(np.mean(acc)), 3),
                             n_turn=len(turn), log_loss_turn=round(float(np.mean(ll_turn)), 3) if turn else None)
    return {"summary": summary, "preds": res}


# ---------------------------------------------------------------- significance-checked selection
GROUPS = {
    "last decision": ["last_up", "last_down"],
    "pace": ["due_up", "due_down", "cyc_up", "cyc_down", "early_up", "early_down"],
    "member guidance": ["g_up", "g_down", "g_hold", "g_up_max", "g_down_max", "g_hold_max", "p_acc", "p_slow"],
    "inflation vs projections": ["d_be", "hicp_gap"],
    "inertia (first move, pause, run length)": ["first_up", "first_down", "paused_up", "paused_down", "run_len"],
    "hawkishness shift": ["hawk", "d_hawk"],
}
CORE = ["pace", "member guidance", "inflation vs projections"]
ADDONS = ["inertia (first move, pause, run length)", "hawkishness shift"]


def _losses(rows, cols, start, how: str = "loyo"):
    """Per-meeting log loss from `start`. "loyo": each meeting predicted by a model fitted
    on every other year (as in the BoE/BoC models); "expanding": fitted on earlier
    meetings only (the real-time backtest)."""
    decided = [r for r in rows if r.meeting.outcome is not None]
    out = []
    for i, r in enumerate(decided):
        if r.meeting.date < start:
            continue
        tr = decided[:i] if how == "expanding" else [x for x in decided if x.meeting.date[:4] != r.meeting.date[:4]]
        p = _fit_predict(tr, r, cols)
        out.append(-math.log(max(p[CLASSES.index(r.meeting.outcome)], 1e-6)))
    return np.array(out)


def _boot(a, b, n=2000, seed=0):
    d = a - b
    rng = np.random.default_rng(seed)
    s = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def select(rows, start: str = "2013-01-01") -> dict:
    """The baseline is "repeat the last decision". The core (pace, member guidance,
    inflation vs projections) is kept as a bundle only if it beats that significantly;
    each add-on is kept only if it significantly lowers log loss on its own. As for the
    BoE and BoC models: leave-one-year-out predictions, paired bootstrap over meetings
    (2000 draws), 95% interval wholly below zero. The real-time (expanding-window)
    result for the core is reported alongside."""
    base = GROUPS["last decision"]
    core = base + [c for g in CORE for c in GROUPS[g]]
    lb, lc = _losses(rows, base, start), _losses(rows, core, start)
    m, lo, hi = _boot(lc, lb)
    eb, ec = _losses(rows, base, start, "expanding"), _losses(rows, core, start, "expanding")
    em, elo, ehi = _boot(ec, eb)
    rep = {"baseline": "repeat the last decision", "baseline_log_loss": round(float(lb.mean()), 4), "n": int(len(lb)),
           "core": {"log_loss": round(float(lc.mean()), 4), "change": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
                    "significant": hi < 0, "groups": [],
                    "realtime": {"baseline_log_loss": round(float(eb.mean()), 4), "log_loss": round(float(ec.mean()), 4),
                                 "change": round(em, 4), "ci95": [round(elo, 4), round(ehi, 4)], "significant": ehi < 0}},
           "addons": []}
    for g in CORE:      # each core group's contribution, for the record
        rest = [c for c in core if c not in GROUPS[g]]
        mm, l2, h2 = _boot(lc, _losses(rows, rest, start))
        rep["core"]["groups"].append({"group": g, "contribution": round(mm, 4), "ci95": [round(l2, 4), round(h2, 4)],
                                      "significant_alone": h2 < 0})
    if not hi < 0:
        core, lc = list(base), lb
    cur, cl = list(core), lc
    for g in ADDONS:
        nl = _losses(rows, cur + GROUPS[g], start)
        mm, l2, h2 = _boot(nl, cl)
        keep = h2 < 0
        rep["addons"].append({"group": g, "log_loss": round(float(nl.mean()), 4), "change": round(mm, 4),
                              "ci95": [round(l2, 4), round(h2, 4)], "kept": keep})
        if keep:
            cur, cl = cur + GROUPS[g], nl
    rep["features"] = cur
    rep["log_loss"] = round(float(cl.mean()), 4)
    return rep


def ranges(rows, cols, n: int = 300, seed: int = 0) -> dict:
    """80% range of the live forecast from refits on resampled histories (meetings)."""
    decided = [r for r in rows if r.meeting.outcome is not None]
    rng = np.random.default_rng(seed)
    acc = []
    for _ in range(n):
        sub = [decided[i] for i in rng.integers(0, len(decided), len(decided))]
        acc.append(_fit_predict(sub, rows[-1], cols))
    acc = np.array(acc)
    return {c: [round(float(np.percentile(acc[:, j], 10)), 3), round(float(np.percentile(acc[:, j], 90)), 3)]
            for j, c in enumerate(("cut", "hold", "hike"))}


def calibration(rows, cols, start: str = "2013-01-01") -> list[dict]:
    """How often the most likely outcome happened, by stated confidence (leave-one-year-out)."""
    decided = [r for r in rows if r.meeting.outcome is not None and r.meeting.date >= start]
    allr = [r for r in rows if r.meeting.outcome is not None]
    pm, hit = [], []
    for r in decided:
        p = _fit_predict([x for x in allr if x.meeting.date[:4] != r.meeting.date[:4]], r, cols)
        pm.append(float(p.max()))
        hit.append(CLASSES[int(np.argmax(p))] == r.meeting.outcome)
    pm, hit = np.array(pm), np.array(hit)
    out = []
    for lo, hi in ((0.0, 0.8), (0.8, 0.95), (0.95, 1.01)):
        m = (pm >= lo) & (pm < hi)
        if m.sum():
            out.append({"band": f"{lo:.0%}–{min(hi, 1):.0%}", "n": int(m.sum()), "stated": round(float(pm[m].mean()), 3),
                        "right": round(float(hit[m].mean()), 3)})
    return out


def inertia_stats(meetings) -> dict:
    """How the Governing Council paces a cycle."""
    d = [(m.date, m.outcome) for m in meetings if m.outcome is not None]
    follow, first, pause, runs = [], [], [], []
    i = 0
    while i < len(d):
        if d[i][1]:
            sgn, j = d[i][1], i
            while j + 1 < len(d) and d[j + 1][1] == sgn:
                j += 1
            runs.append(j - i + 1)
            i = j + 1
        else:
            i += 1
    for i, (dd, u) in enumerate(d[:-1]):
        if not u:
            continue
        follow.append(d[i + 1][1] == u)
        rec = [v for x, v in d[:i] if (dt.date.fromisoformat(dd) - dt.date.fromisoformat(x)).days < 365 and v == u]
        if not rec:
            first.append(d[i + 1][1] == u)
    for i in range(2, len(d)):
        if d[i - 2][1] and not d[i - 1][1]:
            pause.append(d[i][1] == d[i - 2][1])
    f = lambda x: {"p": round(float(np.mean(x)), 3), "n": len(x)}
    return {"since": d[0][0], "after_move": f(follow), "after_first_move": f(first), "after_pause": f(pause),
            "mean_run": round(float(np.mean(runs)), 2), "runs": sorted(runs)}


def forecast(rows: list[Row], cols: list[str]) -> np.ndarray:
    decided = [r for r in rows if r.meeting.outcome is not None]
    return _fit_predict(decided, rows[-1], cols)


def pace_history(rows: list[Row]) -> dict:
    """How often has the ECB broken an established cadence - and did members' language
    flag it? Established-cycle meetings only."""
    est = [r for r in rows if r.meeting.outcome is not None and r.feats["pace_established"]]
    out = {"due": {"moved": 0, "held": 0, "reversed": 0}, "not_due": {"moved": 0, "held": 0, "reversed": 0}}
    flags = {"accelerated": [], "on_cadence_hold": [], "stopped_when_due": [], "on_cadence_move": []}
    for r in est:
        d, o, f = r.feats["pace_dir"], r.meeting.outcome, r.feats
        key = "due" if f["pace_due"] else "not_due"
        out[key]["moved" if o == d else ("held" if o == 0 else "reversed")] += 1
        sig = dict(date=r.meeting.date, g_dir=round(f["g_up"] - f["g_down"], 3) * (1 if d > 0 else -1),
                   p_acc=round(f["p_acc"], 3), p_slow=round(f["p_slow"], 3), n=f["g_n"])
        if f["pace_due"]:
            flags["on_cadence_move" if o == d else "stopped_when_due"].append(sig)
        else:
            flags["accelerated" if o == d else "on_cadence_hold"].append(sig)
    avg = {k: {m: round(float(np.mean([s[m] for s in v])), 3) for m in ("g_dir", "p_acc", "p_slow")} if v else None
           for k, v in flags.items()}
    return {"counts": out, "language": avg, "cases": {k: [s["date"] for s in v] for k, v in flags.items()}}
