"""Build site/nextmeeting.json: probability distribution for the next ECB meeting.

    python scripts/next_meeting.py              # uses cached guidance + breakevens
    python scripts/next_meeting.py --refresh    # classify new core docs (Jev) + refetch breakevens

See src/ecblock/policy/framework.py for the method. The headline model is chosen by a
significance check, re-run daily (framework.select): the core (pace, member guidance,
inflation vs projections) must beat "repeat the last decision" significantly, and each
add-on (inertia, hawkishness shift) must significantly improve on that by itself.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

from ecblock.config import PROCESSED
from ecblock.policy import breakeven as be_mod
from ecblock.policy import calendar as cal
from ecblock.policy import framework as fw
from ecblock.policy import guidance as gd
from ecblock.roster_gc import canon
from ecblock.schema import load_corpus

OUT = ROOT / "site" / "nextmeeting.json"
_KEY = re.compile(r"\b(interest rates?|rates|hike|hiking|raise|increase|cut|cutting|lower|tighten\w*|"
                  r"restrictive|accommodat\w*|december|next meeting|coming meetings|further (?:step|move)|"
                  r"pause|patient|another|step|meeting.by.meeting|data.dependen\w*|screw)\b", re.I)


def key_quote(text: str, n: int = 2, max_len: int = 320) -> str:
    sents = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text))
    scored = sorted(((len(_KEY.findall(s)), i, s) for i, s in enumerate(sents) if 30 < len(s) < 400),
                    key=lambda x: (-x[0], x[1]))
    pick = sorted(scored[:n], key=lambda x: x[1])
    return " … ".join(s for k, _, s in pick if k)[:max_len * n]


def size_split(p: np.ndarray, rows, row) -> dict:
    """Split P(hike)/P(cut) into 25bp vs 50bp+ using the 'larger move' guidance share
    and whether the cycle's last move was already 50bp+."""
    f = row.feats
    last = next((r.meeting.bp for r in reversed(rows) if r.meeting.bp), 0) or 0
    directional = f["g_up"] + f["g_down"]
    share = f["g_large"] / directional if directional > 0.05 else 0.0
    base = 0.5 if abs(last) >= 50 else 0.08
    big = float(np.clip(0.6 * share + 0.4 * base, 0.0, 0.9))
    cut, hold, hike = p.tolist()
    return {"cut_large": round(cut * big, 4), "cut": round(cut * (1 - big), 4), "hold": round(hold, 4),
            "hike": round(hike * (1 - big), 4), "hike_large": round(hike * big, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    today = dt.date.today().isoformat()

    corpus = load_corpus(PROCESSED / "corpus.jsonl")
    meetings = cal.build(corpus)
    nxt = cal.next_meeting(meetings, today)
    if args.refresh:
        from ecblock.process.anonymize import Anonymizer
        from ecblock.process.roster import build_roster
        core = [s for s in corpus if gd.is_core(s) and s.is_policy and s.text and "2010-05-28" <= s.date <= today]
        anon = Anonymizer(build_roster([s.speaker for s in corpus]))
        gd.classify(core, anon)
        council_docs = [s for s in corpus if gd.is_council(s) and s.text and "2010-01-01" <= s.date <= today]
        gd.classify(council_docs, anon, decisions={m.date: m.bp for m in meetings if m.bp is not None})
    guidance = gd.load_cache()
    council = gd.load_cache(gd.COUNCIL_CACHE)
    be_data = be_mod.load(refresh=args.refresh)
    macro = json.loads((ROOT / "site" / "macro.json").read_text(encoding="utf-8"))["series"]
    hicp = {d[:7]: v for d, v in macro["hicp_headline"]["data"]}

    dfr = sorted((d, v) for d, v in macro["deposit_rate"]["data"])
    rows = fw.build_rows(meetings, corpus, guidance, be_data, hicp, today, council=council, dfr=dfr)
    live = rows[-1]
    assert live.meeting.date == nxt.date, (live.meeting.date, nxt.date)
    sel = fw.select(rows)
    best = "Selected (significance-checked)"
    fw.APPROACHES[best] = sel["features"]
    bt = fw.backtest(rows)
    summ = bt["summary"]

    approaches = []
    for name, cols in fw.APPROACHES.items():
        p = fw.forecast(rows, cols)
        approaches.append({"name": name, "features": cols,
                           "probs": {"cut": round(p[0], 4), "hold": round(p[1], 4), "hike": round(p[2], 4)},
                           **summ[name]})
    p_best = fw.forecast(rows, fw.APPROACHES[best])
    rng80 = fw.ranges(rows, sel["features"])
    calib = fw.calibration(rows, sel["features"])
    inertia = fw.inertia_stats(meetings)
    inertia["since_2015"] = fw.inertia_stats([m for m in meetings if m.date >= "2015-01-01"])

    # ---- signals for display
    f = live.feats
    ps = fw.pace_state([m for m in meetings if m.bp is not None and m.date < nxt.date])
    prev = next(m for m in reversed(meetings) if m.bp is not None and m.date < nxt.date)
    docs = []
    for s in sorted(live.docs, key=lambda s: s.date):
        g = guidance.get(s.id)
        docs.append({"date": s.date, "speaker": canon(s.speaker), "inst": s.institution,
                     "type": s.source_type, "title": s.title[:140], "url": s.source_url,
                     "score": None if s.mu is None else round((s.mu - 50) / 5, 2),
                     "next": g["next"] if g else None, "pace": g["pace"] if g else None,
                     "quote": key_quote(s.text)})
    # the Council's own guidance: last meeting's statement + Q&A, and any account since
    hi = min(today, nxt.quiet_start)
    label = {"mp_statement": "ECB Governing Council — statement", "mp_qa": "ECB Governing Council — press conference Q&A",
             "mp_account": "ECB Governing Council — account"}
    for s in sorted((s for s in corpus if s.id in council and (
            (s.date == prev.date and s.source_type in ("mp_statement", "mp_qa"))
            or (s.source_type == "mp_account" and prev.date < s.date <= hi))), key=lambda s: s.date):
        g = council[s.id]
        docs.append({"date": s.date, "speaker": label[s.source_type], "inst": "European Central Bank",
                     "type": "council", "title": s.title[:140], "url": s.source_url,
                     "score": None if s.mu is None else round((s.mu - 50) / 5, 2),
                     "next": g["next"], "pace": g["pace"], "quote": key_quote(s.text)})
    docs.sort(key=lambda d: d["date"])
    lastp = next(m for m in reversed(meetings) if m.projection and m.bp is not None and m.date < nxt.date)
    dbe, isin = be_mod.change(be_data, lastp.date, today)
    be_ser = be_data["breakeven"].get(isin, {}) if isin else {}
    start = (dt.date.fromisoformat(lastp.date) - dt.timedelta(days=120)).isoformat()
    projm = next((m for m in reversed(meetings) if m.projection and m.proj and m.date < nxt.date), None)
    hm = sorted(hicp)[-1]

    # what the model was reading at each meeting (for the misses table)
    sig = {r.meeting.date: {"g_up": round(r.feats["g_up"], 2), "g_down": round(r.feats["g_down"], 2),
                            "g_hold": round(r.feats["g_hold"], 2), "g_n": r.feats["g_n"],
                            "c_up": round(r.feats["c_up"], 2), "c_down": round(r.feats["c_down"], 2),
                            "c_hold": round(r.feats["c_hold"], 2), "prev": r.feats["last_up"] - r.feats["last_down"]}
           for r in rows}
    out = {
        "generated": today,
        "next_meeting": {"date": nxt.date, "projection": nxt.projection, "quiet_start": nxt.quiet_start,
                         "window": [prev.date, min(today, nxt.quiet_start)], "last_decision": prev.bp,
                         "last_decision_date": prev.date},
        "headline": {"model": best, "probs": size_split(p_best, rows, live),
                     "probs3": {"cut": round(p_best[0], 4), "hold": round(p_best[1], 4), "hike": round(p_best[2], 4)},
                     "range80": rng80},
        "selection": sel,
        "calibration": calib,
        "inertia": {**inertia, "now": {k: live.feats[k] for k in ("last_up", "last_down", "first_up", "first_down",
                                                                   "paused_up", "paused_down", "run_len")}},
        "approaches": approaches,
        "signals": {
            "pace": {**ps, "moves": [(m.date, m.bp) for m in meetings if m.bp and m.date >= "2022-01-01"]},
            "guidance": {"docs": docs, "g_up": f["g_up"], "g_down": f["g_down"], "g_hold": f["g_hold"],
                         "p_acc": f["p_acc"], "p_slow": f["p_slow"], "n": f["g_n"]},
            "hawk": {"level": round(f["hawk"], 2), "change": round(f["d_hawk"], 2)},
            "inflation": {"proj_date": lastp.date, "d_be": dbe, "isin": isin,
                          "maturity": be_data["maturity"].get(isin), "be_now": be_mod._at(be_ser, today) if be_ser else None,
                          "be_at_proj": be_mod._at(be_ser, lastp.date) if be_ser else None,
                          "series": sorted((d, v) for d, v in be_ser.items() if d >= start),
                          "proj": projm.proj if projm else {}, "proj_from": projm.date if projm else None,
                          "hicp_month": hm, "hicp": hicp[hm], "hicp_gap": f["hicp_gap"]},
        },
        "pace_history": fw.pace_history(rows),
        "backtest": {"start": "2013-01-01",
                     "series": [{"date": d, "outcome": y, "p": [round(x, 3) for x in p], **sig.get(d, {})}
                                for d, y, p in bt["preds"][best]]},
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    h = out["headline"]["probs3"]
    print(f"[next meeting] {nxt.date}: {best} -> cut {h['cut']:.0%} hold {h['hold']:.0%} hike {h['hike']:.0%} "
          f"| wrote {args.out}")


if __name__ == "__main__":
    main()
