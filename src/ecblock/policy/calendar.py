"""ECB meeting calendar, decision history and staff projections - from the corpus.

Every monetary-policy meeting since 2010 has a statement in the corpus, so the
decision history (bp change in the key rates) and the staff inflation projections
are parsed straight from those texts; nothing here needs an external source.

Projection meetings: staff projections are published four times a year (March,
June, September, December), so every other meeting carries a new forecast.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

from ..schema import Speech

# announced decision dates not yet in the corpus (statements arrive on the day)
UPCOMING = ["2026-12-17", "2027-02-04", "2027-03-18", "2027-04-29", "2027-06-10",
            "2027-07-22", "2027-09-09", "2027-10-28", "2027-12-16"]
QUIET_DAYS = 7          # members stop signalling a week before each meeting
PROJECTION_MONTHS = {3, 6, 9, 12}

_WORDS = {"five": 5, "ten": 10, "fifteen": 15, "twenty": 20, "twenty-five": 25, "twenty five": 25,
          "thirty-five": 35, "forty": 40, "fifty": 50, "sixty": 60, "seventy-five": 75,
          "seventy five": 75, "one hundred": 100}
_MOVE = re.compile(r"decided (?:today )?to (raise|increase|lower|reduce|cut) (?:the )?(?:three )?"
                   r"(?:key (?:ecb )?interest rates|deposit facility rate|interest rates? on the (?:main refinancing operations|deposit facility))"
                   r"[^.]{0,160}?(?:by\s+)?(\d{1,3}|" + "|".join(sorted(_WORDS, key=len, reverse=True)) +
                   r")\s*(basis points|bp|percentage points?)", re.I)
_HOLD = re.compile(r"decided to (?:keep|leave) the (?:three )?key (?:ecb )?interest rates unchanged|"
                   r"key (?:ecb )?interest rates (?:will )?remain unchanged|interest rates unchanged", re.I)


@dataclass
class Meeting:
    date: str
    bp: int | None = None                 # change in the key rates (deposit facility leg)
    projection: bool = False              # new staff projections released at this meeting
    proj: dict[int, float] = field(default_factory=dict)   # year -> headline HICP projection
    statement_mu: float | None = None

    @property
    def d(self) -> dt.date:
        return dt.date.fromisoformat(self.date)

    @property
    def quiet_start(self) -> str:
        return (self.d - dt.timedelta(days=QUIET_DAYS)).isoformat()

    @property
    def outcome(self) -> int | None:      # -1 cut, 0 hold, +1 hike
        return None if self.bp is None else (self.bp > 0) - (self.bp < 0)


def parse_decision(text: str) -> int | None:
    t = re.sub(r"\s+", " ", text[:8000])
    m = _MOVE.search(t)
    if m:
        v = m.group(2).lower()
        bp = _WORDS.get(v) or int(v)
        if m.group(3).lower().startswith("percentage"):
            bp *= 100
        return bp if m.group(1).lower() in ("raise", "increase") else -bp
    if _HOLD.search(t):
        return 0
    return None


def parse_projection(text: str) -> dict[int, float]:
    """Headline HICP projections {year: value} from a projection-meeting statement.

    Handles the formats used since 2010: "3.0 per cent in 2026, 2.5 per cent in
    2027"; ranges "between 1.1% and 1.7% in 2013" (midpoint); "1.4 % and 1.3 % in
    2013 and 2014, respectively"; and "1.7% in 2018, 2019 and 2020"."""
    t = re.sub(r"\s+", " ", text)
    pct = r"(-?\d+(?:\.\d+)?)\s*(?:%|per\s?cent)"
    for sent in re.split(r"(?<=[.;])\s+", t):
        s = sent.lower()
        if "inflation" not in s or not re.search(r"projection|staff (?:now )?(?:expect|see|foresee)", s):
            continue
        # headline only: stop at a core / GDP clause, skip sentences that open with core
        if re.search(r"excluding energy|underlying|core inflation|gdp", s.split("inflation")[0][-80:] + "inflation"[:0]):
            if re.search(r"excluding energy|underlying|core", s.split("inflation")[0][-60:]):
                continue
        main = re.split(r"excluding energy|underlying inflation|core inflation|real gdp|economic growth",
                        sent, flags=re.I)[0]
        out: dict[int, float] = {}
        m = re.search(rf"{pct}\s+and\s+{pct}\s+(?:in|for)\s+(20\d\d)\s+and\s+(20\d\d),?\s+respectively", main, re.I)
        if m:
            out = {int(m.group(3)): float(m.group(1)), int(m.group(4)): float(m.group(2))}
        m = m or re.search(rf"{pct}\s+(?:in|for)\s+(20\d\d),\s+(20\d\d)\s+and\s+(20\d\d)", main, re.I)
        if m and not out:
            out = {int(m.group(k)): float(m.group(1)) for k in (2, 3, 4)}
        if not out:
            for mm in re.finditer(rf"(?:between\s+(-?\d+(?:\.\d+)?)\s*(?:%|per\s?cent)?\s+and\s+)?"
                                  rf"{pct}\s+(?:in|for)\s+(20\d\d)", main, re.I):
                lo, hi, yr = mm.group(1), float(mm.group(2)), int(mm.group(3))
                out[yr] = round((float(lo) + hi) / 2, 2) if lo else hi
        if len(out) >= 2:
            return out
    return {}


def build(corpus: list[Speech], since: str = "2010-01-01") -> list[Meeting]:
    st = sorted((s for s in corpus if s.source_type == "mp_statement" and s.date >= since),
                key=lambda s: s.date)
    out: list[Meeting] = []
    for s in st:
        bp = parse_decision(s.text)
        if bp is None:
            bp = 0                      # unparsed statements are all holds (verified 2010-2021)
        m = Meeting(s.date, bp, projection=int(s.date[5:7]) in PROJECTION_MONTHS,
                    statement_mu=s.mu)
        if m.projection:
            m.proj = parse_projection(s.text)
        out.append(m)
    last = out[-1].date if out else since
    for d in UPCOMING:
        if d > last:
            out.append(Meeting(d, None, projection=int(d[5:7]) in PROJECTION_MONTHS))
    return out


def next_meeting(meetings: list[Meeting], today: str | None = None) -> Meeting:
    today = today or dt.date.today().isoformat()
    return next(m for m in meetings if m.date >= today and m.bp is None)
