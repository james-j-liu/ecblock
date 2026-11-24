"""Market-implied euro-area inflation from German inflation-linked Bunds.

The Bundes €i bonds are indexed to euro-area HICP (ex tobacco), so
    breakeven = nominal Bund yield - real yield of the €i bond (same maturity)
is a market measure of expected euro-area HICP inflation over the bond's life.
Both legs come from the Bundesbank's free SDMX API:
  - real yields of each €i bond (BBSSY, daily yield by ISIN)
  - the nominal zero-coupon Bund curve at 1-15y (BBSIS, Svensson term structure),
    interpolated linearly to each bond's remaining maturity.
Available daily from March 2012. Cached to data/processed/breakeven.json.

Caveats: €i breakevens carry liquidity and inflation-risk premia, and short-dated
ones are noisy, so the framework uses the *change* in a fixed bond's breakeven
(e.g. since the last staff projections) rather than its level.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import time
import urllib.request

from ..config import PROCESSED

API = "https://api.statistiken.bundesbank.de/rest/data"
LINKERS = {  # ISIN -> maturity date (Bundes €i)
    "DE0001030542": "2023-04-15",
    "DE0001030559": "2030-04-15",
    "DE0001030567": "2026-04-15",
    "DE0001030575": "2046-04-15",
    "DE0001030583": "2033-04-15",
}
CACHE = PROCESSED / "breakeven.json"


def _get(url: str, tries: int = 4) -> str:
    for k in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=120).read().decode("utf-8-sig")
        except Exception:  # noqa: BLE001 - flaky TLS on this API; retry
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))
    return ""


def _series(url: str) -> dict[str, float]:
    out = {}
    text = _get(url)
    sep = ";" if text[:200].count(";") > text[:200].count(",") else ","   # locale-dependent
    for r in csv.DictReader(io.StringIO(text), delimiter=sep):
        try:
            out[r["TIME_PERIOD"]] = float(r["OBS_VALUE"])
        except (ValueError, KeyError):
            pass
    return out


def fetch() -> dict:
    nominal = {}
    for y in range(1, 16):
        key = f"D.I.ZST.ZI.EUR.S1311.B.A604.R{y:02d}XX.R.A.A._Z._Z.A"
        nominal[y] = _series(f"{API}/BBSIS/{key}?format=sdmx_csv&detail=dataonly&startPeriod=2010-01-01")
    real = {isin: _series(f"{API}/BBSSY/D.REN.EUR..{isin}.?format=sdmx_csv&detail=dataonly&startPeriod=2012-01-01")
            for isin in LINKERS}
    be: dict[str, dict[str, float]] = {}
    for isin, rmap in real.items():
        mat = dt.date.fromisoformat(LINKERS[isin])
        for d, ry in rmap.items():
            tau = (mat - dt.date.fromisoformat(d)).days / 365.25
            if not 1.0 <= tau <= 15.0:          # very short / long: too noisy or off-curve
                continue
            lo, hi = int(tau), min(int(tau) + 1, 15)
            n_lo, n_hi = nominal[lo].get(d), nominal[hi].get(d)
            if n_lo is None or n_hi is None:
                continue
            nom = n_lo + (n_hi - n_lo) * (tau - lo)
            be.setdefault(isin, {})[d] = round(nom - ry, 3)
    # short nominal yields (1y, 2y) are kept as a market-pricing proxy for the rate path
    data = {"fetched": dt.date.today().isoformat(), "maturity": LINKERS, "breakeven": be,
            "nominal": {"1": nominal[1], "2": nominal[2]}}
    CACHE.write_text(json.dumps(data, indent=0), encoding="utf-8")
    return data


def load(refresh: bool = False) -> dict:
    if refresh or not CACHE.exists():
        try:
            return fetch()
        except Exception as e:  # noqa: BLE001 - fall back to the cached copy
            print(f"[breakeven] fetch failed ({type(e).__name__}: {e}); using cache")
    return json.loads(CACHE.read_text(encoding="utf-8"))


def _at(series: dict[str, float], date: str, back: int = 10) -> float | None:
    d = dt.date.fromisoformat(date)
    for k in range(back):
        v = series.get((d - dt.timedelta(days=k)).isoformat())
        if v is not None:
            return v
    return None


def change(data: dict, since: str, until: str, target_tau: float = 4.0) -> tuple[float | None, str | None]:
    """Breakeven change (pp) between two dates on ONE bond: the €i whose remaining
    maturity at `since` is closest to target_tau (and still >1y at `until`)."""
    best, isin_best = None, None
    for isin, ser in data["breakeven"].items():
        mat = dt.date.fromisoformat(data["maturity"][isin])
        tau0 = (mat - dt.date.fromisoformat(since)).days / 365.25
        a, b = _at(ser, since), _at(ser, until)
        if a is None or b is None:
            continue
        if best is None or abs(tau0 - target_tau) < best[0]:
            best, isin_best = (abs(tau0 - target_tau), round(b - a, 3)), isin
    return (best[1], isin_best) if best else (None, None)


def level(data: dict, date: str, target_tau: float = 4.0) -> tuple[float | None, float | None]:
    """Breakeven level on the bond closest to target_tau at `date` -> (value, maturity)."""
    best = None
    for isin, ser in data["breakeven"].items():
        mat = dt.date.fromisoformat(data["maturity"][isin])
        tau = (mat - dt.date.fromisoformat(date)).days / 365.25
        v = _at(ser, date)
        if v is not None and (best is None or abs(tau - target_tau) < abs(best[1] - target_tau)):
            best = (v, round(tau, 1))
    return best if best else (None, None)
