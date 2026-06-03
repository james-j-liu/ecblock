"""Euro-area macro context for the judge (analogue of FedLock's PCE/unemp/GDP/VIX).

Series pulled from the ECB Data Portal REST API and cached to data/raw. For each
speech date we report the most recent observation on or before that date, so the
judge sees only information available at the time.

Series (euro-area aggregate):
  - core_hicp   : HICP excluding energy & food, annual % change   (~Core PCE)
  - unemployment: unemployment rate, %                            (~UNRATE)
  - gdp_growth  : real GDP, annual % change                       (~GDP growth)
  - vstoxx      : EURO STOXX 50 implied vol (optional, best-effort) (~VIX)

If a series can't be fetched, it degrades to "n/a" rather than failing.
"""
from __future__ import annotations

from datetime import date
from io import StringIO

import pandas as pd
import requests

from ..config import RAW, cfg

ECB_API = "https://data-api.ecb.europa.eu/service/data"


def _split_flow_key(series_key: str) -> tuple[str, str]:
    # SDW key like "ICP.M.U2.N.XEF000.4.ANR" -> flow="ICP", key="M.U2.N.XEF000.4.ANR"
    parts = series_key.split(".", 1)
    return parts[0], parts[1]


def _fetch_series(series_key: str) -> pd.Series | None:
    flow, key = _split_flow_key(series_key)
    cache = RAW / f"macro_{flow}_{key.replace('.', '_')}.csv"
    if cache.exists():
        df = pd.read_csv(cache)
    else:
        url = f"{ECB_API}/{flow}/{key}"
        try:
            r = requests.get(url, params={"format": "csvdata"}, timeout=60)
            r.raise_for_status()
            df = pd.read_csv(StringIO(r.text))
            cache.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache, index=False)
        except Exception as e:  # noqa: BLE001 - degrade gracefully
            print(f"[macro] failed to fetch {series_key}: {e}")
            return None
    if "TIME_PERIOD" not in df or "OBS_VALUE" not in df:
        return None
    idx = pd.to_datetime(df["TIME_PERIOD"].astype(str), errors="coerce", format="mixed")
    s = pd.Series(pd.to_numeric(df["OBS_VALUE"], errors="coerce").values, index=idx)
    s = s[~s.index.isna()].dropna().sort_index()
    return s


class MacroContext:
    def __init__(self):
        series_cfg = cfg()["macro"]["series"]
        self.series: dict[str, pd.Series] = {}
        for name, key in series_cfg.items():
            if not key:
                continue
            s = _fetch_series(key)
            if s is not None and len(s):
                self.series[name] = s

    def as_of(self, d: str) -> dict[str, float | None]:
        ts = pd.Timestamp(d)
        out: dict[str, float | None] = {}
        for name, s in self.series.items():
            prior = s[s.index <= ts]
            out[name] = float(prior.iloc[-1]) if len(prior) else None
        return out

    def string(self, d: str) -> str:
        v = self.as_of(d)
        bits = []
        labels = {
            "core_hicp": "Core HICP infl",
            "unemployment": "Unemployment",
            "gdp_growth": "GDP growth (YoY)",
            "vstoxx": "VSTOXX",
        }
        for k, lab in labels.items():
            if v.get(k) is not None:
                suffix = "%" if k != "vstoxx" else ""
                bits.append(f"{lab}: {v[k]:.1f}{suffix}")
        return "; ".join(bits) if bits else "n/a"


if __name__ == "__main__":
    mc = MacroContext()
    print("Loaded series:", list(mc.series))
    for d in ("2008-09-15", "2015-03-05", "2022-07-21", "2024-06-06"):
        print(d, "->", mc.string(d))
