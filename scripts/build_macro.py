"""Build site/macro.json: macro/market series to overlay on the Timeline tab.

Series:
  deposit_rate   ECB deposit facility rate (%)        ECB FM.B.U2.EUR.4F.KR.DFR.LEV (event/step)
  hicp_headline  HICP all-items, annual % change      Eurostat prc_hicp_manr CP00 / EA
  hicp_core      HICP ex energy & food, annual % chg   Eurostat prc_hicp_manr TOT_X_NRG_FOOD / EA
  eur_10y        Euro area 10Y AAA gov bond yield     ECB YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y (daily->monthly)

HICP is sourced from Eurostat. For each month we use the FINAL value
(prc_hicp_manr) when available, and fall back to the FLASH estimate
(ei_cphi_m, the euro-indicators dataset) for the most recent months that the
final series has not yet reached - so the series runs to the latest flash month
(currently April 2026) instead of stopping at the last finalised month.
Flash vs final agree to <=0.1pp over 2009-2025 (recent months exact), and the
final ECB/Eurostat series matched over 2009-2025 (core: 0 diffs; headline: one
0.1pp revision), so the join is seamless.

The 10Y AAA government yield is the freely-available official proxy for a 10Y EUR
rate (true 10Y EUR swap rates are not distributed free on the ECB portal).
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import requests

from ecblock.macro.euro_macro import _fetch_series

START = "2010-01-01"  # a little before the timeline window for visual lead-in
TODAY = date.today().isoformat()
OUT = Path("site/macro.json")

ES = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"

# name: (source, key, label, unit, shape, extend_to_today)
# for source "hicp", key is (final_coicop, flash_indic)
SPEC = {
    "deposit_rate":  ("ecb", "FM.B.U2.EUR.4F.KR.DFR.LEV",
                      "ECB deposit facility rate", "%", "step", True),
    "hicp_headline": ("hicp", ("CP00", "TOTAL"),
                      "HICP headline (YoY)", "%", "line", False),
    "hicp_core":     ("hicp", ("TOT_X_NRG_FOOD", "CP-HI00XEF"),
                      "HICP core, ex energy & food (YoY)", "%", "line", False),
    "eur_10y":       ("ecb", "YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y",
                      "Euro area 10Y yield (AAA gov)", "%", "line", False),
}


def _es_series(dataset: str, params: dict) -> dict:
    """Fetch a single Eurostat series -> {period: value}."""
    p = {"format": "JSON", "lang": "EN", "geo": "EA", "sinceTimePeriod": "2009-01"}
    p.update(params)
    j = requests.get(ES + dataset, params=p, timeout=60).json()
    if not j.get("value"):
        return {}
    inv = {v: k for k, v in j["dimension"]["time"]["category"]["index"].items()}
    return {inv[int(i)]: v for i, v in j["value"].items()}


def hicp_series(final_coicop: str, flash_indic: str) -> pd.Series | None:
    """Final HICP (prc_hicp_manr) extended with flash (ei_cphi_m) for months the
    final series has not yet reached."""
    try:
        final = _es_series("prc_hicp_manr", {"freq": "M", "unit": "RCH_A", "coicop": final_coicop})
        flash = _es_series("ei_cphi_m", {"unit": "RT12", "indic": flash_indic})
    except Exception as e:  # noqa: BLE001
        print(f"[hicp] failed {final_coicop}/{flash_indic}: {e}")
        return None
    merged = dict(final)
    added = [m for m in flash if m not in merged]
    for m in added:
        merged[m] = flash[m]
    if added:
        print(f"       (+{len(added)} flash months: {min(added)}..{max(added)})")
    if not merged:
        return None
    s = pd.Series(merged)
    s.index = pd.to_datetime(s.index + "-01")
    return s.sort_index()


def main():
    series_out = {}
    for name, (source, key, label, unit, shape, extend) in SPEC.items():
        s = hicp_series(*key) if source == "hicp" else _fetch_series(key)
        if s is None or not len(s):
            print(f"[skip] {name}: no data")
            continue
        s = s.sort_index()
        anchor = s[s.index < pd.Timestamp(START)]
        anchor_val = float(anchor.iloc[-1]) if len(anchor) else None
        s = s[s.index >= pd.Timestamp(START)]
        if name == "eur_10y":  # daily -> month-end for a clean, light line
            s = s.resample("ME").last().dropna()
        data = [[d.strftime("%Y-%m-%d"), round(float(v), 3)] for d, v in s.items()]
        # anchor the start so step/line series span the whole window
        if anchor_val is not None and (not data or data[0][0] > START):
            data.insert(0, [START, round(anchor_val, 3)])
        # carry a step series (the policy rate) forward to today: an unchanged
        # rate is still the rate in force, so the line should reach the present
        if extend and data and data[-1][0] < TODAY:
            data.append([TODAY, data[-1][1]])
        series_out[name] = {"label": label, "unit": unit, "shape": shape, "data": data}
        print(f"[ok]   {name}: {len(data)} points, {data[0][0]}..{data[-1][0]}, "
              f"last={data[-1][1]} (src={source})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"series": series_out}, ensure_ascii=False), encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
