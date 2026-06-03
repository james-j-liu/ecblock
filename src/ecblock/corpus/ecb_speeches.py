"""Load ECB Executive Board speeches from the official CSV dataset.

https://www.ecb.europa.eu/press/key/shared/data/all_ECB_speeches.csv
Pipe-delimited, UTF-8 (no BOM), columns: date|speakers|title|subtitle|contents
Executive Board members only.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd
import requests

from ..config import RAW, cfg
from ..schema import ST_SPEECH, Speech

CSV_PATH = RAW / "all_ECB_speeches.csv"


def download(force: bool = False) -> Path:
    if CSV_PATH.exists() and not force:
        return CSV_PATH
    url = cfg()["corpus"]["ecb_speeches_csv_url"]
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    # many of these servers reject the default python-requests UA (esp. from CI)
    r = requests.get(url, headers={"User-Agent":
                     "Mozilla/5.0 (compatible; ECBLock/1.0; +https://github.com/james-j-liu/ecblock)"},
                     timeout=120)
    r.raise_for_status()
    CSV_PATH.write_bytes(r.content)
    return CSV_PATH


_INST = "European Central Bank"


def _clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def load(force_download: bool = False) -> list[Speech]:
    path = download(force=force_download)
    df = pd.read_csv(path, sep="|", dtype=str, engine="python",
                     quoting=csv.QUOTE_NONE, on_bad_lines="warn")
    df = df.fillna("")
    out: list[Speech] = []
    for _, row in df.iterrows():
        contents = _clean(row.get("contents", ""))
        if len(contents) < 200:  # skip presentation-only / empty stubs
            continue
        # speakers may be comma-separated; take the primary
        speaker = _clean(row.get("speakers", "")).split(",")[0].strip()
        if not speaker:
            continue
        out.append(
            Speech(
                date=_clean(row.get("date", ""))[:10],
                speaker=speaker,
                title=_clean(row.get("title", "")),
                text=contents,
                source_type=ST_SPEECH,
                institution=_INST,
                orig_language="en",
            )
        )
    return out


if __name__ == "__main__":
    sp = load()
    print(f"Loaded {len(sp)} ECB Executive Board speeches")
    if sp:
        by_speaker: dict[str, int] = {}
        for s in sp:
            by_speaker[s.speaker] = by_speaker.get(s.speaker, 0) + 1
        print(f"{len(by_speaker)} distinct speakers")
        print("Date range:", min(s.date for s in sp), "to", max(s.date for s in sp))
        for name, c in sorted(by_speaker.items(), key=lambda x: -x[1])[:10]:
            print(f"  {c:4d}  {name}")
