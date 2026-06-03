"""Euro-area national central bank (NCB) speeches -> Speech records.

The Governing Council is the ECB Executive Board plus the governors of the 20
euro-area NCBs. The Board is covered by the official ECB speeches CSV; the NCB
side comes from the BIS "central bankers' speeches" dataset, a precompiled
full-text English extract of speeches since 1996 (all central banks worldwide).

We download the bulk zip once, then keep only rows whose description names a
euro-area NCB. BIS provides the English text, so these need no translation.

Source: https://www.bis.org/cbspeeches/download.htm  (speeches.zip)
CSV columns: url,title,description,date,text,author
The institution is not a column; it is named inside `description`
("... Governor of the Deutsche Bundesbank, at ..."), so we match on that.
"""
from __future__ import annotations

import csv
import io
import re
import zipfile

import requests

from ..config import RAW
from ..schema import ST_SPEECH, Speech

csv.field_size_limit(10_000_000)

BIS_ZIP = "https://www.bis.org/speeches/speeches.zip"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
CACHE = RAW / "bis"

# canonical institution -> description substrings BIS uses for it
EURO_NCB: dict[str, list[str]] = {
    "Nationale Bank van België / Banque Nationale de Belgique":
        ["National Bank of Belgium", "Banque Nationale de Belgique", "Nationale Bank van Belgi"],
    "Deutsche Bundesbank": ["Deutsche Bundesbank"],
    "Eesti Pank": ["Bank of Estonia", "Eesti Pank"],
    "Central Bank of Ireland": ["Central Bank of Ireland"],
    "Bank of Greece": ["Bank of Greece"],
    "Banco de España": ["Bank of Spain", "Banco de Espa"],
    "Banque de France": ["Banque de France", "Bank of France"],
    "Hrvatska narodna banka": ["Croatian National Bank", "Hrvatska narodna"],
    "Banca d'Italia": ["Bank of Italy", "Banca d'Italia"],
    "Central Bank of Cyprus": ["Central Bank of Cyprus"],
    "Latvijas Banka": ["Bank of Latvia", "Latvijas Banka"],
    "Lietuvos bankas": ["Bank of Lithuania", "Lietuvos bankas"],
    "Banque centrale du Luxembourg": ["Central Bank of Luxembourg", "Banque centrale du Luxembourg"],
    "Central Bank of Malta": ["Central Bank of Malta"],
    "De Nederlandsche Bank": ["De Nederlandsche Bank", "Netherlands Bank"],
    "Oesterreichische Nationalbank":
        ["National Bank of Austria", "Oesterreichische", "Austrian National Bank"],
    "Banco de Portugal": ["Bank of Portugal", "Banco de Portugal"],
    "Banka Slovenije": ["Bank of Slovenia", "Banka Slovenije"],
    "Národná banka Slovenska": ["National Bank of Slovakia"],
    "Suomen Pankki / Bank of Finland": ["Bank of Finland", "Suomen Pankki"],
}
_NCB_RE = {inst: re.compile("|".join(re.escape(p) for p in pats))
           for inst, pats in EURO_NCB.items()}
_WS = re.compile(r"\s+")


def _institution(description: str) -> str | None:
    for inst, rx in _NCB_RE.items():
        if rx.search(description):
            return inst
    return None


def _zip_bytes(use_cache: bool) -> bytes:
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_f = CACHE / "speeches.zip"
    if use_cache and cache_f.exists():
        return cache_f.read_bytes()
    r = requests.get(BIS_ZIP, headers={"User-Agent": UA}, timeout=600)
    r.raise_for_status()
    cache_f.write_bytes(r.content)
    return r.content


def load(use_cache: bool = True, min_chars: int = 400) -> list[Speech]:
    raw = _zip_bytes(use_cache)
    z = zipfile.ZipFile(io.BytesIO(raw))
    out: list[Speech] = []
    for member in z.namelist():
        if not member.endswith(".csv"):
            continue
        reader = csv.DictReader(io.StringIO(z.read(member).decode("utf-8")))
        for row in reader:
            desc = row.get("description") or ""
            inst = _institution(desc)
            if not inst:
                continue
            text = (row.get("text") or "").strip()
            if len(text) < min_chars:
                continue
            speaker = _WS.sub(" ", (row.get("author") or "").strip())
            if not speaker:
                continue
            out.append(Speech(
                date=(row.get("date") or "")[:10],
                speaker=speaker,
                title=(row.get("title") or "").strip(),
                text=text,
                source_type=ST_SPEECH,
                institution=inst,
                source_url=(row.get("url") or "").strip(),
                orig_language="en",
            ))
    return out


if __name__ == "__main__":
    import collections
    sp = load()
    print(f"Loaded {len(sp)} euro-area NCB speeches")
    if sp:
        print("Date range:", min(s.date for s in sp), "..", max(s.date for s in sp))
        ci = collections.Counter(s.institution for s in sp)
        print("\nBy institution:")
        for inst, n in ci.most_common():
            print(f"  {n:4d}  {inst}")
        ca = collections.Counter(s.speaker for s in sp)
        print("\nTop speakers:")
        for name, n in ca.most_common(12):
            print(f"  {n:4d}  {name}")
