"""Governing Council membership roster (single source of truth).

The corpus is assembled from the BIS speech database, which publishes *every*
speech from each euro-area national central bank. That sweeps in NCB deputy
governors, board members, and the occasional non-euro official who spoke at an
NCB event - none of whom sit on (or vote in) the ECB Governing Council.

The Governing Council is exactly: the six Executive Board members plus the
governors of the euro-area national central banks. Only these people (and the
"ECB council" composite) belong in the scoring system. This module is the
authoritative membership list; the pipeline filters the scoring pool through
`is_gc`, and the site builds its Rankings "Members" filter from `to_dict()`
(emitted into data.json meta), so the two never drift.

CURRENT = ECB Executive Board + euro-area NCB governors in office as of May 2026
(source: ecb.europa.eu Governing Council page). FORMER = people who were
Executive Board members or NCB governors during 2010-2026 but have since left.
"""
from __future__ import annotations

ECB_COUNCIL = "ECB council"

CURRENT_GC = {
    # Executive Board
    "Christine Lagarde", "Luis de Guindos", "Piero Cipollone", "Frank Elderson",
    "Philip R. Lane", "Isabel Schnabel",
    # NCB governors
    "Pierre Wunsch", "Dimitar Radev", "Joachim Nagel", "Madis Müller",
    "Gabriel Makhlouf", "Yannis Stournaras", "José Luis Escrivá",
    "François Villeroy de Galhau", "Boris Vujčić", "Fabio Panetta",
    "Christodoulos Patsalides", "Mārtiņš Kazāks", "Gediminas Šimkus",
    "Gaston Reinesch", "Alexander Demarco", "Olaf Sleijpen", "Martin Kocher",
    "Álvaro Santos Pereira", "Primož Dolenc", "Peter Kažimír", "Olli Rehn",
}

FORMER_GC = {
    # former Executive Board
    "Jean-Claude Trichet", "Mario Draghi", "Vítor Constâncio",
    "José Manuel González-Páramo", "Gertrude Tumpel-Gugerell",
    "Lorenzo Bini Smaghi", "Jürgen Stark", "Yves Mersch", "Jörg Asmussen",
    "Benoît Cœuré", "Peter Praet", "Sabine Lautenschläger",
    # former NCB governors
    "Axel A Weber", "Jens Weidmann", "Christian Noyer", "Ignazio Visco",
    "Miguel Fernández Ordóñez", "Luis M Linde", "Pablo Hernández de Cos",
    "Klaas Knot", "Patrick Honohan", "George A Provopoulos",
    "Carlos da Silva Costa", "Mário Centeno", "Ewald Nowotny", "Robert Holzmann",
    "Luc Coene", "Jan Smets", "Erkki Liikanen", "Athanasios Orphanides",
    "Constantinos Herodotou", "Josef Bonnici", "Edward Scicluna", "Mario Vella",
    "Boštjan Vasle", "Bostjan Jazbec", "Andres Lipstok", "Ardo Hansson",
    "Ilmārs Rimšēvičs", "Vitas Vasiliauskas",
}

# Spelling variants that fuzzy matching (see _fold) can't resolve on its own.
ALIASES = {
    "Philip R Lane": "Philip R. Lane",
    "Jose Luis Escrivá": "José Luis Escrivá",
}

# ---- live membership --------------------------------------------------------
# The static sets above go stale whenever a governor changes (four did between May
# and Sept 2026), and anyone missing from the roster is silently dropped from the
# scoring pool. sync_from_ecb() reads the ECB's Governing Council page and records
# the live membership in a small committed state file; on import that state
# overrides CURRENT_GC, and people who drop off the page move to FORMER_GC.
import datetime as _dt
import html as _html
import json as _json
import re as _re
import unicodedata as _ud
from pathlib import Path as _Path

STATE = _Path(__file__).resolve().parents[2] / "data" / "processed" / "roster_state.json"
GC_PAGE = "https://www.ecb.europa.eu/ecb/decisions/govc/html/index.en.html"
_STATIC_CURRENT = frozenset(CURRENT_GC)
ROLES: dict[str, str] = {}      # current member -> "Governor, Eesti Pank" etc.


def _fold(name: str) -> str:
    """Accent/case/initial/title-insensitive key: 'Mr Philip R Lane' == 'Philip R. Lane'."""
    n = "".join(c for c in _ud.normalize("NFKD", name) if not _ud.combining(c)).lower()
    n = _re.sub(r"\b(?:mr|mrs|ms|dr|prof)\b\.?", " ", n)
    return " ".join(t for t in _re.sub(r"[^a-z]+", " ", n).split() if len(t) > 1)


def _apply_state() -> None:
    try:
        st = _json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    live = st.get("current") or {}
    if len(live) < 20:                       # never trust a partial parse
        return
    FORMER_GC.update((_STATIC_CURRENT | set(st.get("former", []))) - set(live))
    CURRENT_GC.clear()
    CURRENT_GC.update(live)
    ROLES.clear()
    ROLES.update(live)
    _rebuild_index()


def _rebuild_index() -> None:
    global _INDEX
    _INDEX = {_fold(n): n for n in (CURRENT_GC | FORMER_GC)}
    _INDEX.update({_fold(a): c for a, c in ALIASES.items()})


def sync_from_ecb(timeout: int = 30) -> list[str]:
    """Refresh roster_state.json from the ECB page; return human-readable changes."""
    import requests
    r = requests.get(GC_PAGE, timeout=timeout,
                     headers={"User-Agent": "Mozilla/5.0 (compatible; ECBLock/1.0)"})
    r.raise_for_status()
    # one member card = name, then the member's portrait, then their role. Matching
    # the card (not role wording - Lithuania's is "Chairman of the Board") keeps page
    # headings out and every member in.
    card = (r'<div class="title">([^<]+)</div></div><div class="content-box">'
            r'<div data-image="[^"]*/shared/img/[^"]+"></div><p>([^<]+)</p>')
    members = {_html.unescape(n).strip(): _html.unescape(r_).strip()
               for n, r_ in _re.findall(card, r.text)}
    if not 20 <= len(members) <= 40:
        raise RuntimeError(f"GC page parse found {len(members)} members; not updating")
    try:
        old = _json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = {"current": {n: "" for n in _STATIC_CURRENT}, "former": []}
    prev = set(old.get("current", {}))
    changes = [f"joined: {n} ({members[n]})" for n in sorted(set(members) - prev)]
    changes += [f"left: {n}" for n in sorted(prev - set(members))]
    former = sorted(set(old.get("former", [])) | (prev - set(members)))
    STATE.write_text(_json.dumps({"fetched": _dt.date.today().isoformat(),
                                  "current": dict(sorted(members.items())),
                                  "former": former}, ensure_ascii=False, indent=1),
                     encoding="utf-8")
    _apply_state()
    return changes


def canon(name: str) -> str:
    n = ALIASES.get(name, name)
    if n in CURRENT_GC or n in FORMER_GC or n == ECB_COUNCIL:
        return n
    return _INDEX.get(_fold(n), n)


def is_current_gc(name: str) -> bool:
    n = canon(name)
    return n == ECB_COUNCIL or n in CURRENT_GC


def is_gc(name: str) -> bool:
    """True for any Governing Council member (current or former) and the council."""
    n = canon(name)
    return n == ECB_COUNCIL or n in CURRENT_GC or n in FORMER_GC


def to_dict() -> dict:
    """Roster payload for the site (embedded in data.json meta)."""
    return {
        "current": sorted(CURRENT_GC),
        "former": sorted(FORMER_GC - CURRENT_GC),
        "aliases": ALIASES,
    }


_rebuild_index()
_apply_state()
