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

# Spelling variants in the corpus that refer to one canonical person.
ALIASES = {
    "Philip R Lane": "Philip R. Lane",
    "Jose Luis Escrivá": "José Luis Escrivá",
}


def canon(name: str) -> str:
    return ALIASES.get(name, name)


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
        "former": sorted(FORMER_GC),
        "aliases": ALIASES,
    }
