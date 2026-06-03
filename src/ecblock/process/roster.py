"""Roster of ECB / euro-area officials, used to drive anonymization.

The roster is the union of:
  - speaker names observed in the built corpus, and
  - a curated seed list of well-known ECB Presidents/Board members and euro-area
    NCB governors whose names recur inside other people's speeches.

We keep this list deliberately broad: the anonymizer needs *every* name that a
judge might recognise, not just the speech's own author. Place/concept names
that collide with surnames are protected by EXCLUSIONS.
"""
from __future__ import annotations

# Curated seed of names likely to appear inside speeches/Q&A. Extend freely.
SEED_OFFICIALS: list[str] = [
    # ECB Presidents
    "Wim Duisenberg", "Jean-Claude Trichet", "Mario Draghi", "Christine Lagarde",
    # ECB Vice-Presidents
    "Christian Noyer", "Lucas Papademos", "Vítor Constâncio", "Luis de Guindos",
    # ECB Executive Board (selected, historical + current)
    "Otmar Issing", "Tommaso Padoa-Schioppa", "Sirkka Hämäläinen", "Eugenio Domingo Solans",
    "Gertrude Tumpel-Gugerell", "José Manuel González-Páramo", "Lorenzo Bini Smaghi",
    "Jürgen Stark", "Peter Praet", "Benoît Cœuré", "Yves Mersch", "Sabine Lautenschläger",
    "Philip Lane", "Isabel Schnabel", "Fabio Panetta", "Frank Elderson", "Piero Cipollone",
    # National central bank governors (euro area, selected recent + notable)
    "Jens Weidmann", "Joachim Nagel", "Axel Weber", "Ernst Welteke",
    "François Villeroy de Galhau", "Klaas Knot", "Ignazio Visco", "Fabio Panetta",
    "Pablo Hernández de Cos", "José Luis Escrivá", "Olli Rehn", "Erkki Liikanen",
    "Yannis Stournaras", "Gabriel Makhlouf", "Philip Lane", "Boštjan Vasle",
    "Pierre Wunsch", "Robert Holzmann", "Ewald Nowotny", "Mário Centeno",
    "Carlos Costa", "Madis Müller", "Gediminas Šimkus", "Mārtiņš Kazāks",
    "Boris Vujčić", "Constantinos Herodotou", "Edward Scicluna", "Peter Kažimír",
    "Gaston Reinesch", "Bostjan Jazbec", "Marko Kranjec",
]

# Strings that look like surnames but must NOT be redacted.
EXCLUSIONS: set[str] = {
    "phillips", "taylor", "lucas", "fisher", "wald", "cos", "lane",
    # ECB-specific concept/place names
    "sintra", "frankfurt", "maastricht", "schengen", "lisbon",
}

TITLES = [
    "President", "Vice-President", "Vice President", "Governor", "Deputy Governor",
    "Chair", "Chairman", "Director", "Board member", "Executive Board member",
    "Mr", "Mr.", "Ms", "Ms.", "Mrs", "Mrs.", "Dr", "Dr.", "Sir", "Madame", "Monsieur",
]


def build_roster(corpus_speakers: list[str]) -> list[str]:
    names = set(SEED_OFFICIALS)
    for s in corpus_speakers:
        if s and s != "ECB council":
            names.add(s)
    return sorted(names)
