r"""Player-name matching across sources.

Sackmann: full names ("Juan Martin del Potro"). tennis-data.co.uk: "Surname I." ("Del Potro J.M.",
"Ramos-Vinolas A."). Kalshi: full names in yes_sub_title ("Alexander Zverev").
Strategy: normalise (strip accents, lowercase, hyphens -> spaces), then build keys
(surname, first initial) for every split of a Sackmann name, and look up candidates; a
name can map to several ids (e.g. two "Kuznetsov A."), and the join then disambiguates
on the opponent and the date window.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from collections import defaultdict


def Normalise(name):
    if not isinstance(name, str):
        return ""
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii").lower()
    text = text.replace("-", " ").replace("'", "").replace("`", "")
    text = re.sub(r"[^a-z. ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def ParseTennisDataName(name):
    """'Del Potro J.M.' -> ('del potro', 'j'); 'Federer R.' -> ('federer', 'r')."""
    text = Normalise(name)
    tokens = text.split(" ")
    initials = []
    while tokens and re.fullmatch(r"([a-z]\.)+|[a-z]", tokens[-1]):
        initials.insert(0, tokens.pop())
    surname = " ".join(t.strip(".") for t in tokens).strip()
    first_initial = initials[0][0] if initials else ""
    return surname, first_initial


def SackmannKeys(full_name):
    tokens = [t.strip(".") for t in Normalise(full_name).split(" ") if t.strip(".")]
    keys = set()
    if not tokens:
        return keys
    first = tokens[0][0]
    for k in range(1, len(tokens)):
        surname = " ".join(tokens[k:])
        keys.add((surname, first))
        keys.add((surname, ""))
    if len(tokens) >= 2:  # east-asian order or swapped: surname first
        keys.add((tokens[0], tokens[1][0]))
    return keys


class NameIndex:
    def __init__(self, id_name_pairs):
        self.by_key = defaultdict(set)
        self.by_full = defaultdict(set)
        self.names = {}
        for pid, name in id_name_pairs:
            if not isinstance(name, str):
                continue
            self.names[pid] = name
            self.by_full[Normalise(name)].add(pid)
            for key in SackmannKeys(name):
                self.by_key[key].add(pid)
        self.surnames_by_initial = defaultdict(set)
        for (surname, initial) in self.by_key:
            if initial:
                self.surnames_by_initial[initial].add(surname)

    def LookupFull(self, full_name):
        """Kalshi-style full names: exact normalised, then surname+initial keys."""
        exact = self.by_full.get(Normalise(full_name))
        if exact:
            return set(exact)
        tokens = Normalise(full_name).split(" ")
        if len(tokens) >= 2:
            hits = self.by_key.get((" ".join(tokens[1:]), tokens[0][0])) or self.by_key.get((tokens[-1], tokens[0][0]))
            if hits:
                return set(hits)
        return set()

    def LookupTennisData(self, name):
        surname, initial = ParseTennisDataName(name)
        if not surname:
            return set()
        for key in ((surname, initial), (surname.split(" ")[0], initial), (surname.split(" ")[-1], initial)):
            hits = self.by_key.get(key)
            if hits:
                return set(hits)
        close = difflib.get_close_matches(surname, self.surnames_by_initial.get(initial, ()), n=2, cutoff=0.85)
        out = set()
        for match in close:
            out |= self.by_key[(match, initial)]
        return out