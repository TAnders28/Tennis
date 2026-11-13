r"""Surface lookup for upcoming events from tournament names (Kalshi gives no surface).

Uses the most recent Sackmann surface for the best token-overlap tourney_name, plus a small
manual table for season-end events. Returns (surface, source); source "guess_default_hard"
means nothing matched and the model used Hard - shown in the PredictDay table.
"""
from __future__ import annotations

import re

MANUAL = {"atp finals": "Hard", "world tour finals": "Hard", "wta finals": "Hard", "next gen": "Hard", "davis cup": "Hard",
          "billie jean king": "Hard", "united cup": "Hard", "laver cup": "Hard",
          "us open men": "Hard", "us open women": "Hard", "french open": "Clay", "roland garros": "Clay", "wimbledon": "Grass",
          "australian open": "Hard"}
DROP = {"atp", "wta", "challenger", "open", "championships", "international", "men", "women", "singles", "the", "de", "of",
        "tour", "masters", "125", "250", "500", "1000", "cup", "presented", "by"}


def Tokens(name):
    return {t for t in re.sub(r"[^a-z0-9 ]", " ", str(name).lower()).split() if t not in DROP and len(t) > 2}


class SurfaceLookup:
    def __init__(self, frame):
        recent = frame.sort_values("date").drop_duplicates("tourney_name", keep="last")
        self.entries = [(Tokens(n), s) for n, s in zip(recent["tourney_name"], recent["surface_norm"]) if s != "Unknown"]

    def Get(self, competition):
        text = str(competition or "").lower()
        for key, surface in MANUAL.items():
            if key in text:
                return surface, "manual_table"
        tokens = Tokens(text)
        best, best_score = None, 0
        for entry_tokens, surface in self.entries:
            score = len(tokens & entry_tokens)
            if score > best_score:
                best, best_score = surface, score
        if best:
            return best, "sackmann_name_match"
        return "Hard", "guess_default_hard"