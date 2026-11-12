"""Offline demo backend: one squad + three condition images -> three XIs with extracted rules.

Uses only cached/mock outputs (no live API). Importable without Streamlit so it can be tested.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from src.data.splits import SPLIT_DATE, holdout_matches, load_tables
from src.evaluation import arms as A
from src.rules.checker import rules_for_label
from src.selection.value import compute_values
from src.vlm.client import VLMClient

DEMO_CONDITIONS = [
    ("green", "pitch_green_00"),
    ("dry", "pitch_dry_00"),
    ("flat", "pitch_flat_00"),
]


@lru_cache(maxsize=1)
def _ctx():
    t = load_tables()
    A._deliveries.set(t["deliveries"])
    values = compute_values(SPLIT_DATE, t["matches"], t["deliveries"])["value"].to_dict()
    names = (t["xi"].dropna(subset=["player_id"])
             .drop_duplicates("player_id").set_index("player_id")["player_name"].to_dict())
    return t, values, names


def demo_teams(n: int = 12) -> list[str]:
    t, _, _ = _ctx()
    hm = holdout_matches(t["matches"])
    return sorted(pd.concat([hm["team1"], hm["team2"]]).value_counts().head(n).index.tolist())


def _latest_match(team: str):
    t, _, _ = _ctx()
    hm = holdout_matches(t["matches"])
    sub = hm[(hm["team1"] == team) | (hm["team2"] == team)].sort_values("date")
    return sub.iloc[-1] if len(sub) else None


def three_xis(team: str, backend: str = "mock") -> dict:
    """Return {condition: {rules, xi_names, n_pace, n_spin}} for the three demo conditions."""
    t, values, names = _ctx()
    m, xi = t["matches"], t["xi"]
    match = _latest_match(team)
    if match is None:
        return {}
    players = A.build_players(team, match["date"], m, xi, values)
    client = VLMClient(backend=backend)
    out = {"team": team, "date": str(match["date"])[:10], "conditions": {}}
    for label, img_id in DEMO_CONDITIONS:
        scn = {"id": f"demo_{img_id}", "pitch_class": label, "sky_class": "clear",
               "has_image": True, "image_path": f"data/synthetic/images/{img_id}.png"}
        runs = client.run(scn, mode="label", k=5)
        from collections import Counter

        perceived = Counter(r["pitch"] for r in runs if r.get("schema_ok")).most_common(1)[0][0]
        rules = rules_for_label(perceived)["rules"]
        axi, log = A.arm_with_rules(players, rules)
        out["conditions"][label] = {
            "perceived": perceived,
            "rules": rules,
            "xi_names": [names.get(p, p) for p in axi],
            "n_pace": sum(A.bowl_type(p) == "pace" for p in axi),
            "n_spin": sum(A.bowl_type(p) == "spin" for p in axi),
            "log": log,
        }
    return out
