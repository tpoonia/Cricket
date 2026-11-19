"""Offline demo backend: one squad + three condition images -> three XIs with extracted rules.

Uses only cached/mock outputs (no live API). Importable without Streamlit so it can be tested.

Import safety: the MILP selector depends on PuLP, which is not installed in every
interpreter (e.g. the system Anaconda python3). The PuLP-dependent import
(``src.evaluation.arms``) is therefore done lazily inside the functions that need it, so
this module — and the PuLP-free helpers below (``win_probability``, ``demo_conditions``) —
import cleanly everywhere. ``HAS_SOLVER`` reports whether live selection is available.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.data.splits import SPLIT_DATE, holdout_matches, load_tables
from src.rules.checker import rules_for_label
from src.selection.value import compute_values
from src.vlm.client import VLMClient

ROOT = Path(__file__).resolve().parent.parent
SET_R_PATH = ROOT / "data" / "results" / "raw" / "set_r_results.json"

DEMO_CONDITIONS = [
    ("green", "pitch_green_00"),
    ("dry", "pitch_dry_00"),
    ("flat", "pitch_flat_00"),
]

# Arms used for the "win probability, before and after the rule" comparison.
# These are recorded condition-aware simulator means (data/results/raw/set_r_results.json),
# never live or invented numbers.
WIN_PROB_ARMS = {
    "before": "B3_cricket",   # cricket rules only (no environment rule)
    "after": "Ours1_label",   # + the rulebook's environment constraint (our pipeline)
    "wrong": "B7_swap",       # reference: a deliberately wrong pace<->spin rule
}

try:  # live MILP selection needs PuLP; the demo stays useful without it.
    import pulp  # noqa: F401

    HAS_SOLVER = True
except Exception:  # pragma: no cover - environment dependent
    HAS_SOLVER = False


def _arms():
    """Lazy import of the PuLP-dependent selection arms (raises if PuLP is missing)."""
    from src.evaluation import arms as A

    return A


@lru_cache(maxsize=1)
def _ctx():
    A = _arms()
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
    """Return {condition: {...}} for the three demo conditions.

    Each condition carries the perceived label, the extracted rulebook rules, and the
    environment-rule XI (``xi_names``/``n_pace``/``n_spin``) plus — added for the demo's
    before/after view — the cricket-only baseline XI (``base_*``). Requires PuLP.
    """
    A = _arms()
    t, values, names = _ctx()
    m, xi = t["matches"], t["xi"]
    match = _latest_match(team)
    if match is None:
        return {}
    players = A.build_players(team, match["date"], m, xi, values)
    base_xi = A.arm_b3_cricket(players)  # cricket rules only — the "before the rule" XI
    client = VLMClient(backend=backend)
    out = {"team": team, "date": str(match["date"])[:10], "conditions": {}}
    for label, img_id in DEMO_CONDITIONS:
        scn = {"id": f"demo_{img_id}", "pitch_class": label, "sky_class": "clear",
               "has_image": True, "image_path": f"data/synthetic/images/{img_id}.png"}
        runs = client.run(scn, mode="label", k=5)
        from collections import Counter

        perceived = Counter(r["pitch"] for r in runs if r.get("schema_ok")).most_common(1)[0][0]
        entry = rules_for_label(perceived)
        rules = entry["rules"]
        axi, log = A.arm_with_rules(players, rules)
        out["conditions"][label] = {
            "perceived": perceived,
            "strength": entry.get("strength"),
            "rules": rules,
            "xi_names": [names.get(p, p) for p in axi],
            "xi_types": [A.bowl_type(p) for p in axi],
            "n_pace": sum(A.bowl_type(p) == "pace" for p in axi),
            "n_spin": sum(A.bowl_type(p) == "spin" for p in axi),
            # cricket-only baseline (the XI "before" the environment rule is applied)
            "base_xi_names": [names.get(p, p) for p in base_xi],
            "base_xi_types": [A.bowl_type(p) for p in base_xi],
            "base_n_pace": sum(A.bowl_type(p) == "pace" for p in base_xi),
            "base_n_spin": sum(A.bowl_type(p) == "spin" for p in base_xi),
            "changed": sorted(set(axi) ^ set(base_xi)) != [],
            "log": log,
        }
    return out


@lru_cache(maxsize=16)
def win_probability(team: str) -> dict:
    """Recorded condition-aware win probability, before vs after the environment rule.

    PuLP-free: reads the frozen simulator results in data/results/raw/set_r_results.json
    and averages ``p_win_ca`` per arm for ``team`` (falling back to all teams if the team
    has no rows). Returns means for the cricket-only baseline (before), our rule-aware
    pipeline (after) and, as a reference, a deliberately wrong rule.
    """
    rows = json.loads(SET_R_PATH.read_text())
    team_rows = [r for r in rows if r.get("team") == team]
    scope_rows, scope = (team_rows, team) if team_rows else (rows, "all teams")

    def _mean(arm: str) -> dict | None:
        ar = [r for r in scope_rows if r.get("arm") == arm]
        if not ar:
            return None
        return {
            "p_win": round(sum(r["p_win_ca"] for r in ar) / len(ar), 4),
            "n": len(ar),
        }

    out = {"team": team, "scope": scope, "bars": {}}
    for role, arm in WIN_PROB_ARMS.items():
        m = _mean(arm)
        if m is not None:
            out["bars"][role] = {"arm": arm, **m}
    return out


def perceive(label: str, image_id: str | None = None, backend: str = "mock") -> dict:
    """PuLP-free: run the (default: offline mock) VLM reader on a demo condition image.

    Returns the perceived pitch label and a confidence — "the AI names the condition".
    No network/API: the mock reader is deterministic and offline.
    """
    from collections import Counter

    img = image_id or f"pitch_{label}_00"
    scn = {"id": f"demo_{img}", "pitch_class": label, "sky_class": "clear",
           "has_image": True, "image_path": f"data/synthetic/images/{img}.png"}
    runs = VLMClient(backend=backend).run(scn, mode="label", k=5)
    ok = [r for r in runs if r.get("schema_ok")]
    perceived = Counter(r["pitch"] for r in ok).most_common(1)[0][0] if ok else label
    conf = round(sum(r.get("confidence", 0.0) for r in ok) / len(ok), 2) if ok else 0.0
    return {"perceived": perceived, "confidence": conf, "k": len(runs), "backend": backend}


def demo_conditions() -> list[dict]:
    """PuLP-free: the three demo conditions with their frozen-rulebook rules + strength.

    Lets the UI show the perception label and the rule bounds even where the MILP solver
    (PuLP) is unavailable.
    """
    out = []
    for label, img_id in DEMO_CONDITIONS:
        entry = rules_for_label(label)
        out.append({
            "label": label,
            "image_id": img_id,
            "image_path": f"data/synthetic/images/{img_id}.png",
            "strength": entry.get("strength"),
            "rules": entry.get("rules", []),
        })
    return out
