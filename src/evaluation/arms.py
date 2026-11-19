"""Selection arms (§3.4). Each returns an XI (list of player ids) for a team at a match.

Non-VLM arms are fully real (no API). VLM arms (B1, Ours-1/2) and CLIP (B6) take extracted
rules/labels from the VLM client (mock offline) and reuse the same MILP machinery.
"""
from __future__ import annotations

import pandas as pd

from src.conditions.player_types import bowl_type
from src.rules.checker import check
from src.selection.milp import MIN_BATTERS, InfeasibleSelection, Player, select_xi
from src.selection.pools import candidate_pool, team_previous_matches
from src.selection.roles import role_table


def build_players(team: str, date, matches, xi, values: dict,
                  keepers_asof: set | None = None,
                  batters_asof: set | None = None) -> list[Player]:
    """Pool players with value, roles and bowl type (the shared input to every MILP arm).

    `keepers_asof` / `batters_asof` (optional) are precomputed sets of keeper / batter player ids
    (as-of a fixed date, leakage-safe for all holdout matches) to avoid rescanning deliveries per
    call. When not supplied, both are computed per `date` through the leakage-safe gate.
    """
    from src.selection.roles import batter_candidates, bowling_options, keeper_candidates

    pool = candidate_pool(team, date, matches, xi)
    dl = _deliveries.get()
    if keepers_asof is not None:
        keepers = {p for p in pool if p in keepers_asof}
    else:
        keepers = keeper_candidates(pool, date, matches, dl)
    if batters_asof is not None:
        batters = {p for p in pool if p in batters_asof}
    else:
        batters = batter_candidates(pool, date, matches, dl)
    options = bowling_options(team, pool, date, matches, dl)
    players = []
    for pid in pool:
        players.append(
            Player(
                pid=pid,
                value=float(values.get(pid, 0.0)),
                is_keeper=pid in keepers,
                is_bowling_option=pid in options,
                bowl_type=bowl_type(pid),
                is_batter=pid in batters,
            )
        )
    return players


class _DeliveriesHolder:
    """Avoid threading the big deliveries frame through every call."""
    _dl = None

    def set(self, dl):
        self._dl = dl

    def get(self):
        return self._dl


_deliveries = _DeliveriesHolder()


def available_counts(players: list[Player]) -> dict:
    return {
        "n_pace": sum(p.bowl_type == "pace" for p in players),
        "n_spin": sum(p.bowl_type == "spin" for p in players),
    }


# ---- arm selectors ----
def arm_b2_ml_only(players: list[Player]) -> list[str]:
    """Top 11 by value, ignoring cricket rules."""
    return [p.pid for p in sorted(players, key=lambda z: z.value, reverse=True)[:11]]


def _select_relaxing(players: list[Player], env_rules=None):
    """Select, relaxing cricket minima to what the pool actually offers (logged in `.relaxations`).

    Associate-team pools may lack a detected keeper (no pre-split stumping), <5 detected bowling
    options, or <5 detected batters; we relax those minima to what the pool holds rather than fail,
    and record each relaxation on the returned SelectionResult.
    """
    n_keeper = sum(p.is_keeper for p in players)
    n_bowl = sum(p.is_bowling_option for p in players)
    n_bat = sum(p.is_batter for p in players)
    mk = 1 if n_keeper >= 1 else 0
    mb = min(5, n_bowl)
    mbat = min(MIN_BATTERS, n_bat)
    relax = []
    if mk < 1:
        relax.append("keeper rule relaxed: no keeper detected in pool")
    if mb < 5:
        relax.append(f"bowling-option rule relaxed 5->{mb} (pool has {n_bowl})")
    if mbat < MIN_BATTERS:
        relax.append(f"batter rule relaxed {MIN_BATTERS}->{mbat} (pool has {n_bat})")
    try:
        res = select_xi(players, env_rules=env_rules, min_keepers=mk,
                        min_bowling_options=mb, min_batters=mbat)
    except InfeasibleSelection:
        res = select_xi(players, env_rules=None, min_keepers=mk,
                        min_bowling_options=mb, min_batters=mbat)
        relax.append("env rules dropped: infeasible with cricket rules alone")
    res.relaxations = relax + list(res.relaxations)
    return res


def arm_b3_cricket(players: list[Player]) -> list[str]:
    return _select_relaxing(players).xi


def arm_with_rules(players: list[Player], rules: list[dict]) -> tuple[list[str], list[str]]:
    """MILP + env rules, with feasibility fallback via the checker. Returns (xi, log)."""
    avail = available_counts(players)
    checked = check(rules, mode="rules", available=avail)
    try:
        res = _select_relaxing(players, env_rules=checked["rules"])
    except InfeasibleSelection:
        res = _select_relaxing(players)  # last-resort: cricket rules only
        checked["log"].append("MILP infeasible even after relaxation; fell back to B3")
    checked["log"].extend(res.relaxations)
    return res.xi, checked["log"]


def perturb_rule(rules: list[dict], swap: bool = False, delta: int = 0) -> list[dict]:
    """B7 noisy rules: optionally swap pace<->spin and/or shift bounds by delta."""
    out = []
    for r in rules:
        feat = r["feature"]
        if swap:
            feat = "n_spin" if feat == "n_pace" else "n_pace"
        lo = max(0, min(11, r["lo"] + delta))
        hi = max(lo, min(11, r["hi"] + delta))
        out.append({"feature": feat, "lo": lo, "hi": hi})
    return out


def habitual_spin_count(team: str, date, matches, xi) -> int:
    """Mean number of spinners in the team's last 10 real XIs (rounded)."""
    prev = team_previous_matches(team, date, matches, 10)
    counts = []
    for mid in prev["match_id"]:
        ids = xi[(xi["match_id"] == mid) & (xi["team"] == team)]["player_id"].dropna()
        counts.append(sum(bowl_type(p) == "spin" for p in ids))
    return int(round(sum(counts) / len(counts))) if counts else 0


def arm_habit(players: list[Player], team: str, date, matches, xi) -> list[str]:
    h = habitual_spin_count(team, date, matches, xi)
    rules = [{"feature": "n_spin", "lo": h, "hi": h}]
    return arm_with_rules(players, rules)[0]


def real_xi(team: str, match_id: str, xi) -> list[str]:
    return xi[(xi["match_id"] == match_id) & (xi["team"] == team)]["player_id"].dropna().tolist()
