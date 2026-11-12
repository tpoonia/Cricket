"""Selection arms (§3.4). Each returns an XI (list of player ids) for a team at a match.

Non-VLM arms are fully real (no API). VLM arms (B1, Ours-1/2) and CLIP (B6) take extracted
rules/labels from the VLM client (mock offline) and reuse the same MILP machinery.
"""
from __future__ import annotations

import pandas as pd

from src.conditions.player_types import bowl_type
from src.rules.checker import check
from src.selection.milp import InfeasibleSelection, Player, select_xi
from src.selection.pools import candidate_pool, team_previous_matches
from src.selection.roles import role_table


def build_players(team: str, date, matches, xi, values: dict,
                  keepers_asof: set | None = None) -> list[Player]:
    """Pool players with value, roles and bowl type (the shared input to every MILP arm).

    `keepers_asof` (optional) is a precomputed set of keeper player ids (as-of a fixed date,
    leakage-safe for all holdout matches) to avoid rescanning deliveries per call.
    """
    from src.selection.roles import bowling_options, keeper_candidates

    pool = candidate_pool(team, date, matches, xi)
    dl = _deliveries.get()
    if keepers_asof is not None:
        keepers = {p for p in pool if p in keepers_asof}
    else:
        keepers = keeper_candidates(pool, date, matches, dl)
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
    """Select, relaxing cricket minima to what the pool actually offers (logs implicitly).

    Associate-team pools may lack a detected keeper (no pre-split stumping) or <5 detected
    bowling options; we relax those minima rather than fail.
    """
    mk = 1 if sum(p.is_keeper for p in players) >= 1 else 0
    mb = min(5, sum(p.is_bowling_option for p in players))
    try:
        return select_xi(players, env_rules=env_rules, min_keepers=mk, min_bowling_options=mb)
    except InfeasibleSelection:
        return select_xi(players, env_rules=None, min_keepers=mk, min_bowling_options=mb)


def arm_b3_cricket(players: list[Player]) -> list[str]:
    return _select_relaxing(players).xi


def arm_with_rules(players: list[Player], rules: list[dict]) -> tuple[list[str], list[str]]:
    """MILP + env rules, with feasibility fallback via the checker. Returns (xi, log)."""
    avail = available_counts(players)
    checked = check(rules, mode="rules", available=avail)
    try:
        xi = _select_relaxing(players, env_rules=checked["rules"]).xi
    except InfeasibleSelection:
        xi = _select_relaxing(players).xi  # last-resort: cricket rules only
        checked["log"].append("MILP infeasible even after relaxation; fell back to B3")
    return xi, checked["log"]


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
