"""F8 + §3.1#9: MILP XI selector (PuLP + HiGHS).

Pick exactly 11 players from the pool maximising total value, subject to cricket rules
(>=1 keeper, >=5 bowling options) and optional environment rules on pace/spin counts.

Environment rule schema (from the rule checker): a list of
  {"feature": "n_pace"|"n_spin", "lo": int, "hi": int}
added as  lo <= sum(x_i for players of that type) <= hi.

`select_xi` returns a SelectionResult with the chosen ids, the realised counts, and the
objective value. Legality is also independently re-checkable via `verify_xi` (Hard rule #8).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pulp

MIN_KEEPERS = 1
MIN_BOWLING_OPTIONS = 5
XI_SIZE = 11


@dataclass
class Player:
    pid: str
    value: float
    is_keeper: bool
    is_bowling_option: bool
    bowl_type: str  # "pace" | "spin" | "unknown"


@dataclass
class SelectionResult:
    xi: list[str]
    objective: float
    n_pace: int
    n_spin: int
    n_keeper: int
    n_bowling_option: int
    status: str
    relaxations: list[str] = field(default_factory=list)


def _type_members(players: list[Player], feature: str) -> list[Player]:
    kind = "pace" if feature == "n_pace" else "spin"
    return [p for p in players if p.bowl_type == kind]


def select_xi(
    players: list[Player],
    env_rules: list[dict] | None = None,
    min_keepers: int = MIN_KEEPERS,
    min_bowling_options: int = MIN_BOWLING_OPTIONS,
) -> SelectionResult:
    """Solve the MILP. env_rules is a validated list of {feature, lo, hi}."""
    env_rules = env_rules or []
    prob = pulp.LpProblem("xi_selection", pulp.LpMaximize)
    x = {p.pid: pulp.LpVariable(f"x_{p.pid}", cat="Binary") for p in players}

    # Objective: maximise total value
    prob += pulp.lpSum(p.value * x[p.pid] for p in players)

    # Exactly 11
    prob += pulp.lpSum(x.values()) == XI_SIZE, "xi_size"
    # >=1 keeper
    prob += pulp.lpSum(x[p.pid] for p in players if p.is_keeper) >= min_keepers, "keepers"
    # >=5 bowling options
    prob += (
        pulp.lpSum(x[p.pid] for p in players if p.is_bowling_option) >= min_bowling_options,
        "bowling_options",
    )
    # Environment rules
    for i, r in enumerate(env_rules):
        members = _type_members(players, r["feature"])
        s = pulp.lpSum(x[p.pid] for p in members)
        if r.get("lo") is not None:
            prob += (s >= r["lo"], f"env_{i}_lo")
        if r.get("hi") is not None:
            prob += (s <= r["hi"], f"env_{i}_hi")

    status = _solve(prob)
    if pulp.LpStatus[status] != "Optimal":
        raise InfeasibleSelection(
            f"MILP not optimal (status={pulp.LpStatus[status]}); "
            f"pool={len(players)} rules={env_rules}"
        )

    chosen = [p for p in players if x[p.pid].value() > 0.5]
    return _result(chosen, pulp.value(prob.objective), pulp.LpStatus[status])


class InfeasibleSelection(Exception):
    pass


def _solve(prob: pulp.LpProblem) -> int:
    # Prefer HiGHS (fast); fall back to the bundled CBC.
    try:
        solver = pulp.HiGHS(msg=False)
    except Exception:
        solver = pulp.PULP_CBC_CMD(msg=False)
    return prob.solve(solver)


def _result(chosen: list[Player], objective: float, status: str,
            relaxations: list[str] | None = None) -> SelectionResult:
    return SelectionResult(
        xi=[p.pid for p in chosen],
        objective=float(objective),
        n_pace=sum(p.bowl_type == "pace" for p in chosen),
        n_spin=sum(p.bowl_type == "spin" for p in chosen),
        n_keeper=sum(p.is_keeper for p in chosen),
        n_bowling_option=sum(p.is_bowling_option for p in chosen),
        status=status,
        relaxations=relaxations or [],
    )


def verify_xi(
    xi_ids: list[str],
    players: list[Player],
    env_rules: list[dict] | None = None,
    min_keepers: int = MIN_KEEPERS,
    min_bowling_options: int = MIN_BOWLING_OPTIONS,
) -> dict:
    """Independently recompute legality + value from ids (never trust the solver's claim)."""
    env_rules = env_rules or []
    by_id = {p.pid: p for p in players}
    sel = [by_id[i] for i in xi_ids if i in by_id]
    n_pace = sum(p.bowl_type == "pace" for p in sel)
    n_spin = sum(p.bowl_type == "spin" for p in sel)
    checks = {
        "size_ok": len(sel) == XI_SIZE and len(set(xi_ids)) == XI_SIZE,
        "all_in_pool": all(i in by_id for i in xi_ids),
        "keeper_ok": sum(p.is_keeper for p in sel) >= min_keepers,
        "bowling_ok": sum(p.is_bowling_option for p in sel) >= min_bowling_options,
    }
    for r in env_rules:
        cnt = n_pace if r["feature"] == "n_pace" else n_spin
        if r.get("lo") is not None:
            checks[f"{r['feature']}>={r['lo']}"] = cnt >= r["lo"]
        if r.get("hi") is not None:
            checks[f"{r['feature']}<={r['hi']}"] = cnt <= r["hi"]
    return {
        "legal": all(checks.values()),
        "checks": checks,
        "n_pace": n_pace,
        "n_spin": n_spin,
        "value": float(sum(p.value for p in sel)),
    }
