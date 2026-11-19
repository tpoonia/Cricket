"""F8 tests: MILP selector on toy squads."""
from __future__ import annotations

import pytest

from src.evaluation.arms import _select_relaxing
from src.selection.milp import (
    MIN_BATTERS,
    InfeasibleSelection,
    Player,
    select_xi,
    verify_xi,
)


def make_pool(n_batters: int = 7):
    """18-player toy pool: 6 pace, 5 spin, 7 non-bowlers; keepers among non-bowlers.

    `n_batters` controls how many of the 7 non-bowlers are flagged as batters (default 7),
    so a caller can build a pool that cannot satisfy MIN_BATTERS=5.
    """
    players = []
    # high-value batters (non-bowlers), one is keeper
    for i in range(7):
        players.append(
            Player(f"bat{i}", value=10 - i, is_keeper=(i in (0, 1)),
                   is_bowling_option=False, bowl_type="unknown", is_batter=(i < n_batters))
        )
    for i in range(6):
        players.append(
            Player(f"pace{i}", value=5 - i * 0.3, is_keeper=False,
                   is_bowling_option=True, bowl_type="pace")
        )
    for i in range(5):
        players.append(
            Player(f"spin{i}", value=4 - i * 0.3, is_keeper=False,
                   is_bowling_option=True, bowl_type="spin")
        )
    return players


def test_basic_legal_xi():
    res = select_xi(make_pool())
    assert len(res.xi) == 11
    assert res.n_keeper >= 1
    assert res.n_bowling_option >= 5
    assert res.n_batter >= MIN_BATTERS
    v = verify_xi(res.xi, make_pool())
    assert v["legal"], v["checks"]


def test_batter_rule_enforced():
    """Every legal XI must contain >= MIN_BATTERS batters, and verify_xi re-checks it."""
    pool = make_pool()
    res = select_xi(pool)
    assert res.n_batter >= MIN_BATTERS
    v = verify_xi(res.xi, pool)
    assert v["checks"]["batter_ok"]
    assert v["n_batter"] >= MIN_BATTERS
    # An XI that swaps a batter out for a non-batter bowler fails the batter check.
    batters_in = [i for i in res.xi if i.startswith("bat")]
    non_in = [p.pid for p in pool if not p.is_batter and p.pid not in res.xi]
    tampered = [i for i in res.xi if i != batters_in[0]] + [non_in[0]]
    v2 = verify_xi(tampered, pool)
    if sum(p.is_batter for p in pool if p.pid in tampered) < MIN_BATTERS:
        assert not v2["checks"]["batter_ok"]


def test_batter_rule_infeasible_when_too_few():
    """A pool with < MIN_BATTERS batters cannot satisfy the rule without relaxation."""
    pool = make_pool(n_batters=3)
    with pytest.raises(InfeasibleSelection):
        select_xi(pool)  # needs 5 batters, only 3 exist


def test_batter_rule_relaxed_and_logged():
    """When the pool has too few batters, selection relaxes the minimum and logs it."""
    pool = make_pool(n_batters=3)
    res = _select_relaxing(pool)
    assert len(res.xi) == 11
    assert res.n_batter == 3  # took every available batter
    assert any("batter rule relaxed" in r for r in res.relaxations), res.relaxations


def test_objective_prefers_high_value():
    res = select_xi(make_pool())
    # bat0 (value 10) must be selected
    assert "bat0" in res.xi


def test_env_rule_min_spin_changes_xi():
    pool = make_pool()
    base = select_xi(pool)
    forced = select_xi(pool, env_rules=[{"feature": "n_spin", "lo": 3, "hi": 11}])
    assert forced.n_spin >= 3
    assert forced.n_spin >= base.n_spin
    assert verify_xi(forced.xi, pool, env_rules=[{"feature": "n_spin", "lo": 3, "hi": 11}])["legal"]
    # forcing extra spin cannot increase the objective
    assert forced.objective <= base.objective + 1e-6


def test_env_rule_max_pace():
    pool = make_pool()
    res = select_xi(pool, env_rules=[{"feature": "n_pace", "lo": 0, "hi": 1}])
    assert res.n_pace <= 1


def test_infeasible_raises():
    pool = make_pool()
    # impossible: need >=6 spin but also >=6 pace while only 11 slots and >=1 keeper
    with pytest.raises(InfeasibleSelection):
        select_xi(
            pool,
            env_rules=[
                {"feature": "n_spin", "lo": 6, "hi": 11},
                {"feature": "n_pace", "lo": 6, "hi": 11},
            ],
        )


def test_verify_rejects_tampered_xi():
    pool = make_pool()
    res = select_xi(pool)
    bad = res.xi[:-1] + ["spin0" if "spin0" not in res.xi else "pace0"]
    # duplicate injection -> size check fails
    bad = res.xi[:-1] + [res.xi[0]]
    assert not verify_xi(bad, pool)["legal"]
