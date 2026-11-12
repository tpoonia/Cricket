"""M5 tests: condition-aware + planted simulators."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.data.splits import SPLIT_DATE, load_tables
from src.simulation.condition import (
    PLANTED,
    fit_condition_multipliers,
    win_probability_planted,
)
from src.simulation.sim import build_rates, win_probability
from src.simulation.validate import eligible_holdout, evaluate

PROCESSED = Path("data/processed")
pytestmark = pytest.mark.skipif(
    not (PROCESSED / "matches.parquet").exists(), reason="run ETL first"
)


@pytest.fixture(scope="module")
def ctx():
    t = load_tables()
    tb = build_rates(SPLIT_DATE, t["matches"], t["deliveries"])
    cond = fit_condition_multipliers(t["matches"], t["deliveries"], SPLIT_DATE)
    return t, tb, cond


def _xis(t, m):
    xi = t["xi"]
    t1 = xi[(xi["match_id"] == m["match_id"]) & (xi["team"] == m["team1"])]["player_id"].dropna().tolist()
    t2 = xi[(xi["match_id"] == m["match_id"]) & (xi["team"] == m["team2"])]["player_id"].dropna().tolist()
    return t1, t2


def test_planted_flat_equals_blind(ctx):
    t, tb, _ = ctx
    m = eligible_holdout(t["matches"], min_prior_matches=10).iloc[0]
    t1, t2 = _xis(t, m)
    blind = win_probability(t1, t2, 1, tb, n=2000, seed=1)
    flat = win_probability_planted(t1, t2, 1, tb, "flat", n=2000, seed=1)
    assert flat == pytest.approx(blind)  # empty multipliers -> identical


def test_planted_dry_changes_prediction(ctx):
    t, tb, _ = ctx
    m = eligible_holdout(t["matches"], min_prior_matches=10).iloc[0]
    t1, t2 = _xis(t, m)
    blind = win_probability(t1, t2, 1, tb, n=3000, seed=1)
    dry = win_probability_planted(t1, t2, 1, tb, "dry", n=3000, seed=1)
    assert dry != blind  # a spin-boosting pitch shifts the prediction
    assert "dry" in PLANTED and PLANTED["dry"]


def test_condition_aware_beats_coin_flip_and_matches_blind(ctx):
    t, tb, cond = ctx
    from src.simulation.condition import win_probability_ca

    sample = eligible_holdout(t["matches"], min_prior_matches=10).head(80)
    blind = evaluate(tb, t["matches"], t["xi"], sample, n=1500)
    ca = evaluate(tb, t["matches"], t["xi"], sample, n=1500,
                  win_prob_fn=win_probability_ca, pass_venue=True, cond_mult=cond)
    assert ca["brier"] < 0.25 and ca["auc"] > 0.5
    # at least as good as blind within Monte-Carlo noise
    assert ca["brier"] <= blind["brier"] + 0.01
    assert ca["auc"] >= blind["auc"] - 0.02
