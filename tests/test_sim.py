"""F9 tests: blind simulator sanity + beats coin flip on held-out matches."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.data.splits import SPLIT_DATE, load_tables
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
    return t, tb


def test_common_random_numbers(ctx):
    t, tb = ctx
    xi = t["xi"]
    m = eligible_holdout(t["matches"], min_prior_matches=10).iloc[0]
    t1 = xi[(xi["match_id"] == m["match_id"]) & (xi["team"] == m["team1"])]["player_id"].dropna().tolist()
    t2 = xi[(xi["match_id"] == m["match_id"]) & (xi["team"] == m["team2"])]["player_id"].dropna().tolist()
    p_a = win_probability(t1, t2, 1, tb, n=1500, seed=42)
    p_b = win_probability(t1, t2, 1, tb, n=1500, seed=42)
    assert p_a == p_b  # identical seed -> identical result (CRN)
    assert 0.0 <= p_a <= 1.0


def test_beats_coin_flip(ctx):
    t, tb = ctx
    sample = eligible_holdout(t["matches"], min_prior_matches=10).head(80)
    res = evaluate(tb, t["matches"], t["xi"], sample, n=1500)
    assert res["n_matches"] >= 60
    assert res["brier"] < 0.25, res
    assert res["auc"] > 0.5, res
