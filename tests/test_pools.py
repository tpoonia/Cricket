"""F5 tests: candidate pool + coverage."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.splits import holdout_matches, load_tables
from src.selection.pools import POOL_MAX, candidate_pool, pool_coverage

PROCESSED = Path("data/processed")
pytestmark = pytest.mark.skipif(
    not (PROCESSED / "matches.parquet").exists(), reason="run ETL first"
)


@pytest.fixture(scope="module")
def tables():
    return load_tables()


def test_pool_is_capped_and_from_history(tables):
    m, xi = tables["matches"], tables["xi"]
    # pick a recent India match in the holdout set
    hm = holdout_matches(m)
    row = hm[(hm["team1"] == "India") | (hm["team2"] == "India")].iloc[0]
    pool = candidate_pool("India", row["date"], m, xi)
    assert 0 < len(pool) <= POOL_MAX
    assert len(pool) == len(set(pool))  # no duplicates


def test_pool_excludes_future(tables):
    m, xi = tables["matches"], tables["xi"]
    hm = holdout_matches(m)
    row = hm[(hm["team1"] == "India") | (hm["team2"] == "India")].iloc[0]
    pool = set(candidate_pool("India", row["date"], m, xi))
    # none of the pool players may come *only* from a future match: check ids all appear
    # in some earlier India match
    from src.data.splits import history_before

    hist_ids = set(history_before(row["date"], m)["match_id"])
    earlier_india = set(
        xi[(xi["match_id"].isin(hist_ids)) & (xi["team"] == "India")]["player_id"].dropna()
    )
    assert pool.issubset(earlier_india)


def test_pool_coverage_reasonable(tables):
    m, xi = tables["matches"], tables["xi"]
    hm = holdout_matches(m)
    sample = hm.head(60)
    cov = pool_coverage(sample, m, xi)
    # last-10-match pool should recover most of a real XI
    assert cov["coverage"] > 0.7, cov
