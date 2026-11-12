"""F6 tests: roles on a real recent match."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.data.splits import holdout_matches, load_tables
from src.selection.pools import candidate_pool
from src.selection.roles import bowling_options, keeper_candidates, role_table

PROCESSED = Path("data/processed")
pytestmark = pytest.mark.skipif(
    not (PROCESSED / "matches.parquet").exists(), reason="run ETL first"
)


@pytest.fixture(scope="module")
def tables():
    return load_tables()


def _india_match(m):
    hm = holdout_matches(m)
    return hm[(hm["team1"] == "India") | (hm["team2"] == "India")].iloc[0]


def test_roles_present_and_sane(tables):
    m, xi, dl = tables["matches"], tables["xi"], tables["deliveries"]
    row = _india_match(m)
    pool = candidate_pool("India", row["date"], m, xi)
    rt = role_table("India", pool, row["date"], m, dl)
    assert len(rt) == len(pool)
    # India pools in the modern era contain at least one keeper and >=5 bowling options
    assert rt["is_keeper"].sum() >= 1, "expected a keeper in a modern India pool"
    assert rt["is_bowling_option"].sum() >= 5, "expected >=5 bowling options"
    assert rt["bowl_type"].isin(["pace", "spin", "unknown"]).all()


def test_keeper_has_stumping_evidence(tables):
    m, xi, dl = tables["matches"], tables["xi"], tables["deliveries"]
    row = _india_match(m)
    pool = candidate_pool("India", row["date"], m, xi)
    keepers = keeper_candidates(pool, row["date"], m, dl)
    assert keepers  # non-empty for India


def test_bowling_options_subset_of_pool(tables):
    m, xi, dl = tables["matches"], tables["xi"], tables["deliveries"]
    row = _india_match(m)
    pool = candidate_pool("India", row["date"], m, xi)
    opts = bowling_options("India", pool, row["date"], m, dl)
    assert opts.issubset(set(pool))
