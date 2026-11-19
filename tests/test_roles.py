"""F6 tests: roles on a real recent match."""
from __future__ import annotations

from pathlib import Path

import pytest

import pandas as pd

from src.data.splits import holdout_matches, load_tables
from src.selection.pools import candidate_pool
from src.selection.roles import (
    batter_candidates,
    batters_before,
    bowling_options,
    keeper_candidates,
    role_table,
)

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
    # a modern India pool has at least 5 established top-7 batters
    assert rt["is_batter"].sum() >= 5, "expected >=5 batters in a modern India pool"
    assert rt["bowl_type"].isin(["pace", "spin", "unknown"]).all()


def test_batters_subset_of_pool(tables):
    m, xi, dl = tables["matches"], tables["xi"], tables["deliveries"]
    row = _india_match(m)
    pool = candidate_pool("India", row["date"], m, xi)
    bats = batter_candidates(pool, row["date"], m, dl)
    assert bats.issubset(set(pool))
    assert bats  # non-empty for a modern India pool


def test_batter_rule_no_future_leakage(tables):
    """is_batter must be computed from matches strictly BEFORE the target date only.

    Extends the leakage guarantee to the batter rule: every qualifying batter must have actually
    batted before the cutoff — nothing from on/after the target date is read."""
    from src.data.splits import deliveries_before

    m, xi, dl = tables["matches"], tables["xi"], tables["deliveries"]
    early = pd.Timestamp("2010-01-01")
    bats = batters_before(early, m, dl)
    batted_before = set(deliveries_before(early, m, dl)["batter_id"].dropna())
    assert bats.issubset(batted_before)


def _synthetic_innings(match_id, innings, order):
    """One ball per batter, in batting order, for a toy innings."""
    return [
        {"match_id": match_id, "innings": innings, "is_super_over": False,
         "batter_id": pid, "over": pos, "ball": 1}
        for pos, pid in enumerate(order)
    ]


def test_batter_rule_leakage_synthetic():
    """Deterministic, data-free proof: innings dated on/after the target date are never read.

    `late` opens the batting (top-7) in three matches that all fall ON/AFTER the cutoff. With a
    leakage-safe gate he must NOT be classified a batter; `early`, who opened three times before
    the cutoff, must be."""
    import pandas as pd

    matches = pd.DataFrame([
        {"match_id": "m1", "date": pd.Timestamp("2009-01-01")},
        {"match_id": "m2", "date": pd.Timestamp("2009-02-01")},
        {"match_id": "m3", "date": pd.Timestamp("2009-03-01")},
        {"match_id": "m4", "date": pd.Timestamp("2010-06-01")},  # on/after cutoff
        {"match_id": "m5", "date": pd.Timestamp("2010-07-01")},
        {"match_id": "m6", "date": pd.Timestamp("2010-08-01")},
    ])
    rows = []
    for mid in ("m1", "m2", "m3"):          # before cutoff: "early" opens
        rows += _synthetic_innings(mid, 0, ["early", "x2", "x3"])
    for mid in ("m4", "m5", "m6"):          # on/after cutoff: "late" opens
        rows += _synthetic_innings(mid, 0, ["late", "y2", "y3"])
    deliveries = pd.DataFrame(rows)

    bats = batters_before(pd.Timestamp("2010-01-01"), matches, deliveries)
    assert "early" in bats          # 3 top-7 innings before the cutoff
    assert "late" not in bats       # his innings are all in the future -> not read


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
