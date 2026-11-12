"""F7 tests: player value, EB shrinkage, leakage-safety."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.splits import load_tables
from src.selection.value import compute_values

PROCESSED = Path("data/processed")
pytestmark = pytest.mark.skipif(
    not (PROCESSED / "matches.parquet").exists(), reason="run ETL first"
)


@pytest.fixture(scope="module")
def tables():
    return load_tables()


def test_value_is_sum_of_components(tables):
    v = compute_values(pd.Timestamp("2025-01-01"), tables["matches"], tables["deliveries"])
    assert ((v["bat_value"] + v["bowl_value"] - v["value"]).abs() < 1e-9).all()


def test_top_batter_positive(tables):
    v = compute_values(pd.Timestamp("2025-01-01"), tables["matches"], tables["deliveries"])
    xi = tables["xi"]
    kohli = xi[xi["player_name"] == "V Kohli"]["player_id"].iloc[0]
    assert v.loc[kohli, "bat_value"] > 5  # a genuinely strong T20I batter


def test_leakage_future_player_absent(tables):
    """A player whose first appearance is after the cutoff must not have a value."""
    m, xi = tables["matches"], tables["xi"]
    early = pd.Timestamp("2006-01-01")
    v = compute_values(early, m, tables["deliveries"])
    # everyone in v must have batted/bowled before 2006 -> ids are few
    assert len(v) < 400  # only very early players
    # and no id appears that only exists in later matches
    from src.data.splits import deliveries_before

    valid_ids = set(deliveries_before(early, m, tables["deliveries"])["batter_id"].dropna()) | set(
        deliveries_before(early, m, tables["deliveries"])["bowler_id"].dropna()
    )
    assert set(v.index).issubset(valid_ids)
