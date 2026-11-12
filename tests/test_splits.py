"""F3 tests: chronological split + leakage guard (Hard rule #1)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.splits import (
    SPLIT_DATE,
    assert_no_leakage,
    deliveries_before,
    history_before,
    holdout_matches,
    load_tables,
    train_matches,
)

PROCESSED = Path("data/processed")
pytestmark = pytest.mark.skipif(
    not (PROCESSED / "matches.parquet").exists(), reason="run ETL first"
)


@pytest.fixture(scope="module")
def tables():
    return load_tables()


def test_split_boundary(tables):
    m = tables["matches"]
    assert train_matches(m)["date"].max() < SPLIT_DATE
    assert holdout_matches(m)["date"].min() >= SPLIT_DATE
    assert len(train_matches(m)) + len(holdout_matches(m)) == len(m)


def test_history_before_is_strict(tables):
    m = tables["matches"]
    target = pd.Timestamp("2025-06-01")
    hist = history_before(target, m)
    assert hist["date"].max() < target
    # strictness: a match exactly on the target date is excluded
    assert (hist["date"] >= target).sum() == 0


def test_deliveries_before_excludes_super_over(tables):
    m, dl = tables["matches"], tables["deliveries"]
    target = pd.Timestamp("2024-01-01")
    d = deliveries_before(target, m, dl)
    assert not d["is_super_over"].any()
    hist_ids = set(history_before(target, m)["match_id"])
    assert set(d["match_id"]).issubset(hist_ids)


def test_leakage_guard_raises(tables):
    m = tables["matches"]
    target = pd.Timestamp("2020-01-01")
    # Pass a future match id as if a feature had used it -> must raise.
    future_id = m[m["date"] >= target].iloc[0]["match_id"]
    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert_no_leakage([future_id], target, m)


def test_leakage_guard_passes_for_valid_history(tables):
    m = tables["matches"]
    target = pd.Timestamp("2020-01-01")
    past_ids = history_before(target, m)["match_id"].tolist()
    assert_no_leakage(past_ids, target, m)  # must not raise
