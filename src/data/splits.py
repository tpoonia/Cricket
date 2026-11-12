"""F3: chronological split + leakage-safe history access.

Every feature, venue prior, player value and simulator multiplier MUST pull its input
through `history_before(target_date, ...)` so it can only ever see matches strictly before
the match being evaluated (Hard rule #1). `assert_no_leakage` is the guard a test uses.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROCESSED = Path("data/processed")
SPLIT_DATE = pd.Timestamp("2025-01-01")  # train/features < this; test (Set R) >= this


def load_tables(processed: str | Path = PROCESSED) -> dict[str, pd.DataFrame]:
    processed = Path(processed)
    out = {}
    for name in ("matches", "xi", "deliveries"):
        out[name] = pd.read_parquet(processed / f"{name}.parquet")
    out["matches"]["date"] = pd.to_datetime(out["matches"]["date"])
    return out


def _as_ts(date) -> pd.Timestamp:
    return pd.Timestamp(date)


def train_matches(matches: pd.DataFrame, split: pd.Timestamp = SPLIT_DATE) -> pd.DataFrame:
    return matches[matches["date"] < split]


def holdout_matches(matches: pd.DataFrame, split: pd.Timestamp = SPLIT_DATE) -> pd.DataFrame:
    """Set R candidates: matches on/after the split date."""
    return matches[matches["date"] >= split]


def history_before(target_date, matches: pd.DataFrame) -> pd.DataFrame:
    """Matches strictly before `target_date`. The '<' (not '<=') is the no-leakage boundary."""
    return matches[matches["date"] < _as_ts(target_date)]


def deliveries_before(
    target_date,
    matches: pd.DataFrame,
    deliveries: pd.DataFrame,
    include_super_over: bool = False,
) -> pd.DataFrame:
    """Delivery rows from matches strictly before `target_date` (super overs excluded by default)."""
    hist_ids = set(history_before(target_date, matches)["match_id"])
    dl = deliveries[deliveries["match_id"].isin(hist_ids)]
    if not include_super_over:
        dl = dl[~dl["is_super_over"]]
    return dl


def assert_no_leakage(used_match_ids, target_date, matches: pd.DataFrame) -> None:
    """Raise if any used match is dated on/after the target match (leakage)."""
    target = _as_ts(target_date)
    used = matches[matches["match_id"].isin(set(used_match_ids))]
    bad = used[used["date"] >= target]
    if len(bad):
        raise AssertionError(
            f"LEAKAGE: {len(bad)} match(es) dated on/after {target.date()} used as history "
            f"(e.g. {bad.iloc[0]['match_id']} @ {bad.iloc[0]['date'].date()})"
        )
