"""F6: player roles — keeper, bowling option, pace/spin.

Rules (DECISIONS D6, logged):
  Keeper        = a player credited with >=1 stumping (as the dismissing fielder) in earlier
                  matches. Stumpings uniquely identify the wicket-keeper.
  Bowling option= bowled >= 6 legal deliveries (1 over) in >= 2 of the team's last 10 matches
                  (earlier matches only).
  Pace/spin     = from F4 player_types.
"""
from __future__ import annotations

import pandas as pd

from src.conditions.player_types import bowl_type
from src.data.splits import deliveries_before
from src.selection.pools import team_previous_matches

MIN_BALLS = 6
MIN_MATCHES_AS_BOWLER = 2
LOOKBACK = 10


def keeper_candidates(
    pool_ids: list[str], target_date, matches: pd.DataFrame, deliveries: pd.DataFrame
) -> set[str]:
    """Pool players with stumping evidence before target_date."""
    dl = deliveries_before(target_date, matches, deliveries)
    st = dl[dl["wicket_kind"] == "stumped"].dropna(subset=["fielders"])
    keepers: set[str] = set()
    for fids in st["fielders"]:
        for pid in str(fids).split(","):
            keepers.add(pid)
    return {p for p in pool_ids if p in keepers}


def bowling_options(
    team: str,
    pool_ids: list[str],
    target_date,
    matches: pd.DataFrame,
    deliveries: pd.DataFrame,
    min_balls: int = MIN_BALLS,
    min_matches: int = MIN_MATCHES_AS_BOWLER,
    lookback: int = LOOKBACK,
) -> set[str]:
    """Pool players who bowled >= min_balls legal deliveries in >= min_matches of the team's
    last `lookback` matches (earlier only)."""
    prev = team_previous_matches(team, target_date, matches, lookback)
    prev_ids = set(prev["match_id"])
    dl = deliveries[deliveries["match_id"].isin(prev_ids) & (~deliveries["is_super_over"])]
    legal = dl[~dl["extra_type"].isin(["wides", "noballs"])]
    # legal balls per (bowler, match)
    by = legal.groupby(["bowler_id", "match_id"]).size()
    qualifying = by[by >= min_balls].reset_index()
    counts = qualifying.groupby("bowler_id").size()
    options = set(counts[counts >= min_matches].index)
    return {p for p in pool_ids if p in options}


def role_table(
    team: str, pool_ids: list[str], target_date, matches: pd.DataFrame, deliveries: pd.DataFrame
) -> pd.DataFrame:
    """One row per pool player: is_keeper, is_bowling_option, bowl_type."""
    keepers = keeper_candidates(pool_ids, target_date, matches, deliveries)
    options = bowling_options(team, pool_ids, target_date, matches, deliveries)
    rows = []
    for pid in pool_ids:
        rows.append(
            {
                "player_id": pid,
                "is_keeper": pid in keepers,
                "is_bowling_option": pid in options,
                "bowl_type": bowl_type(pid),
            }
        )
    return pd.DataFrame(rows).set_index("player_id")
