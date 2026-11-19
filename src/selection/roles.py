"""F6: player roles — keeper, bowling option, batter, pace/spin.

Rules (DECISIONS D6, logged):
  Keeper        = a player credited with >=1 stumping (as the dismissing fielder) in earlier
                  matches. Stumpings uniquely identify the wicket-keeper.
  Bowling option= bowled >= 6 legal deliveries (1 over) in >= 2 of the team's last 10 matches
                  (earlier matches only).
  Batter        = batted in the top 7 in >= 50% of his earlier innings, with >= 3 innings
                  (earlier matches only; a player with < 3 innings is not counted as a batter).
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

# Batter rule thresholds (fixed): top-7 position, >=50% of innings, >=3 innings minimum.
BATTER_TOP_N = 7
BATTER_MIN_SHARE = 0.5
BATTER_MIN_INNINGS = 3


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


def _innings_batting_position(dl: pd.DataFrame) -> pd.DataFrame:
    """Batting position of every batter in every innings of `dl`.

    No position column exists in the ball-by-ball data, so position is inferred from the order
    in which batters first FACE a ball within an innings: the batter who faces the first ball is
    #1, the next new batter to face is #2, and so on. A batter cannot face a ball before an
    earlier-order batter who is still at the crease, so this recovers the batting order.

    Returns a frame with columns [match_id, innings, batter_id, position].
    """
    d = dl.dropna(subset=["batter_id"])[["match_id", "innings", "batter_id", "over", "ball"]].copy()
    if d.empty:
        return pd.DataFrame(columns=["match_id", "innings", "batter_id", "position"])
    # a monotone key for "when in the innings" the ball occurred (ball can exceed 6 with extras)
    d["ball_order"] = d["over"].astype("int64") * 1000 + d["ball"].astype("int64")
    first = (d.groupby(["match_id", "innings", "batter_id"], as_index=False)["ball_order"].min())
    first["position"] = (
        first.groupby(["match_id", "innings"])["ball_order"].rank(method="first").astype(int)
    )
    return first[["match_id", "innings", "batter_id", "position"]]


def batters_before(
    target_date, matches: pd.DataFrame, deliveries: pd.DataFrame,
    top_n: int = BATTER_TOP_N, min_share: float = BATTER_MIN_SHARE,
    min_innings: int = BATTER_MIN_INNINGS,
) -> set[str]:
    """All player ids that qualify as batters using ONLY matches strictly before target_date.

    A player is a batter if he batted in the top `top_n` in at least `min_share` of his earlier
    innings, with at least `min_innings` innings. Leakage-safe: inputs come through
    `deliveries_before` (the same gate every other feature uses)."""
    dl = deliveries_before(target_date, matches, deliveries)
    pos = _innings_batting_position(dl)
    if pos.empty:
        return set()
    pos["top"] = pos["position"] <= top_n
    agg = pos.groupby("batter_id").agg(innings=("position", "size"), top=("top", "sum"))
    agg = agg[agg["innings"] >= min_innings]
    qualifies = agg[(agg["top"] / agg["innings"]) >= min_share]
    return set(qualifies.index)


def batter_candidates(
    pool_ids: list[str], target_date, matches: pd.DataFrame, deliveries: pd.DataFrame
) -> set[str]:
    """Pool players who qualify as batters before target_date (subset of `batters_before`)."""
    batters = batters_before(target_date, matches, deliveries)
    return {p for p in pool_ids if p in batters}


def role_table(
    team: str, pool_ids: list[str], target_date, matches: pd.DataFrame, deliveries: pd.DataFrame
) -> pd.DataFrame:
    """One row per pool player: is_keeper, is_bowling_option, is_batter, bowl_type."""
    keepers = keeper_candidates(pool_ids, target_date, matches, deliveries)
    options = bowling_options(team, pool_ids, target_date, matches, deliveries)
    batters = batter_candidates(pool_ids, target_date, matches, deliveries)
    rows = []
    for pid in pool_ids:
        rows.append(
            {
                "player_id": pid,
                "is_keeper": pid in keepers,
                "is_bowling_option": pid in options,
                "is_batter": pid in batters,
                "bowl_type": bowl_type(pid),
            }
        )
    return pd.DataFrame(rows).set_index("player_id")
