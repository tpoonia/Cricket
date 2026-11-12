"""F7: player value from earlier matches only, empirical-Bayes shrunk.

v_i = batting value + bowling value, where
  batting value  = (shrunk mean batting runs per match) - population mean      [runs above avg/match]
  bowling value  = (population economy - shrunk economy) * balls bowled per match  [runs saved/match]

Shrinkage (EB-style) uses a fixed prior weight k (DECISIONS D7, k=10 "pseudo-matches";
bowling uses k*24 pseudo-balls = k four-over spells). All inputs come from
`deliveries_before(target_date, ...)` so there is no leakage.
"""
from __future__ import annotations

import pandas as pd

from src.data.splits import deliveries_before

K_MATCHES = 10
BALLS_PER_SPELL = 24  # a bowler's typical 4-over quota
# Extras charged to the bowler for economy (byes/legbyes are not the bowler's fault).
BOWLER_CHARGED_EXTRAS = {"wides", "noballs"}


def compute_values(
    target_date,
    matches: pd.DataFrame,
    deliveries: pd.DataFrame,
    k: int = K_MATCHES,
) -> pd.DataFrame:
    """Return a DataFrame indexed by player_id with bat_value, bowl_value, value."""
    dl = deliveries_before(target_date, matches, deliveries)
    if dl.empty:
        return pd.DataFrame(columns=["bat_value", "bowl_value", "value"]).rename_axis("player_id")

    # ---- Batting: runs per match ----
    bat = dl.dropna(subset=["batter_id"]).groupby("batter_id").agg(
        runs=("runs_batter", "sum"),
        matches=("match_id", "nunique"),
    )
    bat["pm"] = bat["runs"] / bat["matches"]
    pop_bat = bat["runs"].sum() / bat["matches"].sum()  # runs per player-match, population
    bat["shrunk"] = (bat["matches"] * bat["pm"] + k * pop_bat) / (bat["matches"] + k)
    bat["bat_value"] = bat["shrunk"] - pop_bat

    # ---- Bowling: economy -> runs saved per match ----
    dl = dl.copy()
    dl["charged"] = dl["runs_batter"] + dl["runs_extras"].where(
        dl["extra_type"].isin(BOWLER_CHARGED_EXTRAS), 0
    )
    dl["legal"] = (~dl["extra_type"].isin(BOWLER_CHARGED_EXTRAS)).astype(int)
    bowl = dl.dropna(subset=["bowler_id"]).groupby("bowler_id").agg(
        conceded=("charged", "sum"),
        balls=("legal", "sum"),
        matches=("match_id", "nunique"),
    )
    bowl = bowl[bowl["balls"] > 0]
    pop_econ = bowl["conceded"].sum() / bowl["balls"].sum()  # runs per legal ball, population
    pseudo = k * BALLS_PER_SPELL
    bowl["econ"] = bowl["conceded"] / bowl["balls"]
    bowl["econ_shrunk"] = (bowl["conceded"] + pseudo * pop_econ) / (bowl["balls"] + pseudo)
    bowl["bpm"] = bowl["balls"] / bowl["matches"]
    bowl["bowl_value"] = (pop_econ - bowl["econ_shrunk"]) * bowl["bpm"]

    # ---- Combine ----
    out = pd.DataFrame(index=sorted(set(bat.index) | set(bowl.index)))
    out.index.name = "player_id"
    out["bat_value"] = bat["bat_value"].reindex(out.index).fillna(0.0)
    out["bowl_value"] = bowl["bowl_value"].reindex(out.index).fillna(0.0)
    out["value"] = out["bat_value"] + out["bowl_value"]
    return out


def values_for_pool(
    pool_ids: list[str], target_date, matches, deliveries, k: int = K_MATCHES
) -> dict[str, float]:
    """value per pool player (0.0 for a player with no prior record)."""
    vals = compute_values(target_date, matches, deliveries, k)
    return {pid: float(vals["value"].get(pid, 0.0)) for pid in pool_ids}
