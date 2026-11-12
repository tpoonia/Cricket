"""F5: candidate pool per team per match.

Pool = players who appeared for that team in its previous 10 T20Is (earlier matches only),
capped at 18 by most-recent appearance. "Pool coverage" = % of the real XI inside the pool.
"""
from __future__ import annotations

import pandas as pd

from src.data.splits import history_before

POOL_LOOKBACK_MATCHES = 10
POOL_MAX = 18


def team_previous_matches(
    team: str, target_date, matches: pd.DataFrame, n: int = POOL_LOOKBACK_MATCHES
) -> pd.DataFrame:
    """The team's most recent `n` matches strictly before target_date."""
    hist = history_before(target_date, matches)
    played = hist[(hist["team1"] == team) | (hist["team2"] == team)]
    return played.sort_values("date", ascending=False).head(n)


def candidate_pool(
    team: str,
    target_date,
    matches: pd.DataFrame,
    xi: pd.DataFrame,
    n: int = POOL_LOOKBACK_MATCHES,
    cap: int = POOL_MAX,
) -> list[str]:
    """Player ids eligible for `team` at `target_date`, newest appearance first, capped."""
    prev = team_previous_matches(team, target_date, matches, n)
    if prev.empty:
        return []
    order = {mid: i for i, mid in enumerate(prev["match_id"])}  # 0 = most recent
    rows = xi[(xi["match_id"].isin(order)) & (xi["team"] == team)].copy()
    rows["recency"] = rows["match_id"].map(order)
    # earliest recency (most recent match) wins for ordering; dedupe players
    rows = rows.dropna(subset=["player_id"]).sort_values("recency")
    seen: dict[str, int] = {}
    for pid in rows["player_id"]:
        if pid not in seen:
            seen[pid] = len(seen)
        if len(seen) >= cap:
            break
    return list(seen.keys())


def pool_coverage(
    target_matches: pd.DataFrame, matches: pd.DataFrame, xi: pd.DataFrame
) -> dict:
    """Over a set of matches, the fraction of each real XI that lies inside the computed pool."""
    covered = 0
    total = 0
    per_match = []
    for _, mm in target_matches.iterrows():
        for team in (mm["team1"], mm["team2"]):
            pool = set(candidate_pool(team, mm["date"], matches, xi))
            real = xi[(xi["match_id"] == mm["match_id"]) & (xi["team"] == team)]
            real_ids = set(real["player_id"].dropna())
            if not real_ids:
                continue
            c = len(real_ids & pool)
            covered += c
            total += len(real_ids)
            per_match.append(c / len(real_ids))
    return {
        "coverage": covered / total if total else 0.0,
        "n_team_matches": len(per_match),
        "mean_per_xi": float(pd.Series(per_match).mean()) if per_match else 0.0,
    }
