"""M5: condition-aware and planted simulators (built on the blind F9 machinery).

Condition-aware: per-venue {pace, spin} multipliers on wicket chance and boundary runs,
fitted on an earlier window than the venue priors (train era, strictly before the holdout),
shrunk toward 1.0 so sparse venues stay neutral.

Planted (Set S): fixed multipliers keyed by the synthetic scenario's TRUE pitch class, e.g.
dry -> spin wickets +15%, pace -5% (DECISIONS; used only on synthetic images).
"""
from __future__ import annotations

import pandas as pd

from src.conditions.player_types import load_player_types
from src.data.splits import SPLIT_DATE, deliveries_before
from src.simulation.sim import win_probability

SHRINK_BALLS = 3500.0  # pseudo-balls toward multiplier 1.0 (gentle; venue effect ~cancels for win-prob)
BOWLER_WICKETS = {"bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"}

# Planted multipliers by true pitch class (Set S). Keys: (type, "wkt"|"run").
PLANTED = {
    "dry": {("spin", "wkt"): 1.15, ("pace", "wkt"): 0.95,
            ("spin", "run"): 0.95, ("pace", "run"): 1.03},
    "slow": {("spin", "wkt"): 1.12, ("pace", "wkt"): 0.97,
             ("spin", "run"): 0.95, ("pace", "run"): 1.0},
    "green": {("pace", "wkt"): 1.15, ("spin", "wkt"): 0.92,
              ("pace", "run"): 0.97, ("spin", "run"): 1.03},
    "flat": {},
    "wet": {},
}


def _typed_legal(dl: pd.DataFrame) -> pd.DataFrame:
    types = load_player_types()["bowl_type"]
    legal = dl[~dl["extra_type"].isin(["wides", "noballs"])].copy()
    legal = legal.dropna(subset=["bowler_id"])
    legal["btype"] = legal["bowler_id"].map(lambda p: types.get(p, "unknown"))
    return legal[legal["btype"] != "unknown"]


def fit_condition_multipliers(
    matches: pd.DataFrame, deliveries: pd.DataFrame, fit_before=SPLIT_DATE
) -> dict[str, dict]:
    """Return {venue -> {(type, 'wkt'|'run'): multiplier}} fitted before `fit_before`."""
    dl = deliveries_before(fit_before, matches, deliveries)
    legal = _typed_legal(dl)
    legal = legal.merge(matches[["match_id", "venue"]], on="match_id", how="left")

    out: dict[str, dict] = {}
    for btype in ("pace", "spin"):
        sub = legal[legal["btype"] == btype]
        g_balls = len(sub)
        g_wkt = sub["wicket_kind"].isin(BOWLER_WICKETS).sum()
        g_runs = sub["runs_batter"].sum()
        g_wkt_rate = g_wkt / g_balls if g_balls else 0.0
        g_run_rate = g_runs / g_balls if g_balls else 0.0
        for venue, vg in sub.groupby("venue"):
            n = len(vg)
            v_wkt_rate = vg["wicket_kind"].isin(BOWLER_WICKETS).sum() / n
            v_run_rate = vg["runs_batter"].sum() / n
            # shrink the RATE toward the global rate, then take the ratio -> multiplier
            w = n / (n + SHRINK_BALLS)
            wkt_mult = (w * v_wkt_rate + (1 - w) * g_wkt_rate) / g_wkt_rate if g_wkt_rate else 1.0
            run_mult = (w * v_run_rate + (1 - w) * g_run_rate) / g_run_rate if g_run_rate else 1.0
            d = out.setdefault(venue, {})
            d[(btype, "wkt")] = float(wkt_mult)
            d[(btype, "run")] = float(run_mult)
    return out


def win_probability_ca(team1_xi, team2_xi, bat_first, tb, venue=None,
                       cond_mult: dict | None = None, n=10000, seed=0) -> float:
    """Condition-aware win probability for a real match at `venue`."""
    vm = (cond_mult or {}).get(venue, {}) if venue is not None else {}
    return win_probability(team1_xi, team2_xi, bat_first, tb, n=n, seed=seed, venue_mult=vm)


def win_probability_planted(team1_xi, team2_xi, bat_first, tb, pitch_class,
                            n=10000, seed=0) -> float:
    """Planted-simulator win probability for a synthetic scenario of a known class."""
    vm = PLANTED.get(pitch_class, {})
    return win_probability(team1_xi, team2_xi, bat_first, tb, n=n, seed=seed, venue_mult=vm)
