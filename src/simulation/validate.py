"""Validate a simulator's win-probability calibration on held-out matches.

Compares Brier score and ROC-AUC against the coin-flip baselines (0.25 / 0.5).
Used by F9 (blind) and M5 (condition-aware).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data.splits import SPLIT_DATE, holdout_matches, load_tables
from src.simulation.sim import SimTables, bat_first_team, build_rates, win_probability


def _prior_match_counts(matches: pd.DataFrame) -> pd.Series:
    """Number of pre-split matches each team played (for the 'established teams' filter)."""
    train = matches[matches["date"] < SPLIT_DATE]
    teams = pd.concat([train["team1"], train["team2"]])
    return teams.value_counts()


def eligible_holdout(matches: pd.DataFrame, min_prior_matches: int = 0) -> pd.DataFrame:
    """Clean matches with a decisive full-length result (no D/L, no 'no result').

    min_prior_matches>0 keeps only matches where BOTH teams played at least that many
    matches before the split (Set-R-style 'established teams' filter).
    """
    hm = holdout_matches(matches)
    base = (
        (hm["overs_limit"] == 20)
        & hm["method"].isna()
        & hm["result"].isna()
        & hm["winner"].notna()
    )
    hm = hm[base]
    hm = hm[(hm["winner"] == hm["team1"]) | (hm["winner"] == hm["team2"])]
    if min_prior_matches > 0:
        counts = _prior_match_counts(matches)
        keep = hm.apply(
            lambda r: counts.get(r["team1"], 0) >= min_prior_matches
            and counts.get(r["team2"], 0) >= min_prior_matches,
            axis=1,
        )
        hm = hm[keep]
    return hm


def _xi(xi: pd.DataFrame, match_id: str, team: str) -> list[str]:
    return xi[(xi["match_id"] == match_id) & (xi["team"] == team)]["player_id"].dropna().tolist()


def evaluate(
    tb: SimTables,
    matches: pd.DataFrame,
    xi: pd.DataFrame,
    sample: pd.DataFrame,
    n: int = 4000,
    win_prob_fn=win_probability,
    pass_venue: bool = False,
    **kwargs,
) -> dict:
    preds, ys = [], []
    for seed, (_, m) in enumerate(sample.iterrows()):
        t1x = _xi(xi, m["match_id"], m["team1"])
        t2x = _xi(xi, m["match_id"], m["team2"])
        if len(t1x) < 7 or len(t2x) < 7:
            continue
        bf = bat_first_team(m)
        call_kwargs = dict(kwargs)
        if pass_venue:
            call_kwargs["venue"] = m["venue"]
        p1 = win_prob_fn(t1x, t2x, bf, tb, n=n, seed=seed, **call_kwargs)
        preds.append(p1)
        ys.append(1 if m["winner"] == m["team1"] else 0)
    preds, ys = np.array(preds), np.array(ys)
    brier = float(np.mean((preds - ys) ** 2))
    auc = float(roc_auc_score(ys, preds)) if len(set(ys)) > 1 else float("nan")
    return {
        "n_matches": int(len(preds)),
        "brier": brier,
        "auc": auc,
        "beats_coinflip_brier": brier < 0.25,
        "beats_coinflip_auc": auc > 0.5,
        "base_rate": float(ys.mean()),
    }


if __name__ == "__main__":
    t = load_tables()
    tb = build_rates(SPLIT_DATE, t["matches"], t["deliveries"])
    sample = eligible_holdout(t["matches"]).head(120)
    res = evaluate(tb, t["matches"], t["xi"], sample, n=3000)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()})
