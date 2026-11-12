"""Set R: real-match test set + pre-registered proxy-truth rules (§3.3, DECISIONS D10).

Truth is the venue's earlier spin share, thresholded two ways (fixed before seeing results).
Venue priors are taken as-of the split date (strictly before every holdout match -> leakage-safe
and efficient).
"""
from __future__ import annotations

import pandas as pd

from src.conditions.venue_prior import all_venue_spin_shares, venue_prior_rule
from src.data.splits import SPLIT_DATE
from src.simulation.validate import eligible_holdout

# Pre-registered thresholds (upper spin-share -> n_spin>=2 ; lower -> n_pace>=3) (DECISIONS D10)
TRUTH_THRESHOLDS = {"A": (0.45, 0.30), "B": (0.50, 0.25)}


def build_set_r(matches: pd.DataFrame, min_prior: int = 10, n: int | None = 250,
                seed: int = 0) -> pd.DataFrame:
    """Clean, decisive, full-length holdout matches between established teams."""
    elig = eligible_holdout(matches, min_prior_matches=min_prior)
    if n is not None and len(elig) > n:
        elig = elig.sort_values("date").head(n)  # earliest n (stable, pre-registered)
    return elig.reset_index(drop=True)


def venue_priors_asof(matches: pd.DataFrame, deliveries: pd.DataFrame,
                      asof=SPLIT_DATE) -> dict[str, dict]:
    """Spin share per venue computed once as-of `asof` (leakage-safe for all holdout matches)."""
    return all_venue_spin_shares(asof, matches, deliveries)


def truth_rule(venue: str, priors: dict, setting: str = "A") -> list[dict]:
    upper, lower = TRUTH_THRESHOLDS[setting]
    share = priors.get(venue)
    if share is None:
        return []
    return venue_prior_rule(share, upper, lower)
