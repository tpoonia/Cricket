"""#4: venue spin prior — spin share of overs and wickets, earlier matches only, shrunk
toward the host-country average and then the global average.

Used (a) for arm B5 (venue-prior rules, no image) and (b) to derive Set R "truth" rules
(DECISIONS D10). Leakage-safe via `deliveries_before`.
"""
from __future__ import annotations

import pandas as pd

from src.conditions.player_types import load_player_types
from src.data.splits import deliveries_before

K_COUNTRY = 200.0   # pseudo-balls toward country prior
K_GLOBAL = 2000.0   # pseudo-balls toward global prior
# bowler-credited dismissals (exclude run out / retirements / obstruction)
BOWLER_WICKETS = {"bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"}


def _spin_flags(deliveries: pd.DataFrame) -> pd.Series:
    types = load_player_types()["bowl_type"]
    return deliveries["bowler_id"].map(lambda p: types.get(p, "unknown"))


def _shares(dl: pd.DataFrame) -> tuple[float, float, int, int]:
    """Return (spin_balls, total_typed_balls, spin_wkts, total_typed_wkts)."""
    legal = dl[~dl["extra_type"].isin(["wides", "noballs"])].copy()
    bt = _spin_flags(legal)
    typed = bt != "unknown"
    spin_balls = int((bt[typed] == "spin").sum())
    total_balls = int(typed.sum())
    wk = legal[legal["wicket_kind"].isin(BOWLER_WICKETS)].copy()
    wbt = _spin_flags(wk)
    wtyped = wbt != "unknown"
    spin_wkts = int((wbt[wtyped] == "spin").sum())
    total_wkts = int(wtyped.sum())
    return spin_balls, total_balls, spin_wkts, total_wkts


def venue_spin_share(
    venue: str,
    target_date,
    matches: pd.DataFrame,
    deliveries: pd.DataFrame,
    country_map: dict | None = None,
) -> dict:
    """Shrunk spin share of overs and wickets at `venue` before `target_date`."""
    hist = matches[matches["date"] < pd.Timestamp(target_date)]
    all_ids = set(hist["match_id"])
    dl_all = deliveries[deliveries["match_id"].isin(all_ids) & (~deliveries["is_super_over"])]

    g_sb, g_tb, g_sw, g_tw = _shares(dl_all)
    p_global_overs = g_sb / g_tb if g_tb else 0.33
    p_global_wkts = g_sw / g_tw if g_tw else 0.33

    venue_ids = set(hist[hist["venue"] == venue]["match_id"])
    dl_v = dl_all[dl_all["match_id"].isin(venue_ids)]
    v_sb, v_tb, v_sw, v_tw = _shares(dl_v)

    # country level
    country = (country_map or {}).get(venue)
    if country and "country" in matches.columns:
        c_ids = set(hist[hist["country"] == country]["match_id"])
    else:
        c_ids = set()
    if c_ids:
        dl_c = dl_all[dl_all["match_id"].isin(c_ids)]
        c_sb, c_tb, c_sw, c_tw = _shares(dl_c)
    else:
        c_sb, c_tb, c_sw, c_tw = 0, 0, 0, 0

    def two_stage(vs, vt, cs, ct, pg):
        p_country = (cs + K_GLOBAL * pg) / (ct + K_GLOBAL) if (ct + K_GLOBAL) else pg
        return (vs + K_COUNTRY * p_country) / (vt + K_COUNTRY) if (vt + K_COUNTRY) else pg

    return {
        "venue": venue,
        "overs_spin_share": two_stage(v_sb, v_tb, c_sb, c_tb, p_global_overs),
        "wickets_spin_share": two_stage(v_sw, v_tw, c_sw, c_tw, p_global_wkts),
        "n_balls": v_tb,
        "raw_overs_spin_share": (v_sb / v_tb) if v_tb else None,
        "global_overs_spin_share": p_global_overs,
    }


def venue_prior_rule(share: dict, upper: float, lower: float) -> list[dict]:
    """Map a venue spin share to a rule (DECISIONS D10 thresholds)."""
    s = share["overs_spin_share"]
    if s >= upper:
        return [{"feature": "n_spin", "lo": 2, "hi": 11}]
    if s <= lower:
        return [{"feature": "n_pace", "lo": 3, "hi": 11}]
    return []
