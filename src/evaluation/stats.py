"""Decision metrics + statistics for Set R (§3.5).

Unit = team-match (repeated sim runs already averaged into one p_win per arm). Paired bootstrap
95% CI (resampled by match cluster) and a Wilcoxon signed-rank test for ΔP(win) vs B3, with Holm
correction over the pre-registered arm comparisons. Also: XI change rate, regret vs B4, overlap
with the real XI, and rule-violation rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

BASELINE = "B3_cricket"
ORACLE = "B4_oracle"
# pre-registered primary comparisons (arms vs B3) for Holm correction
PRIMARY_ARMS = ["B4_oracle", "B5_prior", "Habit", "B7_swap", "Ours1_label_mock", "Ours2_rules_mock"]


def _pivot(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """rows = team-match (key), columns = arm, values = col."""
    df = df.copy()
    df["key"] = df["match_id"] + "|" + df["team"]
    return df.pivot_table(index=["key", "match_id"], columns="arm", values=col, aggfunc="first")


def paired_bootstrap_ci(diffs: pd.Series, match_ids: pd.Series,
                        n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """95% CI for the mean paired difference, resampling by match cluster."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"d": diffs.values, "m": match_ids.values}).dropna()
    groups = [g["d"].values for _, g in df.groupby("m")]
    if not groups:
        return (np.nan, np.nan, np.nan)
    means = np.empty(n_boot)
    k = len(groups)
    for b in range(n_boot):
        idx = rng.integers(0, k, k)
        means[b] = np.concatenate([groups[i] for i in idx]).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float(df["d"].mean())


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    n = len(items)
    adj = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        a = min(1.0, (n - i) * p)
        running = max(running, a)
        adj[k] = running
    return adj


def decision_metrics(df: pd.DataFrame, sim: str = "p_win_ca") -> pd.DataFrame:
    """Per-arm metrics vs B3 (and regret vs B4) under the chosen simulator column."""
    pv = _pivot(df, sim)
    xi_pv = _pivot(df, "xi")
    overlap_pv = _pivot(df, "overlap_real")
    legal_pv = _pivot(df, "legal")
    match_ids = pv.reset_index()["match_id"]

    rows = []
    pvals = {}
    for arm in pv.columns:
        if arm == BASELINE:
            continue
        d = pv[arm] - pv[BASELINE]
        valid = d.dropna()
        changed = (xi_pv[arm] != xi_pv[BASELINE]) if arm in xi_pv else pd.Series(False, index=pv.index)
        lo, hi, mean = paired_bootstrap_ci(d, match_ids)
        # Wilcoxon on nonzero differences
        nz = valid[valid != 0]
        try:
            w_p = wilcoxon(nz).pvalue if len(nz) >= 5 else np.nan
        except ValueError:
            w_p = np.nan
        regret = (pv[ORACLE] - pv[arm]).mean() if ORACLE in pv else np.nan
        row = {
            "arm": arm,
            "n": int(valid.shape[0]),
            "xi_change_rate": float(changed.mean()),
            "delta_pwin_mean": mean,
            "ci_lo": lo, "ci_hi": hi,
            "wilcoxon_p": w_p,
            "regret_vs_B4": float(regret) if regret == regret else np.nan,
            "overlap_real_mean": float(overlap_pv[arm].mean()) if arm in overlap_pv else np.nan,
            "legal_rate": float(legal_pv[arm].mean()) if arm in legal_pv and legal_pv[arm].notna().any() else np.nan,
        }
        rows.append(row)
        if arm in PRIMARY_ARMS and w_p == w_p:
            pvals[arm] = w_p

    out = pd.DataFrame(rows).set_index("arm")
    adj = holm(pvals)
    out["holm_p"] = [adj.get(a, np.nan) for a in out.index]
    return out.sort_values("delta_pwin_mean", ascending=False)


def changed_only(df: pd.DataFrame, arm: str, sim: str = "p_win_ca") -> pd.DataFrame:
    """Subset where `arm`'s XI differs from B3, for change-conditional reporting."""
    pv = _pivot(df, sim)
    xi_pv = _pivot(df, "xi")
    mask = xi_pv[arm] != xi_pv[BASELINE]
    return pv[mask]
