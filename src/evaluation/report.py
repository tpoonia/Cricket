"""Aggregate every headline number into results/metrics.json (the paper pulls from this).

Reproducible from cached parquet results; recomputes simulator validation and extraction metrics.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.splits import SPLIT_DATE, load_tables
from src.evaluation.extraction import run_set_s_extraction
from src.evaluation.stats import decision_metrics
from src.simulation.condition import fit_condition_multipliers, win_probability_ca
from src.simulation.sim import build_rates
from src.simulation.validate import eligible_holdout, evaluate

RESULTS = Path("results")
METRICS = RESULTS / "metrics.json"


def _num(x):
    import numpy as np

    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def build_metrics(sim_n: int = 4000, sim_matches: int = 250) -> dict:
    t = load_tables()
    m, xi, dl = t["matches"], t["xi"], t["deliveries"]
    tb = build_rates(SPLIT_DATE, m, dl)
    cond = fit_condition_multipliers(m, dl, SPLIT_DATE)

    sample = eligible_holdout(m, min_prior_matches=10).head(sim_matches)
    blind = evaluate(tb, m, xi, sample, n=sim_n)
    ca = evaluate(tb, m, xi, sample, n=sim_n, win_prob_fn=win_probability_ca,
                  pass_venue=True, cond_mult=cond)

    out = {
        "simulator_validation": {
            "blind": {k: _num(v) for k, v in blind.items()},
            "condition_aware": {k: _num(v) for k, v in ca.items()},
            "coin_flip_reference": {"brier": 0.25, "auc": 0.5},
        }
    }

    # Set R decision metrics
    setr = pd.read_parquet(RESULTS / "raw/setr_results.parquet")
    for sim in ("p_win_ca", "p_win_blind"):
        dm = decision_metrics(setr, sim=sim)
        out.setdefault("set_r_decision", {})[sim] = {
            arm: {c: _num(dm.loc[arm, c]) for c in
                  ["n", "xi_change_rate", "delta_pwin_mean", "ci_lo", "ci_hi",
                   "wilcoxon_p", "holm_p", "regret_vs_B4", "overlap_real_mean"]}
            for arm in dm.index
        }
    out["set_r_n_matches"] = int(setr["match_id"].nunique())

    # Set S extraction
    mf = pd.read_csv("data/synthetic/manifest.csv")
    out["set_s_extraction"] = {
        "label": run_set_s_extraction(mf, mode="label", k=5),
        "rules": run_set_s_extraction(mf, mode="rules", k=5),
    }

    # Set S planted decision
    sets_path = RESULTS / "raw/sets_decision.parquet"
    if sets_path.exists():
        sd = pd.read_parquet(sets_path)
        pv = sd.pivot_table(index=["match_id", "pitch_class"], columns="arm", values="p_win_planted")
        base = "B3_cricket"
        out["set_s_decision_planted"] = {
            arm: _num((pv[arm] - pv[base]).dropna().mean())
            for arm in pv.columns if arm != base
        }
        out["set_s_decision_planted"]["n_carriers"] = int(sd["match_id"].nunique())

    RESULTS.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(out, indent=2, default=_num))
    return out


if __name__ == "__main__":
    met = build_metrics()
    print(f"wrote {METRICS}")
    print("blind:", met["simulator_validation"]["blind"]["brier"],
          "ca:", met["simulator_validation"]["condition_aware"]["brier"])
