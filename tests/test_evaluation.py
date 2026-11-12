"""M7 tests: arms produce legal XIs; stats helpers correct."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.stats import decision_metrics, holm, paired_bootstrap_ci

PROCESSED = Path("data/processed")


def test_holm_monotone_and_bounded():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert all(0 <= v <= 1 for v in adj.values())
    # most significant stays smallest after correction
    assert adj["a"] <= adj["b"] <= adj["c"]
    assert adj["a"] == pytest.approx(0.03)  # 3 * 0.01


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    diffs = pd.Series(rng.normal(0.05, 0.1, 200))
    mids = pd.Series([f"m{i//2}" for i in range(200)])
    lo, hi, mean = paired_bootstrap_ci(diffs, mids, n_boot=2000)
    assert lo < mean < hi
    assert mean == pytest.approx(0.05, abs=0.03)


def test_decision_metrics_on_crafted_df():
    # two matches, two teams each; B4 beats B3 by +0.1 consistently
    rows = []
    for mi in range(4):
        for arm, p in [("B3_cricket", 0.5), ("B4_oracle", 0.6), ("B5_prior", 0.6)]:
            rows.append({"match_id": f"m{mi//2}", "team": f"t{mi}", "arm": arm,
                         "p_win_ca": p, "p_win_blind": p, "xi": arm, "overlap_real": 7,
                         "legal": True})
    df = pd.DataFrame(rows)
    out = decision_metrics(df, sim="p_win_ca")
    assert out.loc["B4_oracle", "delta_pwin_mean"] == pytest.approx(0.1)
    assert out.loc["B4_oracle", "xi_change_rate"] == 1.0


@pytest.mark.skipif(not (PROCESSED / "matches.parquet").exists(), reason="run ETL first")
def test_arms_produce_legal_xi():
    from src.data.splits import SPLIT_DATE, holdout_matches, load_tables
    from src.evaluation import arms as A
    from src.selection.milp import verify_xi
    from src.selection.value import compute_values

    t = load_tables()
    m, xi, dl = t["matches"], t["xi"], t["deliveries"]
    A._deliveries.set(dl)
    values = compute_values(SPLIT_DATE, m, dl)["value"].to_dict()
    row = holdout_matches(m)
    row = row[(row["team1"] == "India") | (row["team2"] == "India")].iloc[0]
    team = "India"
    players = A.build_players(team, row["date"], m, xi, values)
    xi_b3 = A.arm_b3_cricket(players)
    assert len(xi_b3) == 11
    assert verify_xi(xi_b3, players)["legal"]
    xi_b4 = A.arm_with_rules(players, [{"feature": "n_spin", "lo": 2, "hi": 11}])[0]
    assert len(xi_b4) == 11
