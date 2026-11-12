"""M8: figures + metrics exist and are well-formed."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIG = Path("paper/figures")
METRICS = Path("results/metrics.json")


@pytest.mark.skipif(not METRICS.exists(), reason="run src.evaluation.report first")
def test_metrics_json_wellformed():
    met = json.loads(METRICS.read_text())
    assert met["simulator_validation"]["blind"]["brier"] < 0.25
    assert met["simulator_validation"]["condition_aware"]["auc"] > 0.5
    assert "set_r_decision" in met and "set_s_extraction" in met


@pytest.mark.skipif(not FIG.exists(), reason="run src.evaluation.figures first")
def test_six_figures_present():
    pngs = sorted(p.name for p in FIG.glob("*.png"))
    assert len(pngs) >= 6
    for stem in ["fig1_pipeline", "fig5_delta_pwin", "fig6_noisy_rule"]:
        assert any(stem in p for p in pngs)
