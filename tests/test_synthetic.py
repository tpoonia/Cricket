"""M3 tests: synthetic image manifest + automated check."""
from __future__ import annotations

from pathlib import Path

import pytest

MANIFEST = Path("data/synthetic/manifest.csv")
IMG_DIR = Path("data/synthetic/images")
pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="run src.synthetic.generate first")


def test_manifest_and_images_present():
    import pandas as pd

    mf = pd.read_csv(MANIFEST)
    assert len(mf) >= 100  # ~110 images
    assert set(mf.columns) >= {"id", "file", "class", "prompt", "generator", "seed"}
    # every manifest row has an image on disk
    for _, r in mf.head(20).iterrows():
        assert (IMG_DIR / r["file"]).exists()


def test_heuristic_rater_recovers_class():
    from src.synthetic.checks import heuristic_label

    assert heuristic_label(IMG_DIR / "pitch_green_00.png", "pitch") == "green"
    assert heuristic_label(IMG_DIR / "pitch_dry_00.png", "pitch") == "dry"
    assert heuristic_label(IMG_DIR / "sky_night_00.png", "sky") == "night"


def test_check_runs_and_reports_kappa():
    from src.synthetic.checks import run_check

    res = run_check()
    assert res["n_classified"] >= 80
    assert -1.0 <= res["rater_b_vs_intended_kappa"] <= 1.0
