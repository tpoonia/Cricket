"""M6 tests: offline demo backend produces three condition-dependent XIs."""
from __future__ import annotations

from pathlib import Path

import pytest

PROCESSED = Path("data/processed")
pytestmark = pytest.mark.skipif(
    not (PROCESSED / "matches.parquet").exists(), reason="run ETL first"
)


def test_three_xis_changes_with_condition():
    from app.demo_backend import three_xis

    res = three_xis("India")
    assert set(res["conditions"]) == {"green", "dry", "flat"}
    green = res["conditions"]["green"]
    dry = res["conditions"]["dry"]
    # green pitch -> more pace; dry pitch -> more spin
    assert green["n_pace"] >= dry["n_pace"]
    assert dry["n_spin"] >= green["n_spin"]
    # every XI is 11 players
    for d in res["conditions"].values():
        assert len(d["xi_names"]) == 11
