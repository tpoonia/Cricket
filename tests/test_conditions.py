"""M1b tests: venue prior, venue coords, weather (cached)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.splits import load_tables

PROCESSED = Path("data/processed")
pytestmark = pytest.mark.skipif(
    not (PROCESSED / "matches.parquet").exists(), reason="run ETL first"
)


@pytest.fixture(scope="module")
def tables():
    return load_tables()


def test_venue_spin_share_bounds_and_ordering(tables):
    from src.conditions.venue_prior import venue_spin_share

    m, dl = tables["matches"], tables["deliveries"]
    D = pd.Timestamp("2025-06-01")
    mcg = venue_spin_share("Melbourne Cricket Ground", D, m, dl)
    kl = venue_spin_share("Bayuemas Oval, Kuala Lumpur", D, m, dl)
    for s in (mcg, kl):
        assert 0.0 <= s["overs_spin_share"] <= 1.0
        assert 0.0 <= s["wickets_spin_share"] <= 1.0
    # Asian slow track bowls more spin than the MCG
    assert kl["overs_spin_share"] > mcg["overs_spin_share"]


def test_venue_prior_rule_thresholds(tables):
    from src.conditions.venue_prior import venue_prior_rule

    assert venue_prior_rule({"overs_spin_share": 0.6}, 0.45, 0.30) == [
        {"feature": "n_spin", "lo": 2, "hi": 11}
    ]
    assert venue_prior_rule({"overs_spin_share": 0.2}, 0.45, 0.30) == [
        {"feature": "n_pace", "lo": 3, "hi": 11}
    ]
    assert venue_prior_rule({"overs_spin_share": 0.38}, 0.45, 0.30) == []


def test_venue_coords_cached():
    from src.conditions.venues import resolve

    g = resolve("Melbourne Cricket Ground", "Melbourne", allow_network=False)
    assert g is not None and g["country"] == "Australia"
    assert -38 < g["lat"] < -37 and 144 < g["lon"] < 145


def test_weather_summary_from_cache():
    from src.conditions.weather import match_weather

    cache = Path("cache/weather")
    if not cache.exists() or not any(cache.glob("*.json")):
        pytest.skip("no cached weather")
    w = match_weather(-37.814, 144.96332, "2025-06-01")  # MCG, cached in dev run
    assert "aft_cloud_low" in w and "eve_humidity" in w
