"""M4 tests: VLM client (mock backend), schema validation, caching, k runs."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.vlm.client import CACHE_ROOT, VLMClient
from src.rules.schema import is_valid


@pytest.fixture(autouse=True)
def _clean_cache():
    mock_dir = CACHE_ROOT / "mock"
    if mock_dir.exists():
        shutil.rmtree(mock_dir)
    yield


def test_label_mode_k_runs_and_vocab():
    c = VLMClient(backend="mock")
    scn = {"id": "s_dry_clear", "pitch_class": "dry", "sky_class": "clear", "has_image": True}
    out = c.run(scn, mode="label", k=5)
    assert len(out) == 5
    for r in out:
        assert r["pitch"] in ["green", "dry", "flat", "slow", "wet", "uninformative"]
        assert r["sky"] in ["clear", "overcast", "rain_now", "night", "uninformative"]
        assert r["schema_ok"]
    # dry pitch is perceived as 'dry' most of the time
    assert sum(r["pitch"] == "dry" for r in out) >= 3


def test_rules_mode_valid_schema():
    c = VLMClient(backend="mock")
    scn = {"id": "s_dry2", "pitch_class": "dry", "sky_class": "overcast", "has_image": True}
    out = c.run(scn, mode="rules", k=5)
    assert len(out) == 5
    for r in out:
        assert is_valid(r["rules"])


def test_cache_is_reused():
    c = VLMClient(backend="mock")
    scn = {"id": "s_cache", "pitch_class": "slow", "sky_class": "clear", "has_image": True}
    first = c.run(scn, mode="label", k=5)
    key_files = list((CACHE_ROOT / "mock").glob("*.json"))
    assert key_files, "a cache file should be written"
    second = c.run(scn, mode="label", k=5)
    assert first == second  # identical from cache


def test_uninformative_abstains_in_rules():
    c = VLMClient(backend="mock")
    scn = {"id": "s_uninf", "pitch_class": "uninformative", "sky_class": "uninformative", "has_image": True}
    out = c.run(scn, mode="rules", k=5)
    # uninformative -> mostly empty rule sets
    empties = sum(len(r["rules"]) == 0 for r in out)
    assert empties >= 3
