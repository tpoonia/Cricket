"""F4 tests: bowling-type classification + manual override."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.conditions.player_types import classify_bowl, classify_hand, load_player_types

META = Path("data/processed/player_meta.csv")
pytestmark = pytest.mark.skipif(not META.exists(), reason="player_meta.csv not exported")


@pytest.mark.parametrize(
    "style,expected",
    [
        ("Slow Left arm Orthodox", "spin"),
        ("Right arm Offbreak", "spin"),
        ("Legbreak Googly", "spin"),
        ("Right arm Fast medium", "pace"),
        ("Right arm Medium", "pace"),
        ("Left arm Fast", "pace"),
        ("", "unknown"),
        (None, "unknown"),
    ],
)
def test_classify_bowl(style, expected):
    assert classify_bowl(style) == expected


@pytest.mark.parametrize(
    "style,expected",
    [("Right hand bat", "right"), ("Left hand bat", "left"), ("", "unknown")],
)
def test_classify_hand(style, expected):
    assert classify_hand(style) == expected


def test_known_internationals():
    t = load_player_types()
    # A few well-known bowlers must resolve (via player_meta or manual override).
    from src.data.splits import load_tables

    xi = load_tables()["xi"]

    def pid(name):
        r = xi[xi["player_name"] == name]["player_id"]
        return r.iloc[0] if len(r) else None

    assert t["bowl_type"].get(pid("R Ashwin")) == "spin"
    assert t["bowl_type"].get(pid("JJ Bumrah")) == "pace"
    assert t["bowl_type"].get(pid("Rashid Khan")) == "spin"
    # manual override applied
    assert t["bowl_type"].get("1a8f879b") == "spin"  # Mujeeb-ur-Rehman
