"""F2 tests: parse Cricsheet JSON on hand-checked matches."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.data.etl import parse_match

RAW = Path("data/raw/t20s_json")
pytestmark = pytest.mark.skipif(not RAW.exists(), reason="Cricsheet data not downloaded")


def test_parse_known_match_1001349():
    """Australia v Sri Lanka, MCG, 2017-02-17 — hand-checked from the JSON."""
    pm = parse_match(RAW / "1001349.json")
    m = pm.match
    assert m["match_id"] == "1001349"
    assert str(m["date"]) == "2017-02-17"
    assert m["venue"] == "Melbourne Cricket Ground"
    assert set([m["team1"], m["team2"]]) == {"Australia", "Sri Lanka"}
    assert m["toss_winner"] == "Sri Lanka"
    assert m["toss_decision"] == "field"
    assert m["winner"] == "Sri Lanka"
    assert m["by_wickets"] == 5
    assert m["overs_limit"] == 20
    assert m["gender"] == "male"
    assert m["match_type"] == "T20"


def test_xi_two_teams_eleven_each_with_ids():
    pm = parse_match(RAW / "1001349.json")
    xi = pm.xi
    teams = {r["team"] for r in xi}
    assert teams == {"Australia", "Sri Lanka"}
    for team in teams:
        players = [r for r in xi if r["team"] == team]
        assert len(players) == 11
        assert all(r["player_id"] for r in players)  # every XI player resolved to an id


def test_deliveries_have_ids_and_runs():
    pm = parse_match(RAW / "1001349.json")
    dl = pm.deliveries
    assert len(dl) > 200  # a full T20 innings pair
    first = dl[0]
    assert first["over"] == 0 and first["ball"] == 1
    assert first["batter_id"] and first["bowler_id"]
    assert first["runs_total"] == first["runs_batter"] + first["runs_extras"]
    assert not any(r["is_super_over"] for r in dl)  # this match has no super over


def test_super_over_flagged():
    """1263472 has a super over innings; those balls must be flagged, not dropped."""
    pm = parse_match(RAW / "1263472.json")
    dl = pm.deliveries
    assert any(r["is_super_over"] for r in dl)
    assert any(not r["is_super_over"] for r in dl)


def test_wickets_and_retirements_recorded():
    pm = parse_match(RAW / "1001349.json")
    dl = pm.deliveries
    wkts = [r for r in dl if r["wicket_kind"]]
    assert wkts, "expected at least one dismissal"
    for w in wkts:
        assert w["player_out"] or w["player_out_name"]
