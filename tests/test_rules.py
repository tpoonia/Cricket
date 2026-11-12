"""M2 tests: schema + rulebook + checker (hand-made cases)."""
from __future__ import annotations

import pytest

from src.rules.checker import check, combine_rules, rules_for_label
from src.rules.schema import RuleError, is_valid, validate_rules


# ---- schema ----
def test_schema_valid_and_invalid():
    assert validate_rules([]) == []
    assert validate_rules([{"feature": "n_spin", "lo": 2, "hi": 11}])
    assert not is_valid([{"feature": "n_spin", "lo": 5, "hi": 2}])  # lo>hi
    assert not is_valid([{"feature": "n_offspin", "lo": 1, "hi": 2}])  # bad feature
    with pytest.raises(RuleError):
        validate_rules([{"feature": "n_pace", "lo": -1, "hi": 3}])


# ---- rulebook labels ----
def test_label_lookup_and_synonyms():
    assert rules_for_label("dry")["rules"] == [{"feature": "n_spin", "lo": 2, "hi": 11}]
    assert rules_for_label("cracked")["label"] == "dry"      # synonym
    assert rules_for_label("grassy")["label"] == "green"     # synonym
    assert rules_for_label("flat")["rules"] == []            # no rule
    assert rules_for_label("banana")["known"] is False       # unknown -> no rule


# ---- combine ----
def test_combine_tighter_bound():
    # dry (n_spin>=2) + slow (n_spin>=2) -> n_spin>=2
    r, _ = combine_rules([rules_for_label("dry"), rules_for_label("slow")])
    assert {"feature": "n_spin", "lo": 2, "hi": 11} in r


def test_combine_clash_priority():
    # green (strong: n_spin<=1) vs dry (strong: n_spin>=2) -> clash, same priority
    out = check(["green", "dry"], mode="label")
    spin = [x for x in out["rules"] if x["feature"] == "n_spin"][0]
    assert spin["lo"] <= spin["hi"]  # resolved, not left infeasible
    assert any("clash" in m for m in out["log"])


# ---- feasibility fallback ----
def test_feasibility_relaxes_when_pool_short():
    # require n_spin>=2 but pool has only 1 spinner -> relax to 1 and log
    out = check(["dry"], mode="label", available={"n_spin": 1, "n_pace": 6})
    spin = [x for x in out["rules"] if x["feature"] == "n_spin"][0]
    assert spin["lo"] == 1
    assert any("relax" in m for m in out["log"])


# ---- rules mode ----
def test_rules_mode_passthrough():
    out = check([{"feature": "n_pace", "lo": 3, "hi": 11}], mode="rules")
    assert out["rules"] == [{"feature": "n_pace", "lo": 3, "hi": 11}]
