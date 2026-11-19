"""Rule schema validation (§3.1 #8, §3.2).

A rule set is a list of dicts {feature, lo, hi} with feature in {n_pace, n_spin} and
0 <= lo <= hi <= 11. An empty list is valid ("no rule").
"""
from __future__ import annotations

FEATURES = {"n_pace", "n_spin"}
XI_SIZE = 11


class RuleError(ValueError):
    pass


def validate_rule(rule: dict) -> dict:
    if not isinstance(rule, dict):
        raise RuleError(f"rule must be a dict, got {type(rule)}")
    if rule.get("feature") not in FEATURES:
        raise RuleError(f"unknown feature: {rule.get('feature')!r}")
    lo, hi = rule.get("lo", 0), rule.get("hi", XI_SIZE)
    if not (isinstance(lo, int) and isinstance(hi, int)):
        raise RuleError(f"lo/hi must be ints: {rule}")
    if not (0 <= lo <= hi <= XI_SIZE):
        raise RuleError(f"require 0 <= lo <= hi <= {XI_SIZE}: {rule}")
    return {"feature": rule["feature"], "lo": lo, "hi": hi}


def validate_rules(rules) -> list[dict]:
    """Validate a rule set; empty/None -> []. Raises RuleError on any bad rule."""
    if not rules:
        return []
    if not isinstance(rules, list):
        raise RuleError(f"rules must be a list, got {type(rules)}")
    return [validate_rule(r) for r in rules]


def is_valid(rules) -> bool:
    try:
        validate_rules(rules)
        return True
    except RuleError:
        return False
