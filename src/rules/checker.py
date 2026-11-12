"""§3.1 #8: rule checker.

- validate the schema (src.rules.schema);
- map condition labels to rules via the frozen rulebook (mode "label");
- combine multiple rule sets by the tighter bound, resolving clashes by fixed priority;
- feasibility fallback: if the pool can't meet a rule, relax it to what is available and log.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import yaml

from src.rules.schema import XI_SIZE, validate_rules

RULEBOOK_PATH = Path("configs/rulebook.yaml")


@lru_cache(maxsize=1)
def load_rulebook(path: str = str(RULEBOOK_PATH)) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def canonical_label(label: str) -> str | None:
    rb = load_rulebook()
    label = (label or "").strip().lower()
    if label in rb["labels"]:
        return label
    return rb.get("synonyms", {}).get(label)


def rules_for_label(label: str) -> dict:
    """Return {rules, strength} for a label (via synonyms). Unknown label -> no rule."""
    rb = load_rulebook()
    canon = canonical_label(label)
    if canon is None:
        return {"rules": [], "strength": "none", "label": label, "known": False}
    entry = rb["labels"][canon]
    return {"rules": entry["rules"], "strength": entry["strength"], "label": canon, "known": True}


def combine_rules(tagged_rulesets: list[dict]) -> tuple[list[dict], list[str]]:
    """Combine rule sets by tighter bound; resolve clashes by priority. Each ruleset is
    {"rules": [...], "strength": "strong"|"medium"|"weak"|"none"}."""
    rb = load_rulebook()
    prio = rb["priority"]
    by_feature: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for rs in tagged_rulesets:
        p = prio.get(rs.get("strength", "none"), 0)
        for r in validate_rules(rs.get("rules", [])):
            by_feature[r["feature"]].append((r["lo"], r["hi"], p))

    combined, log = [], []
    for feat, cons in sorted(by_feature.items()):
        lo = max(c[0] for c in cons)
        hi = min(c[1] for c in cons)
        if lo > hi:
            maxp = max(c[2] for c in cons)
            top = [c for c in cons if c[2] == maxp]
            lo, hi = max(c[0] for c in top), min(c[1] for c in top)
            log.append(f"clash on {feat}: kept priority {maxp} -> [{lo},{hi}]")
            if lo > hi:
                hi = lo
                log.append(f"residual clash on {feat}: set hi=lo={lo}")
        combined.append({"feature": feat, "lo": lo, "hi": hi})
    return combined, log


def feasibility_fallback(rules: list[dict], available: dict[str, int]) -> tuple[list[dict], list[str]]:
    """Relax rules the pool cannot meet; log every relaxation (§3.1 #8)."""
    out, log = [], []
    for r in rules:
        lo, hi = r["lo"], min(r["hi"], XI_SIZE)
        avail = available.get(r["feature"], XI_SIZE)
        if lo > avail:
            log.append(f"relax {r['feature']} lo {lo}->{avail} (pool has {avail})")
            lo = avail
        if lo > hi:
            hi = lo
        out.append({"feature": r["feature"], "lo": lo, "hi": hi})
    # joint: sum of minimums cannot exceed the XI
    total_lo = sum(r["lo"] for r in out)
    if total_lo > XI_SIZE:
        log.append(f"joint infeasible: sum lo={total_lo}>11; scaling down los")
        for r in out:
            r["lo"] = max(0, round(r["lo"] * XI_SIZE / total_lo))
    return out, log


def check(
    inputs,
    mode: str = "label",
    available: dict[str, int] | None = None,
) -> dict:
    """Top-level pipeline.

    mode="label": inputs is a list of condition labels (e.g. ["dry", "overcast"]).
    mode="rules": inputs is a rule set (list of {feature, lo, hi}) OR a list of such sets.
    Returns {rules, log, valid, labels_known}.
    """
    log: list[str] = []
    if mode == "label":
        labels = inputs if isinstance(inputs, list) else [inputs]
        tagged = [rules_for_label(l) for l in labels]
        labels_known = all(t["known"] for t in tagged)
        combined, clog = combine_rules(tagged)
        log += clog
    elif mode == "rules":
        # inputs may be a single rule set or several sets to combine
        sets = inputs if (inputs and isinstance(inputs[0], list)) else [inputs or []]
        tagged = [{"rules": validate_rules(s), "strength": "strong"} for s in sets]
        combined, clog = combine_rules(tagged)
        log += clog
        labels_known = True
    else:
        raise ValueError(f"unknown mode {mode!r}")

    if available is not None:
        combined, flog = feasibility_fallback(combined, available)
        log += flog

    return {"rules": combined, "log": log, "valid": True, "labels_known": labels_known}
