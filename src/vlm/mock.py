"""Deterministic, class-conditioned MOCK VLM backend.

PURPOSE: exercise the full pipeline, demo and extraction metrics offline with NO API key.
It is a stand-in for a real VLM, NOT a measurement of real VLM ability — every mock response
is tagged ("backend":"mock") in the cache and excluded from any headline VLM claim.

Given a scenario's TRUE class it emits a perceived label through a fixed confusion matrix,
with k-run variation from a seeded RNG, so results are reproducible.
"""
from __future__ import annotations

import numpy as np

from src.rules.checker import rules_for_label

# Perception confusion matrices: true class -> {perceived: prob}
PITCH_CONFUSION = {
    "green": {"green": 0.78, "flat": 0.10, "slow": 0.05, "uninformative": 0.07},
    "dry": {"dry": 0.74, "slow": 0.16, "flat": 0.10},
    "flat": {"flat": 0.70, "dry": 0.10, "slow": 0.10, "green": 0.10},
    "slow": {"slow": 0.68, "dry": 0.22, "flat": 0.10},
    "wet": {"wet": 0.85, "uninformative": 0.15},
    "uninformative": {"uninformative": 0.88, "flat": 0.12},
}
SKY_CONFUSION = {
    "clear": {"clear": 0.85, "overcast": 0.10, "uninformative": 0.05},
    "overcast": {"overcast": 0.78, "clear": 0.12, "rain_now": 0.10},
    "rain_now": {"rain_now": 0.80, "overcast": 0.20},
    "night": {"night": 0.88, "clear": 0.12},
    "uninformative": {"uninformative": 0.9, "clear": 0.1},
}


def _sample(dist: dict, rng: np.random.Generator) -> str:
    labels = list(dist)
    probs = np.array([dist[k] for k in labels], dtype=float)
    probs /= probs.sum()
    return labels[int(rng.choice(len(labels), p=probs))]


def _seed(scenario_id: str, draw: int) -> int:
    import hashlib

    h = hashlib.sha256(f"{scenario_id}|{draw}".encode()).hexdigest()
    return int(h[:8], 16)


def mock_label(scenario: dict, draw: int) -> dict:
    rng = np.random.default_rng(_seed(scenario["id"], draw))
    pitch_true = scenario.get("pitch_class", "uninformative")
    sky_true = scenario.get("sky_class", "uninformative")
    pitch = _sample(PITCH_CONFUSION.get(pitch_true, {"uninformative": 1.0}), rng)
    sky = _sample(SKY_CONFUSION.get(sky_true, {"uninformative": 1.0}), rng)
    conf = round(float(0.6 + 0.4 * rng.random()), 2)
    return {"pitch": pitch, "sky": sky, "confidence": conf, "backend": "mock"}


def mock_rules(scenario: dict, draw: int) -> dict:
    """Rules mode: perceive a label, map via rulebook, then add independent direct-rule noise."""
    rng = np.random.default_rng(_seed(scenario["id"], draw) ^ 0x5EED)
    lab = mock_label(scenario, draw)
    rules = list(rules_for_label(lab["pitch"])["rules"])
    sky_rules = rules_for_label(lab["sky"])["rules"]
    rules += sky_rules
    # direct-rule noise: 15% of the time a strong lower bound is softened by 1
    out = []
    for r in rules:
        rr = dict(r)
        if rng.random() < 0.15 and rr["lo"] > 0:
            rr["lo"] = max(0, rr["lo"] - 1)
        out.append(rr)
    return {"rules": out, "backend": "mock"}
