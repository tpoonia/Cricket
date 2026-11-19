"""VLM prompts. Memorisation control (Hard rule #7): no team names, venue names or dates.

Two modes:
  label — name the pitch/sky condition from a fixed vocabulary;
  rules — write pace/spin selection constraints directly.
"""
from __future__ import annotations

PITCH_LABELS = ["green", "dry", "flat", "slow", "wet", "uninformative"]
SKY_LABELS = ["clear", "overcast", "rain_now", "night", "uninformative"]

LABEL_PROMPT = """You are shown a photograph of a cricket pitch and/or the sky above a ground.
Classify the visible CONDITION. Do not guess any team, venue, country or date.

Return STRICT JSON only, no prose:
{{"pitch": "<one of {pitch}>", "sky": "<one of {sky}>", "confidence": <0..1>}}

Definitions:
- pitch green: grassy/moist surface; dry: dry/cracked; flat: hard/even; slow: dusty/worn; wet: covered/waterlogged.
- sky clear; overcast: heavy cloud; rain_now: rain falling; night: floodlit.
- Use "uninformative" if the image does not show the pitch/sky (crowd, wide stadium, blur).
"""

RULES_PROMPT = """You are shown a photograph of a cricket pitch and/or the sky above a ground.
Infer SELECTION CONSTRAINTS on how many pace and spin bowlers an XI should pick, based only on
what the image implies about conditions. Do not name any team, venue, country or date.

Return STRICT JSON only, no prose:
{"rules": [{"feature": "n_pace"|"n_spin", "lo": <int 0..11>, "hi": <int 0..11>}]}
An empty list [] means the image implies no constraint. Keep 0 <= lo <= hi <= 11.
"""

# Set R: no image; weather text + host country + month only.
LABEL_PROMPT_TEXT = """Given only this match-day weather summary, host country and month, infer the
likely PITCH and SKY condition. Do not name any team, venue (beyond host country) or date.

Weather: {weather}
Host country: {host}
Month: {month}

Return STRICT JSON only:
{{"pitch": "<one of {pitch}>", "sky": "<one of {sky}>", "confidence": <0..1>}}
"""

RULES_PROMPT_TEXT = """Given only this match-day weather summary, host country and month, infer
SELECTION CONSTRAINTS on pace vs spin bowler counts. Do not name any team, venue or date.

Weather: {weather}
Host country: {host}
Month: {month}

Return STRICT JSON only:
{{"rules": [{{"feature": "n_pace"|"n_spin", "lo": <int>, "hi": <int>}}]}}
"""


def build_prompt(mode: str, has_image: bool, weather=None, host=None, month=None) -> str:
    if mode == "label":
        if has_image:
            return LABEL_PROMPT.format(pitch=PITCH_LABELS, sky=SKY_LABELS)
        return LABEL_PROMPT_TEXT.format(weather=weather, host=host, month=month,
                                        pitch=PITCH_LABELS, sky=SKY_LABELS)
    if mode == "rules":
        if has_image:
            return RULES_PROMPT
        return RULES_PROMPT_TEXT.format(weather=weather, host=host, month=month)
    raise ValueError(f"unknown mode {mode!r}")
