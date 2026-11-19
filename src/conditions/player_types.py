"""F4: bowling type (pace/spin) and batting hand per Cricsheet player.

Source order (DECISIONS; no scraping):
  (a) cricketdata `player_meta` (bundled lazy data in the R package, exported once to
      data/processed/player_meta.csv) joined on its own `cricsheet_id`;
  (b) R fallback (same table) — already handled by exporting the CSV;
  (c) manual overrides in data/manual/player_types_manual.csv (hand-typed, never scraped).

bowl_type in {pace, spin, unknown}; bat_hand in {right, left, unknown}.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

META_CSV = Path("data/processed/player_meta.csv")
MANUAL_CSV = Path("data/manual/player_types_manual.csv")

# Substrings (lowercase) that mark spin; checked before pace so "slow left arm orthodox" -> spin.
SPIN_TOKENS = ("break", "orthodox", "chinaman", "googly", "spin", "slow left", "slow right")
PACE_TOKENS = ("fast", "medium", "seam", "pace", "quick")


def classify_bowl(style: str | float) -> str:
    if not isinstance(style, str) or not style.strip():
        return "unknown"
    s = style.lower()
    if any(tok in s for tok in SPIN_TOKENS):
        return "spin"
    if any(tok in s for tok in PACE_TOKENS):
        return "pace"
    return "unknown"


def classify_hand(style: str | float) -> str:
    if not isinstance(style, str) or not style.strip():
        return "unknown"
    s = style.lower()
    if "left" in s:
        return "left"
    if "right" in s:
        return "right"
    return "unknown"


@lru_cache(maxsize=1)
def load_player_types(meta_csv: str = str(META_CSV), manual_csv: str = str(MANUAL_CSV)) -> pd.DataFrame:
    """Return DataFrame indexed by Cricsheet player_id with columns bowl_type, bat_hand."""
    meta = pd.read_csv(meta_csv, dtype=str)
    meta = meta.dropna(subset=["cricsheet_id"])
    meta["bowl_type"] = meta["bowling_style"].map(classify_bowl)
    meta["bat_hand"] = meta["batting_style"].map(classify_hand)
    out = meta.set_index("cricsheet_id")[["bowl_type", "bat_hand"]]
    out = out[~out.index.duplicated(keep="first")]
    out.index.name = "player_id"

    manual_path = Path(manual_csv)
    if manual_path.exists():
        man = pd.read_csv(manual_path, dtype=str).dropna(subset=["player_id"]).set_index("player_id")
        for col in ("bowl_type", "bat_hand"):
            if col in man.columns:
                out.loc[man.index.intersection(out.index), col] = man[col]
                new = man.index.difference(out.index)
                for pid in new:
                    out.loc[pid, col] = man.loc[pid, col]
    return out


def bowl_type(player_id: str) -> str:
    t = load_player_types()
    return t["bowl_type"].get(player_id, "unknown")


def coverage(xi: pd.DataFrame, deliveries: pd.DataFrame | None = None) -> dict:
    """Report % of players (and % of bowlers who actually bowled) with a known bowl_type."""
    types = load_player_types()
    players = set(xi["player_id"].dropna())
    known = {p for p in players if types["bowl_type"].get(p, "unknown") != "unknown"}
    out = {
        "n_players": len(players),
        "bowl_type_known_pct": round(100 * len(known) / len(players), 2) if players else 0.0,
    }
    if deliveries is not None:
        bowlers = set(deliveries["bowler_id"].dropna())
        bknown = {p for p in bowlers if types["bowl_type"].get(p, "unknown") != "unknown"}
        out["n_bowlers"] = len(bowlers)
        out["bowler_type_known_pct"] = round(100 * len(bknown) / len(bowlers), 2) if bowlers else 0.0
    return out


if __name__ == "__main__":
    from src.data.splits import load_tables

    t = load_tables()
    print(coverage(t["xi"], t["deliveries"]))
