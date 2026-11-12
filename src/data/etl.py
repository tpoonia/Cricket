"""F2: parse Cricsheet T20I JSON into tidy Parquet tables.

Tables:
  matches     one row per match (id, date, venue, teams, toss, result, method, overs)
  xi          one row per (match, team, player) — the named XI, with player_id
  deliveries  one row per legal/illegal ball (excludes super-over balls, which are flagged)

Super overs are parsed but kept out of `deliveries` feature rows via `is_super_over`.
Rain/D-L matches are flagged (`method`). Retirements are recorded as wicket rows with the
retirement kind so they can be excluded or handled downstream.

Run as a script:  python -m src.data.etl
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/t20s_json")
OUT_DIR = Path("data/processed")

# Wicket kinds that are retirements (not a bowler/fielder dismissal).
RETIREMENT_KINDS = {"retired hurt", "retired out", "retired not out"}


@dataclass
class ParsedMatch:
    match: dict
    xi: list[dict]
    deliveries: list[dict]


def _name_to_id(info: dict) -> dict[str, str]:
    """Registry maps player name -> Cricsheet person id."""
    return dict(info.get("registry", {}).get("people", {}))


def parse_match(path: str | Path) -> ParsedMatch:
    """Parse a single Cricsheet match JSON into (match, xi, deliveries) rows."""
    path = Path(path)
    with open(path) as fh:
        d = json.load(fh)
    info = d["info"]
    match_id = path.stem
    reg = _name_to_id(info)

    dates = info.get("dates", [])
    teams = info.get("teams", [None, None])
    outcome = info.get("outcome", {})
    by = outcome.get("by", {})
    toss = info.get("toss", {})

    match_row = {
        "match_id": match_id,
        "date": pd.to_datetime(dates[0]).date() if dates else pd.NaT,
        "season": str(info.get("season", "")),
        "venue": info.get("venue"),
        "city": info.get("city"),
        "team1": teams[0] if len(teams) > 0 else None,
        "team2": teams[1] if len(teams) > 1 else None,
        "gender": info.get("gender"),
        "match_type": info.get("match_type"),
        "overs_limit": info.get("overs"),
        "toss_winner": toss.get("winner"),
        "toss_decision": toss.get("decision"),
        "winner": outcome.get("winner"),
        "result": outcome.get("result"),            # 'tie' | 'no result' | None
        "method": outcome.get("method"),            # 'D/L' | 'Awarded' | None
        "by_runs": by.get("runs"),
        "by_wickets": by.get("wickets"),
        "eliminator": outcome.get("eliminator"),    # super-over winner, if any
        "event_name": (info.get("event") or {}).get("name"),
    }

    # XI rows
    xi_rows: list[dict] = []
    for team, players in info.get("players", {}).items():
        for name in players:
            xi_rows.append(
                {
                    "match_id": match_id,
                    "team": team,
                    "player_id": reg.get(name),
                    "player_name": name,
                }
            )

    # Delivery rows
    dl_rows: list[dict] = []
    for inn_idx, inn in enumerate(d.get("innings", [])):
        batting_team = inn.get("team")
        is_super = bool(inn.get("super_over", False))
        for ov in inn.get("overs", []):
            over_no = ov["over"]
            for ball_no, dl in enumerate(ov.get("deliveries", []), start=1):
                runs = dl.get("runs", {})
                extras = dl.get("extras", {})
                extra_type = next(iter(extras), None) if extras else None
                wkts = dl.get("wickets", [])
                # A delivery can have >1 wicket (rare); take the primary dismissal.
                wk = wkts[0] if wkts else {}
                dl_rows.append(
                    {
                        "match_id": match_id,
                        "innings": inn_idx,
                        "is_super_over": is_super,
                        "batting_team": batting_team,
                        "over": over_no,
                        "ball": ball_no,
                        "batter_id": reg.get(dl.get("batter")),
                        "bowler_id": reg.get(dl.get("bowler")),
                        "non_striker_id": reg.get(dl.get("non_striker")),
                        "batter": dl.get("batter"),
                        "bowler": dl.get("bowler"),
                        "runs_batter": runs.get("batter", 0),
                        "runs_extras": runs.get("extras", 0),
                        "runs_total": runs.get("total", 0),
                        "extra_type": extra_type,
                        "wicket_kind": wk.get("kind"),
                        "player_out": reg.get(wk.get("player_out")) if wk else None,
                        "player_out_name": wk.get("player_out") if wk else None,
                    }
                )

    return ParsedMatch(match=match_row, xi=xi_rows, deliveries=dl_rows)


def run_etl(raw_dir: str | Path = RAW_DIR, out_dir: str | Path = OUT_DIR,
            gender: str = "male", match_type: str = "T20") -> dict[str, pd.DataFrame]:
    """Parse every match JSON under raw_dir and write three Parquet tables."""
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matches, xi, deliveries = [], [], []
    for path in sorted(raw_dir.glob("*.json")):
        pm = parse_match(path)
        if pm.match["gender"] != gender or pm.match["match_type"] != match_type:
            continue
        matches.append(pm.match)
        xi.extend(pm.xi)
        deliveries.extend(pm.deliveries)

    m = pd.DataFrame(matches).sort_values("date").reset_index(drop=True)
    x = pd.DataFrame(xi)
    dl = pd.DataFrame(deliveries)

    # Normalise dtypes
    m["date"] = pd.to_datetime(m["date"])
    m.to_parquet(out_dir / "matches.parquet", index=False)
    x.to_parquet(out_dir / "xi.parquet", index=False)
    dl.to_parquet(out_dir / "deliveries.parquet", index=False)
    return {"matches": m, "xi": x, "deliveries": dl}


if __name__ == "__main__":
    tables = run_etl()
    m = tables["matches"]
    print(f"matches={len(m)}  xi={len(tables['xi'])}  deliveries={len(tables['deliveries'])}")
    print(f"date range: {m['date'].min().date()} .. {m['date'].max().date()}")
    print(f"super-over delivery rows: {int(tables['deliveries']['is_super_over'].sum())}")
    print(f"D/L matches: {int((m['method'] == 'D/L').sum())}")
