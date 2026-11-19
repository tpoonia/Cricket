"""F9: blind ball-by-ball T20 Monte Carlo simulator (negative control).

Per-ball outcome ~ categorical over {0,1,2,3,4,6,W}. Each batter and bowler has a per-phase
outcome distribution shrunk toward the phase population (K=50 balls). Batter and bowler
distributions are combined multiplicatively (log5-style) and renormalised:

    combined_c  ∝  pop_c * (batter_c/pop_c) * (bowler_c/pop_c)  =  batter_c * bowler_c / pop_c

Extras are added as the per-phase mean extra runs per legal ball (documented approximation).
Simulation is vectorised over N sims; a match seed gives common random numbers across arms.

Phases: powerplay overs 0-5, middle 6-14, death 15-19 (overs are 0-indexed in Cricsheet).
Validated in tests: Brier < 0.25 and AUC > 0.5 vs a coin flip on held-out matches.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.splits import SPLIT_DATE, deliveries_before, history_before

OUTCOMES = ["0", "1", "2", "3", "4", "6", "W"]
RUN_VALUE = np.array([0, 1, 2, 3, 4, 6, 0], dtype=float)
N_OUT = 7
N_PHASE = 3
K_SHRINK = 50.0  # pseudo-balls toward phase population
WICKET_KINDS_OUT = None  # all dismissals count; retirements excluded below
RETIREMENTS = {"retired hurt", "retired out", "retired not out"}
MAX_BALLS = 120
MAX_WICKETS = 10
OVERS_PER_BOWLER = 4


def _phase_of_over(over: int) -> int:
    if over <= 5:
        return 0
    if over <= 14:
        return 1
    return 2


def _outcome_index(dl: pd.DataFrame) -> pd.Series:
    """Map each legal delivery to an outcome index 0..6."""
    out = pd.Series(0, index=dl.index, dtype=int)
    rb = dl["runs_batter"].clip(upper=6)
    out[rb == 1] = 1
    out[rb == 2] = 2
    out[rb == 3] = 3
    out[(rb == 4) | (rb == 5)] = 4
    out[rb == 6] = 5
    is_wkt = dl["wicket_kind"].notna() & ~dl["wicket_kind"].isin(RETIREMENTS)
    out[is_wkt] = 6
    return out


@dataclass
class SimTables:
    idx: dict  # player_id -> int
    bat_dist: np.ndarray  # [P, 3, 7]
    bowl_dist: np.ndarray  # [P, 3, 7]
    pop_phase: np.ndarray  # [3, 7]
    extras_per_ball: np.ndarray  # [3]
    bat_balls: np.ndarray  # [P, 3] balls faced (for attack selection)
    bowl_balls: np.ndarray  # [P, 3] balls bowled
    position: dict  # player_id -> median batting position
    bowl_type_code: np.ndarray  # [P] 0=pace, 1=spin, 2=unknown


def build_rates(before_date=SPLIT_DATE, matches=None, deliveries=None) -> SimTables:
    """Build leakage-safe per-player phase distributions from matches strictly before date."""
    if matches is None or deliveries is None:
        from src.data.splits import load_tables

        t = load_tables()
        matches, deliveries = t["matches"], t["deliveries"]

    dl = deliveries_before(before_date, matches, deliveries)
    # legal balls only (exclude wides/noballs) for outcome categorisation
    legal = dl[~dl["extra_type"].isin(["wides", "noballs"])].copy()
    legal = legal.dropna(subset=["batter_id", "bowler_id"])
    legal["phase"] = legal["over"].map(_phase_of_over)
    legal["oc"] = _outcome_index(legal)

    players = sorted(set(legal["batter_id"]) | set(legal["bowler_id"]))
    idx = {p: i for i, p in enumerate(players)}
    P = len(players)

    # population per phase
    pop_phase = np.zeros((N_PHASE, N_OUT))
    for ph, g in legal.groupby("phase"):
        counts = np.bincount(g["oc"], minlength=N_OUT).astype(float)
        pop_phase[ph] = counts / counts.sum()

    bat_dist = np.tile(pop_phase, (P, 1, 1)).astype(float)
    bowl_dist = np.tile(pop_phase, (P, 1, 1)).astype(float)
    bat_balls = np.zeros((P, N_PHASE))
    bowl_balls = np.zeros((P, N_PHASE))

    # batter distributions
    for (pid, ph), g in legal.groupby(["batter_id", "phase"]):
        i = idx[pid]
        counts = np.bincount(g["oc"], minlength=N_OUT).astype(float)
        n = counts.sum()
        bat_balls[i, ph] = n
        bat_dist[i, ph] = (counts + K_SHRINK * pop_phase[ph]) / (n + K_SHRINK)

    # bowler distributions
    for (pid, ph), g in legal.groupby(["bowler_id", "phase"]):
        i = idx[pid]
        counts = np.bincount(g["oc"], minlength=N_OUT).astype(float)
        n = counts.sum()
        bowl_balls[i, ph] = n
        bowl_dist[i, ph] = (counts + K_SHRINK * pop_phase[ph]) / (n + K_SHRINK)

    # extras per legal ball per phase (all extras as runs)
    extras_per_ball = np.zeros(N_PHASE)
    dl2 = dl.copy()
    dl2["phase"] = dl2["over"].map(_phase_of_over)
    for ph, g in dl2.groupby("phase"):
        legal_n = (~g["extra_type"].isin(["wides", "noballs"])).sum()
        extras_per_ball[ph] = g["runs_extras"].sum() / max(legal_n, 1)

    position = _batting_positions(before_date, matches, deliveries)

    from src.conditions.player_types import load_player_types

    types = load_player_types()["bowl_type"]
    code = {"pace": 0, "spin": 1, "unknown": 2}
    bowl_type_code = np.array([code.get(types.get(p, "unknown"), 2) for p in players])

    return SimTables(idx, bat_dist, bowl_dist, pop_phase, extras_per_ball,
                     bat_balls, bowl_balls, position, bowl_type_code)


def _batting_positions(before_date, matches, deliveries) -> dict:
    """Median batting position per player over matches strictly before before_date."""
    dl = deliveries_before(before_date, matches, deliveries).dropna(subset=["batter_id"])
    dl = dl.sort_values(["match_id", "innings", "over", "ball"])
    first = dl.drop_duplicates(["match_id", "innings", "batter_id"])
    first = first.copy()
    first["pos"] = first.groupby(["match_id", "innings"]).cumcount() + 1
    med = first.groupby("batter_id")["pos"].median()
    return med.to_dict()


def _attack(team_xi: list[str], tb: SimTables, max_bowlers: int = 6) -> list[int]:
    """Choose the bowling attack: XI members with the most historical balls bowled."""
    scored = [(tb.bowl_balls[tb.idx[p]].sum(), tb.idx[p]) for p in team_xi if p in tb.idx]
    scored = [s for s in scored if s[0] > 0]
    scored.sort(reverse=True)
    attack = [i for _, i in scored[:max_bowlers]]
    if attack:
        return attack
    fallback = [tb.idx[p] for p in team_xi if p in tb.idx][:max_bowlers]
    return fallback or [-1]  # -1 => population bowler (no history available)


def _over_schedule(attack: list[int], overs: int = 20) -> list[int]:
    """Round-robin overs among the attack, max 4 overs each, no bowler twice in a row."""
    if not attack:
        return [-1] * overs
    counts = {b: 0 for b in attack}
    sched = []
    last = None
    for _ in range(overs):
        cands = [b for b in attack if counts[b] < OVERS_PER_BOWLER and b != last]
        if not cands:
            cands = [b for b in attack if counts[b] < OVERS_PER_BOWLER] or attack
        b = min(cands, key=lambda z: counts[z])
        sched.append(b)
        counts[b] += 1
        last = b
    return sched


def _order_batting(team_xi: list[str], tb: SimTables) -> list[int]:
    """Order the XI by median historical batting position (unknowns last)."""
    known = [(tb.position.get(p, 99), p) for p in team_xi]
    known.sort()
    return [tb.idx.get(p, -1) for _, p in known]


def simulate_innings(bat_idx: list[int], attack: list[int], tb: SimTables,
                     rng: np.random.Generator, n: int, venue_mult: dict | None = None) -> np.ndarray:
    """Vectorised innings: returns array of N total scores.

    venue_mult (condition-aware / planted): dict mapping (type, "wkt"|"run") -> multiplier,
    type in {"pace","spin"}. Scales the per-ball wicket and boundary probabilities when the
    over's bowler is of that type, then renormalises.
    """
    order = [b for b in bat_idx if b >= 0]
    if len(order) < 2:
        return np.zeros(n)
    order_arr = np.array(order + [order[-1]] * (MAX_WICKETS + 2))  # pad
    sched = _over_schedule(attack)

    # per-over multipliers from the over's bowler type
    wmult = rmult = None
    if venue_mult:
        tname = {0: "pace", 1: "spin", 2: None}
        tcode = [int(tb.bowl_type_code[b]) if b >= 0 else 2 for b in sched]
        wmult = np.array([venue_mult.get((tname[c], "wkt"), 1.0) if tname[c] else 1.0 for c in tcode])
        rmult = np.array([venue_mult.get((tname[c], "run"), 1.0) if tname[c] else 1.0 for c in tcode])

    striker = np.zeros(n, dtype=int)      # index into order_arr
    non_striker = np.ones(n, dtype=int)
    wickets = np.zeros(n, dtype=int)
    total = np.zeros(n, dtype=float)
    alive = np.ones(n, dtype=bool)

    for ball in range(MAX_BALLS):
        over = ball // 6
        phase = _phase_of_over(over)
        bowler = sched[over]
        pop = tb.pop_phase[phase]                          # (7,)
        bowl_d = pop if bowler < 0 else tb.bowl_dist[bowler, phase]  # (7,)
        striker_players = order_arr[striker]               # (N,)
        bat_d = tb.bat_dist[striker_players, phase]        # (N,7)
        combined = bat_d * bowl_d[None, :] / np.where(pop > 0, pop, 1e-9)
        combined = np.clip(combined, 1e-12, None)
        combined /= combined.sum(axis=1, keepdims=True)
        if wmult is not None:
            combined[:, 6] *= wmult[over]
            combined[:, 4] *= rmult[over]
            combined[:, 5] *= rmult[over]
            combined /= combined.sum(axis=1, keepdims=True)
        cdf = np.cumsum(combined, axis=1)
        u = rng.random(n)
        oc = (u[:, None] > cdf).sum(axis=1)                # 0..6
        oc = np.clip(oc, 0, N_OUT - 1)

        runs = RUN_VALUE[oc]
        is_wicket = (oc == 6) & alive
        scored = alive & ~is_wicket
        total += np.where(scored, runs + tb.extras_per_ball[phase], 0.0)

        # wickets
        wickets = np.where(is_wicket, wickets + 1, wickets)
        # new striker = next batter (index = wickets) if still alive
        striker = np.where(is_wicket, np.minimum(wickets + 1, MAX_WICKETS + 1), striker)
        alive = alive & (wickets < MAX_WICKETS)

        # strike rotation on odd runs (only when not a wicket)
        odd = scored & (np.isin(oc, [1, 3]))
        striker, non_striker = (
            np.where(odd, non_striker, striker),
            np.where(odd, striker, non_striker),
        )
        # end of over swap
        if (ball + 1) % 6 == 0:
            striker, non_striker = non_striker.copy(), striker.copy()

    return total


def win_probability(team1_xi, team2_xi, bat_first: int, tb: SimTables,
                    n: int = 10000, seed: int = 0, venue_mult: dict | None = None) -> float:
    """P(team1 wins). bat_first in {1,2}: which team bats first. CRN via seed.

    venue_mult (optional): condition-aware/planted multipliers applied to both innings.
    """
    rng = np.random.default_rng(seed)
    o1, o2 = _order_batting(team1_xi, tb), _order_batting(team2_xi, tb)
    a1, a2 = _attack(team1_xi, tb), _attack(team2_xi, tb)
    if bat_first == 1:
        s1 = simulate_innings(o1, a2, tb, rng, n, venue_mult)
        s2 = simulate_innings(o2, a1, tb, rng, n, venue_mult)
    else:
        s2 = simulate_innings(o2, a1, tb, rng, n, venue_mult)
        s1 = simulate_innings(o1, a2, tb, rng, n, venue_mult)
    wins = (s1 > s2).mean()
    ties = (s1 == s2).mean()
    return float(wins + 0.5 * ties)


def bat_first_team(match_row) -> int:
    """Return 1 if team1 bats first, else 2, from the real toss."""
    tw, dec, t1 = match_row["toss_winner"], match_row["toss_decision"], match_row["team1"]
    if dec == "bat":
        return 1 if tw == t1 else 2
    else:  # field
        return 2 if tw == t1 else 1
