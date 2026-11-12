"""#11: run all arms over Set R and save raw per-(team, match, arm) results.

Fully-real arms (no API): B2, B3, B4/B5 (venue-prior-truth), B7 (noisy), Habit, Ref.
Mock-backed VLM arms (offline demonstration): Ours-1 (label->rulebook), Ours-2 (rules).

Each arm's XI is pitted against the opponent's REAL XI; win probability is computed under the
blind and condition-aware simulators (CRN per match). win-prob is memoised per identical XI.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.splits import SPLIT_DATE, load_tables
from src.evaluation import arms as A
from src.evaluation.setr import build_set_r, truth_rule, venue_priors_asof
from src.selection.milp import Player, verify_xi
from src.simulation.condition import fit_condition_multipliers, win_probability_ca
from src.simulation.sim import bat_first_team, build_rates, win_probability
from src.selection.value import compute_values
from src.vlm.client import VLMClient

RESULTS = Path("results/raw")


def _setr_scenario(match, host_region: str) -> dict:
    """Weak latent condition for the Set-R VLM MOCK (host-region heuristic; does NOT peek at truth)."""
    asian = {"India", "Pakistan", "Sri Lanka", "Bangladesh", "United Arab Emirates", "Oman",
             "Malaysia", "Nepal", "Singapore", "Hong Kong", "Thailand", "Indonesia"}
    pitch = "slow" if host_region in asian else "flat"
    month = str(match["date"])[5:7]
    return {"id": f"setr_{match['match_id']}", "has_image": False,
            "pitch_class": pitch, "sky_class": "clear",
            "host": host_region, "month": month}


def _vlm_rules_from_mock(client: VLMClient, scenario: dict, mode: str, k: int = 5) -> list[dict]:
    """Aggregate k mock runs into one rule set (majority label -> rulebook, or combined rules)."""
    from collections import Counter

    from src.rules.checker import check, rules_for_label

    runs = client.run(scenario, mode=mode, k=k)
    if mode == "label":
        pitches = Counter(r.get("pitch") for r in runs if r.get("schema_ok"))
        if not pitches:
            return []
        top = pitches.most_common(1)[0][0]
        return rules_for_label(top)["rules"]
    else:
        # combine all k rule sets by tighter bound
        sets = [r.get("rules", []) for r in runs if r.get("schema_ok")]
        return check(sets or [[]], mode="rules")["rules"] if sets else []


def run_setr(n_sims: int = 3000, n_matches: int | None = 250, with_vlm: bool = True,
             backend: str = "mock") -> pd.DataFrame:
    t = load_tables()
    m, xi, dl = t["matches"], t["xi"], t["deliveries"]
    A._deliveries.set(dl)

    tb = build_rates(SPLIT_DATE, m, dl)
    cond = fit_condition_multipliers(m, dl, SPLIT_DATE)
    values = compute_values(SPLIT_DATE, m, dl)["value"].to_dict()
    priors = venue_priors_asof(m, dl, SPLIT_DATE)
    setr = build_set_r(m, n=n_matches)

    # precompute keeper set as-of split (leakage-safe for all holdout matches)
    from src.data.splits import deliveries_before

    pre = deliveries_before(SPLIT_DATE, m, dl)
    st = pre[pre["wicket_kind"] == "stumped"].dropna(subset=["fielders"])
    keepers_asof: set = set()
    for f in st["fielders"]:
        keepers_asof.update(str(f).split(","))

    client = VLMClient(backend=backend) if with_vlm else None
    memo: dict = {}

    def wp(xi_ids, opp_ids, bat_first, venue, seed):
        key = (tuple(sorted(xi_ids)), tuple(sorted(opp_ids)), bat_first)
        if key not in memo:
            blind = win_probability(xi_ids, opp_ids, bat_first, tb, n=n_sims, seed=seed)
            ca = win_probability_ca(xi_ids, opp_ids, bat_first, tb, venue=venue,
                                    cond_mult=cond, n=n_sims, seed=seed)
            memo[key] = (blind, ca)
        return memo[key]

    rows = []
    for seed, (_, mm) in enumerate(setr.iterrows()):
        venue = mm["venue"]
        t_rule = truth_rule(venue, priors, "A")
        for persp, (team, opp) in enumerate([(mm["team1"], mm["team2"]), (mm["team2"], mm["team1"])]):
            players = A.build_players(team, mm["date"], m, xi, values, keepers_asof=keepers_asof)
            if len(players) < 11:
                continue
            opp_ids = A.real_xi(opp, mm["match_id"], xi)
            if len(opp_ids) < 7:
                continue
            # perspective team bats first?
            bf_match = bat_first_team(mm)
            bf = 1 if (bf_match == 1) == (persp == 0) else 2
            real_persp_xi = A.real_xi(team, mm["match_id"], xi)
            won = 1 if mm["winner"] == team else 0

            arm_xis = {
                "B2_ml": A.arm_b2_ml_only(players),
                "B3_cricket": A.arm_b3_cricket(players),
                "B4_oracle": A.arm_with_rules(players, t_rule)[0],
                "B5_prior": A.arm_with_rules(players, t_rule)[0],
                "B7_swap": A.arm_with_rules(players, A.perturb_rule(t_rule, swap=True))[0],
                "B7_plus1": A.arm_with_rules(players, A.perturb_rule(t_rule, delta=1))[0],
                "Habit": A.arm_habit(players, team, mm["date"], m, xi),
                "Ref": real_persp_xi,
            }
            if client is not None:
                scn = _setr_scenario(mm, _host(venue, priors))
                r1 = _vlm_rules_from_mock(client, scn, "label")
                r2 = _vlm_rules_from_mock(client, scn, "rules")
                arm_xis["Ours1_label_mock"] = A.arm_with_rules(players, r1)[0]
                arm_xis["Ours2_rules_mock"] = A.arm_with_rules(players, r2)[0]

            for arm, axi in arm_xis.items():
                if len(axi) < 11 and arm != "Ref":
                    continue
                blind, ca = wp(axi, opp_ids, bf, venue, seed)
                vr = verify_xi(axi, players) if arm not in ("Ref", "B2_ml") else {"legal": None, "n_spin": sum(A.bowl_type(p) == "spin" for p in axi), "n_pace": sum(A.bowl_type(p) == "pace" for p in axi)}
                rows.append({
                    "match_id": mm["match_id"], "date": mm["date"], "venue": venue,
                    "team": team, "arm": arm, "won": won, "bat_first": bf,
                    "p_win_blind": blind, "p_win_ca": ca,
                    "n_spin": vr.get("n_spin"), "n_pace": vr.get("n_pace"),
                    "legal": vr.get("legal"),
                    "xi": ",".join(sorted(axi)),
                    "truth_rule": str(t_rule),
                    "overlap_real": len(set(axi) & set(real_persp_xi)),
                })
    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RESULTS / "setr_results.parquet", index=False)
    return df


def _host(venue: str, priors: dict) -> str:
    from src.conditions.venues import country_map

    return country_map().get(venue) or "Unknown"


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    df = run_setr(n_sims=3000, n_matches=n)
    print(f"rows={len(df)}  arms={df['arm'].nunique()}  matches={df['match_id'].nunique()}")
    print(df.groupby("arm")[["p_win_blind", "p_win_ca"]].mean().round(3).to_string())
