"""Rebuild data/results/metrics.json (the headline-numbers file the demo + test_paper_ready read).

Nothing in the repo previously built this artifact; this script makes it reproducible. The batter
rule (and any selection change) can only move the XI-dependent `decision` block, so:

  * `decision.*` is recomputed from data/results/raw/set_r_results.json with src.evaluation.stats
    (the committed decision-metric code);
  * `decision.holm_corrected_p` uses a STANDARD Holm-Bonferroni correction (monotone) over the
    Wilcoxon p-values of every non-baseline arm;
  * `extraction` and `foundations` are carried over unchanged from the current file (they depend
    only on the VLM reader and the data foundations, never on XI selection);
  * `cost` is refreshed from the live cost ledger / cache.

Usage:
    python scripts/build_data_metrics.py                 # rebuild in place
    python scripts/build_data_metrics.py --raw DIR --template FILE --out FILE   # e.g. verify on v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.stats import decision_metrics  # noqa: E402


def _round(x, nd):
    return None if x != x else round(float(x), nd)  # NaN -> null


def standard_holm(pvals: dict[str, float]) -> dict[str, float]:
    """Textbook Holm-Bonferroni: sort ascending, adj_i = min(1, (n-i)*p_i), enforce monotonicity."""
    items = sorted(((k, v) for k, v in pvals.items() if v == v), key=lambda kv: kv[1])
    n = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (n - i) * p))
        adj[k] = running
    return adj


def _decision_block(raw_setr: Path) -> dict:
    df = pd.DataFrame(json.loads(raw_setr.read_text()))
    ca = decision_metrics(df, sim="p_win_ca")
    bl = decision_metrics(df, sim="p_win_blind")

    out = {"set_r_condition_aware": {}, "set_r_blind": {}}
    for arm in ca.index:
        r = ca.loc[arm]
        out["set_r_condition_aware"][arm] = {
            "mean_delta": _round(r["delta_pwin_mean"], 6),
            "ci_lo": _round(r["ci_lo"], 6),
            "ci_hi": _round(r["ci_hi"], 6),
            "wilcoxon_p": _round(r["wilcoxon_p"], 6),
            "xi_change_rate": _round(r["xi_change_rate"], 3),
            "n": int(r["n"]),
        }
    for arm in bl.index:
        r = bl.loc[arm]
        out["set_r_blind"][arm] = {
            "mean_delta": _round(r["delta_pwin_mean"], 6),
            "ci_lo": _round(r["ci_lo"], 6),
            "ci_hi": _round(r["ci_hi"], 6),
            "wilcoxon_p": _round(r["wilcoxon_p"], 6),
        }
    holm = standard_holm({a: float(ca.loc[a, "wilcoxon_p"]) for a in ca.index})
    out["holm_corrected_p"] = {a: _round(holm[a], 6) for a in holm}
    return out


def _cost_block(template_cost: dict) -> dict:
    ledger = ROOT / "cache/cost_ledger.json"
    claude_dir = ROOT / "cache/vlm/claude"
    qwen = ROOT / "data/results/raw/qwen_results.json"
    cost = dict(template_cost)
    if ledger.exists():
        cost["total_api_usd"] = json.loads(ledger.read_text()).get("total_usd", cost.get("total_api_usd"))
    if claude_dir.exists():
        cost["haiku_calls"] = len(list(claude_dir.glob("*.json")))
    if qwen.exists():
        cost["qwen_scenarios"] = json.loads(qwen.read_text()).get("n_scenarios", cost.get("qwen_scenarios"))
    return cost


def build(raw_dir: Path, template: Path, out: Path) -> dict:
    tmpl = json.loads(template.read_text())
    metrics = {
        "extraction": tmpl["extraction"],          # VLM-reader only; XI-independent
        "decision": _decision_block(raw_dir / "set_r_results.json"),
        "foundations": tmpl["foundations"],         # data foundations; XI-independent
        "cost": _cost_block(tmpl.get("cost", {})),
    }
    out.write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data/results/raw"))
    ap.add_argument("--template", default=str(ROOT / "data/results/metrics.json"))
    ap.add_argument("--out", default=str(ROOT / "data/results/metrics.json"))
    a = ap.parse_args()
    m = build(Path(a.raw), Path(a.template), Path(a.out))
    print(f"wrote {a.out}")
    print("Set R arms:", list(m["decision"]["set_r_condition_aware"].keys()))
    print("holm:", m["decision"]["holm_corrected_p"])
