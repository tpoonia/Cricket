"""M8: the six paper figures, one consistent style (dataviz principles, Okabe-Ito CVD-safe palette).

Figures:
  1 pipeline            schematic of the method
  2 three_xis           pace/spin counts for one team under three conditions
  3 perception          per-class perception accuracy (Set S, mock VLM)
  4 extraction_f1       constraint F1 by arm/mode (Set S, mock VLM)
  5 delta_pwin          ΔP(win) vs B3 by arm, blind vs condition-aware (Set R, 95% CI)
  6 noisy_rule          ΔP(win) vs B3 as rule noise increases
All numbers are pulled from results/metrics.json / parquet. Mock VLM figures are labelled.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = Path("paper/figures")
METRICS = Path("results/metrics.json")

# Okabe-Ito (canonical colorblind-safe), fixed order
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]


def _style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "figure.autolayout": True,
    })


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(9, 2.2))
    ax.axis("off")
    steps = ["condition\nimage / weather", "VLM\n(label | rules)", "rulebook +\nrule checker",
             "MILP\nselector", "XI", "simulator\nP(win)", "evaluation"]
    x = 0
    for i, s in enumerate(steps):
        ax.add_patch(plt.Rectangle((x, 0), 1.5, 1, facecolor=OKABE[i % len(OKABE)], alpha=0.18,
                                   edgecolor=OKABE[i % len(OKABE)], linewidth=1.5))
        ax.text(x + 0.75, 0.5, s, ha="center", va="center", fontsize=8.5)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 1.75, 0.5), xytext=(x + 1.5, 0.5),
                        arrowprops=dict(arrowstyle="->", lw=1.2))
        x += 1.75
    ax.set_xlim(-0.1, x); ax.set_ylim(-0.2, 1.2)
    ax.set_title("Seeing conditions → writing constraints → XI → P(win)", fontsize=10)
    _save(fig, "fig1_pipeline")


def fig_three_xis(team="India"):
    from app.demo_backend import three_xis

    res = three_xis(team)
    conds = list(res["conditions"])
    pace = [res["conditions"][c]["n_pace"] for c in conds]
    spin = [res["conditions"][c]["n_spin"] for c in conds]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    import numpy as np

    xpos = np.arange(len(conds))
    w = 0.38
    ax.bar(xpos - w / 2, pace, w, label="pace", color=OKABE[0])
    ax.bar(xpos + w / 2, spin, w, label="spin", color=OKABE[1])
    for i, (p, s) in enumerate(zip(pace, spin)):
        ax.text(i - w / 2, p + 0.1, str(p), ha="center", fontsize=8)
        ax.text(i + w / 2, s + 0.1, str(s), ha="center", fontsize=8)
    ax.set_xticks(xpos); ax.set_xticklabels(conds)
    ax.set_ylabel("count in XI"); ax.legend(frameon=False)
    ax.set_title(f"{team}: XI pace/spin balance by condition")
    _save(fig, "fig2_three_xis")


def fig_perception():
    met = json.loads(METRICS.read_text())
    pca = met["set_s_extraction"]["label"]["per_class_perception_acc"]
    classes = list(pca); vals = [pca[c] for c in classes]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(classes, vals, color=OKABE[2])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("perception accuracy")
    ax.set_title("Per-class perception accuracy (Set S, mock VLM)")
    _save(fig, "fig3_perception")


def fig_extraction_f1():
    met = json.loads(METRICS.read_text())
    ex = met["set_s_extraction"]
    arms = ["label (Ours-1)", "rules (Ours-2)"]
    f1 = [ex["label"]["constraint_f1"], ex["rules"]["constraint_f1"]]
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.bar(arms, f1, color=[OKABE[0], OKABE[3]])
    for i, v in enumerate(f1):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("constraint F1")
    ax.set_title("Extraction constraint F1 (Set S, mock VLM)")
    _save(fig, "fig4_extraction_f1")


def fig_delta_pwin():
    met = json.loads(METRICS.read_text())
    ca = met["set_r_decision"]["p_win_ca"]
    blind = met["set_r_decision"]["p_win_blind"]
    arms = ["B4_oracle", "B5_prior", "Habit", "B7_swap", "Ours1_label_mock", "Ours2_rules_mock"]
    arms = [a for a in arms if a in ca]
    import numpy as np

    xpos = np.arange(len(arms)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 3.4))
    cb = [ca[a]["delta_pwin_mean"] for a in arms]
    bb = [blind[a]["delta_pwin_mean"] for a in arms]
    cerr = [[ca[a]["delta_pwin_mean"] - ca[a]["ci_lo"] for a in arms],
            [ca[a]["ci_hi"] - ca[a]["delta_pwin_mean"] for a in arms]]
    ax.bar(xpos - w / 2, cb, w, yerr=cerr, capsize=3, label="condition-aware", color=OKABE[0])
    ax.bar(xpos + w / 2, bb, w, label="blind", color=OKABE[1])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xpos); ax.set_xticklabels([a.replace("_", "\n") for a in arms], fontsize=7.5)
    ax.set_ylabel("ΔP(win) vs B3"); ax.legend(frameon=False)
    ax.set_title("Decision effect by arm (Set R, 95% CI)")
    _save(fig, "fig5_delta_pwin")


def fig_noisy_rule():
    met = json.loads(METRICS.read_text())
    ca = met["set_r_decision"]["p_win_ca"]
    # ordered by increasing 'wrongness': oracle (0), +1 shift, swap (100% wrong)
    order = [("oracle", "B4_oracle"), ("+1 shift", "B7_plus1"), ("pace/spin swap", "B7_swap")]
    labels = [o[0] for o in order if o[1] in ca]
    vals = [ca[o[1]]["delta_pwin_mean"] for o in order if o[1] in ca]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(labels, vals, "-o", color=OKABE[3], lw=2, markersize=7)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("ΔP(win) vs B3"); ax.set_xlabel("rule correctness →  wrongness")
    ax.set_title("Wrong rules hurt the decision (Set R)")
    _save(fig, "fig6_noisy_rule")


def build_all():
    _style()
    fig_pipeline(); fig_three_xis(); fig_perception()
    fig_extraction_f1(); fig_delta_pwin(); fig_noisy_rule()
    return sorted(p.name for p in FIG.glob("*.png"))


if __name__ == "__main__":
    print("figures:", build_all())
