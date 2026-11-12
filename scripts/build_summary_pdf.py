"""One-page results summary PDF for the TA (M8 deliverable).

Pulls numbers from results/metrics.json and embeds key figures. Uses matplotlib's PdfPages
(no LaTeX / reportlab needed). Output: paper/results_summary.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

METRICS = Path("results/metrics.json")
FIG = Path("paper/figures")
OUT = Path("paper/results_summary.pdf")


def build():
    met = json.loads(METRICS.read_text())
    sv = met["simulator_validation"]
    ca = met["set_r_decision"]["p_win_ca"]
    ex = met["set_s_extraction"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        fig.suptitle("Seeing conditions, writing constraints — results summary",
                     fontsize=14, y=0.98)
        plt.figtext(0.5, 0.955, "VLMs as constraint writers for cricket team selection (T20I). "
                    "Simulator-based, exploratory.", ha="center", fontsize=8.5, style="italic")

        lines = [
            "SIMULATOR VALIDATION (held-out matches, vs coin flip 0.25 / 0.5)",
            f"   blind:           Brier {sv['blind']['brier']:.3f}   AUC {sv['blind']['auc']:.3f}   (n={sv['blind']['n_matches']})",
            f"   condition-aware: Brier {sv['condition_aware']['brier']:.3f}   AUC {sv['condition_aware']['auc']:.3f}   — within MC noise of blind",
            "",
            f"SET R DECISION — ΔP(win) vs B3 (cricket-only), condition-aware, N={met['set_r_n_matches']} matches",
            f"   B4 oracle / B5 prior : {ca['B4_oracle']['delta_pwin_mean']:+.4f}  "
            f"[{ca['B4_oracle']['ci_lo']:+.4f}, {ca['B4_oracle']['ci_hi']:+.4f}]  (XI change {ca['B4_oracle']['xi_change_rate']:.0%})",
            f"   B7 pace/spin swap    : {ca['B7_swap']['delta_pwin_mean']:+.4f}  Holm p={ca['B7_swap']['holm_p']:.1e}  (wrong rules hurt)",
            f"   Habit                : {ca['Habit']['delta_pwin_mean']:+.4f}  Holm p={ca['Habit']['holm_p']:.1e}",
            f"   Ours-1 label (mock)  : {ca['Ours1_label_mock']['delta_pwin_mean']:+.4f}",
            f"   Ours-2 rules (mock)  : {ca['Ours2_rules_mock']['delta_pwin_mean']:+.4f}",
            "",
            "SET S EXTRACTION (mock VLM; real-photo + real-VLM numbers pending key/Ollama)",
            f"   label mode: schema {ex['label']['schema_validity']:.2f}  exact {ex['label']['exact_set_match']:.2f}  "
            f"F1 {ex['label']['constraint_f1']:.2f}  NL4Opt {ex['label']['nl4opt_acc']:.2f}",
            f"   rules mode: exact {ex['rules']['exact_set_match']:.2f}  F1 {ex['rules']['constraint_f1']:.2f}  "
            f"abstention {ex['rules']['abstention_uninformative']:.2f}",
            "",
            "HEADLINE FINDING",
            "   Correct environment rules rarely change a value-maximising XI and give ≈0 win-prob",
            "   gain on real matches; deliberately WRONG rules reliably hurt. Label-mode extraction",
            "   is accurate; rules-mode is noisier. Honest, controlled, simulator-based.",
        ]
        plt.figtext(0.07, 0.90, "\n".join(lines), fontsize=8.6, va="top", family="monospace")

        # embed two figures
        for i, (name, y) in enumerate([("fig5_delta_pwin", 0.30), ("fig1_pipeline", 0.06)]):
            p = FIG / f"{name}.png"
            if p.exists():
                ax = fig.add_axes([0.08, y, 0.84, 0.22 if i == 0 else 0.12])
                ax.imshow(mpimg.imread(p)); ax.axis("off")

        plt.figtext(0.07, 0.025, "Data: Cricsheet (ODC-By), Open-Meteo (CC BY 4.0), cricketdata player_meta. "
                    "No scraping. Human image check pending.", fontsize=7, va="bottom")
        pdf.savefig(fig)
        plt.close(fig)
    return OUT


if __name__ == "__main__":
    print("wrote", build())
