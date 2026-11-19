"""Master orchestrator: reproduce every result from cache (M10).

    python scripts/run_vision_constraints.py --all          # reproduce (reuse cached results)
    python scripts/run_vision_constraints.py --all --force   # recompute the slow simulator runs
    python scripts/run_vision_constraints.py --report        # just rebuild metrics.json + figures + PDFs

Fully-real arms need no API. VLM arms use the real reader given by --backend (claude=Haiku 5.5 by
default, ollama=Qwen3-VL:8b, or mock for a fully-offline run). Add --local-backend ollama for the
hosted-vs-local extraction ablation. Images are the real SDXL-Turbo set in data/synthetic/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROC = ROOT / "data/processed/matches.parquet"
IMGS = ROOT / "data/synthetic/manifest.csv"
SETR = ROOT / "results/raw/setr_results.parquet"
SETS = ROOT / "results/raw/sets_decision.parquet"


def step(msg):
    print(f"\n==> {msg}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--backend", default="claude",
                    help="main VLM reader: claude (Haiku, default) | ollama (Qwen3-VL) | mock (offline)")
    ap.add_argument("--local-backend", default=None,
                    help="optional second reader for the hosted-vs-local ablation, e.g. ollama")
    ap.add_argument("--sims", type=int, default=5000)
    ap.add_argument("--matches", type=int, default=250)
    args = ap.parse_args(argv)

    if args.all:
        if not PROC.exists():
            step("ETL (Cricsheet -> Parquet)")
            from src.data.etl import run_etl
            run_etl()
        if not IMGS.exists():
            step("synthetic images (procedural)")
            from src.synthetic.generate import generate_procedural
            generate_procedural()
            from src.synthetic.checks import build_human_sheet, run_check
            print(run_check()); build_human_sheet()
        if args.force or not SETR.exists():
            step(f"Set R evaluation ({args.matches} matches, n={args.sims})")
            from src.evaluation.vision_constraints import run_setr
            run_setr(n_sims=args.sims, n_matches=args.matches, backend=args.backend)
        if args.force or not SETS.exists():
            step("Set S planted decision")
            from src.evaluation.vision_constraints import run_set_s_decision
            run_set_s_decision(n_sims=args.sims, backend=args.backend)

    step("metrics.json")
    from src.evaluation.report import build_metrics
    build_metrics(backend=args.backend, local_backend=args.local_backend)

    step("figures + summary PDF + paper numbers")
    from src.evaluation.figures import build_all
    print("figures:", build_all())
    from scripts.build_summary_pdf import build as build_pdf
    print("summary:", build_pdf())
    from scripts.build_paper_numbers import build as build_nums
    print("numbers:", build_nums())

    print("\nDone. Compile the paper with:  cd paper && tectonic main.tex   "
          "(or upload paper/ to Overleaf; falls back to pdflatex+bibtex if installed)")


if __name__ == "__main__":
    main()
