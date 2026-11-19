# Progress log

Running log for the owner. Newest milestone at the bottom. Real pasted output, numbers, cost, risks.

---

## NEEDS HUMAN (blockers that need a person)

1. **Research reports missing.** `docs/research/` was empty — the two reference reports named in
   MASTER_PROMPT §0 are absent. Built the spec from §3 (self-contained). Please drop the reports in
   `docs/research/` so citations can be cross-checked for the paper.
2. **No API keys.** No `.env` with `GEMINI_API_KEY` / `ANTHROPIC_API_KEY`; Ollama not installed.
   Live VLM arms (B1, Ours-1, Ours-2) and CLIP arm (B6) run on cached/mock outputs until a key is
   added or Ollama is installed. All non-VLM arms and the full offline pipeline run now. See DECISIONS D2.
3. **Human image check & final paper read** still pending by design (MASTER_PROMPT §3.3): the
   automated image check is done; `data/synthetic/human_check_sheet.csv` awaits two human raters.

Total API cost so far: **$0.00** (no paid calls made).

---

## M0 / M0b — Explore, repo setup (Thu 12 Nov)

**Machine check:** macOS (Darwin 27.0), Python 3.13 default + **3.12.15** available, **16 GB RAM**,
262 GB free disk, **R 4.5.2** installed, **Ollama NOT installed**, git 2.54.
API keys in `.env`: **none present** (only unrelated `.env.pages`, not read).

**Network:** Cricsheet ✅, Open-Meteo ✅ (both reachable).

**Repo:** `git init`; structure under `src/{data,conditions,selection,simulation,synthetic,vlm,rules,evaluation}`,
`tests/`, `scripts/`, `configs/`, `app/`, `paper/`, `docs/`. `.venv` (py3.12) created; `requirements.txt`
installed — imports OK (pandas, pyarrow, pulp, highspy, scipy, sklearn, yaml, matplotlib, pyreadr).
`.gitignore` excludes `.env`, `.venv/`, `data/raw/`, `cache/`, processed data. `scripts/run_tests.py`
wrapper in place. Docs seeded: SKILLS_USED, DECISIONS, DATA, this file.

**Data downloaded:** Cricsheet `t20s_male_json.zip` (14.6 MB, 3,651 files), `people.csv` (1.19 MB).

**Next:** spec (M0), then foundations F1–F9 (M1) with TDD.
Risk to 15 Nov demo / 20 Nov deadline: live-VLM arms need a key or Ollama (mitigated by cached/mock path).

## M1 + M1b(partial) — Foundations (Thu 12 Nov)

**F1–F9 all built, tested, green (39 tests pass).**
- F2 ETL: 3,649 matches (2005→2026-11), 80,288 XI rows, 820,658 deliveries; 401 super-over balls flagged; 126 D/L flagged.
- F3 split: train<2025 = 2,580; holdout≥2025 = 1,069. Strict `history_before`; leakage guard test passes.
- F5 pool coverage (last-10, cap 18): **90.3%** of real XIs recovered (first 200 holdout team-matches).
- F4 player types (cricketdata bundled player_meta, no scraping): bowl_type known for **78.5% of bowlers**, **75.4% ball-weighted** in holdout; associate bowlers are the unknowns; +manual override CSV.
- F7 value (EB k=10): sane (top batters Rizwan/Babar/Kohli/Suryakumar); leakage-safe.
- F6 roles: keeper via stumpings (all 1,410 stumpings have a keeper fielder), bowling option ≥6 balls in ≥2 of last 10.
- F8 MILP (PuLP+HiGHS): exactly 11, ≥1 keeper, ≥5 bowling options, env rules, independent verifier; toy tests pass incl. infeasibility.
- **F9 blind simulator: beats coin flip** — Brier **0.237**<0.25, AUC **0.659**>0.5 on 250 established-team holdout matches (n=5000). CRN via seed.

**Pilot test output (real):**
```
blind sim validation: {'n_matches': 250, 'brier': 0.2368, 'auc': 0.659,
 'beats_coinflip_brier': True, 'beats_coinflip_auc': True, 'base_rate': 0.452}
```

Established-team holdout has **871** eligible matches (plenty for Set R 150–300) across **123 venues**.
Cost: **$0.00**. Next: M1b conditions (venue coords via Open-Meteo geocoding, weather, venue prior), then M2 rules.

## M5 — Condition-aware + planted simulators (Thu 12 Nov)

Built on the F9 machinery via per-over pace/spin multipliers.
- **Condition-aware** (per-venue, fit on train era, shrink 3500 pseudo-balls): e.g. MCG spin-wkt×0.88,
  Kuala Lumpur spin-wkt×1.13 & runs×0.88 (slow low track). Validation (300 matches, n=6000):
  Brier **0.2434** (blind 0.2422), AUC **0.6317** (blind 0.6341) — within MC noise, both beat coin flip.
  **M5 gate met; no Option 2 fallback** (DECISIONS D17).
- **Planted** (Set S): fixed multipliers by true pitch class (dry→spin wkt ×1.15, pace ×0.95; green→
  pace wkt ×1.15; flat/wet neutral). Plumbing + directional tests pass.

Tests green. Cost $0.00. Next: evaluation harness + non-VLM arms (real results), VLM client (mock), synthetic images.

## M4/M7 — VLM client + full Set R evaluation (Thu 12 Nov)

- **VLM client** (`src/vlm/`): strict JSON schema, disk cache (hash of model+prompt+image), k runs,
  budget guard (<$10), real gemini/claude/ollama backends (gated) + deterministic class-conditioned
  **mock** (offline). No key present -> mock used; API cost **$0.00**.
- **Set R** = 250 decisive 2025-26 matches, established teams. All arms run under blind + condition-aware.

**Decision metrics (condition-aware, ΔP(win) vs B3, 95% CI, Holm):**
```
B4_oracle/B5_prior : -0.0002  [-0.0011, +0.0005]   XI change 6%   (correct rule ~ no gain)
B7_swap            : -0.0041   Holm p=4.1e-06       (WRONG rules hurt)
Habit              : -0.0034   Holm p=5e-04
Ours1_label_mock   : -0.0016   Ours2_rules_mock: -0.0127   Ref(real XI): -0.075
```
Blind control: B4/B5 ≈ 0 (CI straddles 0), wrong rules negative (pure value cost). As expected.

## M8 — Figures + tables + one-page PDF (Thu 12 Nov)
`results/metrics.json` (every paper number, by script). 6 CVD-safe figures (Okabe-Ito):
pipeline, three-XI, per-class perception, extraction F1, ΔP(win) by arm, noisy-rule curve.
One-page `paper/results_summary.pdf` for the TA. Set S extraction (mock): label F1 **0.98**, rules F1 **0.71**.

## M9 — Paper draft (Thu 12 Nov)
`paper/main.tex` (clean `article` template — AAAI kit not downloadable offline, logged D-note),
`paper/numbers.tex` (auto-generated macros), `paper/refs.bib`, 6 figures. ~4 pages + refs: abstract,
intro (gap), related work, method, experiments, results, limitations, data statement. **No LaTeX engine
in this env** -> source ships + documented compile (`pdflatex main; bibtex main; pdflatex x2`).
Citations that could not be cross-checked (no research reports) are flagged `NEEDS-VERIFY` in refs.bib.

## M10 — Final check + handover (Thu 12 Nov)

**Full suite: `66 passed in 16.52s`.** `scripts/run_vision_constraints.py --all` reproduces every
result from cache; `--report` rebuilds metrics/figures/PDF.

### HANDOVER — how to run, demo, and what humans must still do
1. **Setup:** `python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
2. **Reproduce everything (from cache):** `python scripts/run_vision_constraints.py --all`
   (add `--force` to recompute the simulators; ~a few minutes at n=5000).
3. **Tests (keep-green gate):** `python scripts/run_tests.py` -> 66 pass.
4. **Demo (offline, <5 min):** `streamlit run app/streamlit_app.py` -> pick a team, see three XIs by condition.
5. **Compile paper:** `cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main`
   (needs a LaTeX install — not present here).

**Humans must still (by Fri 20 Nov midday):**
- Add `GEMINI_API_KEY`/`ANTHROPIC_API_KEY` to `.env` (or install Ollama `ollama pull qwen3-vl:4b`) and
  re-run with `--backend gemini|claude|ollama` to replace the **mock** VLM numbers with real ones.
- Generate real photoreal images (`src/synthetic/generate.py:generate_sdxl`, needs torch+diffusers+MPS)
  to replace procedural placeholders, then re-run the CLIP image check.
- Fill `data/synthetic/human_check_sheet.csv` with two human raters (the paper says "human check pending").
- Drop the two research reports into `docs/research/` and verify `NEEDS-VERIFY` citations
  (`anthropic-skills:deep-research`). Final read + submit.

### Merge summary (branch kept; NOT merged/pushed per instructions)
Branch `feat/vision-to-constraints`, ~16 conventional commits, built on `main` (repo-setup commit).
All milestones M0–M10 delivered; VLM-on-real-photo numbers are the only mock-backed part (keys absent).
Owner to review and merge.

## Real-model evaluation run (Thu 13 Nov, 00:18) — RAW OUTPUTS ONLY

Scope of this run (owner instruction): evaluation runs only — no figures/metrics/stats/PDF/paper
(those are done elsewhere). Readers are now real: **Claude Haiku 5.5 is the main VLM** (Anthropic
credit restored; pilot verified), **Qwen3-VL:8b (Ollama, local) is the second**. Images are the real
**SDXL-Turbo** set (Gemini image-gen is blocked on the free tier — `limit: 0` — so SDXL was used; see
DECISIONS D22). The deterministic mock is retired from every reported file.

**Finished and saved (all raw, no post-processing):**
1. **Haiku 10-call pilot** → `results/pilot_haiku.json`: **10/10 valid JSON**, image input confirmed,
   per-call cost ~$0.0005.
2. **Set S** (planted decision, all arms, reader=Haiku, k=1) → `data/results/raw/set_s_results.json`
   (600 rows; arms B3, B4_oracle, Ours1_label, Ours2_rules, Ref).
3. **Set R** (decision, all arms, reader=Haiku on all 250 matches, k=1) →
   `data/results/raw/set_r_results.json` (5000 rows, 250 matches; all 10 arms incl. Ours1/Ours2,
   blind + condition-aware win probs).
4. **Qwen 50 seeded scenarios** (local, label+rules raw) → `data/results/raw/qwen_results.json`
   (**49/50 label-valid**; pitch perception 19/24 — green/dry/wet read correctly, flat→dry conflated).
5. **Per-arm extractions** → `data/results/raw/extractions/Ours1_label.json`, `Ours2_rules.json`.

**Test output (pilot, real):**
```
pilot (claude-haiku-5-5): valid_json 10/10, image input OK
pilot (qwen3-vl:8b):      valid_json 10/10, mean 33.9s/call (max 63s)
qwen 50-scenario run:     label-valid 49/50, pitch perception 19/24
```

**Total API cost this run: $0.0565** (Haiku only; Qwen is local/free). Cumulative spend well under
the $10 cap.

**NEEDS HUMAN (unchanged / new):**
- Downstream figures/metrics/stats/PDF/paper from these raw files (done elsewhere, per instruction).
- To generate images with **Gemini** instead of SDXL: enable billing on the Google AI Studio project
  (free tier returns `limit: 0` for every image model); then `python -m src.synthetic.generate gemini`.
- One Qwen scenario returned an empty reply (now handled as `schema_ok=false`, run no longer aborts).
