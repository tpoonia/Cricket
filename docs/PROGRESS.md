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
