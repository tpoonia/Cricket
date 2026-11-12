# Spec: Seeing conditions, writing constraints (Option 1)

Status: **approved** (autonomous self-review 2026-11-12; see DECISIONS "Approvals").
Target: NeSy-FM @ AAAI 2027. Class demo Sun 15 Nov; submission Fri 20 Nov midday.
Scope: **men's T20I only**.

## 1. Goal (one sentence)
Test whether a VLM can infer selection *constraints* that a natural condition photo only implies,
whether those constraints are *correct*, and whether they *improve* a cricket XI decision — measured
honestly on a simulator, with strict no-leakage and memorisation controls.

## 2. Architecture (modules, each one job, each tested)

### Foundations (M1)
- `src/data/download.py` (F1) — fetch Cricsheet zip + register into `data/raw/`; record date/licence.
- `src/data/etl.py` (F2) — JSON → Parquet: `matches`, `xi`, `deliveries`. Super overs separate;
  rain/D-L flagged; retirements handled. Tests on 2–3 hand-checked matches.
- `src/data/splits.py` (F3) — `history_before(match_date)` + chronological split (train <2025-01-01,
  test ≥2025-01-01). **Leakage test** fails if any feature reads a match on/after the target date.
- `src/conditions/player_types.py` (F4) — pace/spin + batting hand via cricketdata `player_meta`
  joined on `key_cricinfo`; R fallback; manual CSV fallback. Report coverage %.
- `src/selection/pools.py` (F5) — pool = last-10-match appearances, ≤18. Report pool coverage %.
- `src/selection/roles.py` (F6) — keeper rule; bowling-option rule (≥6 balls in ≥2 of last 10);
  pace/spin from F4. Rules logged in code + DECISIONS.
- `src/selection/value.py` (F7) — `v_i` = batting runs above avg/match + bowling runs saved/match,
  earlier matches only, EB shrinkage (k=10).
- `src/selection/milp.py` (F8) — PuLP+HiGHS: exactly 11, ≥1 keeper, ≥5 bowling options, pool-only,
  maximise Σ v_i x_i. Accepts environment rules. Toy-squad unit tests.
- `src/simulation/sim.py` (F9) — **blind** ball-by-ball MC; phase-shrunk batter/bowler rates;
  ≥10k sims, common random numbers; validated Brier<0.25 & AUC>0.5 vs coin flip on held-out.

### Conditions (M1b)
- `src/conditions/venues.py` — venue → lat/lon (`data/manual/venue_coords.csv`, test venues).
- `src/conditions/weather.py` — Open-Meteo historical: afternoon/evening cloud (low/mid/high),
  precip, humidity, dew point, wind. Cached. Credit Open-Meteo.
- `src/conditions/venue_prior.py` — spin share of overs & wickets per venue, earlier only, shrunk
  to host-country then global.

### Rules (M2)
- `configs/rulebook.yaml` — frozen §3.2 rulebook (label → rules).
- `src/rules/schema.py` — rule schema `[{feature:n_pace|n_spin, lo:int, hi:int}]`, 0≤lo≤hi≤11; empty list valid.
- `src/rules/checker.py` — validate, combine by tighter bound, fixed-priority clash resolution,
  feasibility fallback (relax to available + log).
- `src/selection/milp.py` env rules — `sum(x type) >= lo`, `<= hi`.

### Synthetic (M3)
- `src/synthetic/generate.py` — SDXL (diffusers/MPS) per class; `data/synthetic/manifest.csv`
  (id, class, prompt, generator, seed).
- `src/synthetic/checks.py` — rater A (automated model view, blind), rater B (CLIP/SigLIP zero-shot),
  Cohen's/Fleiss kappa vs intended class; drop images both raters miss; `human_check_sheet.csv` (50).
  Paper says "automated check; human check pending".

### VLM + arms (M4, M7)
- `src/vlm/client.py` — one interface; backends: Gemini Flash-Lite (main), Claude Haiku (2nd family),
  Qwen3-VL via Ollama (local), plus a deterministic **mock** backend for offline dev/tests. Strict
  JSON schema; modes `label` / `rules`; k runs (default 5, temp>0); disk cache keyed by
  hash(model, prompt, image). No team/venue/date in extraction prompts; players anonymised P01…P18 in B1.
- `src/simulation/` — **blind** (control), **condition-aware** (venue×{pace,spin} multipliers, fit on
  earlier window than priors), **planted** (Set S: dry→spin wickets +15%, pace −5%). CRN, ≥10k sims.
- `src/evaluation/vision_constraints.py` — run all arms, compute extraction + decision metrics + stats.

### Demo / outputs (M6, M8, M9)
- `app/streamlit_app.py` — offline: one squad + three condition images → three XIs with extracted
  rules shown; cached outputs only, no live calls.
- `scripts/run_vision_constraints.py --all` — reproduce every result from cache.
- `paper/` — LaTeX (AAAI kit if downloadable else `article`) + figures from `dataviz`; numbers pulled by script.

## 3. Arms (§3.4)
B1 VLM-picks-XI (anon), B2 ML-only top-11, B3 MILP cricket-only, B4 MILP+oracle rules,
B5 MILP+venue-prior rules, B6 CLIP→rulebook→MILP, B7 noisy rules (±1 at 25/50/100%, pace/spin swap),
Ours-1 VLM-labels→rulebook→MILP, Ours-2 VLM-rules→MILP, Ref real XI, Habit (Set R).
Ablations: image-only vs +weather vs text-only; ±reference images; Ours-1 vs Ours-2; hosted vs local.

## 4. Test sets (§3.3)
- **Set S** synthetic (~110 images, known truth) + weather-contradicts-photo probe.
- **Set R** 150–300 real T20Is 2025–26, teams ≥10 earlier matches; inputs = weather JSON + host + month;
  truth = venue-spin-share rule (thresholds fixed in DECISIONS D10, two settings); also report
  "agreement with selectors" (not correctness).

## 5. Metrics & stats (§3.5)
Extraction: schema validity, exact set match, constraint P/R/F1, NL4Opt Acc, interval IoU/containment/width,
abstention, per-class perception accuracy, k-run consistency (Fleiss' kappa), feasibility rate.
Decision: XI change rate vs B3, ΔP(win) vs B3, regret vs B4, rule-violation rate, overlap with real XI;
reported under blind (~0 expected), condition-aware (Set R), planted (Set S).
Stats: unit = match/scenario (runs averaged first); paired bootstrap 95% CI (10k, by series),
Wilcoxon for ΔP(win), McNemar for paired yes/no, Holm over 5–8 pre-registered comparisons; 20-match pilot.
Memorisation probe: ask each model "who played for X vs Y on date D?" and flag recalls.

## 6. Hard rules (enforced)
No leakage (tested) · keep green · no scraping · secrets only in `.env` · budget <$10 (cache all,
10-call pilot first) · generator family ≠ reader family · no team/venue/date in extraction prompts;
P01…P18 anon · recompute every XI's legality/value · outputs honest (simulator-based, exploratory).

## 7. Fallback (§4)
If condition-aware simulator (M5) fails by Sat 14 Nov night → Option 2: downstream only on Set S +
planted sim; Set R becomes an extraction + selector-agreement study. Log in DECISIONS.

## 8. Self-review (brainstorming)
Conflicts checked: (a) F4 appears as both foundation and §3.1 #1 — treat as one module, built in M1,
coverage reported once. (b) "value = batting + bowling" vs simulator — value drives MILP objective;
simulator is a separate evaluator (no shared fitted params with condition-aware priors beyond
documented windows). (c) No keys → mock/cached VLM path keeps every downstream arm runnable and all
non-VLM arms fully real. No open conflicts block the build.
