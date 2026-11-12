# MASTER PROMPT (autonomous, NEW PROJECT FROM SCRATCH): Option 1 — "Seeing conditions, writing constraints"

Save this file as `MASTER_PROMPT.md` in an **empty new project folder** (with `.env` and `docs/research/`), start Claude Code in that folder (auto mode on) and type:
**"Read MASTER_PROMPT.md and execute it completely and autonomously."**

---

## AUTONOMOUS MODE (read first)

I, the project owner, am handing you the whole project. **Do not stop to ask me questions.**

- **Approvals are already given.**
  - This prompt **is** the approved design.
  - Your spec is approved once you have self-reviewed it.
  - Your plan is approved once `agentic-sdlc:reviewing-plan` passes.
  - Execution method: **subagent-driven development**.
  - Wherever a skill says "wait for the human partner", record your decision in `docs/DECISIONS.md` (decision, reason, alternatives) and continue.
- **When something is unclear,** pick the most reasonable option that keeps the deadlines, log it in `docs/DECISIONS.md`, and keep going.
- **Keep a running log** at `docs/PROGRESS.md`. After each milestone, record what was done, real pasted test output, numbers, cost spent, and risks. This is how I follow your work.
- **Never stop until** milestone M10 is done, or a true blocker in section 5 makes progress impossible. Even then, finish everything that doesn't depend on the blocker.
- **Use every skill.** At the very start, list every skill available in this session in `docs/SKILLS_USED.md`. For each one, write whether you used it, where, and why or why not. Use every skill that applies to this project. Do not force skills that clearly don't apply (for example job search or PowerPoint); just mark them "not applicable" with a reason.

---

## 0. Who you are and what we are building

You are starting a **brand-new project in an empty folder**. There is no existing code. You build everything: data download, cleaning, player values, the selector, the simulator, the vision pipeline, the evaluation, the demo and the paper. First run `git init`, create a Python 3.12 virtual environment (`.venv`), `requirements.txt`, `.gitignore` (including `.env`, `.venv/`, `data/raw/`, `cache/`), `README.md` and `scripts/run_tests.py` (pytest wrapper). Make the first commit on `main`, then do all work on branch `feat/vision-to-constraints`.

**Paper:** "Seeing conditions, writing constraints: VLMs as constraint writers for cricket team selection", a short workshop paper for NeSy-FM @ AAAI 2027.
- Class demo: **Sun 15 Nov 2026**.
- Submission: **Fri 20 Nov 2026, midday**.

**Research gap:** Foundation models already turn *written* problems into optimisation models, and MM-OptBench (2026) reads numbers drawn in diagrams. To our knowledge, no work tests whether a vision-language model (VLM) can *infer* constraints that a natural photo only implies (a dry, cracked pitch implies more spinners). Nor has any work tested whether such constraints are correct, and whether they actually improve the decision.

**Pipeline (Option 1):**
```
condition input (pitch/sky image, or weather JSON)
  -> VLM  (mode "label": names the condition | mode "rules": writes rules directly)
  -> fixed RULEBOOK (label -> rules)            [skipped in mode "rules"]
  -> RULE CHECKER (validate, combine, feasibility fallback + log)
  -> MILP selector (cricket rules + environment rules; objective: player value = batting + bowling)
  -> XI
  -> SIMULATOR -> P(win) vs opponent's real XI
  -> EVALUATION (are rules correct? do they help?)
```

**What exists at the start:** only `MASTER_PROMPT.md`, `.env`, and two research reports in `docs/research/`:
- `Vision to constraints cricket selection.md`, the **main design reference**;
- `LLM cricket team selection plan.md`.

Read both fully before designing. Everything else you build (section 3.0).

**Scope:** **T20I only**, men's internationals. Do not build ODI or Test parts.

---

## 1. Hard rules (never break these)

1. **No leakage.** Every feature, venue prior, player value and simulator multiplier uses only matches strictly **before** the match being evaluated. Write a test for this.
2. **Keep it green.** `python scripts/run_tests.py` must pass after every task. Never commit failing tests.
3. **No scraping.** Do not scrape ESPNcricinfo, Cricbuzz or Getty. Use only Cricsheet (ODC-By), Open-Meteo (CC BY 4.0, attribute it), the `cricketdata` `player_meta` table, and images we generate.
4. **Secrets.** API keys go in `.env` only: `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, and an image-generation key if used. Add `.env` to `.gitignore`. Never print or commit keys.
5. **Budget.** Total API spend must stay **under $10**. Cache every API response to disk, keyed by a hash of model, prompt and image, so re-runs cost nothing. First run a **10-call pilot** per model, then log the real token counts and cost in `docs/PROGRESS.md`, and scale up only if the projected total stays under $10.
6. **Different families.** The model that *generates* the synthetic images must come from a different family than the models that *read* them. For example, generate with FLUX or SDXL and read with Gemini, Claude Haiku and Qwen3-VL.
7. **Memorisation control.** Extraction prompts must not contain team names, venue names or dates. In the "VLM picks the XI" baseline, anonymise players as P01…P18.
8. **Verify, don't trust.** Recompute every XI's legality and value from player IDs. Never use a model's own claim that its XI is valid.
9. **Be honest in outputs.** Results are simulator-based and exploratory. Never write that the system wins real matches.

---

## 2. Process: use these skills in this order

Before each step, invoke the named skill with the Skill tool and follow it. If a skill is not installed, say so in one line and follow the same step by hand. Codex users: do the same steps without the skill tool.

| Step | Skill(s) | What to do |
|---|---|---|
| Start | `superpowers:using-superpowers` | Load the skill system. |
| Throughout | `anthropic-skills:karpathy-guidelines` | Small, surgical changes. State assumptions. Define verifiable success for each task. |
| A. Setup + read | `anthropic-skills:project-builder` | Read both research reports and this prompt fully. Check the machine: Python version, RAM, free disk, whether `ollama` and R are installed, and which API keys exist in `.env` (check presence only, never print them). Set up the repo as described in section 0. Write findings to `docs/PROGRESS.md`. |
| B. Design spec | `superpowers:brainstorming` | The design is already agreed (section 3 below), so do **not** re-ask settled questions. Resolve any conflicts yourself and log them. Write the spec to `docs/superpowers/specs/2026-11-12-vision-to-constraints-design.md`, self-review it, then mark it approved in `docs/DECISIONS.md` and continue. |
| C. Plan | `superpowers:writing-plans`, then `agentic-sdlc:reviewing-plan` | Turn the approved spec into a task plan with the milestones in section 4. Review the plan; fix blockers with `agentic-sdlc:receiving-plan-review`. Choose **subagent-driven** execution and continue. |
| D. Workspace | `superpowers:using-git-worktrees` | Work on a new branch, `feat/vision-to-constraints`, not on main. |
| E. Build | `superpowers:subagent-driven-development`; `superpowers:dispatching-parallel-agents` for independent tasks | Run M1–M8. Run data tasks in parallel where they don't share files. |
| Each task | `superpowers:test-driven-development` + `agentic-sdlc:testing-pytest` | Write a failing test first, then the code, then make it pass. |
| Any bug | `superpowers:systematic-debugging` | Find the root cause before fixing. Never patch symptoms. |
| Synthetic images | `image-generation:image-generation` | Write a strong prompt for each class, and keep prompts in the manifest. |
| Figures | `dataviz` | Make every paper figure with one consistent style. |
| One-page summaries / PDF | `anthropic-skills:pdf` | Build the results summary PDF for the TA. |
| Citations check | `anthropic-skills:deep-research` (only if needed) | Verify any citation marked "from memory" in the reports before it goes in the paper. |
| Review | `superpowers:requesting-code-review` + `agentic-sdlc:reviewing-code`; respond with `superpowers:receiving-code-review` | After M4 and after M8. |
| Before saying "done" | `superpowers:verification-before-completion` | Run the tests and the scripts, and show the real output. |
| Commits | `agentic-sdlc:crafting-commits` | Small conventional commits, one per task. |
| Finish | `superpowers:finishing-a-development-branch` | Keep the branch and do **not** merge to main or push. Write a merge summary in `docs/PROGRESS.md`; I will merge. |

Skip skills that don't fit this work, such as pptx, docx, xlsx, job search and browser tools. Using them would waste time.

---

## 3. The agreed design (put this into the spec)

### 3.0 Foundations to build first (milestone M1, no existing code)

| # | Module (suggested path) | Job |
|---|---|---|
| F1 | `src/data/download.py` | Download Cricsheet `t20s_male_json.zip` (men's T20Is) and `people.csv` (register) from cricsheet.org into `data/raw/` (git-ignored). Record the download date and the licence (ODC-By) in `docs/DATA.md`. |
| F2 | `src/data/etl.py` | Parse the JSON into tidy Parquet tables: `matches` (id, date, venue, city, teams, toss, result, method such as D/L, overs limit), `xi` (match, team, player_id), `deliveries` (match, innings, over, ball, batter, bowler, runs_batter, extras, wicket kind, player_out). Handle super overs (keep them separate and exclude from features), rain-shortened matches (flag them) and retirements. Write tests on 2–3 hand-checked matches. |
| F3 | `src/data/splits.py` | Chronological split. Train/features: matches before 2025-01-01. Test: 2025-01-01 onward (Set R candidates). Add a helper `history_before(match_date)` used by **every** feature, and a leakage test that fails if any feature uses a match on or after the target date. |
| F4 | `src/conditions/player_types.py` | Bowling type (pace/spin) and batting hand. Try in order: (a) download `player_meta.rda` from the `cricketdata` GitHub repo and read it with `pyreadr`, joining via Cricsheet `people.csv` `key_cricinfo`; (b) R, if installed; (c) hand-fill missing test-set bowlers in `data/manual/player_types_manual.csv` from public squad pages, typing them yourself and never scraping. Report coverage %. |
| F5 | `src/selection/pools.py` | Candidate pool per team per match = players who appeared for that team in its previous 10 T20Is (earlier matches only), at most 18 players. Report "pool coverage": the % of the real XI inside the pool. |
| F6 | `src/selection/roles.py` | Keeper = player with stumpings or keeping evidence in earlier matches (log the rule). Bowling option = bowled at least N overs in the last M matches (choose N and M, log them). Pace/spin from F4. |
| F7 | `src/selection/value.py` | Player value v_i = batting runs above average per match + bowling runs saved per match, from earlier matches only, shrunk toward the population mean for players with few matches (empirical Bayes or a fixed prior weight; log it). |
| F8 | `src/selection/milp.py` | MILP selector with PuLP + HiGHS or CBC (`pip install pulp highspy`): exactly 11 players, at least 1 keeper, at least 5 bowling options, all from the pool; maximise Σ v_i x_i. Accepts extra environment rules (section 3.1, #9). Unit tests on toy squads. |
| F9 | `src/simulation/sim.py` | Ball-by-ball T20 Monte Carlo simulator. Per-ball outcome probabilities (0, 1, 2, 3, 4, 6, wicket, extra) from a simple model: batter and bowler rates shrunk toward phase averages (powerplay 1–6, middle 7–15, death 16–20), combined multiplicatively and renormalised. Batting order from earlier batting positions. Real toss result kept. ≥10,000 sims per match, with common random numbers across arms. **Validate on held-out matches**: win-probability Brier score and AUC vs a coin flip (0.250 / 0.5). It must beat the coin flip; log the numbers. This is the **blind** simulator. |

Keep every module small with one job, a docstring, and tests. Reuse these foundations everywhere; never fork logic.


### 3.1 Components (one job each, tested on their own)

| # | Module (suggested path) | Job |
|---|---|---|
| 1 | `src/conditions/player_types.py` | Join bowling type (pace/spin) and batting hand to every Cricsheet player via the `key_cricinfo` ID and the `cricketdata` `player_meta` table. Report the % of players covered. Fill the remaining test-set players by hand in `data/manual/player_types_manual.csv`. |
| 2 | `src/conditions/venues.py` | Map Cricsheet venue IDs to lat/lon in `data/manual/venue_coords.csv`, hand-filled from Wikipedia for the test venues. |
| 3 | `src/conditions/weather.py` | Open-Meteo historical API. For each match: afternoon and evening mean cloud cover (low/mid/high), precipitation, humidity, dew point and wind. Cache all calls. Credit Open-Meteo. |
| 4 | `src/conditions/venue_prior.py` | Spin share of overs and wickets at each venue, from earlier matches only, shrunk toward the host-country average and then the global average. |
| 5 | `configs/rulebook.yaml` | Fixed rulebook, frozen **before** any model run (see 3.2). |
| 6 | `src/synthetic/` | Generate the images and `data/synthetic/manifest.csv` (id, class, prompt, generator, seed), plus a human-check sheet (CSV), an automated visual check and a Cohen's kappa script. |
| 7 | `src/vlm/client.py` | One interface for Gemini Flash-Lite (main), Claude Haiku (second family) and Qwen3-VL via Ollama (local). Strict JSON schema. It has two modes, `label` and `rules`, runs k times per input (default 5, temperature above 0) and caches to disk. |
| 8 | `src/rules/checker.py` | Validate the schema (known features, 0 ≤ lo ≤ hi ≤ 11). Combine rules by taking the tighter bound. Resolve clashes with a fixed priority. If the squad can't meet a rule, relax it to what is available and log that. |
| 9 | `src/selection/milp.py` (env rules) | Add environment rules to the F8 MILP: `sum(x_i for type) >= lo` and `<= hi`. Objective stays player value from F7. |
| 10 | `src/simulation/` | Three simulators: **blind** (F9, as a negative control), **condition-aware** (venue × {pace, spin} multipliers on runs and wicket chance, fitted on an earlier time window than the venue priors) and **planted** (for synthetic images, e.g. dry → spin wickets +15%, pace −5%). Use common random numbers across arms and ≥10,000 simulations per match. |
| 11 | `src/evaluation/vision_constraints.py` | Run all arms and compute all metrics and statistics. |
| 12 | `app/streamlit_app.py` | Offline Streamlit demo: one squad and three condition images give three XIs, with the extracted rules shown between image and team. Use only cached outputs, with no live API calls. |

### 3.2 Rulebook v1 (freeze before running any model)

| Condition label | Rule | Strength |
|---|---|---|
| green / grassy / moist pitch | n_pace ≥ 3, n_spin ≤ 1 | strong |
| dry / cracked pitch | n_spin ≥ 2 | strong |
| slow / dusty pitch | n_spin ≥ 2 | medium |
| flat / hard pitch | no rule | — |
| wet / covered pitch | no rule (match-length issue, not selection) | — |
| overcast sky | n_pace ≥ 3 | weak |
| clear sky, rain now, floodlit night | no rule | — |
| uninformative image | no rule | — |

Rule schema: `[{"feature": "n_pace" | "n_spin", "lo": int, "hi": int}]`. An empty list is a valid answer.

### 3.3 Two test sets

- **Set S (synthetic, known truth):**
  - Pitch classes: green, dry-cracked, flat-hard, slow-dusty, wet-covered.
  - Sky classes: clear, overcast, light rain, night-floodlit.
  - About 10 images per class (about 90), plus about 20 uninformative images (crowd, wide stadium, blur).
  - Pair pitch and sky images into scenarios.
  - A small probe set where the weather JSON contradicts the photo.
  - **Image check (autonomous version):**
    1. You (Claude Code) view each image and label it blind, without looking at the manifest. Record this as **rater A** ("automated model rater").
    2. A CLIP/SigLIP zero-shot classifier is **rater B**.
    3. Report kappa between A, B and the intended class. Drop images where both raters disagree with the intended class.
    4. Write `data/synthetic/human_check_sheet.csv` with 50 images for two team members to label later.
    5. The paper must say "automated check; human check pending" until humans fill it in. Never present the automated check as a human check.
- **Set R (real matches, proxy truth):**
  - 150–300 men's T20Is from 2025–26 between teams with at least 10 earlier matches.
  - Inputs to the model: weather JSON, host country and month only. No venue prior, which is used for the truth.
  - Truth is a data-derived rule from the venue's earlier spin share:
    - above the upper threshold → `n_spin ≥ 2`;
    - below the lower threshold → `n_pace ≥ 3`;
    - otherwise no rule.
  - Fix the thresholds before looking at results, and test two settings.
  - Also report agreement with the real XI's pace/spin count, labelled as "agreement with selectors", not as correctness.

### 3.4 Arms

| Arm | Description |
|---|---|
| B1 | Image + context → VLM picks XI directly (anonymised players) |
| B2 | ML only: top 11 by value, no rules |
| B3 | MILP with cricket rules only |
| B4 | MILP + oracle (true) environment rules |
| B5 | MILP + venue-prior rules (no image) |
| B6 | CLIP/SigLIP zero-shot labels → rulebook → MILP |
| B7 | Noisy rules: oracle bounds ±1 at 25/50/100%, and pace/spin swapped |
| Ours-1 | VLM labels → rulebook → MILP |
| Ours-2 | VLM writes rules → MILP |
| Ref | Real selectors' XI |
| Habit | Team's usual spin count over the previous 10 matches (Set R only) |

**Ablations:**
1. Image only vs image + weather text vs text only.
2. With vs without 1–2 reference images per class in the prompt.
3. Ours-1 vs Ours-2.
4. Hosted vs local model.

### 3.5 Metrics and statistics

- **Extraction:**
  - schema validity, with failures counted over a fixed denominator;
  - exact set match;
  - constraint precision, recall and F1, and NL4Opt accuracy Acc = 1 − Σmin(FP+FN, D)/ΣD;
  - interval IoU, containment and width;
  - abstention on "no rule" inputs;
  - per-class perception accuracy;
  - consistency across k runs (Fleiss' kappa), and whether 5/5 agreement predicts correctness;
  - feasibility rate.
- **Decision:**
  - XI change rate against B3;
  - ΔP(win) against B3;
  - regret against B4;
  - rule-violation rate, by our verifier;
  - overlap with the real XI.

  Report all decision metrics under the **blind** simulator (expect about 0) and the **condition-aware** simulator (Set R), and under the **planted** simulator (Set S).
- **Statistics:**
  - Match or scenario is the unit, with repeated runs averaged first.
  - Paired bootstrap 95% CI (10,000 resamples, by series when matches cluster) plus a Wilcoxon signed-rank test for ΔP(win).
  - McNemar's test for paired yes/no outcomes, and paired bootstrap for F1.
  - Holm correction over 5–8 primary comparisons fixed in advance.
  - Report results both for all matches and for matches where the XI changed.
  - Run a 20-match pilot to estimate the spread before full runs.
- **Memorisation probe:** ask each model "Who played for X vs Y on date D?" and flag matches it recalls.

---

## 4. Milestones and checkpoints

| Milestone | Deadline | Done when |
|---|---|---|
| **M0** Explore + spec approved | Thu 12 Nov | Spec file written; I approved it |
| **M0b** Repo setup | Thu 12 Nov | git, venv, requirements, run_tests.py, first commit |
| **M1** Foundations F1–F9 | Fri 13 Nov | ETL tests pass; leakage test passes; player-type coverage % and pool coverage % reported; MILP tests pass; blind simulator beats coin flip on held-out matches (Brier and AUC logged) |
| **M1b** Conditions data: venues, weather, venue prior | Fri 13 Nov | Coverage % reported; leakage test passes |
| **M2** Rulebook + schema + rule checker + MILP env rules | Sat 14 Nov (morning) | Unit tests on hand-made cases pass; all tests green |
| **M3** Synthetic images + manifest + automated check + human-check sheet | Fri 13 Nov (run in parallel with M1) | About 110 images on disk; automated kappa reported; sheet ready for humans later (do not wait) |
| **M4** VLM client + 10-call pilot per model + first 30 scenarios end to end | Sat 14 Nov | Real cost/token report; 30 scenarios image → rules → XI |
| **M5** Simulators: condition-aware + planted | Sat 14 Nov | Condition-aware simulator is at least as good as the blind one on held-out Brier and AUC; planted works on Set S |
| **M6** Offline demo | Sat 14 Nov night | Runs offline in under 5 minutes |
| **M7** Full runs + evaluation + stats | Mon 16 – Tue 17 Nov | All raw outputs saved with model IDs and dates; every number produced by a script |
| **M8** Figures + results tables + one-page PDF summary | Tue 17 Nov | 6 figures: pipeline, three-XI example, per-class accuracy, F1 by arm, ΔP(win) by arm under both simulators, noisy-rule curve |
| **M9** Paper draft | Wed 18 Nov | `paper/` with a LaTeX source (AAAI 2027 author kit if downloadable, otherwise a clean `article` template; log which) and a compiled PDF of 4 pages + references. Sections: abstract, introduction (gap), related work, method, experiments, results, limitations, data statement (Cricsheet ODC-By, Open-Meteo CC BY 4.0, image generator licence, no scraping). Every number is pulled from result files by script. Citations marked "from memory" in the reports are verified (`anthropic-skills:deep-research`) or removed. |
| **M10** Final check + handover | Thu 19 Nov | Full test suite passes. `scripts/run_vision_constraints.py --all` reproduces every result from cache. `docs/PROGRESS.md` ends with a handover: how to run, how to demo, what humans must still do (human image check, final read, submission by Fri 20 Nov midday). |

**Fallback rule:** if M5 (condition-aware simulator) fails by **Sat 14 Nov night**, switch to Option 2. Report downstream results only on Set S with the planted simulator, and present Set R as an extraction and selector-agreement study. Log the switch and the reason in `docs/DECISIONS.md`.

**If behind, cut in this order** (new-project version: first make sure M1 and the M6 demo exist):
1. the second hosted model (keep it to 50 scenarios);
2. the local model beyond 30 scenarios;
3. soft-constraint and value-multiplier variants;
4. k = 5, reduced to k = 3.

**Never cut:**
- the rulebook;
- Set S and its image check;
- the extraction metrics;
- arms B1, B3, B4, B5 and Ours-1;
- the condition-aware and blind simulators;
- the leakage and memorisation controls.

---

## 5. Decide yourself vs true blockers

**Decide yourself and log it in `docs/DECISIONS.md`:**
- library and file choices;
- thresholds (fixed before seeing results);
- which model to use;
- how to handle odd matches (rain-shortened, super overs: exclude and log);
- prompt wording;
- what to cut when behind (follow the cut order);
- **switching to the Option 2 fallback** if M5 fails by Sat 14 Nov night.

**API keys:**
- If `GEMINI_API_KEY` or `ANTHROPIC_API_KEY` is missing, do **not** stop. Use the local Qwen3-VL via Ollama (`ollama pull qwen3-vl:8b`, or `:4b` if RAM is under 16 GB) for all VLM arms, and log it.
- If there is no image-generation key, generate images locally with SDXL via `diffusers` on Apple MPS and log the speed.
- If a local install fails, debug it with `superpowers:systematic-debugging` before giving up.

**True blockers.** Do these never, whatever happens:
- deleting files outside this project folder;
- pushing to any remote or merging to main;
- spending over **$10** in API costs (stop paid calls and continue locally);
- scraping restricted sites;
- committing secrets.

If a blocker would be the only way forward, write it in `docs/PROGRESS.md` under **"NEEDS HUMAN"** and continue with everything else.

## 6. Progress log (at each milestone, in `docs/PROGRESS.md`)

Keep it short:
- what was done;
- test output (real, pasted);
- numbers produced so far;
- cost spent;
- what's next;
- any risk to the 15 Nov demo or the 20 Nov deadline.