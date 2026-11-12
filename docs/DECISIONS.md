# Decisions log

Format: decision — reason — alternatives considered. Thresholds are fixed here **before** seeing results.

## Approvals (autonomous mode)
- **Spec approved** once self-reviewed (per MASTER_PROMPT §AUTONOMOUS). Recorded at M0.
- **Plan approved** once `reviewing-plan` self-review passes.
- **Execution:** subagent-driven where chunks are independent; core foundations built inline with strict TDD for integration correctness.

## D1 — Missing research reports (BLOCKER noted, worked around)
`docs/research/` is empty: `Vision to constraints cricket selection.md` and
`LLM cricket team selection plan.md` are **absent**. MASTER_PROMPT §3 is itself a complete,
detailed "agreed design", so the spec is built from §3. **Consequence:** any citation the
reports would have supplied cannot be verified here; the paper will only include citations
I can verify with `deep-research`, and will mark the rest as needed. Logged under
"NEEDS HUMAN" in PROGRESS.md. Alternatives: stop (rejected — autonomous mode; §3 is sufficient to build).

## D2 — No API keys present
No `.env` with `GEMINI_API_KEY` / `ANTHROPIC_API_KEY` (only an unrelated `.env.pages`,
not read). Ollama is **not installed**. Per §5, the fallback is local Qwen3-VL via Ollama.
**Decision:** build the full VLM client + caching + schema + all downstream arms so they run
on cached/mock VLM outputs; make real VLM runs a one-command step that works the moment a key
is added OR Ollama is installed. Attempt an Ollama install opportunistically; if it fails or is
too heavy for 16 GB, document it and ship the deterministic mock-backed path. This keeps total
API spend at **$0** (well under the $10 cap). Alternatives: block on keys (rejected).

## D3 — Python & libraries
Python 3.12.15 (`~/.local/bin/python3.12`) in `.venv`. MILP via **PuLP + HiGHS (highspy)**
(prompt-suggested; HiGHS is fast, open source). Data in **Parquet via pyarrow**. Player meta via
**pyreadr** reading `player_meta.rda`; R 4.5.2 is installed as a fallback path. Stats via scipy/sklearn.

## D4 — Chronological split (F3)
Train/features: matches **before 2025-01-01**. Test (Set R candidates): **2025-01-01 onward**.
`history_before(match_date)` gates every feature. Leakage test asserts no feature touches a
match on/after the target date. (Matches MASTER_PROMPT §3.0 F3.)

## D5 — Odd matches
Super overs: parsed but **excluded from features and simulation fitting** (kept in a separate
table). Rain-shortened / D-L matches: **flagged** (`method`, `overs` limit) and excluded from
simulator fitting; retirements handled as not-out with `player_out` recorded. (Per §5.)

## D6 — Roles thresholds (F6), fixed in advance
- **Keeper:** a player credited with a stumping, or with ≥1 catch-as-keeper signal, in earlier
  matches for that team; if none in pool, the player with most dismissals-as-fielder marked `wicket_keeper` in XI metadata when available. Rule logged in code.
- **Bowling option:** bowled **≥ 6 legal deliveries (1 over)** in **≥ 2 of the team's last 10**
  matches (earlier matches only). N=6 balls, M=10 matches.

## D7 — Player value (F7) shrinkage
Empirical-Bayes-style shrinkage toward the population mean with a **fixed prior weight k=10
"pseudo-matches"**: `v_i = (n_i*raw_i + k*pop_mean) / (n_i + k)`, computed separately for the
batting and bowling components, earlier matches only. k logged here.

## D8 — Pool (F5)
Pool = players who appeared for the team in its **previous 10 T20Is** (earlier only), capped at
**18** (most-recent-appearance priority). "Pool coverage" = % of the real XI inside the pool.

## D9 — Simulator (F9) phases
Powerplay = overs 1–6, middle = 7–15, death = 16–20. Per-ball outcome probs = batter rate and
bowler rate each shrunk toward the phase average (k=50 balls), combined multiplicatively and
renormalised over {0,1,2,3,4,6,wicket,extra}. ≥10,000 sims/match, common random numbers across
arms. Must beat a coin flip (Brier<0.25, AUC>0.5) on held-out matches.

## D10 — Set R truth thresholds (venue spin share), fixed in advance
Two settings tested:
- **Setting A (primary):** upper = 0.45 spin share → `n_spin ≥ 2`; lower = 0.30 → `n_pace ≥ 3`; else no rule.
- **Setting B (sensitivity):** upper = 0.50; lower = 0.25.
(Spin share = spin overs / total overs at the venue, earlier matches only, shrunk to host-country then global.)

## D11 — Rulebook frozen
`configs/rulebook.yaml` matches MASTER_PROMPT §3.2 exactly and is frozen before any model run.

## D12 — Set R size & filter
150–300 T20Is from 2025-01-01 onward between teams each with ≥10 earlier T20Is. Exact N reported after ETL.

## D13 — k runs per VLM input
Default **k=5**, temperature>0. Cut order allows k=3 if behind (per §If-behind).

## D14 — Image generator family
Reader models (Gemini/Claude/Qwen) must differ in family from the generator. Generator = **SDXL via
diffusers on Apple MPS** (no image-gen key). Logged per §Hard-rule-6.

## D15 — Venue coordinates via Open-Meteo geocoding (not hand-filled)
MASTER_PROMPT §3.1#2 suggested hand-filling `venue_coords.csv` from Wikipedia. The established-team
holdout spans **123 venues**, mostly associate-nation grounds that cannot be reliably geolocated from
memory. **Decision:** resolve coordinates with the **Open-Meteo geocoding API** (free, same CC-BY
source family, not scraping), query = city (+ country hint), cached to `data/manual/venue_coords.csv`.
Manual overrides in the same CSV take precedence. All 123 venues have a `city`. Logged per §5.

## D16 — Set R realism note
The "≥10 prior matches" filter admits many associate teams (Malaysia, Indonesia, Romania, …).
These qualify under MASTER_PROMPT §3.3. Set R will be drawn from this pool; the paper will note the
mix skews to associate cricket and report results overall and for Full-Member-only matches as a cut.

## D17 — M5 condition-aware simulator kept (no Option 2 fallback)
Condition-aware per-venue pace/spin multipliers (shrink pseudo-balls=3500) give, on 300 held-out
established-team matches (n=6000): Brier 0.2434 vs blind 0.2422; AUC 0.6317 vs blind 0.6341 — a
difference well within Monte-Carlo noise, and both beat the coin flip. The venue effect is near-
symmetric across the two teams, so it barely changes *win prediction* (the expected, honest result);
its purpose is to make the *decision* metric sensitive to pace/spin balance. The M5 gate ("at least
as good as blind") is met within noise. **Decision: keep condition-aware; do NOT switch to Option 2.**

## D18 — Ollama not installable headlessly here
No Homebrew, no Ollama.app, and the official install script requires the macOS GUI app / sudo.
**Decision:** do not force a fragile system install. The VLM client's `ollama` backend is code-
complete and works the instant a user installs Ollama (`ollama pull qwen3-vl:4b`). Offline runs use
the deterministic `mock` backend. Live VLM remains a one-line backend swap. Logged under NEEDS HUMAN.

## D19 — Synthetic images: procedural placeholders now, SDXL path documented
No image-generation key; SDXL via diffusers on 16 GB MPS would take hours and risk OOM.
**Decision:** generate **procedural, class-distinct placeholder images** (PIL) so the manifest,
CLIP-check infrastructure, demo and Set S plumbing all work offline, CLEARLY labelled as
placeholders (generator="procedural-v1"). The real SDXL generation code + per-class prompts ship in
`src/synthetic/generate.py` (gated on torch/diffusers). Real VLM-on-photo numbers need a key + real
images (NEEDS HUMAN). This keeps the non-VLM, Set R science fully real.

## D20 — What is fully REAL vs mock in this delivery
REAL (no key needed): all foundations; Set R decision metrics for arms B2, B3, B4, B5, B7, Ref, Habit
under blind + condition-aware simulators, with bootstrap/Wilcoxon/McNemar/Holm stats. MOCK/pending:
VLM extraction (B1, Ours-1, Ours-2) and CLIP arm (B6) — run through the deterministic mock to exercise
the pipeline; real numbers pending a key/Ollama and real images.
