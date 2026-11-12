# Seeing conditions, writing constraints

VLMs as constraint writers for cricket team selection (T20I men's internationals).
Short workshop paper target: **NeSy-FM @ AAAI 2027**.

> Results in this repo are **simulator-based and exploratory**. The system is not
> claimed to win real matches.

## Research question

Foundation models can turn *written* problems into optimisation models, and MM-OptBench
(2026) reads numbers drawn in diagrams. We test whether a vision-language model (VLM) can
*infer* constraints that a natural photo only **implies** (a dry, cracked pitch implies more
spinners), whether those constraints are **correct**, and whether they **improve the decision**.

## Pipeline

```
condition input (pitch/sky image, or weather JSON)
  -> VLM  (mode "label" | mode "rules")
  -> fixed RULEBOOK (label -> rules)        [skipped in mode "rules"]
  -> RULE CHECKER (validate, combine, feasibility fallback + log)
  -> MILP selector (cricket + environment rules; objective: player value)
  -> XI
  -> SIMULATOR -> P(win) vs opponent's real XI
  -> EVALUATION (are rules correct? do they help?)
```

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # optional: add API keys for live VLM arms
```

## Run

```bash
python scripts/run_tests.py                 # test suite (keep-it-green gate)
python src/data/download.py                  # download Cricsheet into data/raw/ (git-ignored)
python src/data/etl.py                       # parse JSON -> Parquet tables
python scripts/run_vision_constraints.py --all   # reproduce all results from cache
streamlit run app/streamlit_app.py           # offline demo
```

## Data & licences

See `docs/DATA.md`. Cricsheet (ODC-By), Open-Meteo (CC BY 4.0), cricketdata `player_meta`.
No scraping of ESPNcricinfo / Cricbuzz / Getty.

## Project docs

- `docs/PROGRESS.md` — running log (what was done, test output, numbers, cost, risks).
- `docs/DECISIONS.md` — every decision, reason, alternatives.
- `docs/SKILLS_USED.md` — which skills were used, where, why.
- `docs/superpowers/specs/` — the design spec.
