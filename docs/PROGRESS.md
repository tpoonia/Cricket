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
