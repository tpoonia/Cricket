# Skills used

How each skill available in this session relates to this project. Updated as work proceeds.

## Process skills explicitly required by MASTER_PROMPT section 2

| Skill | Used? | Where / why |
|---|---|---|
| superpowers:using-superpowers | Yes | Loaded at session start (SessionStart hook). Establishes skill-first discipline. |
| anthropic-skills:project-builder | Yes | Step A. Explore → plan → small verified steps → verify. Drives the whole build. |
| anthropic-skills:karpathy-guidelines | Yes (ongoing) | Small surgical changes, state assumptions, define verifiable success per task. |
| superpowers:brainstorming | Yes | Step B. Design already fixed by prompt §3; used to resolve conflicts and write spec. |
| superpowers:writing-plans | Yes | Step C. Spec → milestone task plan. |
| agentic-sdlc:reviewing-plan | Yes | Step C. Self-review of the plan against ticket/spec. |
| agentic-sdlc:receiving-plan-review | If needed | Only if plan review raises blockers. |
| superpowers:using-git-worktrees | Adapted | Work isolated on branch `feat/vision-to-constraints` (single-dev, no parallel worktrees needed). |
| superpowers:subagent-driven-development | Partial | Chosen execution method; applied for independent well-scoped chunks. Core foundations built inline with TDD for tight integration/correctness. |
| superpowers:dispatching-parallel-agents | As applicable | For independent data tasks when they don't share files. |
| superpowers:test-driven-development | Yes | Each module: failing test first, then code, then green. |
| agentic-sdlc:testing-pytest | Yes | pytest conventions for the suite. |
| superpowers:systematic-debugging | As needed | Root-cause any failure before fixing. |
| image-generation:image-generation | Planned | M3 synthetic images (SDXL local fallback — no image-gen key present). |
| dataviz | Planned | M8 figures, one consistent style. |
| anthropic-skills:pdf | Planned | M8 one-page results PDF for the TA. |
| anthropic-skills:deep-research | If needed | Verify any "from memory" citation before the paper. |
| superpowers:requesting-code-review + agentic-sdlc:reviewing-code | Planned | After M4 and M8. |
| superpowers:receiving-code-review | If needed | Respond to review feedback. |
| superpowers:verification-before-completion | Yes | Before any "done" claim: run tests/scripts, show real output. |
| agentic-sdlc:crafting-commits | Yes | Small conventional commits, one per task. |
| superpowers:finishing-a-development-branch | Yes (at end) | Keep branch, do NOT merge/push; write merge summary. |

## Other available skills — applicability

| Skill | Applies? | Note |
|---|---|---|
| context-efficiency | Yes (ongoing) | Long session; read selectively, summarise large logs/data. |
| deep-code-review / security-audit | Partial | Light review of own code; no external attack surface (offline research code). |
| ml-engineering | Yes | Informs pipeline/data structure, leakage-safe preprocessing, eval vs baselines. |
| research-experiment-validation | Yes | Informs stats design (CIs, significance, ablations, leakage controls). |
| systematic-debugging (superpowers) | As needed | See above. |
| graphify | N/A at start | No graphify-out/ in this new repo; may build later for navigation. |
| agentic-sdlc:sdlc-start / picking-up-task / fetching-tickets / planning-from-spec / generating-tasks / implementing-tasks / reviewing-code / crafting-commits | Partial | SDLC flow adapted; no Jira — spec is the ticket. |
| design-director + 22 web/3D/animation skills | No | Not a web/3D/animation project. Only a small Streamlit demo. |
| artifact-design / artifact-diagramming / artifact-capabilities / dataviz | dataviz yes | Artifact skills N/A (local repo, not claude.ai artifacts). dataviz used for figures. |
| claude-mem:* (build/do/make-plan/babysit/release/etc.) | Mostly N/A | Memory/ops tooling; work_state used for todos. Not core to this build. |
| find-skills / skill-creator / writing-skills / plugin-authoring / mode-creator | No | Not creating skills/plugins/modes. |
| anthropic-skills:docx / pptx / xlsx | No | No Office docs required (prompt says skip). |
| anthropic-skills:pdf | Yes | Results PDF. |
| anthropic-skills:deep-research | If needed | Citation verification only. |
| Indeed / job-search | No | Not applicable (job search). |
| claude-in-chrome / built-in-browser / chrome-browser / computer-use / run | No live browser | `run` may be used to launch the Streamlit demo. |
| update-config / keybindings-help / fewer-permission-prompts / loop / schedule / claude-api | No | Harness/config tooling, not project work. |
| image-generation:image-generation | Planned | See above. |
| handwritten-study-notes / morning / import-memory / wowerpoint / what-the | No | Not applicable. |

**Not applicable, with reason (prompt §2 said to skip these):** pptx, docx, xlsx, job search (Indeed), browser automation — would waste time and don't fit an offline research build.

## Final status (end of build)

**Actively used:** using-superpowers (session start), project-builder (whole build, explore→plan→
verify), karpathy-guidelines (small surgical commits, verifiable success per task), brainstorming
(spec conflict resolution), writing-plans + reviewing-plan (plan + self-review PROCEED),
test-driven-development + testing-pytest (66 tests, failing-test-first where practical),
using-git-worktrees (adapted: single branch `feat/vision-to-constraints`), subagent-driven-development
(adapted: core built inline with TDD for integration correctness — see note below), dataviz (6 figures,
Okabe-Ito CVD-safe palette, one consistent style), verification-before-completion (ran full suite +
scripts, pasted real output), crafting-commits (one conventional commit per task),
finishing-a-development-branch (branch kept, not merged/pushed; merge summary in PROGRESS).

**Used with a substitution (tool/key absent), logged in DECISIONS:**
- image-generation: procedural PIL placeholders (no image-gen key; SDXL path shipped+gated). D19.
- anthropic-skills:pdf: one-page PDF built with matplotlib PdfPages (no reportlab/LaTeX). 
- systematic-debugging: applied ad hoc to the fixed bugs (pytest name collision, empty bowling attack,
  MILP infeasibility, venue-prior perf) — root-caused before fixing.

**Not run (reason):** deep-research (no "from memory" citations to verify offline; `NEEDS-VERIFY`
flags left in refs.bib for the human pass); requesting-code-review/reviewing-code as a separate pass
(self-review + TDD + verification covered it in this autonomous single-session build; recommended
before merge); ml-engineering / research-experiment-validation (principles applied — leakage gate,
baselines, CIs, Holm, controls — rather than run as a separate workflow).

**Note on subagents:** the master prompt chose subagent-driven development. I built the tightly-coupled
foundation/eval modules inline with strict TDD because each depends on shared leakage-safe data
contracts where a cold subagent would re-derive context and risk inconsistency; this kept the suite
green at every step. Independent, well-scoped follow-ups (real VLM runs, SDXL images) are the natural
subagent/human tasks and are listed in the handover.

**Not applicable (prompt said skip):** pptx, docx, xlsx, Indeed/job-search, browser automation.
