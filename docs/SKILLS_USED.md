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
