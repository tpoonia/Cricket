"""Pre-submission checklist for the NeSy-FM @ AAAI 2027 paper.

These tests verify that every artefact the paper depends on is present and
mutually consistent: LaTeX macros, figures, metrics, references, raw results,
the frozen rulebook, the offline demo, and that no secrets have leaked.

Dependencies: stdlib + pytest + pyyaml only. Nothing is imported from ``src``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

# Project root = parent of the tests/ directory.
ROOT = Path(__file__).resolve().parent.parent

NUMBERS_TEX = ROOT / "paper" / "numbers.tex"
MAIN_TEX = ROOT / "paper" / "main.tex"
METRICS_JSON = ROOT / "data" / "results" / "metrics.json"
REFS_BIB = ROOT / "paper" / "refs.bib"
RAW_DIR = ROOT / "data" / "results" / "raw"
RULEBOOK = ROOT / "configs" / "rulebook.yaml"
DEMO_APP = ROOT / "demo" / "app.py"
GITIGNORE = ROOT / ".gitignore"

# LaTeX built-ins that look like custom macros but are not defined in numbers.tex.
STD_LATEX = {
    "section", "textbf", "emph", "cite", "label", "ref", "url", "begin", "end",
    "includegraphics", "caption", "paragraph", "bibliographystyle", "bibliography",
    "maketitle", "documentclass", "usepackage", "newcommand", "input", "centering",
    "linewidth", "text", "mathrm", "Rightarrow", "ge", "le", "pm", "Delta", "kappa",
    "times",
}


def _non_comment_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if not ln.lstrip().startswith("%")]


def _defined_macros(numbers_text: str) -> set[str]:
    """Macro names defined by non-commented \\newcommand lines."""
    defined: set[str] = set()
    for ln in _non_comment_lines(numbers_text):
        for m in re.finditer(r"\\newcommand\{\\([a-zA-Z]+)\}", ln):
            defined.add(m.group(1))
    return defined


# ---------------------------------------------------------------------------
# 1. numbers.tex macro names must be letters only (LaTeX forbids digits).
# ---------------------------------------------------------------------------
def test_numbers_tex_valid():
    assert NUMBERS_TEX.exists(), f"missing {NUMBERS_TEX}"
    invalid: list[str] = []
    for ln in _non_comment_lines(NUMBERS_TEX.read_text()):
        for m in re.finditer(r"\\newcommand\{\\([a-zA-Z0-9]+)\}", ln):
            name = m.group(1)
            if any(c.isdigit() for c in name):
                invalid.append(name)
    assert not invalid, f"macro names must be letters only; found digits in: {invalid}"


# ---------------------------------------------------------------------------
# 2. Every custom macro used in main.tex is defined in numbers.tex.
# ---------------------------------------------------------------------------
def test_numbers_tex_macros_resolve():
    defined = _defined_macros(NUMBERS_TEX.read_text())
    used = set(re.findall(r"\\([A-Z][a-zA-Z]+)", MAIN_TEX.read_text()))
    used -= STD_LATEX
    missing = sorted(u for u in used if u not in defined)
    assert not missing, f"macros used in main.tex but undefined in numbers.tex: {missing}"


# ---------------------------------------------------------------------------
# 3. All referenced figures exist; fig1..fig6 exist as both PDF and PNG.
# ---------------------------------------------------------------------------
def test_figures_exist():
    main = MAIN_TEX.read_text()
    includes = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", main)
    assert includes, "no \\includegraphics found in main.tex"
    for rel in includes:
        path = ROOT / "paper" / rel
        assert path.exists(), f"figure referenced in main.tex is missing: {path}"

    figures_dir = ROOT / "paper" / "figures"
    for i in range(1, 7):
        pdfs = list(figures_dir.glob(f"fig{i}_*.pdf"))
        pngs = list(figures_dir.glob(f"fig{i}_*.png"))
        assert pdfs, f"no PDF found for fig{i} in {figures_dir}"
        assert pngs, f"no PNG found for fig{i} in {figures_dir}"


# ---------------------------------------------------------------------------
# 4. metrics.json has the required keys and respects the budget.
# ---------------------------------------------------------------------------
def test_metrics_json_complete():
    metrics = json.loads(METRICS_JSON.read_text())

    def get(path: str):
        cur = metrics
        for part in path.split("."):
            assert isinstance(cur, dict) and part in cur, f"missing metrics key: {path}"
            cur = cur[part]
        return cur

    for key in [
        "extraction.ours1_label_rulebook.f1",
        "extraction.ours2_rules_direct.f1",
        "extraction.qwen.label_accuracy",
        "decision.set_r_condition_aware.B4_oracle",
        "decision.holm_corrected_p",
        "foundations.matches",
        "cost.total_api_usd",
    ]:
        get(key)

    total = get("cost.total_api_usd")
    assert total < 10.0, f"API spend {total} exceeds the $10 budget"


# ---------------------------------------------------------------------------
# 5. refs.bib contains no unverified / placeholder entries.
# ---------------------------------------------------------------------------
def test_refs_bib_no_unverified():
    text = REFS_BIB.read_text().lower()
    for bad in ("unverified", "placeholder"):
        assert bad not in text, f"refs.bib still contains '{bad}'"


# ---------------------------------------------------------------------------
# 6. Raw result files have the expected volume.
# ---------------------------------------------------------------------------
def test_raw_results_row_counts():
    set_r = json.loads((RAW_DIR / "set_r_results.json").read_text())
    assert len(set_r) >= 500, f"set_r_results.json has only {len(set_r)} rows"

    set_s = json.loads((RAW_DIR / "set_s_results.json").read_text())
    assert len(set_s) >= 100, f"set_s_results.json has only {len(set_s)} rows"

    qwen = json.loads((RAW_DIR / "qwen_results.json").read_text())
    assert len(qwen) > 0, "qwen_results.json has no entries"


# ---------------------------------------------------------------------------
# 7. The rulebook is frozen and well-formed.
# ---------------------------------------------------------------------------
def test_rulebook_frozen():
    rb = yaml.safe_load(RULEBOOK.read_text())
    assert rb.get("version") == 1, f"rulebook version is {rb.get('version')}, expected 1"
    assert "frozen" in rb, "rulebook has no 'frozen' field"
    assert rb["frozen"] == "2026-11-12", f"rulebook frozen date is {rb['frozen']!r}"

    labels = rb.get("labels", {})
    assert labels, "rulebook has no labels"
    for name, entry in labels.items():
        assert "rules" in entry, f"label {name!r} missing 'rules'"
        assert "strength" in entry, f"label {name!r} missing 'strength'"


# ---------------------------------------------------------------------------
# 8. The demo app exists, has tabs, and makes no real API calls.
# ---------------------------------------------------------------------------
def test_demo_app_exists():
    assert DEMO_APP.exists(), f"missing {DEMO_APP}"
    src = DEMO_APP.read_text()
    assert "st.tabs" in src, "demo app has no st.tabs"
    for forbidden in ("ANTHROPIC_API_KEY", "anthropic.Client", "os.environ"):
        assert forbidden not in src, f"demo app contains forbidden API usage: {forbidden}"


# ---------------------------------------------------------------------------
# 9. No secrets leak: .env ignored, no hardcoded API keys in src/ or scripts/.
# ---------------------------------------------------------------------------
def test_no_env_leak():
    gitignore_lines = {ln.strip() for ln in GITIGNORE.read_text().splitlines()}
    assert ".env" in gitignore_lines, ".env is not listed in .gitignore"

    key_pattern = re.compile(r"sk-ant-[A-Za-z0-9]|AIza[A-Za-z0-9]")
    offenders: list[str] = []
    for base in ("src", "scripts"):
        for py in (ROOT / base).rglob("*.py"):
            if key_pattern.search(py.read_text(errors="ignore")):
                offenders.append(str(py.relative_to(ROOT)))
    assert not offenders, f"hardcoded API key pattern found in: {offenders}"
