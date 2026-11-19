"""NeSy Cricket XI — a polished three-part class demo (NeSy-FM @ AAAI 2027).

The story in one line: *a photo of the pitch should change who you pick* — this
pipeline reads a condition photo, writes a rule, and picks a legal team, end to end.

Three parts (tabs), one pipeline:
  ①  The Story       — hero + a 4-step stepper with plain-language captions.
  ②  Under the Hood  — the backend made visible: the label, the rule (bounds),
                        the XI table with pace/spin counts, and a win-probability
                        bar before vs after the rule (from app/demo_backend.py).
  ③  Why this matters — the three findings and a clear "simulation only" notice.

ZERO API calls. Perception uses the offline deterministic **mock** reader (default);
win probabilities are the recorded, frozen simulator means in
data/results/raw/set_r_results.json; every number is read from data files — nothing
is hardcoded or invented, and missing data shows a warning, never a fake value.

Run (full experience, incl. live MILP team selection — needs the PuLP solver):
    .venv/bin/python -m streamlit run demo/app.py

It also runs under a plain interpreter:
    python3 -m streamlit run demo/app.py
If that interpreter has no PuLP, every panel still renders; only the live MILP XI
table is replaced by a short notice (the recorded win-probability bars still show).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make src/ and app/ importable regardless of where streamlit is launched from.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# `streamlit run demo/app.py` registers THIS script as module ``app``, which shadows the
# real ``app/`` package — so ``import app.demo_backend`` fails under `streamlit run` (though
# not under pytest/AppTest, which use a different module name). Import it by directory.
sys.path.insert(0, str(ROOT / "app"))
from demo_backend import (  # noqa: E402
    HAS_SOLVER,
    demo_conditions,
    perceive,
    three_xis,
    win_probability,
)

# ---------------------------------------------------------------------------
# Streamlit-version-aware width kwarg (1.40 uses use_container_width;
# 1.49+ prefers width="stretch"). Keeps the app warning-free on either venv.
# ---------------------------------------------------------------------------
_ST_VER = tuple(int(p) for p in st.__version__.split(".")[:2])


def wide(stretch: bool = True) -> dict:
    if _ST_VER >= (1, 49):
        return {"width": "stretch" if stretch else "content"}
    return {"use_container_width": bool(stretch)}


# ---------------------------------------------------------------------------
# Okabe-Ito palette — colour-vision-deficiency safe, identical to the paper figures.
# ---------------------------------------------------------------------------
BLUE = "#0072B2"     # before / primary / structure
GREEN = "#009E73"    # after (our rule) / positive
VERM = "#D55E00"     # wrong rule / removed
AMBER = "#E69F00"    # accent / perception
PURPLE = "#CC79A7"
INK = "#13232c"
MUTED = "#5b6b73"
SURFACE = "#ffffff"
PANEL = "#eef3f6"
LINE = "rgba(19,35,44,.12)"

IMG_DIR = ROOT / "data" / "synthetic" / "images"
RESULTS = ROOT / "data" / "results"
RULEBOOK_FROZEN = "2026-11-12"

st.set_page_config(page_title="NeSy Cricket XI", page_icon="🏏", layout="wide")

# ---------------------------------------------------------------------------
# Design system — one headline font (Space Grotesk), one body font (Inter),
# card layout, Okabe-Ito tokens, subtle fade/stagger motion (reduced-motion safe),
# WCAG-AA contrast, visible focus rings.
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
      :root {{
        --blue:{BLUE}; --green:{GREEN}; --verm:{VERM}; --amber:{AMBER}; --purple:{PURPLE};
        --ink:{INK}; --muted:{MUTED}; --surface:{SURFACE}; --panel:{PANEL}; --line:{LINE};
        --head:'Space Grotesk', ui-sans-serif, system-ui, sans-serif;
        --body:'Inter', ui-sans-serif, system-ui, -apple-system, sans-serif;
        --r:16px; --dur:.42s; --ease:cubic-bezier(.2,.7,.2,1);
        color-scheme: light;  /* fallback if the light theme config isn't picked up */
      }}
      [data-testid="stHeader"] {{ background: transparent; }}
      html, body, [class*="st-"], .stMarkdown, p, li, div {{ font-family: var(--body); }}
      h1, h2, h3, h4, h5 {{ font-family: var(--head); letter-spacing:-.01em; color:var(--ink); }}
      .stApp {{ background:
        radial-gradient(1200px 520px at 15% -10%, rgba(0,114,178,.06), transparent 60%),
        radial-gradient(1000px 520px at 100% 0%, rgba(0,158,115,.06), transparent 55%),
        #f7fafc; }}
      .block-container {{ padding-top: 1.3rem; max-width: 1140px; }}

      /* ---- entrance motion (subtle; disabled for reduced-motion) ---- */
      @keyframes riseIn {{ from {{ opacity:0; transform: translateY(12px); }}
                           to {{ opacity:1; transform:none; }} }}
      @media (prefers-reduced-motion: no-preference) {{
        .hero, div[data-testid="stVerticalBlockBorderWrapper"] {{
          animation: riseIn var(--dur) var(--ease) both;
        }}
        /* stagger the step cards left-to-right */
        div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-child(1) div[data-testid="stVerticalBlockBorderWrapper"] {{ animation-delay:.02s; }}
        div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-child(2) div[data-testid="stVerticalBlockBorderWrapper"] {{ animation-delay:.10s; }}
        div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-child(3) div[data-testid="stVerticalBlockBorderWrapper"] {{ animation-delay:.18s; }}
        div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-child(4) div[data-testid="stVerticalBlockBorderWrapper"] {{ animation-delay:.26s; }}
        .wfill {{ animation: grow var(--dur) var(--ease) both; }}
        @keyframes grow {{ from {{ width:0; }} to {{ width: var(--w); }} }}
      }}

      /* ---- hero ---- */
      .hero {{
        background: linear-gradient(120deg, {BLUE} 0%, #065b8c 52%, #023a5b 100%);
        color:#fff; border-radius:22px; padding:1.7rem 2rem; margin:.2rem 0 1rem 0;
        box-shadow:0 18px 44px rgba(2,58,91,.28);
      }}
      .hero .kick {{ text-transform:uppercase; letter-spacing:.16em; font-size:.72rem;
        font-weight:700; color:#bfe3f6; margin-bottom:.55rem; }}
      .hero h1 {{ color:#fff; font-size:2.15rem; line-height:1.1; margin:0 0 .5rem 0; }}
      .hero p {{ color:#eaf5fc; font-size:1.12rem; line-height:1.5; margin:0; max-width:64ch; }}
      .hero .chips {{ margin-top:1rem; display:flex; gap:.5rem; flex-wrap:wrap; }}
      .hchip {{ background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.28);
        color:#fff; border-radius:999px; padding:.3rem .8rem; font-size:.82rem; font-weight:600; }}

      /* ---- step cards (inside bordered containers) ---- */
      .steptop {{ display:flex; align-items:center; gap:.6rem; margin:-.1rem 0 .55rem 0; }}
      .snum {{ flex:0 0 auto; width:1.9rem; height:1.9rem; border-radius:50%;
        display:inline-flex; align-items:center; justify-content:center; font-family:var(--head);
        font-weight:700; color:#fff; background:var(--blue); font-size:1rem; }}
      .snum.g {{ background:var(--green); }} .snum.a {{ background:var(--amber); }}
      .snum.p {{ background:var(--purple); }}
      .stitle {{ font-family:var(--head); font-weight:600; font-size:1.0rem; color:var(--ink); line-height:1.15; }}
      .scap {{ color:var(--muted); font-size:.88rem; line-height:1.45; margin:.5rem 0 0 0; }}

      /* ---- badges & chips ---- */
      .badge {{ display:inline-block; padding:.2rem .66rem; border-radius:8px;
        font-size:.8rem; font-weight:700; }}
      .badge.perc {{ background:rgba(230,159,0,.16); color:#7a5300; border:1px solid rgba(230,159,0,.45); }}
      .badge.ok   {{ background:rgba(0,158,115,.15); color:#005e45; border:1px solid rgba(0,158,115,.4); }}
      .badge.grey {{ background:rgba(19,35,44,.07); color:var(--muted); border:1px solid var(--line); }}
      .cons {{ display:inline-block; font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
        background:#e9f2f8; color:#06415f; border:1px solid rgba(0,114,178,.28);
        border-radius:9px; padding:.26rem .66rem; margin:.18rem .32rem .18rem 0; font-size:.95rem; font-weight:600; }}
      .cons.none {{ background:#f1f1f1; color:#555; border-color:#ddd; }}

      /* ---- player rows ---- */
      .pl {{ display:flex; justify-content:space-between; gap:.5rem; padding:.3rem .6rem;
        border-radius:9px; margin:.2rem 0; font-size:.9rem; background:#f1f4f7; color:var(--ink);
        border:1px solid var(--line); }}
      .pl .t {{ color:var(--muted); font-size:.76rem; font-weight:700; text-transform:uppercase; letter-spacing:.03em; }}
      .pl.add {{ background:#ddf1ea; border-color:var(--green); color:#07422f; font-weight:600; }}
      .pl.add .t {{ color:#07422f; }}
      .pl.rem {{ background:#fae1d4; border-color:var(--verm); color:#7a3400; text-decoration:line-through; }}
      .pl.rem .t {{ color:#7a3400; }}

      /* ---- count pills ---- */
      .counts {{ display:flex; gap:.5rem; flex-wrap:wrap; margin:.5rem 0 .2rem; }}
      .cpill {{ border-radius:11px; padding:.35rem .7rem; font-size:.84rem; font-weight:700;
        border:1px solid var(--line); background:#fff; color:var(--ink); }}
      .cpill b {{ font-family:var(--head); }}
      .cpill.pace {{ border-color:rgba(0,114,178,.4); background:rgba(0,114,178,.08); }}
      .cpill.spin {{ border-color:rgba(204,121,167,.5); background:rgba(204,121,167,.1); }}

      /* ---- win-probability bars ---- */
      .wrow {{ margin:.55rem 0; }}
      .wlab {{ display:flex; justify-content:space-between; font-size:.86rem; margin-bottom:.22rem; color:var(--ink); }}
      .wlab .nm {{ font-weight:600; }} .wlab .vl {{ font-family:var(--head); font-weight:700; }}
      .wtrack {{ height:16px; border-radius:999px; background:#e7edf1; border:1px solid var(--line); overflow:hidden; }}
      .wfill {{ height:100%; width: var(--w); border-radius:999px; }}
      .wfill.before {{ background:var(--blue); }}
      .wfill.after  {{ background:var(--green); }}
      .wfill.wrong  {{ background:var(--verm); opacity:.85; }}

      /* ---- notes / findings ---- */
      .note {{ background:#eaf3f9; border:1px solid rgba(0,114,178,.25); border-left:5px solid var(--blue);
        border-radius:12px; padding:.85rem 1.05rem; color:var(--ink); font-size:.95rem; line-height:1.5; }}
      .note.warn {{ background:#fdf0e8; border-color:rgba(213,94,0,.35); border-left-color:var(--verm); }}
      .fcard h4 {{ margin:.1rem 0 .35rem 0; font-size:1.02rem; }}
      .fbig {{ font-family:var(--head); font-weight:700; font-size:1.7rem; line-height:1; }}
      .fbig.b {{ color:var(--blue); }} .fbig.g {{ color:var(--green); }} .fbig.r {{ color:var(--verm); }}

      a, .stMarkdown a {{ color:var(--blue); }}
      :focus-visible {{ outline:3px solid var(--amber); outline-offset:2px; border-radius:6px; }}
      [data-testid="stImage"] img {{ border-radius:12px; }}
      .stTabs [data-baseweb="tab"] {{ font-family:var(--head); font-weight:600; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# Cached loaders (PuLP-free)
# ===========================================================================
@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    return json.loads((RESULTS / "metrics.json").read_text())


@st.cache_resource(show_spinner="Loading squads and player pools…")
def get_teams() -> list[str]:
    from src.data.splits import holdout_matches, load_tables

    t = load_tables()
    hm = holdout_matches(t["matches"])
    return sorted(pd.concat([hm["team1"], hm["team2"]]).value_counts().head(12).index.tolist())


@st.cache_data(show_spinner="Running the MILP team selector…")
def cached_three_xis(team: str) -> dict:
    return three_xis(team)  # requires PuLP; only called when HAS_SOLVER


metrics: dict = {}
try:
    metrics = load_metrics()
except Exception as exc:  # pragma: no cover
    st.error(f"Could not load data/results/metrics.json: {exc}")


# ===========================================================================
# Small helpers
# ===========================================================================
_FEAT = {"n_pace": "seamers", "n_spin": "spinners"}


def constraint_math(r: dict) -> str:
    f, lo, hi = r.get("feature"), int(r.get("lo", 0)), int(r.get("hi", 11))
    if lo <= 0 and hi >= 11:
        return f"{f} (free)"
    if lo > 0 and hi >= 11:
        return f"{f} ≥ {lo}"
    if lo <= 0 and hi < 11:
        return f"{f} ≤ {hi}"
    return f"{lo} ≤ {f} ≤ {hi}"


def rule_chips(rules: list[dict]) -> str:
    if not rules:
        return "<span class='cons none'>∅ no environment constraint</span>"
    return " ".join(f"<span class='cons'>{constraint_math(r)}</span>" for r in rules)


def plain_rule(label: str, rules: list[dict]) -> str:
    if not rules:
        return (f"A <b>{label}</b> pitch is neutral — the rulebook adds no constraint, "
                "so the optimiser is free to pick the best available XI.")
    parts = []
    for r in rules:
        name = _FEAT.get(r["feature"], r["feature"])
        lo, hi = int(r["lo"]), int(r["hi"])
        if lo > 0 and hi >= 11:
            parts.append(f"at least <b>{lo} {name}</b>")
        elif lo <= 0 and hi < 11:
            parts.append(f"at most <b>{hi} {name}</b>")
        elif lo > 0 and hi < 11:
            parts.append(f"between <b>{lo} and {hi} {name}</b>")
    return f"A <b>{label}</b> pitch means the team must play " + " and ".join(parts) + "."


def players_html(names: list[str], types: list[str], mark: set[str], klass: str) -> str:
    out = ""
    for nm, ty in zip(names, types):
        cls = f"pl {klass}" if nm in mark else "pl"
        out += f"<div class='{cls}'><span>{nm}</span><span class='t'>{ty}</span></div>"
    return out


def count_pills(n_pace: int, n_spin: int) -> str:
    return (f"<div class='counts'><span class='cpill pace'>🔵 <b>{n_pace}</b> pace</span>"
            f"<span class='cpill spin'>🟣 <b>{n_spin}</b> spin</span></div>")


def win_bars(wp: dict) -> str:
    """Render before / after (and a muted wrong-rule reference) as scaled bars."""
    bars = wp.get("bars", {})
    spec = [
        ("before", "before", "Before — cricket rules only", "No environment rule"),
        ("after", "after", "After — our rule applied", "Rulebook constraint added"),
        ("wrong", "wrong", "If the rule were wrong (pace↔spin swap)", "Reference only"),
    ]
    html = ""
    for role, cls, title, sub in spec:
        b = bars.get(role)
        if not b:
            continue
        pct = max(0.0, min(1.0, b["p_win"])) * 100
        html += (
            f"<div class='wrow'><div class='wlab'><span class='nm'>{title}</span>"
            f"<span class='vl'>{b['p_win']*100:.1f}%</span></div>"
            f"<div class='wtrack'><div class='wfill {cls}' style='--w:{pct:.1f}%'></div></div>"
            f"<div class='scap' style='margin-top:.15rem'>{sub} · arm <code>{b['arm']}</code></div></div>"
        )
    return html


# ===========================================================================
# HERO
# ===========================================================================
st.markdown(
    """
    <div class='hero'>
      <div class='kick'>Neuro-symbolic team selection · NeSy-FM @ AAAI 2027</div>
      <h1>A photo of the pitch should change who you pick.</h1>
      <p>A vision-language model reads the conditions, a frozen rulebook turns that reading
      into a hard constraint, and an optimiser picks the best <b>legal</b> starting XI —
      then a simulator checks whether it was worth it. No human in the loop.</p>
      <div class='chips'>
        <span class='hchip'>🔒 Zero API calls · fully offline</span>
        <span class='hchip'>🧪 Simulation study — not match advice</span>
        <span class='hchip'>❄️ Rulebook frozen """ + RULEBOOK_FROZEN + """</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- shared controls (apply to every tab) ----------------------------------
try:
    teams = get_teams()
except Exception as exc:
    st.error(f"Could not load squads: {exc}")
    st.stop()

conds = {c["label"]: c for c in demo_conditions()}
cond_labels = list(conds)  # green, dry, flat
_pretty = {"green": "🌱 Green (seaming) pitch", "dry": "🏜️ Dry (turning) pitch",
           "flat": "➖ Flat (neutral) pitch"}

cc = st.columns([2, 3])
with cc[0]:
    team = st.selectbox("Team", teams, index=teams.index("India") if "India" in teams else 0)
with cc[1]:
    label = st.radio("Pitch condition", cond_labels,
                     format_func=lambda c: _pretty.get(c, c), horizontal=True)

cond = conds[label]
perc = perceive(label, cond["image_id"])            # offline mock reading (PuLP-free)
wp = win_probability(team)                           # recorded Set R means (PuLP-free)
live = cached_three_xis(team) if HAS_SOLVER else None
lc = live["conditions"][label] if live else None     # this condition's selection


# ===========================================================================
# TABS
# ===========================================================================
tab_story, tab_hood, tab_why = st.tabs(
    ["①  The Story", "②  Under the Hood", "③  Why this matters"]
)

# ---------------------------------------------------------------------------
# ①  THE STORY — hero already shown; here the 4-step stepper + plain captions
# ---------------------------------------------------------------------------
with tab_story:
    st.markdown(f"#### From photo to team — for a **{label}** pitch, **{team}**")
    st.caption("Four steps, left to right. Every step is real code running offline — "
               "open **Under the Hood** for the numbers behind each one.")

    s1, s2, s3, s4 = st.columns(4)

    with s1:
        with st.container(border=True):
            st.markdown("<div class='steptop'><span class='snum'>1</span>"
                        "<span class='stitle'>A photo of the pitch</span></div>",
                        unsafe_allow_html=True)
            img = IMG_DIR / f"{cond['image_id']}.png"
            if img.exists():
                st.image(str(img), **wide())
            else:
                st.warning("image missing")
            st.markdown("<p class='scap'>A single image of the ground — the only input. "
                        "(Synthetic SDXL-Turbo condition image from the study.)</p>",
                        unsafe_allow_html=True)

    with s2:
        with st.container(border=True):
            st.markdown("<div class='steptop'><span class='snum a'>2</span>"
                        "<span class='stitle'>The AI names the condition</span></div>",
                        unsafe_allow_html=True)
            st.markdown(f"<span class='badge perc'>reads “{perc['perceived']}”</span>",
                        unsafe_allow_html=True)
            st.markdown(f"<p class='scap'>An offline vision-language reader looks at the photo "
                        f"and calls it <b>{perc['perceived']}</b> "
                        f"(confidence {perc['confidence']:.2f}, {perc['k']} samples). "
                        "No text, no network — just the picture.</p>", unsafe_allow_html=True)

    with s3:
        with st.container(border=True):
            st.markdown("<div class='steptop'><span class='snum g'>3</span>"
                        "<span class='stitle'>The rulebook writes the rule</span></div>",
                        unsafe_allow_html=True)
            st.markdown(rule_chips(cond["rules"]), unsafe_allow_html=True)
            st.markdown(f"<p class='scap'>{plain_rule(label, cond['rules'])} "
                        f"<span class='badge grey'>strength · {cond.get('strength','?')}</span></p>",
                        unsafe_allow_html=True)

    with s4:
        with st.container(border=True):
            st.markdown("<div class='steptop'><span class='snum p'>4</span>"
                        "<span class='stitle'>Team picked & win chance</span></div>",
                        unsafe_allow_html=True)
            if lc is not None:
                st.markdown(count_pills(lc["n_pace"], lc["n_spin"]), unsafe_allow_html=True)
                after = wp.get("bars", {}).get("after", {})
                if after:
                    st.markdown(f"<p class='scap'>The optimiser picks the best legal XI "
                                f"(<b>{lc['n_pace']} pace, {lc['n_spin']} spin</b>). "
                                f"Its simulated win chance is "
                                f"<b>{after['p_win']*100:.1f}%</b>.</p>", unsafe_allow_html=True)
            else:
                after = wp.get("bars", {}).get("after", {})
                pct = f"{after['p_win']*100:.1f}%" if after else "—"
                st.markdown(f"<span class='badge grey'>win chance {pct}</span>",
                            unsafe_allow_html=True)
                st.markdown("<p class='scap'>Live team selection needs the PuLP solver "
                            "(run via <code>.venv</code>). The recorded win chance is shown; "
                            "see <b>Under the Hood</b>.</p>", unsafe_allow_html=True)

    st.markdown(
        "<div class='note'>The surprise: a strong squad is often <b>already</b> "
        "condition-appropriate, so the correct rule can reshape the lineup yet barely move the "
        "win chance. The rule earns its keep by stopping <i>wrong</i> picks — see "
        "<b>Why this matters</b>.</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# ②  UNDER THE HOOD — the backend made visible (from app/demo_backend.py)
# ---------------------------------------------------------------------------
with tab_hood:
    st.markdown(f"#### The backend, made visible — **{team}**, **{label}** pitch")
    st.caption("Everything here comes from app/demo_backend.py: the perceived label, the "
               "rulebook's rule, the XI the optimiser returns, and the recorded win probability.")

    cL, cR = st.columns(2)
    with cL:
        with st.container(border=True):
            st.markdown("**1 · The label the reader produced**")
            st.markdown(f"<span class='badge perc'>● mock reader (offline)</span> &nbsp; "
                        f"perceived <b>{perc['perceived']}</b> · confidence "
                        f"{perc['confidence']:.2f} · {perc['k']} samples", unsafe_allow_html=True)
            st.caption("The default reader is a deterministic offline mock — nothing is sent "
                       "anywhere. In the paper this slot is the real Claude Haiku 5.5 reading.")
    with cR:
        with st.container(border=True):
            st.markdown("**2 · The rule the frozen rulebook wrote**")
            st.markdown(rule_chips(cond["rules"]), unsafe_allow_html=True)
            st.markdown(f"<span class='badge grey'>label · {label}</span> "
                        f"<span class='badge grey'>strength · {cond.get('strength','?')}</span>",
                        unsafe_allow_html=True)
            st.caption(f"Mapping is fixed: rulebook v1, frozen {RULEBOOK_FROZEN}, before any run.")

    st.markdown("##### 3 · The XI the optimiser picked — before vs after the rule")
    if lc is not None:
        base_set = set(lc["base_xi_names"])
        after_set = set(lc["xi_names"])
        added = after_set - base_set
        removed = base_set - after_set
        xc = st.columns(2)
        with xc[0]:
            with st.container(border=True):
                st.markdown("**Before** — cricket rules only")
                st.markdown(count_pills(lc["base_n_pace"], lc["base_n_spin"]), unsafe_allow_html=True)
                st.markdown(players_html(lc["base_xi_names"], lc["base_xi_types"], removed, "rem"),
                            unsafe_allow_html=True)
        with xc[1]:
            with st.container(border=True):
                st.markdown("**After** — the environment rule added")
                st.markdown(count_pills(lc["n_pace"], lc["n_spin"]), unsafe_allow_html=True)
                st.markdown(players_html(lc["xi_names"], lc["xi_types"], added, "add"),
                            unsafe_allow_html=True)
        if added or removed:
            st.markdown(
                f"<p class='scap'>The rule swapped <b>{len(added)}</b> player(s): "
                f"pace {lc['base_n_pace']}→{lc['n_pace']}, spin {lc['base_n_spin']}→{lc['n_spin']}. "
                "Green = swapped in, struck-through red = swapped out.</p>",
                unsafe_allow_html=True)
        else:
            st.success("The rule did not change the XI — the value-maximising team already "
                       "satisfied it. (The common case; see Why this matters.)")

        met = pd.DataFrame([
            {"XI": "Before (cricket only)", "pace": lc["base_n_pace"], "spin": lc["base_n_spin"]},
            {"XI": "After (with rule)", "pace": lc["n_pace"], "spin": lc["n_spin"]},
        ])
        st.dataframe(met, hide_index=True, **wide())
        if lc.get("log"):
            st.caption("Checker log: " + "; ".join(lc["log"]))
    else:
        st.markdown(
            "<div class='note warn'><b>Live MILP selection is unavailable in this interpreter.</b> "
            "The optimiser uses the PuLP solver, which isn’t installed here. Run "
            "<code>.venv/bin/python -m streamlit run demo/app.py</code> to see the before/after XI "
            "table. The recorded win probabilities below still work.</div>",
            unsafe_allow_html=True)

    st.markdown("##### 4 · Win probability — before vs after the rule")
    if wp.get("bars"):
        st.markdown(win_bars(wp), unsafe_allow_html=True)
        st.caption(f"Mean condition-aware win probability over the frozen Set R simulator runs · "
                   f"scope: {wp['scope']} (n={wp['bars'].get('before',{}).get('n','?')} matches). "
                   "Real recorded numbers — nothing live or invented.")
        bb = wp["bars"].get("before", {}).get("p_win")
        aa = wp["bars"].get("after", {}).get("p_win")
        if bb is not None and aa is not None:
            d = aa - bb
            st.markdown(f"<span class='badge {'ok' if abs(d)<1e-9 else 'grey'}'>"
                        f"Δ after − before = {d:+.4f}</span>", unsafe_allow_html=True)
    else:
        st.warning("No Set R simulator rows available for this team.")


# ---------------------------------------------------------------------------
# ③  WHY THIS MATTERS — the three findings + the simulation-only notice
# ---------------------------------------------------------------------------
with tab_why:
    st.markdown("#### What we actually found")
    ext = metrics.get("extraction", {})
    dec = metrics.get("decision", {}).get("set_r_condition_aware", {})
    holm = metrics.get("decision", {}).get("holm_corrected_p", {})
    f1_label = ext.get("ours1_label_rulebook", {}).get("f1")
    f1_rules = ext.get("ours2_rules_direct", {}).get("f1")
    oracle = dec.get("B4_oracle", {})
    swap = dec.get("B7_swap", {})

    fc = st.columns(3)
    with fc[0]:
        with st.container(border=True):
            st.markdown("<div class='fcard'><h4>1 · Naming beats writing bounds</h4></div>",
                        unsafe_allow_html=True)
            if f1_label is not None and f1_rules is not None:
                st.markdown(f"<span class='fbig b'>{f1_label:.2f}</span> "
                            f"<span class='scap'>F1, name-the-condition</span>",
                            unsafe_allow_html=True)
                st.markdown(f"<p class='scap'>vs <b>{f1_rules:.2f}</b> when the model writes the "
                            "numeric bounds directly. Reading a label and mapping it through a "
                            "frozen rulebook recovers the right constraint far more reliably.</p>",
                            unsafe_allow_html=True)
            else:
                st.warning("extraction F1 missing in metrics.json")
    with fc[1]:
        with st.container(border=True):
            st.markdown("<div class='fcard'><h4>2 · Correct rules rarely change the XI</h4></div>",
                        unsafe_allow_html=True)
            if oracle:
                st.markdown(f"<span class='fbig g'>{oracle.get('xi_change_rate',0)*100:.0f}%</span> "
                            f"<span class='scap'>of matches change</span>", unsafe_allow_html=True)
                st.markdown(f"<p class='scap'>With the <i>correct</i> rule (oracle), the team changes "
                            f"only ~{oracle.get('xi_change_rate',0)*100:.0f}% of the time and win "
                            f"probability moves by ΔP ≈ <b>{oracle.get('mean_delta',0):+.4f}</b>. "
                            "A value-maximising XI is usually already condition-appropriate.</p>",
                            unsafe_allow_html=True)
            else:
                st.warning("B4_oracle missing in metrics.json")
    with fc[2]:
        with st.container(border=True):
            st.markdown("<div class='fcard'><h4>3 · Wrong rules reliably hurt</h4></div>",
                        unsafe_allow_html=True)
            if swap:
                p = holm.get("B7_swap")
                psig = ("Holm p < 1e-5" if (p is not None and p < 1e-5)
                        else (f"Holm p = {p:.1e}" if p is not None else "Holm-corrected"))
                st.markdown(f"<span class='fbig r'>{swap.get('mean_delta',0):+.4f}</span> "
                            f"<span class='scap'>ΔP(win)</span>", unsafe_allow_html=True)
                st.markdown(f"<p class='scap'>A deliberately wrong pace↔spin swap significantly "
                            f"lowers win probability ({psig}). The rule earns its keep by "
                            "preventing bad picks, not by improving good ones.</p>",
                            unsafe_allow_html=True)
            else:
                st.warning("B7_swap missing in metrics.json")

    st.markdown("#### The honest take")
    st.markdown(
        "<div class='note warn'><b>Simulation only.</b> Every number here comes from a validated "
        "ball-by-ball <i>simulator</i> on synthetic condition images, under a frozen rulebook and "
        "leakage-safe splits. It is a controlled study of <b>whether seeing conditions should "
        "reshape selection</b> — <u>not</u> a claim about winning real matches, and not selection "
        "advice for any real team.</div>", unsafe_allow_html=True)

    cost = metrics.get("cost", {})
    found = metrics.get("foundations", {})
    st.caption(
        f"Provenance · {found.get('matches','—')} T20Is (Cricsheet) · readers: Claude Haiku 5.5 "
        f"(paper) + local Qwen3-VL:8b · solver: PuLP + HiGHS · total API spend "
        f"${cost.get('total_api_usd',0):.4f} · this demo makes zero API calls."
    )
