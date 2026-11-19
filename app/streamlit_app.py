"""Offline Streamlit demo (#12): one squad + three condition images -> three XIs.

No live API calls — uses the deterministic mock VLM and cached data. Run:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.demo_backend import DEMO_CONDITIONS, demo_teams, three_xis  # noqa: E402

st.set_page_config(page_title="Seeing conditions, writing constraints", layout="wide")
st.title("Seeing conditions → writing constraints → XI")
st.caption(
    "Offline demo. A (mock) VLM reads a pitch image, a frozen rulebook turns the label into "
    "pace/spin constraints, and a MILP picks the XI from the team's pool. "
    "Results are simulator-based and exploratory — not a claim of winning real matches."
)

team = st.selectbox("Team", demo_teams(), index=0)

if st.button("Pick XIs for three conditions", type="primary"):
    with st.spinner("Selecting…"):
        res = three_xis(team)
    if not res:
        st.warning("No recent match found for this team.")
    else:
        st.write(f"**{res['team']}** — pool as of {res['date']}")
        cols = st.columns(len(DEMO_CONDITIONS))
        for col, (label, img_id) in zip(cols, DEMO_CONDITIONS):
            d = res["conditions"][label]
            with col:
                img = Path(f"data/synthetic/images/{img_id}.png")
                if img.exists():
                    st.image(str(img), caption=f"condition: {label}")
                st.markdown(f"**Perceived:** {d['perceived']}")
                st.markdown(f"**Rules:** `{d['rules']}`")
                st.markdown(f"**pace {d['n_pace']} · spin {d['n_spin']}**")
                for i, name in enumerate(d["xi_names"], 1):
                    st.write(f"{i}. {name}")
                if d["log"]:
                    st.caption("checker log: " + "; ".join(d["log"]))

st.divider()
st.caption("Data: Cricsheet (ODC-By), Open-Meteo (CC BY 4.0), cricketdata player_meta. No scraping.")
