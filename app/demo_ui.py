"""Shared layout helpers: styling, error display, building picker, headline numbers, split-policy box."""
from __future__ import annotations

import functools

import pandas as pd
import streamlit as st

from scripts import demo_core as core

CSS = """
<style>
html, body, [class*="css"] { font-size: 18px; }
.block-container { padding-top: 2.2rem; max-width: 1500px; }
[data-testid="stMetricValue"] { font-size: 2.9rem; font-weight: 600; }
[data-testid="stMetricLabel"] p { font-size: 1.05rem; color: #B9BFC8; }
[data-testid="stCaptionContainer"] p { font-size: 1.0rem; color: #B9BFC8; }
h1 { font-size: 2.3rem; }
.policy-label { display: inline-block; padding: 0.15rem 0.7rem; border-radius: 0.3rem; font-weight: 600;
                background: #D8DBDF; color: #0B0C0E; letter-spacing: 0.02em; }

/* Greys: dark main page #2B2E33, cards #363A41, light-grey sidebar #BDC1C7 with black text */
[data-testid="stSidebar"] { background: #BDC1C7; }
[data-testid="stSidebar"] * { color: #0B0C0E; }
[data-testid="stSidebar"] hr { border-color: #8E949C; }
[data-testid="stSidebar"] a[aria-current="page"] { background: #A3A8AF; }
[data-testid="stSidebar"] button { background: #E4E6E9; border: 1px solid #7C838C; }
[data-testid="stPlotlyChart"], [data-testid="stMetric"], [data-testid="stDataFrame"] {
    background: #363A41; border: 1px solid #4A4F57; border-radius: 0.5rem; padding: 0.6rem 0.8rem; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def show_file_error(exc: core.DemoFileError) -> None:
    """A friendly message plus the command to run, instead of a traceback."""
    st.error(exc.message)
    if exc.command:
        st.markdown("To fix this, run:" if exc.command.startswith(("python", "pip")) else "To fix this:")
        st.code(exc.command, language="bash" if exc.command.startswith(("python", "pip")) else None)


def guarded(fn):
    """Run a page; show file problems as messages (with the fix) rather than a traceback."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except core.DemoFileError as exc:
            show_file_error(exc)
    return wrapper


def building_picker(key: str, names: list[str] | None = None, label: str = "Building") -> str:
    """Sidebar selectbox; the choice is shared between pages through session_state['building']."""
    names = names if names is not None else core.discover_buildings()
    current = st.session_state.get("building")
    idx = names.index(current) if current in names else 0
    choice = st.sidebar.selectbox(label, names, index=idx, key=f"picker_{key}")
    st.session_state["building"] = choice
    return choice


def building_blurb(building: str, meta: pd.DataFrame, n_rows: int) -> str:
    row = meta[meta["uid"] == building]
    if row.empty:
        return f"{n_rows} hourly rows"
    r = row.iloc[0]
    use = r.get("primaryspaceusage")
    use = use if isinstance(use, str) and use else r.get("primaryspaceuse_abbrev", "")
    tz = r.get("timezone", "")
    return f"{use} | time zone {tz} | {n_rows} hourly rows after feature building"


def headline_row(stats: dict) -> None:
    c = st.columns(5)
    c[0].metric("Hours scored", f"{stats['hours_scored']:,}")
    c[1].metric("Flagged hours", f"{stats['flags']:,}")
    c[2].metric("Anomaly rate", f"{stats['anomaly_rate']:.1%}")
    c[3].metric("Flagged periods", f"{stats['periods']:,}")
    c[4].metric("Drift hours", f"{stats['drift_hours']:,}")


def split_box(view: dict) -> None:
    """The SPLIT POLICY box, built from models/<building>_split.json (or the legacy fallback)."""
    with st.container(border=True):
        st.markdown("#### Split policy")
        st.markdown(f"<span class='policy-label'>{view['label']}</span>", unsafe_allow_html=True)
        st.write(view["message"])
        st.markdown(f"**Training period:** {_fmt(view['train_start'])} to {_fmt(view['train_end'])} (UTC)")
        if view["strategy"] == "holdout" and view["test_start"] is not None:
            st.markdown(f"**Test period:** {_fmt(view['test_start'])} to {_fmt(view['test_end'])} (UTC)")
        else:
            st.markdown("**Test period:** none (no held-out data)")
        for s in view["sentences"]:
            st.write(s)
        if view["strategy"] == "full":
            st.page_link("views/evaluation.py", label="Open the Evaluation page")


def _fmt(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")


def periods_table(periods: pd.DataFrame, height: int = 360) -> None:
    if periods.empty:
        st.write("No hours were flagged.")
        return
    show = periods.rename(columns={
        "start": "Start (UTC)", "end": "End (UTC)", "hours": "Hours", "peak_error": "Peak error (MSE)",
        "peak_normalized_error": "Peak normalized error", "peak_time": "Peak at (UTC)"})
    st.dataframe(
        show, hide_index=True, height=height,
        column_config={
            "Start (UTC)": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
            "End (UTC)": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
            "Peak at (UTC)": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
            "Peak error (MSE)": st.column_config.NumberColumn(format="%.4f"),
            "Peak normalized error": st.column_config.NumberColumn(format="%.1f"),
        })


def timeline_caption(extra: str = "") -> None:
    st.caption("Energy (top), model error against the red threshold (middle), and the same error as a z-score (bottom): "
               "blue dots are flagged hours, grey bands are drift periods, and the first 168 hours are never flagged. " + extra)
