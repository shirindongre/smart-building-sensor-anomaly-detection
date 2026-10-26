"""
Live demo of the smart-building sensor anomaly detector.

Run from the project root:   streamlit run app/app.py

Layout only: the logic is in scripts/demo_core.py. Nothing here trains a model or writes to data/, models/ or scalers/.
"""
from __future__ import annotations

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
_ROOT = _APP_DIR.parent
for _p in (_ROOT, _APP_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Sensor anomaly detection demo", layout="wide", initial_sidebar_state="expanded")

import demo_cache  # noqa: E402
import demo_ui  # noqa: E402
from scripts import demo_core as core  # noqa: E402

demo_ui.inject_css()

pages = st.navigation(
    [
        st.Page("views/building.py", title="Building view", url_path="building", default=True),
        st.Page("views/inject.py", title="Inject and detect", url_path="inject"),
        st.Page("views/new_building.py", title="New building", url_path="new-building"),
        st.Page("views/evaluation.py", title="Evaluation", url_path="evaluation"),
        st.Page("views/replay.py", title="Replay", url_path="replay"),
    ]
)

problems = core.check_environment()
if problems:
    st.title("Setup needed")
    for p in problems:
        demo_ui.show_file_error(p)
    st.stop()

# Warm-up: import TensorFlow, load every model, score every building once. Runs once per server process.
state = demo_cache.warm_state()
if not state["done"]:
    with st.status("Warming up: importing TensorFlow and loading all models...", expanded=True) as status:
        state = demo_cache.warm_up(status=status)
        status.update(label=f"Ready: {len(state['buildings'])} buildings loaded in {state['seconds']:.0f} s",
                      state="complete", expanded=False)

with st.sidebar:
    st.markdown("### Demo status")
    if state["errors"]:
        st.warning(f"{len(state['errors'])} item(s) could not be loaded: " + ", ".join(state["errors"]))
    else:
        st.write(f"{len(state['buildings'])} models loaded in {state['seconds']:.0f} s")
    if st.button("Warm up again", help="Reload data, models and scores from disk (takes about as long as at start-up)"):
        with st.status("Reloading...", expanded=True) as status:
            state = demo_cache.warm_up(force=True, status=status)
            status.update(label=f"Ready in {state['seconds']:.0f} s", state="complete", expanded=False)
        st.rerun()
    with st.expander("Start-up timings and notes"):
        for name, secs in state["steps"]:
            st.write(f"{name}: {secs:.1f} s")
        for name, exc in state["errors"].items():
            st.write(f"{name}: {exc.message}")
        missing = core.configured_without_model()
        if missing:
            st.write("In BUILDINGS but not trained yet: " + ", ".join(missing))
        for name, why in core.incomplete_buildings().items():
            st.write(f"{name}: {why}")
    st.divider()

pages.run()
