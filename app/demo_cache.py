"""Streamlit caching around scripts/demo_core.py: models load once, scoring is cached per building."""
from __future__ import annotations

import time

import streamlit as st

from scripts import demo_core as core


@st.cache_data(show_spinner=False)
def get_meta():
    return core.load_meta()


@st.cache_data(show_spinner=False)
def get_inputs(building: str):
    return core.prepare_inputs(building, get_meta())


@st.cache_resource(show_spinner=False)
def get_detector(building: str):
    return core.load_detector(building)


@st.cache_data(show_spinner=False)
def get_split(building: str) -> dict:
    return core.load_split_info(building, len(get_inputs(building).feats))


@st.cache_data(show_spinner=False)
def get_clean(building: str):
    """Scored history of one building (cached; the first call per building is the slow one)."""
    inputs = get_inputs(building)
    return core.score_features(inputs.feats, get_detector(building), get_split(building)["split_idx"])


@st.cache_data(show_spinner=False, max_entries=64)
def get_injection(building: str, key: tuple):
    """``key`` = tuple of (type, start ISO string, hours, magnitude) so the call is hashable and cached per set of injections."""
    import pandas as pd

    specs = [core.InjectionSpec(t, pd.Timestamp(s), int(h), m) for t, s, h, m in key]
    return core.run_injection(building, get_inputs(building), get_meta(), get_detector(building), specs,
                              get_clean(building), get_split(building)["split_idx"])


@st.cache_resource(show_spinner=False)
def warm_state() -> dict:
    return {"done": False, "seconds": 0.0, "steps": [], "errors": {}, "buildings": []}


def warm_up(force: bool = False, status=None) -> dict:
    """
    Import TensorFlow, load every model and scaler, and score every building once (this also compiles the prediction
    graph, which is what makes the first score slow). Runs once per server process; ``force`` reloads everything.
    """
    if force:
        st.cache_data.clear()
        st.cache_resource.clear()
    state = warm_state()
    if state["done"]:
        return state
    say = (lambda m: status.write(m)) if status is not None else (lambda m: None)
    t_all = time.perf_counter()
    try:
        secs = core.import_tensorflow()
        state["steps"].append(("Import TensorFlow", secs))
        say(f"Imported TensorFlow ({secs:.1f} s)")
        t = time.perf_counter()
        get_meta()
        state["steps"].append(("Load building metadata", time.perf_counter() - t))
    except core.DemoFileError as exc:
        state["errors"]["TensorFlow / metadata"] = exc
        state["done"] = True
        return state
    for b in core.discover_buildings():
        t = time.perf_counter()
        try:
            get_clean(b)  # loads data, features, model, scaler and scores once
            state["buildings"].append(b)
            state["steps"].append((f"{b}: data, model, first score", time.perf_counter() - t))
            say(f"{b} ready ({time.perf_counter() - t:.1f} s)")
        except core.DemoFileError as exc:
            state["errors"][b] = exc
            say(f"{b} could not be loaded: {exc.message}")
    state["seconds"] = time.perf_counter() - t_all
    state["done"] = True
    return state
