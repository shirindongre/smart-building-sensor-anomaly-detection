"""Page 5 (optional): step through a building's hours and watch flags appear."""
import streamlit as st

import demo_cache
import demo_charts
import demo_ui
from scripts import demo_core as core

START = 24 * 8  # replay starts after the warm-up week
SPEEDS = {"6 hours per second": 6, "1 day per second": 24, "3 days per second": 72, "1 week per second": 168}


@demo_ui.guarded
def main():
    st.title("Replay")
    building = demo_ui.building_picker("replay")
    res = demo_cache.get_clean(building)
    inputs = demo_cache.get_inputs(building)
    view = core.split_policy_view(demo_cache.get_split(building), inputs.feats.index)

    if view["strategy"] == "full":
        st.warning("In-sample replay. This model was trained on the whole history it is replaying, so the flags you see "
                   "are not a test of how it would behave on new data.")
    else:
        st.warning(f"Mixed replay. Before {view['test_start']:%Y-%m-%d} the model had seen the data (in-sample); from "
                   "that date on the data was held out of training.")

    n = len(res)
    if st.session_state.get("replay_building") != building or "replay_pos" not in st.session_state:
        st.session_state.update(replay_building=building, replay_pos=START, replay_playing=False)  # start from the beginning

    c = st.columns([1, 1, 2, 3])
    playing = st.session_state.get("replay_playing", False)
    if c[0].button("Pause" if playing else "Play", type="primary", width="stretch"):
        if not playing and st.session_state["replay_pos"] >= n:
            st.session_state["replay_pos"] = START
        st.session_state["replay_playing"] = not playing
        st.rerun()
    if c[1].button("Restart", width="stretch"):
        st.session_state.update(replay_pos=START, replay_playing=False)
        st.rerun()
    speed_label = c[2].selectbox("Speed", list(SPEEDS), index=1, key="replay_speed")
    speed = SPEEDS[speed_label]

    y_max = {1: float(res["energy"].max()) * 1.08,
             2: float(max(res["reconstruction_error"].max(), res["threshold"].max())) * 1.08}
    x_range = (res.index[0], res.index[-1])

    @st.fragment(run_every=1.0 if playing else None)
    def player():
        finished = False
        if st.session_state.get("replay_playing"):
            st.session_state["replay_pos"] = min(int(st.session_state["replay_pos"]) + speed, n)
            if st.session_state["replay_pos"] >= n:
                st.session_state["replay_playing"] = False
                finished = True
        pos = st.slider("Replayed hours (drag to jump)", START, n, key="replay_pos")
        seen = res.iloc[:pos]
        m = st.columns(3)
        m[0].metric("Replayed up to", f"{seen.index[-1]:%Y-%m-%d %H:%M}")
        m[1].metric("Flagged hours so far", f"{int(seen['anomaly_flag'].sum()):,}")
        m[2].metric("Drift hours so far", f"{int(seen['drift_flag'].sum()):,}")
        fig = demo_charts.timeline_figure(seen, normalized=False, x_range=x_range, y_max=y_max, height=560,
                                          split_time=view["test_start"] if view["strategy"] == "holdout" else None,
                                          split_label="Test period starts here (held out of training)")
        st.plotly_chart(fig, width="stretch", key="replay_chart")
        st.caption("The series fills in left to right as time advances; blue dots appear the moment the detector would "
                   "have flagged that hour, using only data up to that hour for the threshold.")
        if finished:
            st.rerun()  # leave play mode (the fragment's refresh interval is fixed when it is defined)

    player()


main()
