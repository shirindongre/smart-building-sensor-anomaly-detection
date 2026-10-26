"""Page 1: timeline, headline numbers, flagged periods and the split policy of one building."""
import streamlit as st

import demo_cache
import demo_charts
import demo_ui
from scripts import demo_core as core


@demo_ui.guarded
def main():
    st.title("Building view")
    building = demo_ui.building_picker("building")

    with st.spinner(f"Scoring {building}..."):
        res = demo_cache.get_clean(building)
    inputs = demo_cache.get_inputs(building)
    view = core.split_policy_view(demo_cache.get_split(building), inputs.feats.index)

    st.caption(demo_ui.building_blurb(building, demo_cache.get_meta(), len(inputs.feats)))
    demo_ui.headline_row(core.headline_stats(res))

    split_time = view["test_start"] if view["strategy"] == "holdout" else None
    label = "Legacy 80/20 split: test period to the right" if view["legacy"] else "Train / test split: test period to the right"
    fig = demo_charts.timeline_figure(res, split_time=split_time, split_label=label)
    st.plotly_chart(fig, width="stretch")
    demo_ui.timeline_caption("Drag to zoom, double-click to reset.")

    demo_ui.split_box(view)

    st.markdown("#### Flagged periods")
    periods = core.flag_periods(res)
    demo_ui.periods_table(periods)
    st.caption("Each row is a run of consecutive flagged hours, with the highest error reached in it. "
               "Click a column header to sort.")


main()
