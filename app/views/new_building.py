"""Page 3: onboard a building that was trained in advance, with measured seconds per step."""
import pandas as pd
import streamlit as st

import demo_cache
import demo_charts
import demo_ui
from scripts import demo_core as core

STEPS_TO_ADD = """\
1. Pick a building name from the first column of data/processed/temp_open_utc_complete.csv.
2. In src/config.py, set BUILDINGS to just that name (training retrains every name in the list).
3. python scripts/Model_Training.py        (minutes per building; writes models/, scalers/ and the loss plot)
4. Put BUILDINGS back to the original list, so the new name is not a regular building.
5. In this app press "Warm up again" in the sidebar, then reopen this page."""


@demo_ui.guarded
def main():
    st.title("New building")
    extras = core.extra_buildings()
    if not extras:
        st.info("No pre-trained building outside BUILDINGS was found in models/. This page onboards a building whose "
                "model was trained in advance, so there is nothing to show yet.")
        st.markdown("To add one:")
        st.code(STEPS_TO_ADD, language="text")
        return

    st.write("Onboarding a building means loading its data, building features, deciding the split policy, loading the "
             "model and scoring. The steps below run for real and are timed.")
    st.warning("Training was done in advance and is not part of the timings. The model and scaler for this building "
               "were trained earlier by scripts/Model_Training.py; the loss curve at the bottom is from that run.")
    building = demo_ui.building_picker("new", extras, label="New building (pre-trained, not in BUILDINGS)")

    store = st.session_state.setdefault("onboarded", {})
    if st.button("Onboard this building", type="primary"):
        with st.spinner(f"Onboarding {building}..."):
            store[building] = core.onboard_building(building, demo_cache.get_meta())
    ob = store.get(building)
    if ob is None:
        st.caption("Press the button to run the five steps and see the building's timeline.")
        return

    m = st.columns(2)
    m[0].metric("Total time to onboard", f"{ob.total_seconds:.1f} s")
    m[1].metric("Training time (done in advance)", "not included")
    steps = pd.DataFrame(ob.steps).rename(columns={"step": "Step", "seconds": "Seconds", "detail": "What happened"})
    st.dataframe(steps[["Step", "Seconds", "What happened"]], hide_index=True, width="stretch",
                 column_config={"Seconds": st.column_config.NumberColumn(format="%.2f")})
    st.caption("Measured now, with nothing cached. The first score in a fresh process also pays a one-off TensorFlow "
               "cost, which the warm-up step at start-up has already paid.")

    view = core.split_policy_view(ob.recorded, ob.inputs.feats.index)
    st.markdown("#### Result")
    demo_ui.headline_row(core.headline_stats(ob.res))
    split_time = view["test_start"] if view["strategy"] == "holdout" else None
    label = "Legacy 80/20 split: test period to the right" if view["legacy"] else "Train / test split: test period to the right"
    st.plotly_chart(demo_charts.timeline_figure(ob.res, split_time=split_time, split_label=label), width="stretch",
                    key="onboard_chart")
    demo_ui.timeline_caption()
    demo_ui.split_box(view)
    st.caption("The split above is the one this model was trained with (the legacy 80/20 fallback when no split file exists); "
               "step 3 shows what the policy would decide today for the same data.")

    st.markdown("#### Training loss (from the earlier training run)")
    loss = core.loss_curve_path(building)
    if loss:
        st.image(str(loss), width=900)
        st.caption("Loss per epoch from the earlier training run. Training is not repeated here.")
    else:
        st.write(f"No loss curve found at output_plots/{building}_loss.png.")


main()
