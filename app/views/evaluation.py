"""Page 4: the stored results of scripts/evaluation.py (nothing is recomputed here)."""
import pandas as pd
import streamlit as st

import demo_charts
import demo_ui
from scripts import demo_core as core


@demo_ui.guarded
def main():
    st.title("Evaluation")
    ev = core.load_evaluation()  # DemoFileError -> friendly message with the command to run

    st.write("The dataset has no labelled anomalies, so the detector is tested by injecting known ones into a copy of the "
             "data and checking whether it finds them. The numbers below are read from files in output_plots/evaluation/ "
             "and are not recomputed in this app.")

    table = core.evaluation_table(ev)
    st.markdown("#### Results per building")
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "Median delay (h)": st.column_config.NumberColumn(format="%.1f"),
            "False-alarm rate (%)": st.column_config.NumberColumn(format="%.1f"),
            "Precision": st.column_config.NumberColumn(format="percent"),
            "AUC, LSTM": st.column_config.NumberColumn(format="%.2f"),
            "AUC, same hour last week": st.column_config.NumberColumn(format="%.2f"),
            "AUC difference": st.column_config.NumberColumn(format="%+.2f"),
        })
    st.caption("Events found = injected anomalies where at least one window was flagged. False-alarm rate = flagged windows "
               "with no injected event nearby, as a share of clean windows. AUC measures how well the error separates "
               "event windows from clean ones (0.5 is chance, 1.0 is perfect); the last column is the LSTM minus the baseline.")

    st.markdown("#### What these numbers do and do not say")
    for line in core.evaluation_caveats(ev):
        st.write("- " + line)

    st.markdown("#### Recall by event type")
    long = core.recall_by_type(ev)
    if long.empty:
        st.write("No per-type recall found in evaluation_summary.csv.")
    else:
        st.plotly_chart(demo_charts.recall_bar_figure(long, core.EVENT_TYPES), width="stretch")
        st.caption("For each building, the share of injected events of each type that the detector flagged. With only a "
                   "handful of events per type, a single miss changes a bar by about 15 to 20 points.")

    st.markdown("#### How the test works")
    st.markdown(
        "- **Blocked testing.** A building has about one year of data, all of which was used to train the production model, "
        "so it has no untouched test period. Instead the year is cut into 14-day blocks; for each fold a temporary model is "
        "trained on the other blocks and tested on the held-out ones, with a gap around each block so the 7-day look-back "
        "features cannot leak. Production models are never changed.\n"
        "- **Injected anomalies.** Spikes, dropouts, flatlines and level shifts are added to the held-out blocks only. These "
        "are synthetic: real faults may look different, and real anomalies are not labelled in this dataset.\n"
        "- **Inert events.** An outage in a building that is idle anyway changes nothing, so events whose real effect is "
        "tiny are marked inert and left out of every metric rather than counted as misses.\n"
        "- **The baseline.** 'Same hour last week' scores each window by how much its energy differs from the same hours a "
        "week earlier. It needs no training; the LSTM is only worth its cost if it clearly beats it.")

    st.markdown("#### Per-building figure")
    names = list(table["Building"])
    building = st.selectbox("Building", names, key="eval_building")
    png = core.evaluation_png(building)
    if png:
        st.image(str(png), width="stretch")
        st.caption("Reconstruction error on the held-out windows with the injected events shaded (green found, amber "
                   "missed, grey inert) and false alarms marked; recall by type underneath.")
    else:
        st.write(f"No figure found for {building}.")
        st.code(core.EVAL_COMMAND, language="bash")
    events_path = core.eval_dir() / f"{building}_events.csv"
    if events_path.is_file():
        with st.expander("Injected events for this building"):
            st.dataframe(pd.read_csv(events_path), hide_index=True, width="stretch")


main()
