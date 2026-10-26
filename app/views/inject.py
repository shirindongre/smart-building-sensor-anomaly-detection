"""Page 2: corrupt a copy of one building's series, rescore it with the saved model and show what the detector finds."""
import pandas as pd
import streamlit as st

import demo_cache
import demo_charts
import demo_ui
from scripts import demo_core as core

TYPE_LABELS = {
    "spike": "Spike (short burst of extra load)",
    "dropout": "Dropout (outage, reading falls to about zero)",
    "flatline": "Flatline (stuck sensor, reading frozen)",
    "level_shift": "Level shift (equipment left on, load stays higher)",
}


def _visible_max(frames, x_range, col):
    vals = []
    for f in frames:
        if x_range is not None:
            f = f[(f.index >= x_range[0]) & (f.index <= x_range[1])]
        if len(f):
            vals.append(float(f[col].max()))
    return max(vals) * 1.08 if vals else None


def _y_limits(clean, inj, x_range):
    frames = [clean, inj]
    energy = _visible_max(frames, x_range, "energy")
    err = max(filter(None, [_visible_max(frames, x_range, "reconstruction_error"), _visible_max(frames, x_range, "threshold")]),
              default=None)
    out = {}
    if energy:
        out[1] = energy
    if err:
        out[2] = err
    return out


@demo_ui.guarded
def main():
    st.title("Inject and detect")
    st.write("Corrupt a copy of this building's hourly energy, rebuild the features, and rescore it with the saved "
             "model. Nothing is written to disk and no model is retrained.")
    building = demo_ui.building_picker("inject")
    inputs = demo_cache.get_inputs(building)
    clean = demo_cache.get_clean(building)
    feats = inputs.feats
    specs = st.session_state.setdefault("injections", {}).setdefault(building, [])
    n = len(specs)

    earliest = feats.index[0] + pd.Timedelta(days=core.MIN_START_DAYS)
    typ = st.selectbox("Anomaly type", core.EVAL["event_types"], format_func=TYPE_LABELS.get, key=f"inj_type_{building}")
    info = core.magnitude_info(typ)
    avoid = [(e.start, e.end) for e in core.build_events(specs, feats.index)]
    sug = core.suggest_start(feats, clean, typ, info["hours"], avoid)

    k = f"{building}_{typ}_{n}"  # a new key after every injection, so the suggested defaults refresh
    c = st.columns([1.2, 1, 1, 2])
    day = c[0].date_input("Start date (UTC)", value=sug.date(), min_value=earliest.date(), max_value=feats.index[-1].date(),
                          key=f"inj_date_{k}")
    hour = c[1].selectbox("Start hour (UTC)", list(range(24)), index=sug.hour, format_func=lambda h: f"{h:02d}:00",
                          key=f"inj_hour_{k}")
    hours = c[2].number_input("Duration (hours)", min_value=info["hours_range"][0], max_value=info["hours_range"][1],
                              value=info["hours"], step=1, key=f"inj_hours_{k}")
    if info["range"]:
        lo, hi = info["range"]
        mag = c[3].slider(info["label"], min_value=lo, max_value=hi, value=float(info["magnitude"]), step=info["step"],
                          key=f"inj_mag_{k}")
    else:
        mag = None
        c[3].markdown("**Setting**")
        c[3].write("The reading freezes at the last value before the start.")
    st.caption(f"Defaults come from the evaluation settings (EVAL in src/config.py). The start is pre-filled with a normal, "
               f"active period away from flagged hours; it cannot be earlier than {earliest:%Y-%m-%d} (the first "
               f"{core.MIN_START_DAYS} days are warm-up).")

    b1, b2, _ = st.columns([1, 1, 4])
    if b1.button("Inject", type="primary", width="stretch"):
        start = pd.Timestamp(day).tz_localize("UTC") + pd.Timedelta(hours=int(hour))
        new = core.InjectionSpec(typ, start, int(hours), None if mag is None else float(mag))
        try:
            core.build_events(specs + [new], feats.index)  # validates start, end and overlaps
            specs.append(new)
            st.session_state.pop("inj_error", None)
        except ValueError as exc:
            st.session_state["inj_error"] = str(exc)
        st.rerun()
    if b2.button("Reset", width="stretch", disabled=not specs):
        specs.clear()
        st.session_state.pop("inj_error", None)
        st.rerun()
    if st.session_state.get("inj_error"):
        st.error(st.session_state["inj_error"])

    if not specs:
        st.info("Choose a type and press Inject. You can inject several anomalies one after the other; Reset removes them.")
        return

    key = tuple((s.type, s.start.isoformat(), s.hours, s.magnitude) for s in specs)
    with st.spinner("Rebuilding features and rescoring (a few seconds)..."):
        result = demo_cache.get_injection(building, key)

    scored = result.event_table[~result.event_table["inert"]]
    found = int(scored["detected"].sum())
    delays = scored.loc[scored["detected"], "delay_hours"]
    new_fa_hours = int(result.new_false_alarms["hours"].sum()) if len(result.new_false_alarms) else 0
    m = st.columns(4)
    m[0].metric("Injections", f"{len(result.event_table)}")
    m[1].metric("Detected", f"{found} of {len(scored)}")
    m[2].metric("Median detection delay", f"{delays.median():.0f} h" if len(delays) else "n/a")
    m[3].metric("New false alarms", f"{new_fa_hours} h")

    for _, r in result.event_table.iterrows():
        what = f"Injection {r['#']} ({r['type'].replace('_', ' ')}, {r['hours']} h from {r['start']:%Y-%m-%d %H:%M} UTC)"
        if r["inert"]:
            st.info(f"{what} changed almost nothing in the data (average change {r['effect_std']:.2f} std), so there was "
                    "nothing to detect. It is not counted.")
        elif r["detected"]:
            st.success(f"{what} was detected {r['delay_hours']:.0f} h after it began. Peak normalized error "
                       f"{r['peak_normalized_error']:.1f}.")
        elif r["drift_only"]:
            st.warning(f"{what} was not flagged as an anomaly; only the drift flag fired.")
        else:
            st.error(f"{what} was missed: no hour in its windows was flagged.")
        if r["note"] and not r["inert"]:
            st.caption(r["note"])

    st.caption(f"New false alarms = flagged hours that the clean run did not flag, away from any injection. "
               f"{result.echo_flag_hours} further new flagged hours fall just after an injection and are not counted: the "
               "model's 24 h and 168 h lag features carry an event forward, so its echo shows up about 7 days later "
               "(the Evaluation page ignores the same week, slightly more narrowly). Detection delay is measured from "
               "the first injected hour to the first flagged window.")
    if len(result.new_false_alarms):
        with st.expander(f"New false-alarm periods ({len(result.new_false_alarms)})"):
            demo_ui.periods_table(result.new_false_alarms, height=240)

    table = result.event_table.copy()
    table["result"] = table.apply(lambda r: "Not scored (inert)" if r["inert"] else (
        "Detected" if r["detected"] else ("Drift flag only" if r["drift_only"] else "Missed")), axis=1)
    with st.expander("Injection list"):
        st.dataframe(table[["#", "type", "start", "end", "hours", "setting", "result", "delay_hours",
                            "peak_normalized_error", "effect_std"]].rename(columns={
                                "type": "Type", "start": "Start (UTC)", "end": "End (UTC)", "hours": "Hours",
                                "setting": "Setting", "result": "Result", "delay_hours": "Delay (h)",
                                "peak_normalized_error": "Peak normalized error",
                                "effect_std": "Change (std of energy)"}),
                     hide_index=True, width="stretch")

    view_choice = st.segmented_control("View", ["Around the injections", "Full history"], default="Around the injections",
                                       key="inj_view")
    x_range = None
    if view_choice != "Full history":
        pad = pd.Timedelta(days=4)
        x_range = (min(s[0] for s in result.spans) - pad, max(s[1] for s in result.spans) + pad)
    limits = _y_limits(result.clean_res, result.inj_res, x_range)

    st.markdown("#### Clean history")
    st.plotly_chart(demo_charts.timeline_figure(result.clean_res, normalized=False, x_range=x_range, y_max=limits,
                                                height=560), width="stretch", key="inj_clean")
    st.caption("The untouched series scored by the same model, for comparison.")
    st.markdown("#### With injected anomalies")
    st.plotly_chart(demo_charts.timeline_figure(result.inj_res, normalized=False, spans=result.spans,
                                                ref_energy=result.clean_res["energy"], x_range=x_range, y_max=limits,
                                                height=560), width="stretch", key="inj_corrupt")
    st.caption("The corrupted copy: amber bands mark the injected hours, the dotted grey line is the original energy, "
               "blue dots are hours the detector flagged.")


main()
