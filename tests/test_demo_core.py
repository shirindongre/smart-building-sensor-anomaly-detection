"""
Tests for scripts/demo_core.py. They need no TensorFlow and no real data: a synthetic building is written to a temporary
project root, and the PCA backend from scripts/evaluation.py stands in for the LSTM.

Run from the project root:   python -m pytest tests
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sklearn.preprocessing import MinMaxScaler  # noqa: E402

from scripts import demo_core as core  # noqa: E402
from scripts.evaluation import PCABackend  # noqa: E402
from scripts.sequence_creations import create_sequences  # noqa: E402
from src.config import BUILDINGS, FEATURE_COLS  # noqa: E402

B = "Office_Synth"


# ----------------------------------------------------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def root(tmp_path_factory):
    """A temporary project root with a synthetic wide table (one building, 330 days) and metadata."""
    r = tmp_path_factory.mktemp("proj")
    rng = np.random.default_rng(0)
    idx = pd.date_range("2015-01-01", periods=24 * 330, freq="h", tz="UTC")
    hour, dow = idx.hour.to_numpy(), idx.dayofweek.to_numpy()
    workday = (dow < 5).astype(float)
    energy = 6 + workday * 18 * np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None) + rng.normal(0, 0.6, len(idx))
    wide = pd.DataFrame({B: np.clip(energy, 0, None)}, index=idx)
    wide.index.name = "timestamp"
    (r / "data" / "processed").mkdir(parents=True)
    (r / "data" / "raw").mkdir(parents=True)
    wide.to_csv(r / "data" / "processed" / "temp_open_utc_complete.csv")
    pd.DataFrame({"uid": [B], "timezone": ["UTC"]}).to_csv(r / "data" / "raw" / "meta_open.csv", index=False)
    (r / "models").mkdir()
    (r / "scalers").mkdir()
    mp = pytest.MonkeyPatch()
    mp.setattr(core, "ROOT", r)
    yield r
    mp.undo()


@pytest.fixture(scope="module")
def inputs(root):
    return core.prepare_inputs(B)


@pytest.fixture(scope="module")
def meta(root):
    return core.load_meta()


@pytest.fixture(scope="module")
def detector(inputs):
    """Stand-in scorer: MinMax scaler + PCA reconstruction (same interface as the LSTM backend)."""
    X = inputs.feats[FEATURE_COLS].to_numpy()
    scaler = MinMaxScaler().fit(X)
    backend = PCABackend(8).fit(create_sequences(scaler.transform(X), core.W), None)
    return core.Detector(scaler, backend, "pca")


@pytest.fixture(scope="module")
def clean(inputs, detector):
    return core.score_features(inputs.feats, detector)


# ----------------------------------------------------------------------------------------------------------------------
# Discovery and friendly errors
# ----------------------------------------------------------------------------------------------------------------------
def test_discovery_needs_model_and_scaler(tmp_path, monkeypatch):
    (tmp_path / "models").mkdir()
    (tmp_path / "scalers").mkdir()
    extra = "Extra_One"
    configured = BUILDINGS[0]
    for name in ("Both_A", extra, configured):
        (tmp_path / "models" / f"{name}_best.h5").write_bytes(b"x")
        (tmp_path / "scalers" / f"{name}_scaler.pkl").write_bytes(b"x")
    (tmp_path / "models" / "OnlyModel_best.h5").write_bytes(b"x")
    (tmp_path / "scalers" / "OnlyScaler_scaler.pkl").write_bytes(b"x")
    monkeypatch.setattr(core, "ROOT", tmp_path)
    assert core.discover_buildings() == sorted(["Both_A", extra, configured])
    assert core.incomplete_buildings() == {"OnlyModel": "scaler missing", "OnlyScaler": "model missing"}
    assert extra in core.extra_buildings() and configured not in core.extra_buildings()


def test_missing_files_give_a_command_not_a_traceback(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "ROOT", tmp_path)
    problems = core.check_environment()
    commands = " ".join(p.command or "" for p in problems)
    assert len(problems) == 3
    assert "temp_open_utc_complete.csv" in commands and "meta_open.csv" in commands and "Model_Training.py" in commands
    with pytest.raises(core.DemoFileError) as e:
        core.load_building_wide("Anything")
    assert "temp_open_utc_complete.csv" in e.value.command
    with pytest.raises(core.DemoFileError) as e:
        core.load_detector("Anything")
    assert "Model_Training.py" in e.value.command and "BUILDINGS" in e.value.command
    with pytest.raises(core.DemoFileError) as e:
        core.load_evaluation()
    assert e.value.command == "python scripts/evaluation.py"


def test_unknown_building_column_is_a_friendly_error(root):
    with pytest.raises(core.DemoFileError) as e:
        core.load_building_wide("Not_A_Building")
    assert "Not_A_Building" in e.value.message


# ----------------------------------------------------------------------------------------------------------------------
# Split policy
# ----------------------------------------------------------------------------------------------------------------------
def test_split_info_legacy_fallback_matches_model_testing(root, inputs):
    n = len(inputs.feats)
    info = core.load_split_info(B, n)
    assert info["legacy"] and info["strategy"] == "holdout" and info["split_idx"] == int(n * 0.8)
    view = core.split_policy_view(info, inputs.feats.index)
    assert view["label"].startswith("HOLDOUT") and "80/20" in view["label"]
    assert view["test_start"] == inputs.feats.index[int(n * 0.8)]
    assert "before the split policy" in " ".join(view["sentences"])
    assert "80/20" in " ".join(view["sentences"])


def test_split_info_from_json_full_history(root, inputs):
    import json

    n = len(inputs.feats)
    meta_json = {"strategy": "full", "split_idx": n, "n_rows": n, "message": "Full-history training for X."}
    (root / "models" / f"{B}_split.json").write_text(json.dumps(meta_json), encoding="utf-8")
    try:
        info = core.load_split_info(B, n)
        assert not info["legacy"] and info["message"] == "Full-history training for X."
        view = core.split_policy_view(info, inputs.feats.index)
        assert view["strategy"] == "full" and view["label"] == "FULL history"
        assert view["test_start"] is None and view["train_end"] == inputs.feats.index[-1]
        text = " ".join(view["sentences"])
        assert "no held-out test period" in text and "in-sample" in text and "blocked folds" in text
        assert "newer rows" in core.load_split_info(B, n + 5)["message"]  # data that arrived after training
    finally:
        (root / "models" / f"{B}_split.json").unlink()


# ----------------------------------------------------------------------------------------------------------------------
# Inputs and scoring
# ----------------------------------------------------------------------------------------------------------------------
def test_prepare_inputs_uses_the_pipeline(inputs):
    assert list(inputs.feats[FEATURE_COLS].columns) == FEATURE_COLS and len(FEATURE_COLS) == 13
    assert not inputs.feats[FEATURE_COLS].isna().any().any()
    assert inputs.raw.index.tz is not None and inputs.load_seconds >= 0


def test_scoring_shape_and_warmup(clean, inputs):
    assert len(clean) == len(inputs.feats) - core.W + 1
    assert {"reconstruction_error", "normalized_error", "threshold", "drift_flag", "anomaly_flag", "split"} <= set(clean.columns)
    assert clean["anomaly_flag"].iloc[:168].sum() == 0 and clean["drift_flag"].iloc[:168].sum() == 0
    assert (clean["split"] == "train").all()
    stats = core.headline_stats(clean)
    assert stats["hours_scored"] == len(clean) and stats["flags"] == int(clean["anomaly_flag"].sum())
    assert stats["anomaly_rate"] == pytest.approx(stats["flags"] / len(clean))


def test_split_label_in_scores(inputs, detector):
    cut = 5000
    res = core.score_features(inputs.feats, detector, split_idx=cut)
    ends = np.arange(len(res)) + core.W - 1
    assert (res["split"].to_numpy() == np.where(ends < cut, "train", "test")).all()


def test_flag_periods_groups_consecutive_hours():
    idx = pd.date_range("2015-01-01", periods=12, freq="h", tz="UTC")
    res = pd.DataFrame({"reconstruction_error": np.arange(12.0), "normalized_error": np.arange(12.0) * 2,
                        "anomaly_flag": [0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0], "drift_flag": [0] * 12}, index=idx)
    p = core.flag_periods(res)
    assert list(p["hours"]) == [3, 1, 2]
    assert p.loc[0, "start"] == idx[1] and p.loc[0, "end"] == idx[3] and p.loc[0, "peak_error"] == 3.0
    assert p.loc[2, "peak_time"] == idx[10]
    assert core.flag_periods(res.assign(anomaly_flag=0)).empty


# ----------------------------------------------------------------------------------------------------------------------
# Injection
# ----------------------------------------------------------------------------------------------------------------------
def test_magnitude_defaults_come_from_eval():
    for typ in core.EVAL["event_types"]:
        info = core.magnitude_info(typ)
        lo, hi = core.EVAL[typ]["hours"]
        assert lo <= info["hours"] <= hi
        if typ == "flatline":
            assert info["magnitude"] is None and info["range"] is None
        else:
            key = {"spike": "amplitude_std", "dropout": "factor", "level_shift": "factor"}[typ]
            a, b = core.EVAL[typ][key]
            assert a <= info["magnitude"] <= b and info["range"][0] <= a and b <= info["range"][1]


def test_spec_validation(inputs):
    idx = inputs.feats.index
    ok = core.InjectionSpec("spike", idx[0] + pd.Timedelta(days=20), 2, 5.0)
    assert core.spec_to_event(ok, idx).start == 480
    early = core.InjectionSpec("spike", idx[0] + pd.Timedelta(days=3), 2, 5.0)
    with pytest.raises(ValueError, match="first 8 days"):
        core.spec_to_event(early, idx)
    late = core.InjectionSpec("dropout", idx[-1], 10, 0.0)
    with pytest.raises(ValueError, match="past the end"):
        core.spec_to_event(late, idx)
    a = core.InjectionSpec("dropout", idx[0] + pd.Timedelta(days=20), 10, 0.0)
    b = core.InjectionSpec("flatline", idx[0] + pd.Timedelta(days=20, hours=5), 10)
    with pytest.raises(ValueError, match="overlap"):
        core.build_events([a, b], idx)
    with pytest.raises(ValueError, match="magnitude"):
        core.spec_to_event(core.InjectionSpec("spike", idx[0] + pd.Timedelta(days=20), 2, None), idx)


@pytest.mark.parametrize("typ", ["spike", "dropout", "flatline", "level_shift"])
def test_suggested_start_is_valid_and_in_a_normal_period(inputs, clean, typ):
    info = core.magnitude_info(typ)
    start = core.suggest_start(inputs.feats, clean, typ, info["hours"])
    spec = core.InjectionSpec(typ, start, info["hours"], info["magnitude"])
    ev = core.spec_to_event(spec, inputs.feats.index)
    assert ev.start >= core.MIN_START_DAYS * 24
    near = clean["anomaly_flag"].to_numpy() | clean["drift_flag"].to_numpy()
    assert not near[max(ev.start - 2 * core.W, 0): ev.end + 1].any()


def test_next_suggestion_avoids_existing_injection(inputs, clean):
    first = core.suggest_start(inputs.feats, clean, "dropout", 12)
    ev = core.spec_to_event(core.InjectionSpec("dropout", first, 12, 0.0), inputs.feats.index)
    second = core.suggest_start(inputs.feats, clean, "dropout", 12, avoid=[(ev.start, ev.end)])
    pos = inputs.feats.index.get_loc(second)
    assert pos > ev.end + core.ECHO_HOURS or pos + 12 < ev.start - core.W


@pytest.fixture(scope="module")
def spike_run(root, inputs, meta, detector, clean):
    start = core.suggest_start(inputs.feats, clean, "spike", 3)
    spec = core.InjectionSpec("spike", start, 3, 9.0)
    return spec, core.run_injection(B, inputs, meta, detector, [spec], clean)


def test_injection_detects_a_large_spike(spike_run, inputs):
    spec, r = spike_run
    row = r.event_table.iloc[0]
    assert row["type"] == "spike" and bool(row["detected"]) and not bool(row["inert"])
    assert 0 <= row["delay_hours"] <= core.W
    assert r.inj_res.index.equals(r.clean_res.index)
    assert r.spans == [(spec.start.floor("h"), spec.start.floor("h") + pd.Timedelta(hours=2))]


def test_injection_corrupts_a_copy_only(spike_run, inputs, root):
    spec, r = spike_run
    start = spec.start.floor("h")
    changed = (r.raw_injected != inputs.raw["energy"]).to_numpy()
    assert changed.sum() == 3 and (r.raw_injected.index[changed] == pd.date_range(start, periods=3, freq="h")).all()
    # the inputs were not modified, and nothing was written under the project root
    fresh = core.prepare_inputs(B)
    pd.testing.assert_frame_equal(fresh.raw, inputs.raw)
    written = {p.name for p in root.rglob("*") if p.is_file()}
    assert written == {"temp_open_utc_complete.csv", "meta_open.csv"}


def test_echo_flags_are_not_counted_as_false_alarms(spike_run, inputs):
    _, r = spike_run
    ev = r.events[0]
    lo, hi = max(ev.start - core.W + 1, 0), ev.end + core.ECHO_HOURS
    for _, p in r.new_false_alarms.iterrows():
        j0, j1 = r.inj_res.index.get_loc(p["start"]), r.inj_res.index.get_loc(p["end"])
        assert j1 < lo or j0 > hi


def test_dropout_and_inert_event(root, inputs, meta, detector, clean):
    start = core.suggest_start(inputs.feats, clean, "dropout", 12)
    drop = core.InjectionSpec("dropout", start, 12, 0.0)
    r = core.run_injection(B, inputs, meta, detector, [drop], clean)
    assert bool(r.event_table.loc[0, "detected"])
    # a 5% load increase for one hour changes (almost) nothing: reported as inert, not as a miss
    tiny = core.InjectionSpec("level_shift", start, 1, 1.05)
    r2 = core.run_injection(B, inputs, meta, detector, [tiny], clean)
    assert bool(r2.event_table.loc[0, "inert"]) and "nothing" in r2.event_table.loc[0, "note"]


def test_two_injections_are_reported_separately(root, inputs, meta, detector, clean):
    s1 = core.suggest_start(inputs.feats, clean, "spike", 2)
    e1 = core.spec_to_event(core.InjectionSpec("spike", s1, 2, 9.0), inputs.feats.index)
    s2 = core.suggest_start(inputs.feats, clean, "dropout", 10, avoid=[(e1.start, e1.end)])
    specs = [core.InjectionSpec("spike", s1, 2, 9.0), core.InjectionSpec("dropout", s2, 10, 0.0)]
    r = core.run_injection(B, inputs, meta, detector, specs, clean)
    assert list(r.event_table["#"]) == [1, 2] and list(r.event_table["type"]) == ["spike", "dropout"]
    assert len(r.spans) == 2


# ----------------------------------------------------------------------------------------------------------------------
# New building flow
# ----------------------------------------------------------------------------------------------------------------------
def test_onboarding_steps_are_timed(root, inputs, detector, monkeypatch):
    monkeypatch.setattr(core, "load_detector", lambda building: detector)  # no TensorFlow in the tests
    ob = core.onboard_building(B)
    assert [s["step"] for s in ob.steps] == ["1. Load data", "2. Build features", "3. Decide split policy",
                                             "4. Load model and scaler", "5. Score"]
    assert all(s["seconds"] >= 0 for s in ob.steps) and ob.total_seconds >= sum(s["seconds"] for s in ob.steps) * 0.9
    assert ob.decision["strategy"] == "full" and "Full-history" in ob.steps[2]["detail"]  # 330 days -> full
    assert len(ob.res) == len(ob.inputs.feats) - core.W + 1


# ----------------------------------------------------------------------------------------------------------------------
# Evaluation files
# ----------------------------------------------------------------------------------------------------------------------
def _write_eval_files(folder: Path, with_summary: bool):
    folder.mkdir(parents=True, exist_ok=True)
    n = 120
    ts = pd.date_range("2015-03-01", periods=n, freq="h", tz="UTC")
    windows = pd.DataFrame({
        "timestamp": ts, "energy": 10.0, "reconstruction_error": np.linspace(0.001, 0.01, n), "normalized_error": 0.0,
        "threshold": 0.005, "anomaly_flag": (np.arange(n) % 10 == 0).astype(int), "drift_flag": 0,
        "end_row": np.arange(n) + 23, "event_id": [0 if 20 <= i < 30 else (1 if 60 <= i < 70 else -1) for i in range(n)],
        "ignore": False, "baseline_score": np.linspace(0, 1, n), "fold": 0})
    events = pd.DataFrame({
        "event_id": [0, 1], "type": ["spike", "dropout"], "start": [ts[20], ts[60]], "end": [ts[22], ts[66]],
        "hours": [3, 7], "magnitude": [1.0, 0.0], "effect_std": [3.0, 2.0], "active_frac": [1.0, 1.0], "inert": False,
        "detected": [True, False], "alerted": [True, False], "delay_hours": [0.0, np.nan], "max_normalized_error": [9.0, 1.0]})
    windows.to_csv(folder / "Extra_Eval_windows.csv", index=False)
    events.to_csv(folder / "Extra_Eval_events.csv", index=False)
    (folder / "Extra_Eval_evaluation.png").write_bytes(b"png")
    if with_summary:
        pd.DataFrame([{
            "building": "Listed_One", "mode": "blocked", "events": 25, "events_detected": 20, "event_recall": 0.8,
            "median_delay_hours": 5.0, "false_alarm_rate_pct": 2.0, "window_precision": 0.7, "auc_normalized_error": 0.8,
            "auc_baseline_same_hour_last_week": 0.79, "recall_spike": 0.9, "recall_dropout": 0.8}]
        ).to_csv(folder / "evaluation_summary.csv", index=False)
        (folder / "evaluation_report.md").write_text(
            "Backend: `lstm`. Event types: spike. Seed 7.\n\n| Building | Mode | Events | Recall | Median delay (h) | "
            "False alarms | Precision | AUC (error) | AUC baseline |\n|---|---|---|---|---|---|---|---|---|\n"
            "| Listed_One | blocked (3 folds, max 30 epochs) | 20/25 | 80% | 5 | 2.0% | 70% | 0.80 | 0.79 |\n",
            encoding="utf-8")


def test_evaluation_summary_plus_rebuilt_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "ROOT", tmp_path)
    _write_eval_files(core.eval_dir(), with_summary=True)
    ev = core.load_evaluation()
    t = core.evaluation_table(ev)
    t = t.set_index("Building")  # buildings outside BUILDINGS are listed alphabetically
    assert list(t.index) == ["Extra_Eval", "Listed_One"]
    assert t.loc["Listed_One", "Run"] == "blocked (3 folds, max 30 epochs)" and t.loc["Listed_One", "Events found"] == "20 of 25"
    assert t.loc["Extra_Eval", "Events found"] == "1 of 2" and "rebuilt" in t.loc["Extra_Eval", "Run"]
    long = core.recall_by_type(ev)
    assert set(long["type"]) == {"spike", "dropout"}
    text = " ".join(core.evaluation_caveats(ev))
    assert "baseline" in text and "rebuilt from saved files" in text and "capped epochs" in text
    assert core.evaluation_png("Extra_Eval") is not None and core.evaluation_png("Nope") is None
    assert not (core.eval_dir() / "Extra_Eval_summary.json").exists()  # nothing is written


def test_evaluation_without_summary_file_uses_saved_files(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "ROOT", tmp_path)
    _write_eval_files(core.eval_dir(), with_summary=False)
    assert list(core.evaluation_table(core.load_evaluation())["Building"]) == ["Extra_Eval"]
