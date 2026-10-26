"""
Logic behind the Streamlit demo (``app/app.py``). No Streamlit here, so everything can be tested without a browser
(see ``tests/test_demo_core.py``; the tests use the PCA stand-in scorer and need no TensorFlow).

Nothing in this module trains a model or writes to disk. It reuses the project's code instead of re-implementing it:

* scripts.preprocessing / scripts.feature_engineering   wide table -> the 13 FEATURE_COLS frame
* scripts.evaluation.score_series / rebuild_features     scoring (with or without injected anomalies)
* scripts.anomaly_injection                              Event, inject, activity_profile
* scripts.splitting.decide_split                         split policy

Main entry points: ``discover_buildings``, ``prepare_inputs``, ``load_detector``, ``score_features``,
``run_injection``, ``onboard_building``, ``load_evaluation``.
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_PROJECT_ROOT, _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts.anomaly_injection import Event, activity_profile, inject  # noqa: E402
from scripts.evaluation import LSTMBackend, rebuild_features, score_series, summarize_tables  # noqa: E402
from scripts.feature_engineering import engineer_features  # noqa: E402
from scripts.preprocessing import preprocess_building  # noqa: E402
from scripts.splitting import decide_split  # noqa: E402
from src.config import BUILDING_SPLIT, BUILDINGS, CONFIG, EVAL, FEATURE_COLS  # noqa: E402

# Project root used for every path below. Tests point it at a temporary folder.
ROOT = _PROJECT_ROOT
W = int(CONFIG["window_size"])
MIN_START_DAYS = 8  # an injection never starts in the first 8 days (168 h look-back + warm-up)
ECHO_HOURS = int(EVAL["purge_after_hours"])  # lag_168 / rolling_mean_168 carry an event forward this many rows


# ----------------------------------------------------------------------------------------------------------------------
# Paths and friendly errors
# ----------------------------------------------------------------------------------------------------------------------
class DemoFileError(Exception):
    """A missing file or unusable input. ``command`` says what to run (shown by the app instead of a traceback)."""

    def __init__(self, message: str, command: str | None = None):
        super().__init__(message)
        self.message = message
        self.command = command


def model_dir() -> Path:
    return ROOT / CONFIG["model_dir"]


def scaler_dir() -> Path:
    return ROOT / "scalers"


def plots_dir() -> Path:
    return ROOT / CONFIG["output_dir"]


def eval_dir() -> Path:
    return plots_dir() / "evaluation"


def wide_path() -> Path:
    return ROOT / CONFIG["data_url"]


def meta_path() -> Path:
    return ROOT / CONFIG["meta_url"]


def _train_hint(building: str) -> str:
    return (f"1) add \"{building}\" to BUILDINGS in src/config.py   2) python scripts/Model_Training.py "
            "(this trains every name in BUILDINGS, so list only the ones you need)")


def check_environment() -> list[DemoFileError]:
    """Problems that stop the whole demo (data files, no models). Empty list = ready."""
    problems = []
    if not wide_path().is_file():
        problems.append(DemoFileError(
            f"Data file not found: {wide_path().relative_to(ROOT)}. data/ is git-ignored, so it must be copied in by hand.",
            "Place temp_open_utc_complete.csv in data/processed/ (see the Data section of README.md)"))
    if not meta_path().is_file():
        problems.append(DemoFileError(
            f"Metadata file not found: {meta_path().relative_to(ROOT)}.",
            "Place meta_open.csv in data/raw/ (see the Data section of README.md)"))
    if not discover_buildings():
        problems.append(DemoFileError(
            f"No building has both a model ({model_dir().name}/<building>_best.h5) and a scaler "
            f"({scaler_dir().name}/<building>_scaler.pkl).",
            "python scripts/Model_Training.py"))
    return problems


# ----------------------------------------------------------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------------------------------------------------------
def _stems(folder: Path, suffix: str) -> set[str]:
    if not folder.is_dir():
        return set()
    return {p.name[: -len(suffix)] for p in folder.iterdir() if p.is_file() and p.name.endswith(suffix)}


def discover_buildings() -> list[str]:
    """Every building with BOTH a model and a scaler on disk (scans models/ and scalers/; nothing is hard-coded)."""
    return sorted(_stems(model_dir(), "_best.h5") & _stems(scaler_dir(), "_scaler.pkl"))


def incomplete_buildings() -> dict[str, str]:
    """Buildings that have a model or a scaler but not both: name -> what is missing."""
    models, scalers = _stems(model_dir(), "_best.h5"), _stems(scaler_dir(), "_scaler.pkl")
    out = {b: "scaler missing" for b in models - scalers}
    out.update({b: "model missing" for b in scalers - models})
    return dict(sorted(out.items()))


def extra_buildings() -> list[str]:
    """Pre-trained buildings that are NOT in BUILDINGS (the 'new building' candidates)."""
    return [b for b in discover_buildings() if b not in BUILDINGS]


def configured_without_model() -> list[str]:
    """Names in BUILDINGS that have no usable model yet."""
    have = set(discover_buildings())
    return [b for b in BUILDINGS if b not in have]


# ----------------------------------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------------------------------
def load_meta() -> pd.DataFrame:
    if not meta_path().is_file():
        raise DemoFileError(f"Metadata file not found: {meta_path().relative_to(ROOT)}.",
                            "Place meta_open.csv in data/raw/ (see the Data section of README.md)")
    return pd.read_csv(meta_path())


def load_building_wide(building: str) -> pd.DataFrame:
    """The wide hourly table (UTC index) reduced to the one building column, so loading takes well under a second."""
    path = wide_path()
    if not path.is_file():
        raise DemoFileError(f"Data file not found: {path.relative_to(ROOT)}.",
                            "Place temp_open_utc_complete.csv in data/processed/ (see the Data section of README.md)")
    try:
        df = pd.read_csv(path, usecols=["timestamp", building])
    except ValueError as exc:
        raise DemoFileError(f"{building} is not a column of {path.name}.",
                            "Check the building name against the columns of the wide table") from exc
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp").sort_index()


@dataclass
class Inputs:
    raw: pd.DataFrame  # hourly UTC energy (column "energy"), as produced by preprocess_building
    feats: pd.DataFrame  # the engineered frame (energy + FEATURE_COLS)
    load_seconds: float
    feature_seconds: float


def prepare_inputs(building: str, meta: pd.DataFrame | None = None) -> Inputs:
    """Wide table -> ``preprocess_building`` -> ``engineer_features`` (the training pipeline's own functions)."""
    t0 = time.perf_counter()
    meta = load_meta() if meta is None else meta
    wide = load_building_wide(building)
    with contextlib.redirect_stdout(io.StringIO()):  # preprocess_building prints a line per building
        raw = preprocess_building(wide, building)
    if raw is None or len(raw) <= W + 168 + 24 * MIN_START_DAYS:
        raise DemoFileError(f"{building} has too little hourly data to score.",
                            "Pick another building, or check data/processed/temp_open_utc_complete.csv")
    if not (meta["uid"] == building).any():
        raise DemoFileError(f"{building} is not listed in {meta_path().name} (needed for its time zone).",
                            "Check data/raw/meta_open.csv")
    t1 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        feats = engineer_features(raw, building, meta)
    return Inputs(raw, feats, t1 - t0, time.perf_counter() - t1)


# ----------------------------------------------------------------------------------------------------------------------
# Split policy
# ----------------------------------------------------------------------------------------------------------------------
def load_split_info(building: str, n_rows: int) -> dict:
    """
    Same logic as ``Model_Testing.load_split_info`` (that module imports Keras at load time, so it cannot be imported
    here without TensorFlow): ``models/<building>_split.json`` if present, otherwise the legacy 80/20 holdout.
    Adds ``legacy: True`` when the json is missing.
    """
    path = model_dir() / f"{building}_split.json"
    if not path.is_file():
        split_idx = int(n_rows * (1.0 - float(CONFIG.get("test_split", 0.2))))
        return {
            "strategy": "holdout",
            "split_idx": split_idx,
            "legacy": True,
            "message": f"No split metadata at {path.name}: assuming the legacy 80/20 holdout. "
                       "Re-run scripts/Model_Training.py to record the split.",
        }
    meta = json.loads(path.read_text(encoding="utf-8"))
    split_idx = int(meta["split_idx"])
    meta["legacy"] = False
    if n_rows < split_idx:
        meta["message"] += f" WARNING: data now has {n_rows} rows, fewer than at training ({split_idx})."
        split_idx = n_rows
    elif n_rows > int(meta.get("n_rows", n_rows)):
        meta["message"] += (f" {n_rows - int(meta['n_rows'])} newer rows than the training data are treated "
                            "as unseen test data.")
    meta["split_idx"] = split_idx
    return meta


def split_policy_view(info: dict, index: pd.DatetimeIndex) -> dict:
    """What the SPLIT POLICY box shows: label, message, training / test dates and the plain-English sentences."""
    n = len(index)
    split_idx = min(max(int(info["split_idx"]), 0), n)
    is_full = info.get("strategy") == "full"
    legacy = bool(info.get("legacy"))
    train_end_pos = (n if is_full else split_idx) - 1
    view = {
        "strategy": "full" if is_full else "holdout",
        "legacy": legacy,
        "label": "FULL history" if is_full else ("HOLDOUT (legacy 80/20 split)" if legacy else "HOLDOUT"),
        "message": info.get("message", ""),
        "train_start": index[0],
        "train_end": index[max(train_end_pos, 0)],
        "test_start": None if is_full or split_idx >= n else index[split_idx],
        "test_end": None if is_full or split_idx >= n else index[-1],
        "split_idx": n if is_full else split_idx,
        "sentences": [],
    }
    if is_full:
        view["sentences"].append(
            "There is no held-out test period. The model was trained on every season, so its flags on this history "
            "are in-sample. Testing is done with blocked folds plus injected anomalies (see the Evaluation page).")
    elif legacy:
        view["sentences"].append(
            "This model was trained before the split policy existed. It used a plain 80/20 time split: trained on the "
            f"first 80% ({_d(view['train_start'])} to {_d(view['train_end'])}), with the last 20% "
            f"({_d(view['test_start'])} to {_d(view['test_end'])}) held out as the test period.")
        view["sentences"].append(
            "With about a year of data that test period is a season the model barely saw, which is why the policy was "
            "changed. Flags before the split date are in-sample.")
    else:
        view["sentences"].append(
            f"The model was trained on {_d(view['train_start'])} to {_d(view['train_end'])} and tested on "
            f"{_d(view['test_start'])} to {_d(view['test_end'])}. Flags in the test period are out-of-sample.")
    return view


def _d(ts) -> str:
    return "n/a" if ts is None else pd.Timestamp(ts).strftime("%Y-%m-%d")


# ----------------------------------------------------------------------------------------------------------------------
# Detector and scoring
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class Detector:
    """A fitted scaler plus anything with ``errors(windows) -> per-window mean squared error`` (LSTM or stand-in)."""

    scaler: object
    backend: object
    name: str = "lstm"


class BatchedLSTMBackend(LSTMBackend):
    """``evaluation.LSTMBackend`` with a larger predict batch: same errors (differences ~1e-8), about 6x faster."""

    batch_size = 1024

    def errors(self, X):
        rec = self.model.predict(X, batch_size=self.batch_size, verbose=0)
        return np.mean((X - rec) ** 2, axis=(1, 2))


def import_tensorflow() -> float:
    """Import TensorFlow / Keras once (the slow part of start-up). Returns the seconds it took."""
    t0 = time.perf_counter()
    try:
        import tensorflow  # noqa: F401
        import keras  # noqa: F401
    except ModuleNotFoundError as exc:
        raise DemoFileError(f"TensorFlow / Keras could not be imported ({exc}).",
                            "pip install -r requirements.txt") from exc
    return time.perf_counter() - t0


def load_detector(building: str) -> Detector:
    """Load models/<building>_best.h5 (compile=False) and scalers/<building>_scaler.pkl."""
    mp = model_dir() / f"{building}_best.h5"
    sp = scaler_dir() / f"{building}_scaler.pkl"
    if not mp.is_file() or not sp.is_file():
        what = mp if not mp.is_file() else sp
        raise DemoFileError(f"No trained {'model' if what is mp else 'scaler'} for {building} ({what.relative_to(ROOT)}).",
                            _train_hint(building))
    import_tensorflow()
    import joblib
    from keras.models import load_model

    scaler = joblib.load(sp)
    n_feat = len(FEATURE_COLS)
    if getattr(scaler, "n_features_in_", n_feat) != n_feat:
        raise DemoFileError(f"The scaler of {building} was fitted on {scaler.n_features_in_} features but the code "
                            f"uses {n_feat}.", _train_hint(building))
    model = load_model(mp, compile=False)
    if model.input_shape[-1] != n_feat:
        raise DemoFileError(f"The model of {building} expects {model.input_shape[-1]} features but the code uses "
                            f"{n_feat}.", _train_hint(building))
    return Detector(scaler, BatchedLSTMBackend(model), "lstm")


def score_features(feats: pd.DataFrame, detector: Detector, split_idx: int | None = None) -> pd.DataFrame:
    """
    ``scripts.evaluation.score_series`` (scale -> windows -> errors -> adaptive threshold, drift, flags), plus a
    ``split`` column: "test" for windows ending at or after feature row ``split_idx`` (default: none are test).
    One row per window, stamped with the window's LAST hour; the first 168 h are never flagged.
    """
    res = score_series(feats, detector.scaler, detector.backend)
    end_pos = np.arange(len(res)) + W - 1
    limit = len(feats) if split_idx is None else int(split_idx)
    res["split"] = np.where(end_pos < limit, "train", "test")
    return res


def headline_stats(res: pd.DataFrame) -> dict:
    n = len(res)
    flags = int(res["anomaly_flag"].sum())
    drift = int(res["drift_flag"].sum())
    return {
        "hours_scored": n,
        "flags": flags,
        "anomaly_rate": flags / n if n else float("nan"),
        "drift_hours": drift,
        "drift_rate": drift / n if n else float("nan"),
        "periods": len(flag_periods(res)),
    }


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(first, last) positions, both inclusive, of each run of True."""
    padded = np.concatenate([[0], mask.astype(int), [0]])
    d = np.diff(padded)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def flag_periods(res: pd.DataFrame, flag: str = "anomaly_flag") -> pd.DataFrame:
    """Runs of consecutive flagged hours: start, end, hours, peak error (MSE), peak normalized error, when it peaked."""
    cols = ["start", "end", "hours", "peak_error", "peak_normalized_error", "peak_time"]
    rows = []
    for a, b in _runs(res[flag].to_numpy() == 1):
        seg = res.iloc[a : b + 1]
        rows.append({
            "start": res.index[a], "end": res.index[b], "hours": b - a + 1,
            "peak_error": float(seg["reconstruction_error"].max()),
            "peak_normalized_error": float(seg["normalized_error"].max()),
            "peak_time": seg["reconstruction_error"].idxmax(),
        })
    return pd.DataFrame(rows, columns=cols)


def drift_periods(res: pd.DataFrame) -> pd.DataFrame:
    p = flag_periods(res, "drift_flag")
    return p[["start", "end", "hours"]]


# ----------------------------------------------------------------------------------------------------------------------
# Injection
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class InjectionSpec:
    type: str
    start: pd.Timestamp  # UTC, first affected hour
    hours: int
    magnitude: float | None = None  # spike: x std of the energy; dropout / level_shift: factor; flatline: unused


def magnitude_info(typ: str) -> dict:
    """Defaults from EVAL (midpoint of the evaluation range) and slider bounds for the UI."""
    mid = lambda r: (float(r[0]) + float(r[1])) / 2.0
    hours = EVAL[typ]["hours"]
    base = {"hours": int(round(mid(hours))), "hours_range": (1, max(int(hours[1]) * 2, 24)), "magnitude": None,
            "label": None, "range": None, "step": None}
    if typ == "spike":
        base.update(magnitude=mid(EVAL["spike"]["amplitude_std"]), range=(1.0, 10.0), step=0.5,
                    label="Spike size (x the standard deviation of the building's energy)")
    elif typ == "dropout":
        base.update(magnitude=mid(EVAL["dropout"]["factor"]), range=(0.0, 0.5), step=0.01,
                    label="Remaining load (fraction of normal; 0 = no reading at all)")
    elif typ == "level_shift":
        base.update(magnitude=mid(EVAL["level_shift"]["factor"]), range=(1.05, 2.5), step=0.05,
                    label="Load multiplier (1.5 = 50% more than normal)")
    return base


def describe_spec(spec: InjectionSpec) -> str:
    if spec.type == "spike":
        return f"+{spec.magnitude:.1f} std"
    if spec.type == "dropout":
        return f"x{spec.magnitude:.2f} of normal load"
    if spec.type == "level_shift":
        return f"x{spec.magnitude:.2f} load"
    return "frozen at last reading"


def spec_to_event(spec: InjectionSpec, index: pd.DatetimeIndex, event_id: int = 0) -> Event:
    """Validate a spec against the feature index and return the ``Event`` (row positions) that ``inject`` expects."""
    if spec.type not in EVAL["event_types"]:
        raise ValueError(f"Unknown anomaly type {spec.type!r}")
    if spec.hours < 1:
        raise ValueError("Duration must be at least 1 hour")
    start = pd.Timestamp(spec.start)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    earliest = index[0] + pd.Timedelta(days=MIN_START_DAYS)
    if start < earliest:
        raise ValueError(f"Start must be after {earliest:%Y-%m-%d %H:%M} UTC (never in the first {MIN_START_DAYS} days: "
                         "the detector is still warming up there)")
    pos = int(index.searchsorted(start.floor("h")))
    end = pos + int(spec.hours) - 1
    if end > len(index) - 1:
        raise ValueError(f"The injection would run past the end of the data ({index[-1]:%Y-%m-%d %H:%M} UTC)")
    if spec.type in ("dropout", "level_shift", "spike") and spec.magnitude is None:
        raise ValueError(f"{spec.type} needs a magnitude")
    mag = float("nan") if spec.type == "flatline" else float(spec.magnitude)
    return Event(event_id, spec.type, pos, end, mag)


def build_events(specs: list[InjectionSpec], index: pd.DatetimeIndex) -> list[Event]:
    events = [spec_to_event(s, index, i) for i, s in enumerate(specs)]
    ordered = sorted(events, key=lambda e: e.start)
    for a, b in zip(ordered, ordered[1:]):
        if b.start <= a.end:
            raise ValueError(f"Injection {a.event_id + 1} and {b.event_id + 1} overlap; move one of them")
    return events


def suggest_start(feats: pd.DataFrame, clean_res: pd.DataFrame, typ: str, hours: int,
                  avoid: list[tuple[int, int]] | None = None) -> pd.Timestamp:
    """
    A start time in a normal period: after the first 8 days, no anomaly / drift flag within a window or the 7-day echo
    around it, building active during the event (for the types where that matters), closest to the middle of the
    history. ``avoid`` = (first_row, last_row) of injections already placed. Relaxes the conditions if nothing fits.
    """
    n = len(feats)
    first = MIN_START_DAYS * 24
    last = n - int(hours) - W
    if last < first:
        raise ValueError("Not enough data to place an injection")
    bad = np.zeros(n, dtype=bool)
    for j in np.flatnonzero(((clean_res["anomaly_flag"] == 1) | (clean_res["drift_flag"] == 1)).to_numpy()):
        bad[j : j + W] = True  # window j covers feature rows j .. j + W - 1
    avoid_mask = np.zeros(n, dtype=bool)
    for a, b in avoid or []:
        avoid_mask[max(a - W, 0) : b + ECHO_HOURS + 1] = True
    act = activity_profile(feats["energy"].to_numpy(), float(EVAL["active_quantile"]))

    def csum(x):
        return np.concatenate([[0], np.cumsum(x.astype(int))])

    c_bad, c_avoid = csum(bad), csum(avoid_mask)
    c_act = np.concatenate([[0], np.cumsum(act)])
    starts = np.arange(first, last + 1)
    h = int(hours)
    lo, hi = np.maximum(starts - W, 0), np.minimum(starts + h + W, n)  # rows a window around the event touches
    clean_ok = (c_bad[hi] - c_bad[lo]) == 0
    free_ok = (c_avoid[hi] - c_avoid[lo]) == 0
    active_ok = (c_act[starts + h] - c_act[starts]) / h >= float(EVAL["min_active_frac"])
    mid = n // 2
    for ok in (clean_ok & free_ok & active_ok, free_ok & active_ok, free_ok, np.ones_like(free_ok)):
        if ok.any():
            cand = starts[ok]
            return feats.index[int(cand[np.argmin(np.abs(cand - mid))])]
    return feats.index[first]


@dataclass
class InjectionResult:
    specs: list
    events: list  # resolved Event objects (magnitude in energy units, effect_std, inert)
    raw_injected: pd.Series  # corrupted COPY of the raw hourly series
    clean_res: pd.DataFrame
    inj_res: pd.DataFrame
    event_table: pd.DataFrame  # one row per injection: detected, delay, peak z, note
    new_false_alarms: pd.DataFrame  # flag periods in the corrupted run that the clean run did not have, away from events
    echo_flag_hours: int  # new flagged hours in the 7 days after an event (lag features carry the event forward)
    spans: list  # (first hour, last hour) of each injection, for shading
    seconds: float = 0.0


def _event_window_bounds(ev: Event, n_res: int) -> tuple[int, int]:
    """Positions in the scored frame of the windows that contain any injected hour (window j ends at feature row j+W-1)."""
    return max(ev.start - W + 1, 0), min(ev.end, n_res - 1)


def run_injection(building: str, inputs: Inputs, meta: pd.DataFrame, detector: Detector, specs: list[InjectionSpec],
                  clean_res: pd.DataFrame | None = None, split_idx: int | None = None) -> InjectionResult:
    """
    Corrupt a COPY of the raw series, rebuild the features, rescore with the same detector, and report per injection
    whether it was found. Event windows, the 7-day echo and the energy std follow ``scripts/evaluation.py``.
    Nothing is written to disk.
    """
    t0 = time.perf_counter()
    feats, raw = inputs.feats, inputs.raw
    events = build_events(specs, feats.index)
    energy_std = float(raw["energy"].std())
    raw_inj, resolved = inject(raw["energy"], feats.index, events, energy_std, float(EVAL["min_effect_std"]))
    feats_inj = rebuild_features(raw_inj, raw, building, meta, feats.index)
    if clean_res is None:
        clean_res = score_features(feats, detector, split_idx)
    inj_res = score_features(feats_inj, detector, split_idx)
    n_res = len(inj_res)
    flag_inj = inj_res["anomaly_flag"].to_numpy() == 1
    flag_clean = clean_res["anomaly_flag"].to_numpy() == 1
    drift_inj = inj_res["drift_flag"].to_numpy() == 1

    near = np.zeros(n_res, dtype=bool)  # event windows plus the echo after them
    in_event = np.zeros(n_res, dtype=bool)
    rows = []
    for k, ev in enumerate(resolved):
        a, b = _event_window_bounds(ev, n_res)
        in_event[a : b + 1] = True
        near[a : min(ev.end + ECHO_HOURS, n_res - 1) + 1] = True  # a window still holds echoed rows until end + 168 + W - 1
        seg = np.arange(a, b + 1)
        hit = seg[flag_inj[a : b + 1]]
        any_alert = bool((flag_inj[a : b + 1] | drift_inj[a : b + 1]).any())
        t_start = feats.index[ev.start]
        delay = float((inj_res.index[hit[0]] - t_start) / pd.Timedelta(hours=1)) if len(hit) else float("nan")
        note = ""
        if ev.inert:
            note = "Changed almost nothing (e.g. the building was idle anyway), so no detector could see it."
        elif len(hit) and flag_clean[hit].all():
            note = "Every flagged window here was already flagged in the clean run, so the injection is not what raised them."
        elif not len(hit) and any_alert:
            note = "Only the drift flag fired (a sustained change is reported as drift, not as an anomaly)."
        rows.append({
            "#": k + 1, "type": ev.type, "start": t_start, "end": feats.index[ev.end], "hours": ev.end - ev.start + 1,
            "setting": describe_spec(specs[k]), "detected": bool(len(hit)), "drift_only": bool(not len(hit) and any_alert),
            "delay_hours": delay, "peak_normalized_error": float(inj_res["normalized_error"].iloc[a : b + 1].max()),
            "effect_std": ev.effect_std, "inert": ev.inert, "note": note,
        })
    new_flag = flag_inj & ~flag_clean
    fa_mask = new_flag & ~near
    fa = inj_res.loc[fa_mask, ["energy", "reconstruction_error", "normalized_error", "threshold", "anomaly_flag", "drift_flag"]]
    fa_periods = flag_periods(inj_res.assign(anomaly_flag=fa_mask.astype(int)))
    echo = int((new_flag & near & ~in_event).sum())
    spans = [(feats.index[e.start], feats.index[e.end]) for e in resolved]
    return InjectionResult(list(specs), resolved, raw_inj, clean_res, inj_res, pd.DataFrame(rows), fa_periods, echo,
                           spans, time.perf_counter() - t0)


# ----------------------------------------------------------------------------------------------------------------------
# New building flow
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class Onboarding:
    building: str
    steps: list = field(default_factory=list)  # {"step", "detail", "seconds"}
    inputs: Inputs | None = None
    decision: dict | None = None  # what decide_split says today
    recorded: dict | None = None  # the split recorded at training time (json or legacy fallback)
    detector: Detector | None = None
    res: pd.DataFrame | None = None
    total_seconds: float = 0.0


def onboard_building(building: str, meta: pd.DataFrame | None = None) -> Onboarding:
    """
    The real onboarding path for a building whose model was trained in advance, with measured seconds per step:
    load data, build features, decide split policy, load model + scaler, score. Nothing is cached, so the seconds are
    what a first-time run costs (TensorFlow itself is imported once at warm-up). Training is NOT part of this.
    """
    meta = load_meta() if meta is None else meta
    ob = Onboarding(building)
    t_all = time.perf_counter()

    t0 = time.perf_counter()
    wide = load_building_wide(building)
    with contextlib.redirect_stdout(io.StringIO()):
        raw = preprocess_building(wide, building)
    if raw is None or len(raw) <= W + 168 + 24 * MIN_START_DAYS:
        raise DemoFileError(f"{building} has too little hourly data to score.", "Check data/processed/temp_open_utc_complete.csv")
    s_load = time.perf_counter() - t0
    ob.steps.append({"step": "1. Load data", "seconds": s_load,
                     "detail": f"Read the building's column from the wide table and resample to hourly: {len(raw)} valid "
                               f"hours, {raw.index[0]:%Y-%m-%d} to {raw.index[-1]:%Y-%m-%d} (UTC)."})

    t0 = time.perf_counter()
    if not (meta["uid"] == building).any():
        raise DemoFileError(f"{building} is not listed in {meta_path().name}.", "Check data/raw/meta_open.csv")
    with contextlib.redirect_stdout(io.StringIO()):
        feats = engineer_features(raw, building, meta)
    s_feat = time.perf_counter() - t0
    ob.inputs = Inputs(raw, feats, s_load, s_feat)
    ob.steps.append({"step": "2. Build features", "seconds": s_feat,
                     "detail": f"Calendar (local time), lag and rolling features: {len(feats)} rows, "
                               f"{len(FEATURE_COLS)} model features."})

    t0 = time.perf_counter()
    ob.decision = decide_split(feats.index, building, CONFIG, BUILDING_SPLIT)
    ob.recorded = load_split_info(building, len(feats))
    s_split = time.perf_counter() - t0
    ob.steps.append({"step": "3. Decide split policy", "seconds": s_split, "detail": ob.decision["message"]})

    t0 = time.perf_counter()
    ob.detector = load_detector(building)
    s_model = time.perf_counter() - t0
    ob.steps.append({"step": "4. Load model and scaler", "seconds": s_model,
                     "detail": f"{(model_dir() / (building + '_best.h5')).name} and "
                               f"{(scaler_dir() / (building + '_scaler.pkl')).name}, trained in advance."})

    t0 = time.perf_counter()
    ob.res = score_features(feats, ob.detector, ob.recorded["split_idx"])
    s_score = time.perf_counter() - t0
    st = headline_stats(ob.res)
    ob.steps.append({"step": "5. Score", "seconds": s_score,
                     "detail": f"{st['hours_scored']} windows scored, {st['flags']} flagged hours "
                               f"({st['anomaly_rate']:.1%}), {st['drift_hours']} drift hours."})
    ob.total_seconds = time.perf_counter() - t_all
    return ob


def loss_curve_path(building: str) -> Path | None:
    p = plots_dir() / f"{building}_loss.png"
    return p if p.is_file() else None


# ----------------------------------------------------------------------------------------------------------------------
# Evaluation files
# ----------------------------------------------------------------------------------------------------------------------
EVAL_COMMAND = "python scripts/evaluation.py"
EVENT_TYPES = list(EVAL["event_types"])


@dataclass
class EvaluationData:
    summary: pd.DataFrame  # one row per building (columns as in evaluation_summary.csv, plus "run")
    report_text: str
    backend: str
    seed: str


def _summarize_saved(building: str, folder: Path) -> tuple[dict, int]:
    """
    ``scripts.evaluation.summarize_tables`` applied to the saved ``<building>_windows.csv`` / ``_events.csv`` in
    ``folder`` (same steps as ``evaluation.rebuild_from_files``, which only reads the project's own folder).
    Returns (summary dict, number of folds).
    """
    rows = pd.read_csv(folder / f"{building}_windows.csv")
    ev = pd.read_csv(folder / f"{building}_events.csv")
    if "inert" not in ev.columns:
        ev["inert"] = False
    ev["inert"] = ev["inert"].astype(bool)
    summary = summarize_tables(rows, ev[~ev["inert"]].copy(), int(ev["inert"].sum()))
    return summary, int(rows["fold"].nunique()) if "fold" in rows.columns else 1


def load_evaluation() -> EvaluationData:
    """
    Read output_plots/evaluation/. ``evaluation_summary.csv`` only lists the buildings of the latest run, so any building
    that has saved ``_windows.csv`` + ``_events.csv`` files but no row there is summarised from those files in memory
    (``scripts.evaluation.rebuild_from_files``; nothing is written). Raises DemoFileError if there are no results.
    """
    d = eval_dir()
    sp = d / "evaluation_summary.csv"
    summary = pd.read_csv(sp) if sp.is_file() else pd.DataFrame()
    rp = d / "evaluation_report.md"
    text = rp.read_text(encoding="utf-8") if rp.is_file() else ""
    runs = {}
    for line in text.splitlines():  # | Building | Mode | Events | ...  ->  "blocked (3 folds, max 30 epochs)"
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and len(cells) >= 9 and cells[0] not in ("Building", "---"):
            runs[cells[0]] = cells[1]
    if len(summary):
        summary["run"] = summary["building"].map(runs).fillna(summary["mode"])
        summary["source"] = "evaluation_summary.csv"
    have = set(summary["building"]) if len(summary) else set()
    rebuilt = []
    names = sorted(_stems(d, "_windows.csv") & _stems(d, "_events.csv") - have,
                   key=lambda b: (b not in BUILDINGS, b))
    for b in names:
        s, folds = _summarize_saved(b, d)
        row = {"building": b, "mode": "blocked" if folds > 1 else "holdout",
               **{k: v for k, v in s.items() if k != "by_type"}}
        for t, v in s["by_type"].items():
            row[f"recall_{t}"] = v["recall"]
        row["run"] = f"{row['mode']}" + (f" ({folds} folds)" if folds > 1 else "") + ", rebuilt from saved files"
        row["source"] = "rebuilt from files"
        rebuilt.append(row)
    if rebuilt:
        summary = pd.concat([summary, pd.DataFrame(rebuilt)], ignore_index=True)
    if summary.empty:
        raise DemoFileError(f"No evaluation results found ({sp.relative_to(ROOT)}).", EVAL_COMMAND)
    order = {b: i for i, b in enumerate(BUILDINGS)}
    summary = summary.assign(_o=summary["building"].map(lambda b: order.get(b, len(order))))         .sort_values(["_o", "building"]).drop(columns="_o").reset_index(drop=True)
    m = re.search(r"Backend: `([^`]*)`", text)
    seed = re.search(r"Seed ([\d, ]+)\.", text)
    return EvaluationData(summary, text, m.group(1) if m else "unknown", seed.group(1) if seed else "unknown")


def evaluation_table(ev: EvaluationData) -> pd.DataFrame:
    """The per-building table the Evaluation page shows."""
    s = ev.summary
    out = pd.DataFrame({
        "Building": s["building"],
        "Events found": s["events_detected"].astype(int).astype(str) + " of " + s["events"].astype(int).astype(str),
        "Median delay (h)": s["median_delay_hours"],
        "False-alarm rate (%)": s["false_alarm_rate_pct"],
        "Precision": s["window_precision"],
        "AUC, LSTM": s["auc_normalized_error"],
        "AUC, same hour last week": s["auc_baseline_same_hour_last_week"],
    })
    out["AUC difference"] = out["AUC, LSTM"] - out["AUC, same hour last week"]
    out["Run"] = s["run"]
    return out


def recall_by_type(ev: EvaluationData) -> pd.DataFrame:
    """Long table: building, type, recall (fraction). Types without events are left out."""
    rows = []
    for _, r in ev.summary.iterrows():
        for t in EVENT_TYPES:
            v = r.get(f"recall_{t}")
            if v is not None and pd.notna(v):
                rows.append({"building": r["building"], "type": t, "recall": float(v)})
    return pd.DataFrame(rows, columns=["building", "type", "recall"])


def evaluation_png(building: str) -> Path | None:
    p = eval_dir() / f"{building}_evaluation.png"
    return p if p.is_file() else None


def evaluation_caveats(ev: EvaluationData) -> list[str]:
    """Plain-language caveats computed from the numbers, so the page never reads better than the files."""
    t = evaluation_table(ev)
    out = []
    n = len(t)
    diff = t["AUC difference"]
    if n:
        better = int((diff > 0.05).sum())
        close = int(((diff >= -0.05) & (diff <= 0.05)).sum())
        worse = int((diff < -0.05).sum())
        out.append(f"AUC against the 'same hour last week' baseline: clearly better (by more than 0.05) in {better} of {n} "
                   f"buildings, within 0.05 in {close}, clearly worse in {worse}. The difference ranges from "
                   f"{diff.min():+.2f} to {diff.max():+.2f}. Where that difference is small, the LSTM is not clearly better than a rule "
                   "that needs no training.")
        per_event = ev.summary["events"].astype(int)
        lo, hi = int(per_event.min()), int(per_event.max())
        out.append(f"Each building has only {lo if lo == hi else f'{lo} to {hi}'} scored events, so one missed event moves "
                   f"recall by about {100 / max(hi, 1):.0f} points. Differences of a few points between buildings or "
                   "types are not reliable.")
        rt = recall_by_type(ev)
        if len(rt):
            avg = rt.groupby("type")["recall"].mean().sort_values()
            out.append(f"Weakest event type on average over buildings: {avg.index[0].replace('_', ' ')} "
                       f"({avg.iloc[0]:.0%}); strongest: {avg.index[-1].replace('_', ' ')} ({avg.iloc[-1]:.0%}).")
        fa = t["False-alarm rate (%)"]
        out.append(f"False alarms are {fa.min():.1f}% to {fa.max():.1f}% of clean windows, i.e. roughly one in every "
                   f"{100 / max(fa.max(), 1e-9):.0f} to {100 / max(fa.min(), 1e-9):.0f} clean hours is flagged.")
    runs = sorted(set(t["Run"]))
    out.append("Runs shown: " + "; ".join(runs) + f". Backend of the stored report: {ev.backend}.")
    if any("epochs" in r for r in runs):
        out.append("A run with capped epochs (the quick setting) is faster to produce but its temporary models may be "
                   "weaker than in a full run.")
    if (ev.summary["source"] == "rebuilt from files").any():
        out.append("Rows marked 'rebuilt from saved files' have no entry in evaluation_summary.csv (it only lists the latest "
                   "run). They were summarised from their saved events and windows files, so they may come from runs with "
                   "different settings, and their backend and epoch cap are not recorded.")
    absent = [b for b in BUILDINGS if b not in set(t["Building"])]
    if absent:
        out.append(f"Listed in BUILDINGS but not evaluated: {', '.join(absent)}.")
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Command line check:  python scripts/demo_core.py --check [Building]
# ----------------------------------------------------------------------------------------------------------------------
def _check(building: str | None) -> None:
    for p in check_environment():
        print(f"PROBLEM: {p.message}  ->  {p.command}")
    found = discover_buildings()
    print(f"Buildings with model + scaler: {', '.join(found) or 'none'}")
    print(f"Not in BUILDINGS (new-building candidates): {', '.join(extra_buildings()) or 'none'}")
    if not found:
        return
    b = building or found[0]
    meta = load_meta()
    t0 = time.perf_counter()
    inputs = prepare_inputs(b, meta)
    print(f"{b}: {len(inputs.feats)} feature rows ({inputs.load_seconds:.2f}s load, {inputs.feature_seconds:.2f}s features)")
    print(f"TensorFlow import {import_tensorflow():.1f}s")
    t0 = time.perf_counter()
    det = load_detector(b)
    info = load_split_info(b, len(inputs.feats))
    res = score_features(inputs.feats, det, info["split_idx"])
    print(f"model + scoring {time.perf_counter() - t0:.1f}s; {headline_stats(res)}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Smoke test of the demo logic (no browser).")
    ap.add_argument("--check", nargs="?", const="", metavar="BUILDING", help="load, score and print numbers")
    a = ap.parse_args()
    if a.check is None:
        ap.print_help()
    else:
        _check(a.check or None)
