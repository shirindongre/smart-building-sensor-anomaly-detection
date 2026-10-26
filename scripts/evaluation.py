"""
Evaluate the anomaly detector with injected anomalies.

The evaluation depends on each building's split policy (``scripts/splitting.py``), so it follows your settings:

* ONE YEAR OF DATA OR LESS  ("full" history)  ->  BLOCKED testing + injected anomalies (always).
    The year is cut into 14-day blocks that are dealt out to K folds. For each fold a temporary model is trained on
    the other blocks (with a purge gap around every held-out block so no information leaks in through the 7-day
    look-back features), anomalies are injected into the held-out blocks, and the model has to find them. Every
    season is in training for every fold, and every block is tested exactly once.
* MORE THAN ONE YEAR        ("holdout")       ->  no blocked testing. The building is evaluated on its chronological
    test period with the production model. Injected anomalies are optional: pass --inject (or set
    EVAL["inject_on_holdout"] = True in src/config.py).

Run from the project root:

    python scripts/evaluation.py                       # all buildings in src/config.py
    python scripts/evaluation.py --buildings Office_Annika
    python scripts/evaluation.py --quick               # 3 folds, 30 epochs max (faster)
    python scripts/evaluation.py --backend pca         # no TensorFlow needed; PCA reconstruction as a quick stand-in
    python scripts/evaluation.py --inject              # also inject into holdout buildings' test period

Outputs go to output_plots/evaluation/: <building>_events.csv, <building>_windows.csv, <building>_evaluation.png,
evaluation_summary.csv and evaluation_report.md. Temporary fold models are NOT saved; production models in models/
are never changed.

How events are scored
---------------------
Each 24-hour window gets one reconstruction error, stamped with the window's last hour. A window is an EVENT window
if it contains any injected hour. Windows that end up to 7 days after an event are IGNORED, because the 24 h / 168 h
lag and rolling features carry the event forward. All other held-out windows are CLEAN.
* inert events    = events that changed (almost) nothing (mean |change| < EVAL["min_effect_std"] std, e.g. a dropout in a
                    building that is idle anyway). They stay in the data and in the events CSV but are not scored; their
                    windows are ignored. Dropout / flatline / level_shift are also placed where the load is active.
* event detected  = at least one of its event windows is flagged as an anomaly (``alerted`` also counts drift flags)
* false alarms    = flagged CLEAN windows (reported as % of clean windows)
* precision       = flagged event windows / all flagged (event + clean) windows
* ROC-AUC         = how well the error separates event windows from clean ones, threshold-free
* baseline AUC    = the same for a trivial detector: |energy - energy one week earlier| (max over the window)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_PROJECT_ROOT, _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import MinMaxScaler  # noqa: E402

from scripts.anomaly_injection import activity_profile, inject, place_events  # noqa: E402
from scripts.anomaly_scoring import add_adaptive_thresholding, build_results_df  # noqa: E402
from scripts.feature_engineering import engineer_features  # noqa: E402
from scripts.preprocessing import preprocess_building  # noqa: E402
from scripts.sequence_creations import create_sequences  # noqa: E402
from scripts.splitting import decide_split, validation_row_mask  # noqa: E402
from src.config import BUILDING_SPLIT, BUILDINGS, CONFIG, EVAL, FEATURE_COLS  # noqa: E402

OUT_DIR = _PROJECT_ROOT / CONFIG["output_dir"] / "evaluation"
W = int(CONFIG["window_size"])


# ----------------------------------------------------------------------------------------------------------------------
# Detector backends: something that learns "normal" windows and returns one reconstruction error per window
# ----------------------------------------------------------------------------------------------------------------------
class PCABackend:
    """Quick stand-in / baseline: PCA reconstruction of the flattened window. No TensorFlow needed."""

    name = "pca"

    def __init__(self, k: int):
        self.k = k
        self.pca = None

    def fit(self, X_fit, X_val, epochs=None, seed=0):
        from sklearn.decomposition import PCA

        flat = X_fit.reshape(len(X_fit), -1)
        self.pca = PCA(n_components=min(self.k, flat.shape[0] - 1, flat.shape[1]), random_state=seed).fit(flat)
        return self

    def errors(self, X):
        flat = X.reshape(len(X), -1)
        rec = self.pca.inverse_transform(self.pca.transform(flat))
        return np.mean((flat - rec) ** 2, axis=1)


class LSTMBackend:
    """The project's LSTM autoencoder, trained freshly for each fold (not saved)."""

    name = "lstm"

    def __init__(self, model=None):
        self.model = model

    def fit(self, X_fit, X_val, epochs=None, seed=0):
        import tensorflow as tf
        from keras.callbacks import EarlyStopping, LambdaCallback

        from scripts.Model_Architechture import build_lstm_autoencoder

        tf.random.set_seed(seed)
        with contextlib.redirect_stdout(io.StringIO()):  # build_lstm_autoencoder prints a model summary
            self.model = build_lstm_autoencoder(
                window_size=X_fit.shape[1],
                n_features=X_fit.shape[2],
                lstm_units=CONFIG["lstm_units"],
                latent_dim=CONFIG["latent_dim"],
                decoder_units=CONFIG["decoder_units"],
                dropout_rate=CONFIG["dropout_rate"],
                learning_rate=CONFIG["learning_rate"],
            )
        self.model.fit(
            X_fit, X_fit,
            validation_data=(X_val, X_val),
            epochs=int(epochs or CONFIG["epochs"]),
            batch_size=CONFIG["batch_size"],
            shuffle=True,
            verbose=0,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=CONFIG["patience"], restore_best_weights=True),
                LambdaCallback(on_epoch_end=lambda ep, logs: print(
                    f"      epoch {ep + 1}: loss {logs.get('loss', float('nan')):.5f}  val_loss {logs.get('val_loss', float('nan')):.5f}",
                    flush=True) if (ep + 1) % 5 == 0 else None),
            ],
        )
        return self

    def errors(self, X):
        rec = self.model.predict(X, verbose=0)
        return np.mean((X - rec) ** 2, axis=(1, 2))


def make_backend(name: str):
    if name == "pca":
        return PCABackend(int(EVAL.get("pca_components", 16)))
    return LSTMBackend()


# ----------------------------------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------------------------------
def window_counts(mask: np.ndarray, w: int = W) -> np.ndarray:
    """Number of True rows inside every window (window i covers rows i .. i + w - 1)."""
    csum = np.concatenate([[0], np.cumsum(mask.astype(int))])
    n = len(mask) - w + 1
    return csum[w : w + n] - csum[:n]


def runs(mask: np.ndarray):
    """(start, end_exclusive) of each run of True in a boolean array."""
    padded = np.concatenate([[0], mask.astype(int), [0]])
    d = np.diff(padded)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def score_series(feats_inj: pd.DataFrame, scaler, backend) -> pd.DataFrame:
    """Scale -> windows -> reconstruction errors -> adaptive threshold / drift / anomaly flags (production logic)."""
    X = scaler.transform(feats_inj[FEATURE_COLS].to_numpy())
    windows = create_sequences(X, W)
    errors = backend.errors(windows)
    res = build_results_df(feats_inj, errors, W, split_idx=len(feats_inj))
    return add_adaptive_thresholding(res)


def baseline_window_score(raw_inj: pd.Series, feature_index: pd.DatetimeIndex, energy_std: float) -> np.ndarray:
    """Trivial detector: |energy - energy 168 h earlier| / std, max over each window. One value per window."""
    resid = (raw_inj - raw_inj.shift(168)).abs() / max(energy_std, 1e-9)
    per_row = resid.reindex(feature_index)
    return per_row.rolling(W, min_periods=1).max().to_numpy()[W - 1 :]


def label_windows(n_rows: int, events, held: np.ndarray, purge_after: int):
    """
    Per window (length n_rows - W + 1): event id (-1 = none), ignore flag, and "evaluable" flag.
    evaluable = the whole window lies inside the held-out rows. Event windows end in [start, end + W - 1];
    windows ending up to ``purge_after`` rows after the event are ignored (feature echo).
    Inert events (they changed (almost) nothing, see anomaly_injection.inject) are neither positives nor clean:
    all their windows are ignored.
    """
    n_w = n_rows - W + 1
    ends = np.arange(n_w) + W - 1
    event_id = np.full(n_w, -1)
    ignore = np.zeros(n_w, dtype=bool)
    for ev in events:
        if getattr(ev, "inert", False):
            ignore |= (ends >= ev.start) & (ends <= ev.end + purge_after)
            continue
        in_event = (ends >= ev.start) & (ends <= ev.end + W - 1)
        event_id[in_event] = ev.event_id
        ignore |= (ends > ev.end + W - 1) & (ends <= ev.end + purge_after)
    evaluable = window_counts(held) == W
    ignore &= event_id < 0
    return event_id, ignore, evaluable


def collect_eval_rows(res: pd.DataFrame, event_id, ignore, evaluable, base_score, fold: int, feature_index) -> pd.DataFrame:
    keep = evaluable
    out = res.loc[keep, ["energy", "reconstruction_error", "normalized_error", "threshold",
                          "anomaly_flag", "drift_flag"]].copy()
    out["end_row"] = (np.flatnonzero(keep) + W - 1)
    out["event_id"] = event_id[keep]
    out["ignore"] = ignore[keep]
    out["baseline_score"] = base_score[keep]
    out["fold"] = fold
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------------------------------
def _auc(y, score):
    y = np.asarray(y).astype(int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, np.asarray(score, dtype=float)))


def compute_metrics(rows: pd.DataFrame, events: list, feature_index) -> tuple[dict, pd.DataFrame]:
    """rows = evaluable windows from all folds; events = resolved Event objects. Returns (summary dict, event table)."""
    rows = rows.copy()
    is_event = rows["event_id"] >= 0
    clean = (~is_event) & (~rows["ignore"])
    scored = is_event | clean  # ignored windows are left out of every metric
    flagged = rows["anomaly_flag"] == 1

    ev_rows, inert_rows = [], []
    for ev in events:
        base = {"event_id": ev.event_id, "type": ev.type, "start": feature_index[ev.start], "end": feature_index[ev.end],
                "hours": ev.end - ev.start + 1, "magnitude": ev.magnitude, "effect_std": ev.effect_std,
                "active_frac": ev.active_frac, "inert": bool(ev.inert)}
        r = rows[rows["event_id"] == ev.event_id]
        if ev.inert:
            inert_rows.append({**base, "detected": np.nan, "alerted": np.nan, "delay_hours": np.nan,
                               "max_normalized_error": np.nan})
            continue
        if r.empty:
            continue
        f_anom = r[r["anomaly_flag"] == 1]
        f_any = r[(r["anomaly_flag"] == 1) | (r["drift_flag"] == 1)]
        ev_rows.append({
            **base,
            "detected": bool(len(f_anom)), "alerted": bool(len(f_any)),
            "delay_hours": float(f_anom["end_row"].min() - ev.start) if len(f_anom) else np.nan,
            "max_normalized_error": float(r["normalized_error"].max()),
        })
    ev_df = pd.DataFrame(ev_rows)
    n_inert = len(inert_rows)
    summary = summarize_tables(rows, ev_df, n_inert)
    if n_inert:  # keep inert events in the table (inert=True, detected empty) so nothing is hidden
        ev_df = pd.concat([ev_df, pd.DataFrame(inert_rows)], ignore_index=True).sort_values("start")
    return summary, ev_df


def summarize_tables(rows: pd.DataFrame, ev_df: pd.DataFrame, n_inert: int) -> dict:
    """
    Summary metrics from the evaluated windows (``rows``) and the table of SCORED (non-inert) events ``ev_df``
    (columns type, detected, alerted, delay_hours). Used both live and when rebuilding from saved CSV files.
    """
    rows = rows.copy()
    is_event = rows["event_id"] >= 0
    clean = (~is_event) & (~rows["ignore"].astype(bool))
    scored = is_event | clean
    flagged = rows["anomaly_flag"] == 1
    n_clean = int(clean.sum())
    fp = int((flagged & clean).sum())
    tp_w = int((flagged & is_event).sum())
    sc = rows[scored]
    y = is_event[scored].astype(int)
    summary = {
        "events_injected": int(len(ev_df) + n_inert),
        "events_inert": int(n_inert),
        "events": int(len(ev_df)),
        "events_detected": int(ev_df["detected"].astype(bool).sum()) if len(ev_df) else 0,
        "event_recall": float(ev_df["detected"].astype(bool).mean()) if len(ev_df) else float("nan"),
        "events_alerted_incl_drift": int(ev_df["alerted"].astype(bool).sum()) if len(ev_df) else 0,
        "median_delay_hours": float(ev_df["delay_hours"].median()) if len(ev_df) else float("nan"),
        "clean_windows": n_clean,
        "false_alarm_windows": fp,
        "false_alarm_rate_pct": 100.0 * fp / n_clean if n_clean else float("nan"),
        "window_precision": tp_w / (tp_w + fp) if (tp_w + fp) else float("nan"),
        "auc_normalized_error": _auc(y, sc["normalized_error"]) if len(sc) else float("nan"),
        "auc_reconstruction_error": _auc(y, sc["reconstruction_error"]) if len(sc) else float("nan"),
        "auc_baseline_same_hour_last_week": _auc(y, sc["baseline_score"].fillna(0)) if len(sc) else float("nan"),
        "ignored_windows": int(rows["ignore"].sum()),
        "evaluated_windows": int(len(rows)),
    }
    by_type = {}
    if len(ev_df):
        for typ, g in ev_df.groupby("type"):
            by_type[typ] = {"events": int(len(g)), "detected": int(g["detected"].astype(bool).sum()),
                            "recall": float(g["detected"].astype(bool).mean()),
                            "median_delay_hours": float(g["delay_hours"].median())}
    summary["by_type"] = by_type
    return summary


# ----------------------------------------------------------------------------------------------------------------------
# Per-building evaluation
# ----------------------------------------------------------------------------------------------------------------------
def load_inputs(building: str, df_wide: pd.DataFrame, meta: pd.DataFrame):
    raw = preprocess_building(df_wide, building)
    if raw is None or len(raw) <= 500:
        return None
    feats = engineer_features(raw, building, meta)
    return raw, feats


def rebuild_features(raw_inj: pd.Series, template_raw: pd.DataFrame, building, meta, feats_index):
    df = template_raw.copy()
    df["energy"] = raw_inj.reindex(df.index).to_numpy()
    f = engineer_features(df, building, meta)
    if not f.index.equals(feats_index):
        raise RuntimeError(f"{building}: injected features have a different index than the clean ones")
    return f


def blocked_layout(n_rows: int, block_rows: int, k_requested: int):
    n_blocks = n_rows // block_rows
    if n_rows - n_blocks * block_rows >= block_rows // 2:
        n_blocks += 1
    k = min(k_requested, n_blocks)
    if n_blocks < 4 or k < 2:
        return None
    block_id = np.minimum(np.arange(n_rows) // block_rows, n_blocks - 1)
    return block_id, np.arange(n_blocks) % k, n_blocks, k


def evaluate_blocked(building, raw, feats, meta, backend_name, folds, epochs, seed, log):
    n = len(feats)
    block_rows = int(EVAL["block_days"]) * 24
    layout = blocked_layout(n, block_rows, folds)
    if layout is None:
        log(f"  [SKIP] {building}: too little data for blocked testing ({n} rows)")
        return None
    block_id, fold_of_block, n_blocks, k = layout
    pb, pa = int(EVAL["purge_before_hours"]), int(EVAL["purge_after_hours"])
    slot_rows = min(int(EVAL["slot_days"]) * 24, block_rows)
    energy_std = float(raw["energy"].std())
    activity = activity_profile(raw["energy"].reindex(feats.index).to_numpy(), float(EVAL["active_quantile"]))
    X_clean_rows = feats[FEATURE_COLS].to_numpy()
    rng = np.random.default_rng(seed)
    all_rows, all_events, next_id = [], [], 0
    for fold in range(k):
        t0 = time.time()
        held = (fold_of_block[block_id] == fold)
        purge = np.zeros(n, dtype=bool)
        for (s, e) in runs(held):
            purge[max(s - pb, 0):s] = True
            purge[e:min(e + pa, n)] = True
        purge &= ~held
        train_rows = ~(held | purge)

        regions = runs(held)
        events = place_events(regions, W, EVAL, rng, slot_rows, first_id=next_id, activity=activity)
        next_id += len(events)
        raw_inj, events = inject(raw["energy"], feats.index, events, energy_std, float(EVAL["min_effect_std"]))
        feats_inj = rebuild_features(raw_inj, raw, building, meta, feats.index)

        scaler = MinMaxScaler().fit(X_clean_rows[train_rows])
        win_clean = create_sequences(scaler.transform(X_clean_rows), W)
        eligible = window_counts(train_rows) == W
        # validation blocks are placed among THIS fold's training rows (a year-wide mask can fall inside held-out blocks)
        val_blocks = np.zeros(n, dtype=bool)
        train_pos = np.flatnonzero(train_rows)
        val_blocks[train_pos[validation_row_mask(len(train_pos), CONFIG["val_fraction"], CONFIG["val_block_days"])]] = True
        fit_w = eligible & (window_counts(val_blocks) == 0)
        val_w = window_counts(val_blocks) == W
        if fit_w.sum() == 0 or val_w.sum() == 0:
            log(f"  [SKIP] fold {fold}: no fit/validation windows")
            continue

        log(f"  fold {fold + 1}/{k}: training on {int(fit_w.sum())} windows (progress printed every 5 epochs)...")
        backend = make_backend(backend_name)
        backend.fit(win_clean[fit_w], win_clean[val_w], epochs=epochs, seed=seed + fold)
        res = score_series(feats_inj, scaler, backend)

        event_id, ignore, evaluable = label_windows(n, events, held, pa)
        base = baseline_window_score(raw_inj, feats.index, energy_std)
        rows = collect_eval_rows(res, event_id, ignore, evaluable, base, fold, feats.index)
        all_rows.append(rows)
        all_events.extend(events)
        det = rows[rows["event_id"] >= 0].groupby("event_id")["anomaly_flag"].max().sum()
        log(f"  fold {fold + 1}/{k}: held-out {int(held.sum()) // 24} d, trained on {int(train_rows.sum()) // 24} d "
            f"({int(fit_w.sum())} windows), {len(events)} events injected, {int(det)} detected "
            f"[{time.time() - t0:.0f}s]")
    if not all_rows:
        return None
    return pd.concat(all_rows).sort_index(), all_events, {"mode": "blocked", "folds": k, "blocks": int(n_blocks)}


def evaluate_holdout(building, raw, feats, meta, decision, backend_name, inject_on, epochs, seed, log):
    """Chronological holdout: production model + scaler (LSTM) or a PCA fitted on the training part (pca)."""
    n = len(feats)
    split_idx = int(decision["split_idx"])
    held = np.zeros(n, dtype=bool)
    held[split_idx:] = True
    train_rows = ~held
    X_rows = feats[FEATURE_COLS].to_numpy()

    if backend_name == "lstm":
        import joblib
        from keras.models import load_model

        mp = _PROJECT_ROOT / CONFIG["model_dir"] / f"{building}_best.h5"
        sp = _PROJECT_ROOT / "scalers" / f"{building}_scaler.pkl"
        if not mp.is_file() or not sp.is_file():
            log(f"  [SKIP] {building}: no production model / scaler ({mp.name}); train it first or use --backend pca")
            return None
        scaler = joblib.load(sp)
        backend = LSTMBackend(load_model(mp, compile=False))
        log(f"  using production model {mp.name}")
    else:
        scaler = MinMaxScaler().fit(X_rows[train_rows])
        win = create_sequences(scaler.transform(X_rows), W)
        fit_w = window_counts(train_rows) == W
        backend = make_backend("pca").fit(win[fit_w], None, seed=seed)

    energy_std = float(raw["energy"].std())
    pa = int(EVAL["purge_after_hours"])
    events, raw_inj, feats_use = [], raw["energy"], feats
    if inject_on:
        slot_rows = int(EVAL["slot_days"]) * 24
        rng = np.random.default_rng(seed)
        activity = activity_profile(raw["energy"].reindex(feats.index).to_numpy(), float(EVAL["active_quantile"]))
        events = place_events(runs(held), W, EVAL, rng, slot_rows, activity=activity)
        raw_inj, events = inject(raw["energy"], feats.index, events, energy_std, float(EVAL["min_effect_std"]))
        feats_use = rebuild_features(raw_inj, raw, building, meta, feats.index)
    res = score_series(feats_use, scaler, backend)
    event_id, ignore, evaluable = label_windows(n, events, held, pa)
    base = baseline_window_score(raw_inj, feats.index, energy_std)
    rows = collect_eval_rows(res, event_id, ignore, evaluable, base, 0, feats.index)
    # natural (no-injection) behaviour on the test period is always worth reporting for holdout buildings
    clean_res = res if not inject_on else score_series(feats, scaler, backend)
    tr = clean_res.iloc[: max(split_idx - W + 1, 0)]
    te = clean_res.iloc[max(split_idx - W + 1, 0):]
    natural = {
        "train_mean_error": float(tr["reconstruction_error"].mean()),
        "test_mean_error": float(te["reconstruction_error"].mean()),
        "test_to_train_error_ratio": float(te["reconstruction_error"].mean() / tr["reconstruction_error"].mean()),
        "train_anomaly_rate": float(tr["anomaly_flag"].mean()),
        "test_anomaly_rate": float(te["anomaly_flag"].mean()),
    }
    log(f"  holdout: train {split_idx // 24} d / test {(n - split_idx) // 24} d; "
        f"test/train error {natural['test_to_train_error_ratio']:.2f}x; test anomaly rate {natural['test_anomaly_rate']:.2%}"
        + (f"; {len(events)} events injected" if inject_on else "; no injection (use --inject to add)"))
    return rows, events, {"mode": "holdout", "injected": bool(inject_on), "natural": natural}


# ----------------------------------------------------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------------------------------------------------
def plot_building(building, rows, ev_df, summary, feats_index, info, path):
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(15, 8), gridspec_kw={"height_ratios": [3, 1.4]})
    full = pd.DataFrame(index=feats_index)
    err = pd.Series(np.nan, index=feats_index)
    thr = pd.Series(np.nan, index=feats_index)
    err.iloc[rows["end_row"].to_numpy()] = rows["reconstruction_error"].to_numpy()
    thr.iloc[rows["end_row"].to_numpy()] = rows["threshold"].to_numpy()
    ax.plot(err.index, err, color="#6B7280", lw=0.8, label="Reconstruction error (held-out windows)")
    ax.plot(thr.index, thr, color="#DC2626", lw=1.3, label="Threshold")
    if len(ev_df):
        for _, e in ev_df.iterrows():
            col = "#9CA3AF" if e["inert"] else ("#16A34A" if e["detected"] else "#F59E0B")
            ax.axvspan(e["start"], e["end"] + pd.Timedelta(hours=W - 1), color=col,
                       alpha=0.35, lw=0)
    clean_fp = rows[(rows["event_id"] < 0) & (~rows["ignore"]) & (rows["anomaly_flag"] == 1)]
    ax.scatter(feats_index[clean_fp["end_row"].to_numpy()], clean_fp["reconstruction_error"], marker="x", color="black", s=22,
               label="False alarm", zorder=5)
    ev_hit = matplotlib.patches.Patch(color="#16A34A", alpha=0.35, label="Injected event - detected")
    ev_miss = matplotlib.patches.Patch(color="#F59E0B", alpha=0.35, label="Injected event - missed")
    h, l = ax.get_legend_handles_labels()
    ev_inert = matplotlib.patches.Patch(color="#9CA3AF", alpha=0.35, label="Injected event - inert (changed ~nothing, not scored)")
    extra = [ev_hit, ev_miss] + ([ev_inert] if len(ev_df) and ev_df["inert"].any() else [])
    ax.legend(handles=h + (extra if len(ev_df) else []), fontsize=8, loc="upper left")
    title_mode = info["mode"] + (f" ({info['folds']} folds)" if info["mode"] == "blocked" else "")
    ax.set_title(f"{building} - {title_mode} evaluation  |  event recall {summary['event_recall']:.0%}, "
                 f"false alarms {summary['false_alarm_rate_pct']:.1f}% of clean windows, "
                 f"AUC {summary['auc_normalized_error']:.2f} (baseline {summary['auc_baseline_same_hour_last_week']:.2f})"
                 if summary["events"] else f"{building} - {title_mode} (no injected events)", fontweight="bold", fontsize=10)
    ax.set_ylabel("MSE"); ax.grid(True, alpha=0.25)
    if summary["events"]:
        types = list(summary["by_type"].keys())
        rec = [summary["by_type"][t]["recall"] for t in types]
        bars = ax2.bar(types, rec, color="#3B82F6")
        for b, t in zip(bars, types):
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                     f"{summary['by_type'][t]['detected']}/{summary['by_type'][t]['events']}", ha="center", fontsize=9)
        ax2.set_ylim(0, 1.2); ax2.set_ylabel("Recall by event type"); ax2.grid(True, axis="y", alpha=0.25)
    else:
        ax2.axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def write_report(all_summaries, path):
    backends = sorted({str(i.get("backend", "?")) for _, i in all_summaries.values()})
    seeds = sorted({str(i["seed"]) for _, i in all_summaries.values() if "seed" in i}) or ["unknown"]
    lines = ["# Evaluation report (injected anomalies)", "",
             f"Backend: `{', '.join(backends)}`. Event types: {', '.join(EVAL['event_types'])}. Seed {', '.join(seeds)}.", "",
             "| Building | Mode | Events | Recall | Median delay (h) | False alarms | Precision | AUC (error) | AUC baseline |",
             "|---|---|---|---|---|---|---|---|---|"]
    for b, (s, info) in all_summaries.items():
        mode = info["mode"] + (f" ({info['folds']} folds" + (f", max {info['epochs_cap']} epochs" if info.get("epochs_cap") else "")
                               + ")" if "folds" in info else "")
        if s["events"]:
            lines.append(f"| {b} | {mode} | {s['events_detected']}/{s['events']} | {s['event_recall']:.0%} | "
                         f"{s['median_delay_hours']:.0f} | {s['false_alarm_rate_pct']:.1f}% | {s['window_precision']:.0%} | "
                         f"{s['auc_normalized_error']:.2f} | {s['auc_baseline_same_hour_last_week']:.2f} |")
        else:
            nat = info.get("natural", {})
            lines.append(f"| {b} | {mode}, no injection | - | - | - | test anomaly rate "
                         f"{nat.get('test_anomaly_rate', float('nan')):.1%} | - | - | - |")
    lines += ["", "Recall by event type:", "", "| Building | " + " | ".join(EVAL["event_types"]) + " |",
              "|---|" + "---|" * len(EVAL["event_types"])]
    for b, (s, _) in all_summaries.items():
        cells = []
        for t in EVAL["event_types"]:
            bt = s["by_type"].get(t)
            cells.append(f"{bt['detected']}/{bt['events']}" if bt else "-")
        lines.append(f"| {b} | " + " | ".join(cells) + " |")
    lines += ["", "Injected / inert events per building: " + ", ".join(
        f"{b} {s['events_injected']} / {s['events_inert']}" for b, (s, _) in all_summaries.items()) +
        ". Inert = changed (almost) nothing, so they are listed in the events CSV but not scored.", "",
        "False alarm = a flagged window with no injected event nearby. Event recall counts an event as found if at "
              "least one of its windows is flagged as an anomaly (drift flags are counted separately in the CSV).",
              "Held-out windows up to 7 days after an event are ignored in every metric (the lag features echo the event)."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_building_summary(b: str, summary: dict, info: dict):
    (OUT_DIR / f"{b}_summary.json").write_text(json.dumps({"summary": summary, "info": info}, indent=1), encoding="utf-8")


def rebuild_from_files(b: str, backend: str):
    """Rebuild a building's summary from its saved <b>_windows.csv and <b>_events.csv (e.g. after an interrupted run)."""
    wp, ep = OUT_DIR / f"{b}_windows.csv", OUT_DIR / f"{b}_events.csv"
    if not (wp.is_file() and ep.is_file()):
        return None
    rows = pd.read_csv(wp)
    ev = pd.read_csv(ep)
    if "inert" not in ev.columns:
        ev["inert"] = False
    ev["inert"] = ev["inert"].astype(bool)
    n_inert = int(ev["inert"].sum())
    scored = ev[~ev["inert"]].copy()
    summary = summarize_tables(rows, scored, n_inert)
    folds = int(rows["fold"].nunique()) if "fold" in rows.columns else 1
    info = {"mode": "blocked" if folds > 1 else "holdout", "backend": backend, "rebuilt_from_files": True}
    if folds > 1:
        info["folds"] = folds
    return summary, info


def write_combined(scope):
    """evaluation_summary.csv + evaluation_report.md for every building in ``scope`` that has a saved summary."""
    all_summaries = {}
    for b in scope:
        p = OUT_DIR / f"{b}_summary.json"
        if p.is_file():
            d = json.loads(p.read_text(encoding="utf-8"))
            all_summaries[b] = (d["summary"], d["info"])
    if not all_summaries:
        return None
    flat = []
    for b, (s, info) in all_summaries.items():
        row = {"building": b, "mode": info["mode"], **{k: v for k, v in s.items() if k != "by_type"}}
        for t, v in s["by_type"].items():
            row[f"recall_{t}"] = v["recall"]
        row.update({f"natural_{k}": v for k, v in info.get("natural", {}).items()})
        flat.append(row)
    pd.DataFrame(flat).to_csv(OUT_DIR / "evaluation_summary.csv", index=False)
    write_report(all_summaries, OUT_DIR / "evaluation_report.md")
    return list(all_summaries)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buildings", nargs="*", default=None, help="default: BUILDINGS in src/config.py")
    ap.add_argument("--backend", choices=["lstm", "pca"], default="lstm")
    ap.add_argument("--inject", action="store_true", help="also inject anomalies into holdout buildings' test period")
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="max epochs per temporary fold model (lstm)")
    ap.add_argument("--quick", action="store_true", help="fewer folds and epochs")
    ap.add_argument("--seed", type=int, default=int(EVAL["seed"]))
    ap.add_argument("--summarize-only", action="store_true",
                    help="no training: rebuild evaluation_report.md / evaluation_summary.csv from the files already in "
                         "output_plots/evaluation (use after an interrupted run). Buildings without a saved summary are "
                         "rebuilt from their <building>_windows.csv and _events.csv")
    args = ap.parse_args(argv)

    folds = args.folds or (int(EVAL["quick_folds"]) if args.quick else int(EVAL["folds"]))
    epochs = args.epochs or (int(EVAL["quick_epochs"]) if args.quick else None)
    inject_on_holdout = bool(args.inject or EVAL.get("inject_on_holdout", False))
    log = lambda m: print(m, flush=True)

    buildings = args.buildings or BUILDINGS
    # the combined report covers every configured building (and any named on the command line) that has results
    scope = list(dict.fromkeys(list(BUILDINGS) + list(buildings)))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.summarize_only:
        for b in scope:
            if not (OUT_DIR / f"{b}_summary.json").is_file():
                rebuilt = rebuild_from_files(b, args.backend)
                if rebuilt is None:
                    log(f"  [no results] {b}")
                    continue
                save_building_summary(b, *rebuilt)
                log(f"  rebuilt summary for {b} from its CSV files")
        done = write_combined(scope)
        log(f"Report written for: {', '.join(done) if done else 'nothing (no results found)'}")
        return
    df_wide = pd.read_csv(_PROJECT_ROOT / CONFIG["data_url"])
    df_wide["timestamp"] = pd.to_datetime(df_wide["timestamp"], utc=True)
    df_wide = df_wide.set_index("timestamp").sort_index()
    meta = pd.read_csv(_PROJECT_ROOT / CONFIG["meta_url"])

    for b in buildings:
        log(f"\n{'=' * 60}\n{b}")
        with contextlib.redirect_stdout(io.StringIO()):
            loaded = load_inputs(b, df_wide, meta)
        if loaded is None:
            log("  [SKIP] missing from the wide dataset or too few rows")
            continue
        raw, feats = loaded
        decision = decide_split(feats.index, b, CONFIG, BUILDING_SPLIT)
        log(f"  split policy: {decision['strategy'].upper()} - {decision['reason']}")

        if decision["strategy"] == "full":
            log(f"  -> BLOCKED testing + injected anomalies ({folds} folds, backend={args.backend})")
            out = evaluate_blocked(b, raw, feats, meta, args.backend, folds, epochs, args.seed, log)
        else:
            log(f"  -> HOLDOUT evaluation (no blocked testing); injection {'ON' if inject_on_holdout else 'OFF'}")
            out = evaluate_holdout(b, raw, feats, meta, decision, args.backend, inject_on_holdout, epochs, args.seed, log)
        if out is None:
            continue
        rows, events, info = out
        info = {**info, "backend": args.backend, "seed": args.seed, "quick": bool(args.quick), "epochs_cap": epochs}
        summary, ev_df = compute_metrics(rows, events, feats.index)

        ev_df.to_csv(OUT_DIR / f"{b}_events.csv", index=False)
        w = rows.copy()
        w.insert(0, "timestamp", feats.index[w["end_row"].to_numpy()])
        w.to_csv(OUT_DIR / f"{b}_windows.csv", index=False)
        plot_building(b, rows, ev_df, summary, feats.index, info, OUT_DIR / f"{b}_evaluation.png")
        save_building_summary(b, summary, info)  # saved per building, so an interrupted run loses nothing
        write_combined(scope)

        if summary["events"]:
            log(f"  RESULT: {summary['events_detected']}/{summary['events']} scored events detected "
                f"({summary['event_recall']:.0%}), median delay {summary['median_delay_hours']:.0f} h, "
                f"false alarms {summary['false_alarm_rate_pct']:.1f}% of {summary['clean_windows']} clean windows, "
                f"AUC {summary['auc_normalized_error']:.2f} vs baseline {summary['auc_baseline_same_hour_last_week']:.2f}")
            for t, v in summary["by_type"].items():
                log(f"      {t:12s} {v['detected']}/{v['events']} detected")
            if summary["events_inert"]:
                log(f"      ({summary['events_inert']} of {summary['events_injected']} injected events were inert and are not scored)")

    done = write_combined(scope)
    if done:
        log(f"\nSaved to {OUT_DIR} (evaluation_report.md covers: {', '.join(done)})")


if __name__ == "__main__":
    main()
