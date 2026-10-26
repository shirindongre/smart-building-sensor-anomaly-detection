"""
Per-building inference: load one processed CSV (energy + engineered features),
apply the saved scaler, build sequences, and run the trained LSTM autoencoder.
No wide multi-building download and no preprocess_building / engineer_features here.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

try:
    import joblib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    joblib = None
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import load_model

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_PLOTS_BASE_DIR = "output_plots/sensor_lstm"


def _sensor_lstm_building_plots_dir(building: str) -> str:
    """Per-building directory for sensor LSTM figures under the project root."""
    return os.path.join(str(_PROJECT_ROOT), _PLOTS_BASE_DIR, building)


for p in (_PROJECT_ROOT, _SCRIPT_DIR):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from src.config import BUILDINGS, CONFIG, FEATURE_COLS

from sequence_creations import create_sequences
from anomaly_scoring import add_adaptive_thresholding, build_results_df  # noqa: F401  (re-exported)

os.makedirs(_PROJECT_ROOT / CONFIG["output_dir"], exist_ok=True)


def load_scaler(path: Path):
    """
    Load a scaler saved as a .pkl.

    Uses joblib when available; falls back to pickle to avoid env mismatch issues.
    """
    if joblib is not None:
        return joblib.load(path)

    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


def _resolve_building_features_path(building: str) -> Path:
    """Prefer ``{building}.csv``, then ``{building}_features.csv`` under data/processed/."""
    processed = _PROJECT_ROOT / "data" / "processed"
    for name in (f"{building}.csv", f"{building}_features.csv"):
        candidate = processed / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No processed file for {building!r} in {processed}. "
        f"Expected {building}.csv or {building}_features.csv (run training export or pipeline)."
    )


def load_building_features(building: str) -> pd.DataFrame:
    """Load single-building table: timestamp + energy + feature columns."""
    path = _resolve_building_features_path(building)
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{path}: missing 'timestamp' column")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing FEATURE_COLS {missing}")
    return df


def load_split_info(building: str, n_rows: int) -> dict:
    """
    Train / test boundary for one building, exactly as used at training time.

    Reads ``models/<building>_split.json`` (written by Model_Training.py). Models trained before that file existed
    fall back to the legacy 80/20 holdout. If the CSV has MORE rows than at training time, the extra (newer) rows
    count as unseen "test" data - this is how future data for a full-history building gets evaluated.
    """
    meta_path = _PROJECT_ROOT / CONFIG["model_dir"] / f"{building}_split.json"
    if not meta_path.is_file():
        split_idx = int(n_rows * (1.0 - float(CONFIG.get("test_split", 0.2))))
        return {
            "strategy": "holdout",
            "split_idx": split_idx,
            "message": f"No split metadata at {meta_path.name}: assuming the legacy 80/20 holdout. "
                       "Re-run scripts/Model_Training.py to record the split.",
        }
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    split_idx = int(meta["split_idx"])
    if n_rows < split_idx:
        meta["message"] += f" WARNING: CSV now has {n_rows} rows, fewer than at training ({split_idx})."
        split_idx = n_rows
    elif n_rows > int(meta.get("n_rows", n_rows)):
        meta["message"] += (f" {n_rows - int(meta['n_rows'])} newer rows than the training data are treated "
                            "as unseen test data.")
    meta["split_idx"] = split_idx
    return meta


def compute_reconstruction_errors(model, sequences: np.ndarray) -> np.ndarray:
    reconstructed = model.predict(sequences, verbose=0)
    errors = np.mean(np.power(sequences - reconstructed, 2), axis=(1, 2))
    return errors


def _iter_true_segments(mask: pd.Series):
    idx = mask.index
    m = mask.to_numpy(dtype=bool, copy=False)
    n = len(m)
    i = 0
    while i < n:
        if not m[i]:
            i += 1
            continue
        start = idx[i]
        j = i + 1
        while j < n and m[j]:
            j += 1
        end = idx[j - 1]
        yield start, end
        i = j


def _shade_mask_regions(ax, mask: pd.Series, *, color: str, alpha: float, label: str):
    first = True
    for start, end in _iter_true_segments(mask):
        ax.axvspan(start, end, color=color, alpha=alpha, lw=0, label=(label if first else None))
        first = False


def save_plots(
    results_df,
    train_errors,
    building,
    plots_output_dir,
):
    """Sensor LSTM diagnostic figures using ``threshold``, ``drift_flag``, ``normalized_error``."""
    err = results_df["reconstruction_error"]
    thr = results_df["threshold"]
    anomaly_mask = results_df["anomaly_flag"] == 1
    normal_mask = results_df["anomaly_flag"] == 0
    drift_mask = results_df["drift_flag"] == 1
    drift_z = float(CONFIG.get("drift_z_threshold", 5.0))
    plots_dir = os.path.normpath(plots_output_dir)
    os.makedirs(plots_dir, exist_ok=True)

    # --- 01: error vs smoothed/frozen threshold, drift bands, anomaly markers ---
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(results_df.index, err, color="#6B7280", lw=0.8, label="Reconstruction error", alpha=0.85)
    _shade_mask_regions(ax, drift_mask, color="#93C5FD", alpha=0.25, label="Drift region")
    ax.plot(
        results_df.index,
        thr,
        color="#DC2626",
        lw=1.6,
        linestyle="-",
        label="Threshold (median + k*MAD, EMA, drift-frozen)",
    )
    over_thr_clear = (err > thr) & (results_df["drift_flag"] == 0)
    ax.fill_between(
        results_df.index,
        err,
        thr,
        where=over_thr_clear,
        color="#FCA5A5",
        alpha=0.45,
        label="Above threshold (non-drift)",
    )
    ax.scatter(
        results_df.index[anomaly_mask],
        err[anomaly_mask],
        color="#B91C1C",
        s=22,
        zorder=5,
        label="Anomaly",
    )
    ax.set_title(f"Reconstruction Error — {building}", fontweight="bold")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("MSE")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "01_error_timeline.png"), dpi=150)
    plt.close()

    # --- 02: energy + error with same threshold context ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(
        results_df.index[normal_mask],
        results_df["energy"][normal_mask],
        color="#6B7280",
        lw=0.8,
        label="Normal",
        alpha=0.9,
    )
    axes[0].scatter(
        results_df.index[anomaly_mask],
        results_df["energy"][anomaly_mask],
        color="#DC2626",
        s=18,
        zorder=5,
        label="Anomaly",
    )
    _shade_mask_regions(axes[0], drift_mask, color="#93C5FD", alpha=0.20, label="Drift region")
    axes[0].set_title(
        f"Energy Consumption — Anomalies Highlighted — {building}", fontweight="bold"
    )
    axes[0].set_ylabel("Energy (raw units)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(results_df.index, err, color="#6B7280", lw=0.8, alpha=0.85, label="Reconstruction error")
    _shade_mask_regions(axes[1], drift_mask, color="#93C5FD", alpha=0.20, label="Drift region")
    axes[1].plot(
        results_df.index,
        thr,
        color="#DC2626",
        lw=1.4,
        label="Threshold",
    )
    axes[1].fill_between(
        results_df.index,
        err,
        thr,
        where=over_thr_clear,
        color="#FCA5A5",
        alpha=0.5,
    )
    axes[1].scatter(
        results_df.index[anomaly_mask],
        err[anomaly_mask],
        color="#B91C1C",
        s=18,
        zorder=5,
        label="Anomaly",
    )
    axes[1].set_title("Reconstruction Error (MSE)")
    axes[1].set_xlabel("Timestamp")
    axes[1].set_ylabel("MSE")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "02_anomaly_on_energy.png"), dpi=150)
    plt.close()

    # --- 03: normalized robust z (stabilized MAD) vs drift gate level ---
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(
        results_df.index,
        results_df["normalized_error"],
        color="#6B7280",
        lw=0.8,
        alpha=0.85,
        label="Normalized error (robust z)",
    )
    _shade_mask_regions(ax, drift_mask, color="#93C5FD", alpha=0.25, label="Drift region")
    ax.axhline(drift_z, color="#DC2626", lw=1.8, linestyle="--", label=f"Drift gate (z = {drift_z:g})")
    ax.fill_between(
        results_df.index,
        results_df["normalized_error"],
        drift_z,
        where=results_df["normalized_error"] > drift_z,
        color="#FDE68A",
        alpha=0.35,
        label="Above drift gate",
    )
    ax.set_title(f"Normalized Error — {building}", fontweight="bold")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Robust z-score: (err - median) / (MAD + ε)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "03_normalized_error.png"), dpi=150)
    plt.close()

    # --- 04: error distribution vs median applied threshold ---
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(train_errors, bins=60, color="#3B82F6", alpha=0.7, label="Train errors", density=True)
    ax.hist(err.values, bins=60, color="#F97316", alpha=0.5, label="All errors", density=True)
    ax.axvline(float(thr.median()), color="#DC2626", lw=2, linestyle="--", label="Median threshold")
    ax.set_title(f"Error Distribution — {building}", fontweight="bold")
    ax.set_xlabel("MSE")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "04_error_distribution.png"), dpi=150)
    plt.close()

    print(f"  Plots saved -> {plots_dir} (01-04_*.png)")


def print_diagnostics(building: str, df: pd.DataFrame):
    err = df["reconstruction_error"]
    ar = float(df["anomaly_flag"].mean())
    dr = float(df["drift_flag"].mean()) if "drift_flag" in df.columns else float("nan")

    print(f"\n=== Diagnostics: {building} ===")
    print(f"Anomaly rate: {ar:.4f}")
    print(f"Drift rate: {dr:.4f}")
    print(f"Mean error: {float(err.mean()):.6g}")
    print(f"Std error: {float(err.std()):.6g}")

    if ar < 0.01:
        print("Note: anomaly rate < 1% (threshold may be strict).")
    if ar > 0.10:
        print("Note: anomaly rate > 10% (threshold may be loose).")


def export_report(results_df, building, output_dir, *, threshold_method: str, robust_z_k: float):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{building}_anomaly_report.csv"
    results_df.to_csv(csv_path)

    total = len(results_df)
    n_anomaly = results_df["anomaly_flag"].sum()
    test_df = results_df[results_df["split"] == "test"]
    n_test_anom = test_df["anomaly_flag"].sum()

    print(
        "  Threshold policy : median + k·MAD (k="
        f"{float(CONFIG.get('threshold_mad_k', 6)):g}), EMA span="
        f"{int(CONFIG.get('threshold_ema_span', 24))}, drift-frozen; drift_gate_z="
        f"{float(CONFIG.get('drift_z_threshold', 5)):g}"
    )
    print(f"  Total windows     : {total}")
    print(f"  Anomalies (all)   : {n_anomaly}  ({100 * n_anomaly / total:.2f} %)")
    if len(test_df):
        print(f"  Anomalies (test)  : {n_test_anom}  ({100 * n_test_anom / len(test_df):.2f} %)")
    else:
        print("  Anomalies (test)  : n/a - no held-out period for this building (full-history training); "
              "all flags above are in-sample. Future data will be evaluated as it arrives.")
    print(f"  Report saved      : {csv_path}")


def export_sensor_outputs(results_df, building):
    out_dir = _PROJECT_ROOT / "data" / "processed" / "sensor_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = results_df.reset_index()
    if "timestamp" not in df.columns:
        df = df.rename(columns={df.columns[0]: "timestamp"})
    df["anomaly_score"] = df["reconstruction_error"]
    df["building_id"] = building
    df = df[
        [
            "timestamp",
            "building_id",
            "reconstruction_error",
            "anomaly_score",
            "anomaly_flag",
            "normalized_error",
            "drift_flag",
        ]
    ]
    save_path = out_dir / f"{building}_anomalies.csv"
    df.to_csv(save_path, index=False)
    print(f"  Sensor output saved -> {save_path}")


def main():
    output_dir = _PROJECT_ROOT / CONFIG["output_dir"]
    model_dir = _PROJECT_ROOT / CONFIG["model_dir"]
    scalers_dir = _PROJECT_ROOT / "scalers"

    threshold_method = CONFIG.get("threshold_method", "rolling_percentile")
    rolling_window_days = int(CONFIG.get("rolling_window_days", 7))
    robust_z_k = float(CONFIG.get("robust_z_k", 3.5))

    for building in BUILDINGS:
        print(f"\n{'=' * 50}")
        print(f"Testing: {building}")

        model_path = model_dir / f"{building}_best.h5"
        if not model_path.is_file():
            print(f"  [SKIP] No model found at {model_path}")
            continue

        try:
            df_features = load_building_features(building)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue
        except ValueError as e:
            print(f"  [SKIP] {e}")
            continue

        print(f"  Loaded features: {df_features.shape} (per-building CSV)")

        scaler_path = scalers_dir / f"{building}_scaler.pkl"
        if not scaler_path.is_file():
            print(f"  [SKIP] No scaler found at {scaler_path}")
            continue
        scaler = load_scaler(scaler_path)

        split_info = load_split_info(building, len(df_features))
        split_idx = int(split_info["split_idx"])
        window = int(CONFIG["window_size"])
        print(f"  {split_info['message']}")

        n_feat = len(FEATURE_COLS)
        scaler_n = getattr(scaler, "n_features_in_", n_feat)
        if scaler_n != n_feat:
            print(
                f"  [SKIP] Scaler was fitted on {scaler_n} features but FEATURE_COLS has {n_feat}. "
                "Re-run scripts/Model_Training.py to rebuild scalers and models."
            )
            continue

        # Scale the whole frame with the TRAIN-fitted scaler, then build windows over the full frame so
        # test windows overlap the train/test boundary instead of being cut off (timestamps stay aligned).
        X_all_sc = scaler.transform(df_features[FEATURE_COLS].to_numpy())
        X_all_seq = create_sequences(X_all_sc, window)

        n_train_windows = max(split_idx - window + 1, 0)  # windows ending strictly before split_idx
        print(f"  Windows total  : {X_all_seq.shape}  (train-only: {n_train_windows})")

        if n_train_windows == 0:
            print("  [SKIP] Not enough rows for at least one training window.")
            continue
        if len(X_all_seq) <= n_train_windows and split_info.get("strategy") != "full":
            print("  [SKIP] Not enough rows for at least one test window.")
            continue

        model = load_model(model_path, compile=False)
        model.compile(optimizer="adam", loss="mse")
        print(f"  Model loaded : {model_path}")

        model_n_feat = model.input_shape[-1]
        if model_n_feat != n_feat:
            print(
                f"  [SKIP] Model expects {model_n_feat} features but FEATURE_COLS has {n_feat}. "
                "Re-run scripts/Model_Training.py to retrain."
            )
            continue

        all_errors = compute_reconstruction_errors(model, X_all_seq)
        train_errors = all_errors[:n_train_windows]

        print(f"\n  Results for {building}:")

        results_df = build_results_df(
            df_features=df_features,
            all_errors=all_errors,
            window_size=window,
            split_idx=split_idx,
        )

        results_df = add_adaptive_thresholding(
            results_df,
            threshold_method=threshold_method,
            rolling_window_days=rolling_window_days,
            robust_z_k=robust_z_k,
        )

        print_diagnostics(building, results_df.reset_index())
        save_plots(
            results_df,
            train_errors,
            building,
            _sensor_lstm_building_plots_dir(building),
        )
        export_report(
            results_df,
            building,
            output_dir,
            threshold_method=threshold_method,
            robust_z_k=robust_z_k,
        )
        export_sensor_outputs(results_df, building)

    print(f"\n{'=' * 50}")
    plots_root = os.path.join(str(_PROJECT_ROOT), _PLOTS_BASE_DIR)
    print(
        f"Testing complete. Sensor LSTM plots under {plots_root}/<building>/; "
        f"reports under {output_dir}; sensor_outputs under data/processed/sensor_outputs/"
    )


if __name__ == "__main__":
    main()
