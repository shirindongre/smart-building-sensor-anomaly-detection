"""
Scoring helpers shared by Model_Testing.py and evaluation.py (no TensorFlow / Keras needed here).

* ``add_adaptive_thresholding`` - turns a series of window reconstruction errors into a rolling robust
  threshold, drift flag and anomaly flag.
* ``build_results_df`` - stamps one error per window with the timestamp of the window's LAST row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONFIG


def _rolling_mad(values: np.ndarray) -> float:
    """MAD within one window: median(|x - median(x)|). Used with rolling(..., apply, raw=True)."""
    med = np.median(values)
    return float(np.median(np.abs(values - med)))


def add_adaptive_thresholding(
    results_df: pd.DataFrame,
    *,
    threshold_method: str = "rolling_percentile",
    rolling_window_days: int = 7,
    robust_z_k: float = 3.5,
    anomaly_rate_drift_threshold: float = 0.30,
) -> pd.DataFrame:
    """
    Post-process reconstruction errors: robust normalization, drift vs anomaly separation,
    smoothed MAD-based threshold with freeze-during-drift.

    ``threshold_method``, ``rolling_window_days``, ``robust_z_k``, and
    ``anomaly_rate_drift_threshold`` are kept for call-site compatibility; tunables are read
    from ``CONFIG`` (see ``error_rolling_window_hours``, ``threshold_mad_k``, etc.).

    No future leakage: rolling statistics on errors use only past data (closed='left').
    """
    if results_df.index.name is None:
        results_df = results_df.copy()
        results_df.index.name = "timestamp"

    if not isinstance(results_df.index, pd.DatetimeIndex):
        raise ValueError("results_df must be indexed by timestamp (DatetimeIndex)")

    if "reconstruction_error" not in results_df.columns:
        raise ValueError("results_df missing required column 'reconstruction_error'")

    df = results_df.sort_index().copy()
    err = df["reconstruction_error"]

    # --- 1) Rolling robust location / scale on reconstruction_error (calendar / time window) ---
    hours = int(CONFIG.get("error_rolling_window_hours", 168))
    roll_spec = f"{hours}h"
    eps = float(CONFIG.get("mad_epsilon", 1e-6))
    past = err.rolling(roll_spec, min_periods=1, closed="left")

    rolling_median = past.median()
    rolling_mad = past.apply(_rolling_mad, raw=True)
    # Legacy diagnostic column (unchanged name) for CSV consumers that expect rolling_p99.
    rolling_p99 = past.quantile(0.99)

    # Early rows: no past window yet — fall back so downstream math stays finite.
    rolling_median = rolling_median.fillna(err)
    rolling_mad = rolling_mad.fillna(0.0)
    rolling_p99 = rolling_p99.fillna(err)

    # A MAD of ~0 (very flat early history) made z-scores explode (+/-1000). Floor it relative to the median.
    mad_floor_frac = float(CONFIG.get("mad_floor_frac", 0.10))
    rolling_mad = np.maximum(rolling_mad, mad_floor_frac * rolling_median)

    df["rolling_median"] = rolling_median.astype(float)
    df["rolling_mad"] = rolling_mad.astype(float)
    df["rolling_p99"] = rolling_p99.astype(float)

    # --- 2) Stabilized robust z-score (same rolling median / MAD + epsilon) ---
    z_clip = float(CONFIG.get("normalized_error_clip", 100.0))
    df["normalized_error"] = (
        ((err - df["rolling_median"]) / (df["rolling_mad"] + eps)).clip(-z_clip, z_clip).astype(float)
    )

    # Warm-up: until a full rolling window of history exists the median / MAD are unreliable.
    # The first ``warmup_hours`` rows are never flagged. (Index is hourly, so rows == hours.)
    warmup = int(CONFIG.get("warmup_hours", hours))
    in_warmup = pd.Series(np.arange(len(df)) < warmup, index=df.index)

    # --- 3) Drift from normalized score only (independent of threshold; no leakage into drift rule) ---
    drift_z = float(CONFIG.get("drift_z_threshold", 5.0))
    sustain = int(CONFIG.get("drift_sustain_points", 24))
    high_norm = (df["normalized_error"] > drift_z).astype(np.int8)
    # Sustained elevation: last ``sustain`` points must all exceed drift_z.
    sustained_ct = high_norm.rolling(sustain, min_periods=sustain).sum()
    df["drift_flag"] = ((sustained_ct >= sustain) & ~in_warmup).astype(int)

    # --- 4) Threshold: median + k * MAD, then EMA smooth ---
    k = float(CONFIG.get("threshold_mad_k", 6.0))
    ema_span = int(CONFIG.get("threshold_ema_span", 24))
    threshold_raw = df["rolling_median"] + k * df["rolling_mad"]
    threshold_smooth = threshold_raw.ewm(span=ema_span, adjust=False).mean()

    # --- 5) Freeze threshold while drift_flag is active (carry last pre-drift smoothed value) ---
    thr_carry = threshold_smooth.mask(df["drift_flag"].astype(bool)).ffill().bfill()
    df["threshold"] = threshold_smooth.where(df["drift_flag"] == 0, thr_carry).astype(float)

    # --- 6) Anomalies only when error exceeds threshold and we are not in a drift regime ---
    df["anomaly_flag"] = ((err > df["threshold"]) & (df["drift_flag"] == 0) & ~in_warmup).astype(int)

    return df


def build_results_df(df_features, all_errors, window_size, split_idx):
    """
    One row per window, stamped with the timestamp of the LAST row in that window.

    ``all_errors`` must come from ``create_sequences`` applied to the whole (scaled) frame, so window
    ``i`` ends at row ``i + window_size - 1`` and len(all_errors) == len(df_features) - window_size + 1.
    A window is "train" only if it ends before ``split_idx`` (it then contains no test rows).
    """
    end_pos = np.arange(len(all_errors)) + window_size - 1
    if len(all_errors) != len(df_features) - window_size + 1:
        raise ValueError(
            f"Expected {len(df_features) - window_size + 1} windows, got {len(all_errors)}; "
            "sequences must be built over the full frame."
        )

    results_df = pd.DataFrame(
        {
            "energy": df_features["energy"].to_numpy()[end_pos],
            "reconstruction_error": all_errors,
            "split": np.where(end_pos < split_idx, "train", "test"),
        },
        index=df_features.index[end_pos],
    )
    results_df.index.name = "timestamp"
    return results_df
