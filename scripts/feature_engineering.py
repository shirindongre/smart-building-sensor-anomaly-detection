import numpy as np
import pandas as pd


def engineer_features(df, building, meta):
    """
    Build model features for one building.

    ``df`` must have a tz-aware UTC DatetimeIndex and an ``energy`` column.

    The index is kept in UTC (continuous, no duplicated / missing hours). Local-time features
    (hour, day of week, ...) are derived from a *converted copy* of the index, so daylight-saving
    changes no longer create a duplicated 01:00 or a missing hour in the time series.
    """
    df = df.sort_index().copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    # Put the series on a strict hourly grid so row-based lags / rolling windows are time-correct.
    # Hours that are genuinely missing become NaN and are dropped at the end.
    df = df.asfreq("1h")

    # get building timezone
    tz = meta.loc[meta["uid"] == building, "timezone"].values[0]

    # local wall-clock time, used ONLY for calendar features
    local_idx = df.index.tz_convert(tz)

    # -------------------
    # time features
    # -------------------
    hour = local_idx.hour.to_numpy()
    day_of_week = local_idx.dayofweek.to_numpy()

    # cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    df["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    df["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)

    # weekend indicator
    df["is_weekend"] = np.isin(day_of_week, [5, 6]).astype(int)

    # -------------------
    # lag features
    # -------------------
    df["lag_1"] = df["energy"].shift(1)
    df["lag_24"] = df["energy"].shift(24)
    df["lag_168"] = df["energy"].shift(168)

    # -------------------
    # rolling statistics
    # -------------------
    df["rolling_mean_24"] = df["energy"].rolling(24).mean()
    df["rolling_std_24"] = df["energy"].rolling(24).std()

    df["rolling_mean_168"] = df["energy"].rolling(168).mean()

    # -------------------
    # change features
    # -------------------
    # (pct_change was removed: it explodes when energy is ~0 and carries no usable signal)
    df["diff_1"] = df["energy"].diff()

    # drop NaN rows created by lag/rolling/gaps and any non-finite values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    return df.sort_index()
