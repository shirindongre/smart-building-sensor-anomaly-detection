"""
Per-building train / test split policy.

Why this exists
---------------
A building with only about one year of data cannot be split 80/20 in time without hiding a whole season from
the model (the first run of Office_Annika showed this: the model never saw most of November-December and its
test error was 4-11x the training error). So the split depends on how much history a building has:

* ``holdout`` - train on the earlier part, keep the LAST part as a held-out test period. Used only when the
  training part still spans at least a year and covers all 12 calendar months.
* ``full``    - train on ALL the data. Used when holdout would leave seasons out of training. There is then no
  held-out period; the building is evaluated on data that arrives later.

``decide_split`` makes the choice (``split_strategy = "auto"``) or follows an explicit setting, globally in
``CONFIG["split_strategy"]`` or per building in ``src.config.BUILDING_SPLIT``.

``validation_row_mask`` / ``split_windows`` pick the validation data used for early stopping: several short
blocks spread over the whole training period (instead of the last 10%, which is always one season), with
windows that touch a block's edge dropped so no row is shared between training and validation windows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _parse_override(setting):
    """A per-building setting is either a strategy string or a dict like {"strategy": "holdout", "holdout_days": 365}."""
    if setting is None:
        return {}
    if isinstance(setting, str):
        return {"strategy": setting}
    return dict(setting)


def decide_split(index: pd.DatetimeIndex, building: str, config: dict, overrides: dict | None = None) -> dict:
    """
    Decide how to split one building's feature frame (rows are hourly, sorted by time).

    Returns a JSON-serialisable dict with: strategy ("full" | "holdout"), split_idx (first test row; equals
    n_rows when strategy == "full"), n_rows, start, end, span_days, train_months, requested, reason, message.
    """
    n = len(index)
    if n == 0:
        raise ValueError(f"{building}: no rows to split")

    ov = _parse_override((overrides or {}).get(building))
    requested = str(ov.get("strategy", config.get("split_strategy", "auto"))).lower()
    if requested not in {"auto", "holdout", "full"}:
        raise ValueError(f"{building}: split strategy must be 'auto', 'holdout' or 'full', got {requested!r}")

    test_split = float(ov.get("test_split", config.get("test_split", 0.2)))
    holdout_days = ov.get("holdout_days", config.get("holdout_days"))
    min_train_days = int(ov.get("min_train_days_for_holdout", config.get("min_train_days_for_holdout", 365)))

    # Candidate holdout split point: last ``holdout_days`` days if given, otherwise the last ``test_split`` fraction.
    if holdout_days:
        split_idx = max(n - int(holdout_days) * 24, 0)
    else:
        split_idx = int(n * (1.0 - test_split))
    split_idx = min(max(split_idx, 0), n)

    def days(a, b):  # whole days covered by rows a..b (inclusive)
        return int((index[b] - index[a]) / pd.Timedelta(days=1)) + 1

    span_days = days(0, n - 1)
    train_days = days(0, split_idx - 1) if split_idx > 0 else 0
    train_months = sorted(int(m) for m in set(index[:split_idx].month)) if split_idx > 0 else []
    missing_months = [m for m in range(1, 13) if m not in train_months]

    if requested == "full":
        strategy, reason = "full", "set explicitly to 'full'"
    elif requested == "holdout":
        strategy, reason = "holdout", "set explicitly to 'holdout'"
        if split_idx == 0 or split_idx == n:
            strategy, reason = "full", "explicit 'holdout' left no train or no test rows"
    else:  # auto
        if split_idx == 0 or split_idx == n:
            strategy, reason = "full", "holdout would leave no train or no test rows"
        elif train_days < min_train_days:
            strategy = "full"
            reason = f"a holdout would leave only {train_days} training days (< {min_train_days})"
        elif missing_months:
            strategy = "full"
            reason = f"a holdout would leave calendar months {missing_months} out of training"
        else:
            strategy, reason = "holdout", f"{train_days} training days cover all 12 calendar months"

    if strategy == "full":
        split_idx = n
        message = (
            f"Full-history training for {building}: {span_days} days of data; {reason}. "
            "All rows are used for training so every season is learned. There is no held-out test period for "
            "this building - its flags on the history are in-sample, and future data will be evaluated as it arrives."
        )
    else:
        test_days = days(split_idx, n - 1)
        warn = f" WARNING: training months missing: {missing_months}." if missing_months else ""
        message = (
            f"Holdout split for {building}: train on the first {train_days} days, test on the last {test_days} days "
            f"({reason}).{warn}"
        )

    return {
        "building": building,
        "strategy": strategy,
        "requested": requested,
        "split_idx": int(split_idx),
        "n_rows": int(n),
        "start": str(index[0]),
        "end": str(index[-1]),
        "span_days": span_days,
        "train_days": train_days if strategy == "holdout" else span_days,
        "train_months": train_months if strategy == "holdout" else sorted(int(m) for m in set(index.month)),
        "reason": reason,
        "message": message,
    }


def validation_row_mask(n_rows: int, val_fraction: float = 0.1, block_days: int = 7, min_block_hours: int = 48) -> np.ndarray:
    """
    Boolean mask over ``n_rows`` training rows marking validation rows: evenly spaced blocks of ``block_days``
    days whose total length is about ``val_fraction`` of the data. Returns all-False if ``val_fraction`` <= 0.
    """
    mask = np.zeros(n_rows, dtype=bool)
    if val_fraction <= 0 or n_rows == 0:
        return mask
    block = max(int(block_days) * 24, min_block_hours)
    n_blocks = max(1, int(round(val_fraction * n_rows / block)))
    # never let validation blocks take more than half the data
    while n_blocks > 1 and n_blocks * block > n_rows // 2:
        n_blocks -= 1
    if block > n_rows // 2:
        block = max(n_rows // 4, 1)
    seg = n_rows / n_blocks
    for k in range(n_blocks):
        start = int(k * seg + max(seg - block, 0) / 2)
        mask[start : min(start + block, n_rows)] = True
    return mask


def split_windows(row_val_mask: np.ndarray, window: int):
    """
    Given the validation-row mask over the rows used to build windows with ``create_sequences`` (window ``i``
    covers rows ``i .. i + window - 1``), return (train_window_idx, val_window_idx).
    A window is validation only if ALL its rows are validation rows, and training only if NONE are; windows
    straddling a block edge are dropped, so train and validation never share a row.
    """
    n_windows = len(row_val_mask) - window + 1
    if n_windows <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    csum = np.concatenate([[0], np.cumsum(row_val_mask.astype(int))])
    in_val = csum[window : window + n_windows] - csum[:n_windows]
    train_idx = np.flatnonzero(in_val == 0)
    val_idx = np.flatnonzero(in_val == window)
    return train_idx, val_idx
