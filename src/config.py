CONFIG = {
    # Wide training source (training only). Inference uses data/processed/{building}.csv per building.
    "data_url": "data/processed/temp_open_utc_complete.csv",
    "meta_url": "data/raw/meta_open.csv",
    "timestamp_col": "timestamp",
    "energy_col": "energy",
    "window_size": 24,  # Sliding window length (timesteps) for LSTM sequences
    # --- Train / test split policy (see scripts/splitting.py) ---
    # "auto": hold out the end of the series ONLY if the training part still spans >= min_train_days_for_holdout
    #         days and covers all 12 calendar months; otherwise train on ALL data ("full").
    # "holdout" / "full": force one behaviour for every building (override per building in BUILDING_SPLIT).
    "split_strategy": "auto",
    "test_split": 0.2,  # fraction held out when a holdout split is used ...
    "holdout_days": None,  # ... or hold out exactly the last N days instead (e.g. 365 for "train on 2 years, test on year 3")
    "min_train_days_for_holdout": 365,
    # Validation for early stopping: blocks spread over the whole training period (not "the last 10%")
    "val_fraction": 0.1,
    "val_block_days": 7,
    "lstm_units": [32, 16],
    "latent_dim": 8,
    "decoder_units": [16, 32],
    "dropout_rate": 0.1,
    "learning_rate": 1e-3,
    "epochs": 100,  # Upper bound; EarlyStopping (patience below) normally stops training sooner
    "batch_size": 32,
    "patience": 8,
    # Sensor LSTM post-processing (Model_Testing): rolling error stats, threshold, drift
    "error_rolling_window_hours": 168,  # 7 days of hourly samples - median / MAD of reconstruction_error
    "mad_epsilon": 1e-6,  # Stabilizes normalized_error = (err - median) / (max(MAD, floor) + eps)
    "mad_floor_frac": 0.10,  # MAD floor = frac * rolling_median; stops z-scores exploding when MAD ~ 0
    "normalized_error_clip": 100.0,  # normalized_error is clipped to [-clip, +clip] for plots / reports
    "warmup_hours": 168,  # No anomaly / drift flags until one full rolling window of history exists
    "threshold_mad_k": 6.0,  # threshold_raw = rolling_median + k * rolling_mad
    "threshold_ema_span": 24,  # Smooth threshold_raw with EWM before drift-freeze
    "drift_z_threshold": 5.0,  # Drift candidate when normalized_error exceeds this
    "drift_sustain_points": 24,  # Drift if condition holds for this many consecutive windows
    "output_dir": "output_plots",
    "seed": 42,
    "model_dir": "models",
}

# 13 features. Removed vs. the first version (17):
#   - month / hour / day_of_week: raw integers duplicate the sin/cos encodings; "month" in particular
#     is out of range in the test period of a 1-year chronological split (it dominated Office_Annika).
#   - pct_change: explodes to 90-190x when energy is ~0 and, after MinMax scaling, carried no signal.
FEATURE_COLS = [
    "energy",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_mean_24",
    "rolling_std_24",
    "rolling_mean_168",
    "diff_1",
]

# Chosen to cover different building types and climates (see docs/dataset-notes.md):
#   2 offices, 1 primary-school classroom building, 1 dormitory, 1 college laboratory.
# Office_Jesus and PrimClass_Jaylin were removed (same types as buildings already here); their models from the
# 26 Oct run are still in models/. Next candidates: UnivClass_Nicholas (college classroom), Office_Emerald (large).
BUILDINGS = [
    "Office_Annika",
    "Office_Cristina",
    "PrimClass_Jolie",
    "UnivDorm_Mitch",
    "UnivLab_Brenna",
]

# Optional per-building override of the split policy. Value: "auto" | "holdout" | "full", or a dict such as
#   {"strategy": "holdout", "holdout_days": 365}   (train on everything before the last year, test on the last year)
# Buildings not listed use CONFIG["split_strategy"].
BUILDING_SPLIT = {
    # "Office_Annika": "full",
    # "SomeBuilding_With3Years": {"strategy": "holdout", "holdout_days": 365},
}

# --- Evaluation with injected anomalies (scripts/evaluation.py) ---
# Which evaluation a building gets follows its split policy (CONFIG["split_strategy"] / BUILDING_SPLIT):
#   * one year of data or less ("full" history)  -> BLOCKED testing + injected anomalies (always)
#   * more than a year ("holdout")               -> the chronological test period; injected anomalies only if
#                                                    asked for (--inject, or inject_on_holdout=True here)
EVAL = {
    "folds": 5,  # blocked testing: number of folds (a temporary model is trained per fold)
    "block_days": 14,  # blocked testing: length of one held-out block; blocks are dealt out to folds in turn
    "purge_before_hours": 24,  # training rows dropped just before a held-out block (window overlap)
    "purge_after_hours": 168,  # ... and just after it (lag_168 / rolling_mean_168 features look back 7 days)
    "seed": 7,
    "inject_on_holdout": False,  # holdout buildings: inject anomalies into the test period?
    "slot_days": 14,  # one injected event per this many days of evaluation period
    # Realism: an outage in a building that is idle anyway (school at night / in holidays) changes nothing.
    "active_quantile": 0.9,  # a building's "active level" = this quantile of its clean energy
    "min_active_frac": 0.30,  # dropout / flatline / level_shift are placed where mean clean load >= this x active level
    "placement_tries": 40,  # random start positions tried per event to satisfy min_active_frac
    "min_effect_std": 0.5,  # events whose mean |change| is below this many std of the energy are "inert": kept in the
    #                         data and listed, but left out of recall / precision / AUC (a detector cannot see them)
    "event_types": ["spike", "dropout", "flatline", "level_shift"],
    # event settings: duration in hours [min, max]; amplitude / factor [min, max]
    "spike": {"hours": [1, 3], "amplitude_std": [3.0, 6.0]},  # + N x (std of the building's energy)
    "dropout": {"hours": [6, 24], "factor": [0.0, 0.1]},  # energy x factor (sensor / supply outage)
    "flatline": {"hours": [12, 36]},  # reading stuck at its last value
    "level_shift": {"hours": [48, 72], "factor": [1.3, 1.6]},  # sustained increase (equipment left on)
    "pca_components": 16,  # --backend pca: quick non-LSTM stand-in / baseline
    "quick_folds": 3,  # --quick
    "quick_epochs": 30,
}
