import json
import os
from pathlib import Path
import sys

#Silence TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = info, 2 = warnings, 3 = errors only


#Silence TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
_SCRIPT_DIR = Path(__file__).resolve().parent
for p in (_PROJECT_ROOT, _SCRIPT_DIR):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
import matplotlib
matplotlib.use("Agg")              # Use non-interactive backend for file saving
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib

import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from scripts.preprocessing import preprocess_building
from scripts.feature_engineering import engineer_features
from scripts.scaling import scale_features
from scripts.sequence_creations import create_sequences
from scripts.splitting import decide_split, split_windows, validation_row_mask

# Global Configurations
from src.config import CONFIG
# Reproducibility
np.random.seed(CONFIG["seed"])
tf.random.set_seed(CONFIG["seed"])

_output_dir = _PROJECT_ROOT / CONFIG["output_dir"]
_output_dir.mkdir(parents=True, exist_ok=True)

#-----------------------------------------------------------
# Data Loading and Preprocessing
#-----------------------------------------------------------


def _resolve_config_path(key: str) -> Path:
    """Resolve a config path from the project root. HTTP URLs are left unchanged."""
    raw = CONFIG[key]
    if str(raw).startswith(("http://", "https://")):
        return Path(raw)
    path = Path(raw)
    if path.is_absolute():
        return path
    return _PROJECT_ROOT / path


# Dataset importing
df_wide = pd.read_csv(_resolve_config_path("data_url"))
df_wide["timestamp"] = pd.to_datetime(df_wide["timestamp"], utc=True)
df_wide = df_wide.set_index("timestamp")
print(df_wide.head())
print(f"Dataset shape: {df_wide.shape}")
print(f"Total buildings: {len(df_wide.columns)}")

META_PATH = _resolve_config_path("meta_url")
meta = pd.read_csv(META_PATH)

from src.config import FEATURE_COLS
from src.config import BUILDINGS
from src.config import BUILDING_SPLIT

selected_buildings = BUILDINGS  # Buildings listed in src/config.py

building_data = {}

for building in selected_buildings:
    df_clean = preprocess_building(df_wide, building)
    if df_clean is None or len(df_clean) <= 500:
        print(f"  [SKIP] {building}: insufficient rows or missing from wide dataset")
        continue

    building_data[building] = df_clean

    df_features = engineer_features(df_clean, building, meta)
    print(f"  Features shape: {df_features.shape}")

    # Per-building CSV for testing (load → scale → sequence → predict; no wide table at inference).
    # Index is tz-aware UTC (continuous hourly grid); local-time features are already baked in.
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = _PROCESSED_DIR / f"{building}.csv"
    df_features.reset_index().to_csv(out_csv, index=False)
    print(f"  Saved {out_csv}")

    # Split policy: holdout (train on the early part, test on the end) or full-history training, depending on
    # how much data the building has (see scripts/splitting.py and the split settings in src/config.py).
    decision = decide_split(df_features.index, building, CONFIG, BUILDING_SPLIT)
    print(f"  {decision['message']}")

    X_train_sc, X_test_sc, scaler, split_idx = scale_features(
        df=df_features,
        feature_cols=FEATURE_COLS,
        split_idx=decision["split_idx"],
    )
    _scalers_dir = _PROJECT_ROOT / "scalers"
    _scalers_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, _scalers_dir / f"{building}_scaler.pkl")

    # Step D — sliding windows
    X_train_seq = create_sequences(X_train_sc, CONFIG["window_size"])
    X_test_seq = create_sequences(X_test_sc, CONFIG["window_size"])

    print(f"  Train windows  : {X_train_seq.shape}")   # (n_windows, window_size, n_features)
    print(f"  Test windows   : {X_test_seq.shape}")

    # Validation windows for early stopping: short blocks spread over the whole training period, so every
    # season is represented in both fitting and validation (instead of "the last 10%", which is one season).
    val_rows = validation_row_mask(len(X_train_sc), CONFIG["val_fraction"], CONFIG["val_block_days"])
    fit_idx, val_idx = split_windows(val_rows, CONFIG["window_size"])
    print(f"  Fit windows    : {len(fit_idx)}  | Validation windows: {len(val_idx)} "
          f"({int(val_rows.sum())} validation rows in blocks)")

    # Split metadata so Model_Testing.py reproduces exactly the same train / test boundary
    _PROJECT_ROOT.joinpath(CONFIG["model_dir"]).mkdir(parents=True, exist_ok=True)
    meta_out = dict(decision, n_fit_windows=int(len(fit_idx)), n_val_windows=int(len(val_idx)),
                    val_fraction=CONFIG["val_fraction"], val_block_days=CONFIG["val_block_days"],
                    window_size=CONFIG["window_size"], n_features=len(FEATURE_COLS))
    with open(_PROJECT_ROOT / CONFIG["model_dir"] / f"{building}_split.json", "w", encoding="utf-8") as fh:
        json.dump(meta_out, fh, indent=2)

    # Store everything needed for model training
    building_data[building] = {
        "df"          : df_features,
        "X_train"     : X_train_seq,
        "X_test"      : X_test_seq,
        "fit_idx"     : fit_idx,
        "val_idx"     : val_idx,
        "decision"    : decision,
        "scaler"      : scaler,
        "split_idx"   : split_idx,
    }


print(f"\nReady: {len(building_data)} buildings")

if not building_data:
    raise SystemExit("No buildings processed — check BUILDINGS and wide dataset columns.")

sample = building_data[next(iter(building_data))]
print("\nTrain shape :", sample["X_train"].shape)
print("Test shape  :", sample["X_test"].shape)
print("Features per step:", sample["X_train"].shape[2])


#-----------------------------------------------------------
# Model Implemenntation
#-----------------------------------------------------------

from scripts.Model_Architechture import build_lstm_autoencoder

trained_models = {}

for building, data in building_data.items():

    print(f"\n{'='*40}")
    print(f"Training model: {building}")

    X_train    = data["X_train"][data["fit_idx"]]   # windows used to update the weights
    X_val      = data["X_train"][data["val_idx"]]   # validation windows (blocks spread over the whole period)
    if len(X_val) == 0 or len(X_train) == 0:
        raise SystemExit(f"{building}: not enough data to build separate fit and validation windows.")

    # build a fresh model for each building
    model = build_lstm_autoencoder(
        window_size   = X_train.shape[1],  # number of timesteps per window
        n_features    = X_train.shape[2],  # number of features per timestep
        lstm_units    = CONFIG["lstm_units"],
        latent_dim    = CONFIG["latent_dim"],
        decoder_units = CONFIG["decoder_units"],
        dropout_rate  = CONFIG["dropout_rate"],
        learning_rate = CONFIG["learning_rate"],
    )

    (_PROJECT_ROOT / CONFIG["model_dir"]).mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=CONFIG["patience"],
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(_PROJECT_ROOT / CONFIG["model_dir"] / f"{building}_best.h5"),
            monitor="val_loss",
            save_best_only=True,
            verbose=0
        ),
    ]

    # autoencoder training — input == target
    history = model.fit(
        X_train, X_train,
        epochs          = CONFIG["epochs"],
        batch_size      = CONFIG["batch_size"],
        validation_data = (X_val, X_val),
        callbacks       = callbacks,
        shuffle         = True,
        verbose         = 1,
    )

    trained_models[building] = {
        "model"  : model,
        "history": history,
    }

    # plot training loss
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"],     label="Train loss")
    plt.plot(history.history["val_loss"], label="Val loss", linestyle="--")
    plt.title(f"Training loss — {building}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    _output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(_output_dir / f"{building}_loss.png")
    plt.close()
    print(f"  Loss plot saved for {building}")

print(f"\nAll done. {len(trained_models)} models trained.")

print("\nSplit strategy per building:")
for b, d in building_data.items():
    dec = d["decision"]
    kind = "FULL history (no held-out test period)" if dec["strategy"] == "full" else "HOLDOUT (last part is test)"
    print(f"  {b:20s} {kind}  - {dec['reason']}")


