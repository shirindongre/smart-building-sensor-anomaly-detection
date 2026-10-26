"""
Preprocessing-only pipeline (no model training).

Reads the wide multi-building CSV, cleans + engineers features for every building in
``src.config.BUILDINGS`` and writes ``data/processed/<building>.csv`` -- the same files that
``Model_Training.py`` writes and ``Model_Testing.py`` reads. Run from the project root:

    python scripts/pipeline.py
"""
import logging
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (_PROJECT_ROOT, _PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts.feature_engineering import engineer_features
from scripts.preprocessing import preprocess_building
from src.config import BUILDINGS, CONFIG

PROCESSED_PATH = _PROJECT_ROOT / "data" / "processed"
LOG_DIR = _PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main() -> None:
    meta = pd.read_csv(_PROJECT_ROOT / CONFIG["meta_url"])

    df_wide = pd.read_csv(_PROJECT_ROOT / CONFIG["data_url"])
    df_wide["timestamp"] = pd.to_datetime(df_wide["timestamp"], utc=True)
    df_wide = df_wide.set_index("timestamp").sort_index()

    for i, building in enumerate(BUILDINGS, start=1):
        try:
            msg = f"Processing {i}/{len(BUILDINGS)}: {building}"
            logging.info(msg)
            print(msg)

            df = preprocess_building(df_wide, building)
            if df is None:
                raise ValueError("building not found in wide dataset")

            df = engineer_features(df, building, meta)

            out_path = PROCESSED_PATH / f"{building}.csv"
            df.reset_index().to_csv(out_path, index=False)  # index = tz-aware UTC 'timestamp'
            logging.info(f"Saved {building} with shape {df.shape} -> {out_path}")

        except Exception as e:  # keep going so one bad building doesn't stop the rest
            logging.error(f"Failed {building}: {e}")
            print("Failed:", building, e)

    print("Pipeline finished.")


if __name__ == "__main__":
    main()
