"""
Optional helper: split the wide multi-building CSV into one ``data/raw/<building>.csv`` per building
listed in ``src.config.BUILDINGS`` (columns: timestamp, energy). The main pipeline does not need these.
Run from the project root:  python scripts/split_buildings.py
"""
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import BUILDINGS, CONFIG

INPUT_PATH = _PROJECT_ROOT / CONFIG["data_url"]
OUTPUT_PATH = _PROJECT_ROOT / "data" / "raw"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

for building in BUILDINGS:
    if building not in df.columns:
        print("Not in wide dataset, skipped:", building)
        continue

    building_df = pd.DataFrame({"timestamp": df["timestamp"], "energy": df[building]})
    building_df.to_csv(OUTPUT_PATH / f"{building}.csv", index=False)
    print("Saved:", building)
