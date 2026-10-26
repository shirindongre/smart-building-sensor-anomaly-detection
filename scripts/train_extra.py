"""
Train models for buildings that are NOT in BUILDINGS (for example extra buildings for the demo's "New building" page).

    python scripts/train_extra.py UnivClass_Nicholas Office_Emerald PrimClass_Uma

This runs scripts/Model_Training.py unchanged, but for the names given on the command line instead of BUILDINGS in
src/config.py, so src/config.py never has to be edited (and cannot be left in the wrong state). It writes, for these
buildings only: models/<b>_best.h5, models/<b>_split.json, scalers/<b>_scaler.pkl, data/processed/<b>.csv and
output_plots/<b>_loss.png. It refuses to run if a model for one of the names already exists, so a typo cannot
overwrite a trained building.
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

names = sys.argv[1:]
if not names:
    raise SystemExit("Usage: python scripts/train_extra.py <Building_1> [<Building_2> ...]")

import src.config as cfg  # noqa: E402

existing = [n for n in names if (ROOT / cfg.CONFIG["model_dir"] / f"{n}_best.h5").is_file()]
if existing:
    raise SystemExit("Refusing to overwrite existing models: " + ", ".join(existing)
                     + ". Move or delete them first if you really want to retrain.")

cfg.BUILDINGS[:] = names  # in place, so `from src.config import BUILDINGS` inside Model_Training sees the new list
print("Training only:", ", ".join(cfg.BUILDINGS), flush=True)
runpy.run_path(str(ROOT / "scripts" / "Model_Training.py"), run_name="__main__")
