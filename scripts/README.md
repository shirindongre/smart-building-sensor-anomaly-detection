Sensor-only pipeline for smart-building-sensor-anomaly-detection.

Source files (not in git - `data/` is ignored; see the root README for where to get them):

- data/processed/temp_open_utc_complete.csv  (wide table, one column per building, UTC)
- data/raw/meta_open.csv

Run these from the project root:

    python scripts/Model_Training.py
    python scripts/Model_Testing.py
    python scripts/evaluation.py      # injected-anomaly evaluation (see below)

Training writes per-building CSVs under data/processed/, scalers under scalers/, and models under models/.
Testing reads those outputs and writes plots and reports under output_plots/.

Conventions
- Timestamps in the per-building CSVs and the anomaly reports are tz-aware UTC on a continuous hourly grid.
  Calendar features (hour, day of week) are computed from building-local time.
- Train / test split is chosen per building (`scripts/splitting.py`, settings in `src/config.py`):
  - buildings with about a year of data are trained on ALL of it (full-history), so every season is learned;
    they have no held-out test period and future data is evaluated as it arrives;
  - buildings whose training part would still span 365+ days and all 12 calendar months use a chronological
    holdout (last 20%, or the last `holdout_days`);
  - override per building with `BUILDING_SPLIT` in `src/config.py`.
  Training writes `models/<building>_split.json`; Model_Testing.py reads it so both use the same boundary.
- Early-stopping validation uses 7-day blocks spread over the whole training period, not the last 10%.
- Each report row is stamped with the timestamp of the LAST hour of its 24-hour window.
- A window is labelled `train` only if it ends before the 80/20 split point.
- Models / scalers trained with a different feature count than `FEATURE_COLS` are skipped by
  Model_Testing.py with a message asking you to re-run training.

Evaluation with injected anomalies (scripts/evaluation.py)
- Follows each building's split policy:
  - one year or less (full history): BLOCKED testing + injected anomalies, always. 14-day blocks are dealt to K folds;
    a temporary model per fold is trained on the other blocks (purge gaps of 24 h before / 168 h after each held-out
    block), anomalies are injected into the held-out blocks only, and the detector has to find them.
  - more than a year (holdout): no blocked testing; the test period is evaluated with the production model.
    Injection is optional: `--inject`, or `EVAL["inject_on_holdout"] = True`.
- Options: `--buildings A B`, `--quick` (3 folds, 30 epochs), `--folds N`, `--backend lstm|pca` (pca needs no TensorFlow),
  `--seed N`. Settings (event types, sizes, block length, purge) are in `EVAL` in `src/config.py`.
- `--summarize-only`: no training; rebuilds the report and summary from the files already in output_plots/evaluation/
  (per-building results are saved as soon as each building finishes, so an interrupted run loses nothing).
- Outputs in output_plots/evaluation/: evaluation_report.md, evaluation_summary.csv, and per building `_events.csv`,
  `_windows.csv`, `_evaluation.png`. Production models are never changed; fold models are not saved.
- Realism: dropout / flatline / level_shift are placed where the load is active (`min_active_frac`); events whose real
  effect is below `min_effect_std` are marked `inert` (kept in the events CSV, not scored).
- Event types: spike, dropout, flatline, level_shift. Metrics: event recall, delay, false-alarm rate, precision,
  ROC-AUC, and a "same hour last week" baseline. Windows up to 7 days after an event are ignored.
- Modules: scripts/anomaly_injection.py (event placement and injection), scripts/anomaly_scoring.py (rolling
  threshold, drift and anomaly flags shared with Model_Testing.py).

Helpers
- scripts/pipeline.py        preprocessing only: wide CSV -> data/processed/<building>.csv (same output as training)
- scripts/split_buildings.py optional: writes data/raw/<building>.csv (timestamp, energy) for each configured building

Fault history and future work: docs/FAULT_TRACKER.md
