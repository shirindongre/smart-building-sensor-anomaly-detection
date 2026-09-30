This is the sensor-only pipeline for smart-building-sensor-anomaly-detection.

Source files (already in this project):

- data/processed/temp_open_utc_complete.csv
- data/raw/meta_open.csv

Run these from the project root:

    python scripts/Model_Training.py
    python scripts/Model_Testing.py

Training writes per-building CSVs under data/processed/, scalers under scalers/, and models under models/. Testing reads those outputs and writes plots and reports under output_plots/.

scripts/split_buildings.py and scripts/pipeline.py are earlier preprocessing helpers. The current training script loads the wide CSV directly and does not use them.
