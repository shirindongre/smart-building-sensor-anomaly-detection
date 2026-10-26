# Smart-building sensor anomaly detection

Unsupervised anomaly detection on hourly building energy data using a per-building LSTM autoencoder.
Reconstruction error is turned into anomaly / drift flags with a rolling median + MAD threshold.

Buildings: see `BUILDINGS` in `src/config.py` (currently 5: 2 offices, 1 primary-school classroom building,
1 dormitory, 1 college laboratory).

## Setup

    python -m venv venv
    venv\Scripts\activate            # Windows (use `source venv/bin/activate` on macOS/Linux)
    pip install -r requirements.txt

## Data

`data/` is git-ignored. Place these files before running (from the Building Data Genome Project):

- `data/processed/temp_open_utc_complete.csv` - wide hourly table (UTC), one column per building
- `data/raw/meta_open.csv` - building metadata (`uid`, `timezone`, ...)

## Run

    python scripts/Model_Training.py   # features, scalers, models, loss plots
    python scripts/Model_Testing.py    # anomaly reports and diagnostic plots
    python scripts/evaluation.py       # injected-anomaly evaluation (recall, false alarms, AUC)

Outputs: `models/`, `scalers/`, `output_plots/`, `data/processed/sensor_outputs/`.

Train/test split: each building gets its own policy. With about a year of data the model trains on all of it
(so every season is learned) and the building is evaluated on data that arrives later; with enough history the
end of the series is held out as a test period. Details in `scripts/README.md`; settings in `src/config.py`.

Evaluation: the dataset has no labelled anomalies, so `scripts/evaluation.py` injects known ones. Buildings with
a year of data or less get blocked testing plus injected anomalies; buildings with more than a year use the
held-out test period, with injection as an option (`--inject`). Details in `scripts/README.md`.

## Running the demo

A locally run Streamlit app for live demonstrations. It needs the data files, the trained models and the scalers
(see Setup and Data above); it never trains, never writes to `data/`, `models/` or `scalers/`, and needs no network.

    venv\Scripts\activate
    pip install -r requirements.txt
    streamlit run app/app.py

Run from the project root. Streamlit opens `http://localhost:8501`. On the first visit a warm-up step imports TensorFlow,
loads every model and scores every building once (about 45 s for five buildings); after that each action takes a few
seconds. "Warm up again" in the sidebar reloads everything from disk.

Pages:

- **Building view** - timeline of energy, reconstruction error, threshold, flags and drift for any building that has a
  model and a scaler, headline numbers, the flagged periods, and the split policy of the building.
- **Inject and detect** - add a spike, dropout, flatline or level shift to a copy of the series, rescore it with the
  saved model, and see whether it was found, how late, and any new false alarms. Several injections, plus Reset.
- **New building** - onboard a building that was trained in advance but is not in `BUILDINGS` (found by scanning
  `models/`), with the real steps timed. To add one: set `BUILDINGS` to just that name, run
  `python scripts/Model_Training.py`, restore `BUILDINGS`, then press "Warm up again".
- **Evaluation** - results of `python scripts/evaluation.py` read from `output_plots/evaluation/`.
- **Replay** - step through a building's hours and watch flags appear (in-sample for full-history buildings).

Tests for the logic behind the demo (no browser, no TensorFlow):

    python -m pytest tests

The logic is in `scripts/demo_core.py`; `app/` only holds layout code. Missing files produce a message that names the
command to run.

## More

- `scripts/README.md` - script details and conventions
- `app/` and `scripts/demo_core.py` - the demo UI (see Running the demo)
- `docs/FAULT_TRACKER.md` - known faults, fixes applied, what still needs verifying, future scope
- `architecture_diagram.html` - model architecture
- `notebooks/` - exploratory notebooks (EDA for all configured buildings)
