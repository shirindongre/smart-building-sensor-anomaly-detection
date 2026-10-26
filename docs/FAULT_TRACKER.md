# Fault tracker - smart-building-sensor-anomaly-detection

Last updated: 2026-10-26 (evaluation framework added; injection made realistic; demo UI added). Update the status column and the counts below whenever a fault changes state.

## Summary

| | Count |
|---|---|
| **Total faults found** | **21** |
| Fixed and confirmed (checked by running code, or by your retraining run on 26 Oct) | 15 |
| Fixed in code, effect still to be confirmed by running on your computer | 4 (F02, F17, F19, F21) |
| Still open | 2 (F18, F20) |

Status key: **Fixed** = changed and checked. **Fixed (retrain)** = changed and tested with the training script's
TensorFlow part stubbed out, but the effect on the models can only be confirmed by re-running
`scripts/Model_Training.py` (F17: `scripts/evaluation.py` with the LSTM backend). **Open** = not fixed yet.

**Current state of the models:** the models in `models/` come from your 26 Oct run (13 features, 80/20 split,
before the per-building split policy). They work with the current code; `Model_Testing.py` reads them with the old
80/20 boundary because they have no `*_split.json` file. Retrain once more to get the new split policy (F02, F19, F21).

---

## Faults and fixing steps

| ID | Severity | Fault | Status |
|---|---|---|---|
| F01 | High | Test-period timestamps shifted 24 h early, last 24 h of data never reported, `split` label off by one | Fixed |
| F02 | High | Office_Annika: train/test mismatch; raw `month` out of range in 85% of test rows, test error 11x train (now 4.5x; the rest is seasonal, see F19) | Fixed (retrain) |
| F03 | High | `pct_change` feature explodes (up to 189x) when energy is ~0 and is useless after MinMax scaling | Fixed |
| F04 | Medium | Daylight saving: duplicated 01:00 hour and a missing hour in 4 of 5 buildings; row-based lags shifted | Fixed |
| F05 | Medium | Threshold instability: MAD = 0 in early rows gave z-scores of -695 to +1104 and 4-17 anomalies flagged in week 1 | Fixed |
| F06 | Medium | `scripts/pipeline.py` could not run (wrong call signature, missing `data/raw/<building>.csv`, `logs/` not created) | Fixed |
| F07 | Low | `architecture_diagram.html` and a code comment described a different model (window 30, 64/32 units, latent 16, mu+3 sigma) | Fixed |
| F08 | Medium | Models under-trained: loss still falling at epoch 25, early stopping never triggered | Fixed |
| F09 | Low | Plot labels wrong: "Energy (scaled)" for raw energy; z-score axis label unclear | Fixed |
| F10 | Medium | `create_sequences` off by one: it dropped the last window of every array | Fixed |
| F11 | Medium | Notebooks stale: covered 3 old buildings, used `StandardScaler` on all columns, referenced missing files | Fixed |
| F12 | Low | Docs stale or unfinished (`dataset-notes.md`, no project README) | Fixed |
| F13 | Low | No requirements file, no setup steps, Windows-only `venv` path check in `Model_Testing.py` | Fixed |
| F14 | Low | Unused config keys (`threshold_sigma`, `model_path`) | Fixed |
| F15 | Low | `split_buildings.py` wrote about 507 files regardless of the building list | Fixed |
| F16 | High | Old models / scalers / reports / plots incompatible with the new 13-feature set | Fixed |
| F17 | High | No ground truth or evaluation metrics: nothing shows the flagged points are real anomalies | Fixed (retrain) |
| F18 | Low | `scripts/Model_Architechture.py` is misspelled (imports depend on the spelling) | Open |
| F19 | Medium | One fixed 80/20 time split for every building: with only about a year of data the test period is a season the model barely saw | Fixed (retrain) |
| F20 | Medium | `data/` is git-ignored but models are committed, so a fresh clone cannot reproduce results (setup now documented in README only) | Open |
| F21 | Medium | Early-stopping validation used the last 10% of the training timeline, which is always one season (so validation was biased and that season was never trained on) | Fixed (retrain) |

### Results of your 26 Oct retraining run (13 features, 80/20 split, before the split policy)

| Building | Test error / train error | Anomaly rate (train / test) | Drift (train / test) |
|---|---|---|---|
| Office_Annika | 4.5x (was 11x) | 3.8% / 7.9% | 1.0% / 5.1% |
| Office_Cristina | 1.4x | 2.7% / 3.5% | 0.4% / 3.5% |
| Office_Jesus | 1.5x | 2.8% / 6.0% | 3.2% / 5.6% |
| PrimClass_Jaylin | 1.5x | 1.9% / 2.2% | 1.2% / 0.8% |
| PrimClass_Jolie | 1.1x | 3.5% / 3.8% | 0.4% / 0.9% |

- No `[SKIP]` lines; reports end at the last hour of the data; no anomalies in the first week; z-scores bounded (-9 to +58). This confirms F01, F05 and F16.
- Annika's loss curve flattened and early stopping ended training at about epoch 95, which confirms F08 for that building (the other four curves were not inspected).
- Annika's remaining gap is seasonal: weekday energy averages about 20 in summer and 44.5 in December, higher than any month in its training part. Error follows the level (0.0011 in summer, 0.0049 in January, 0.0142 in December). Its biggest error days are 24-26 December, probably genuine holiday low-occupancy days (not yet checked against the data).

### Fixing steps and evidence

**F01 - timestamp alignment** (`scripts/Model_Testing.py`, `build_results_df`, `main`)
1. Cause: train and test windows were built separately and then stacked, but timestamps assumed one continuous run.
2. Now the whole frame is scaled once with the train-fitted scaler and windows are built over the full frame, so test windows overlap the split boundary.
3. Each row is stamped with the time of the last hour in its window; a window is `train` only if it ends before the split point.
4. Confirmed by your run: all 5 reports have 8569 windows (8592 - 24 + 1) and end at the last hour of the data.

**F02 - Annika's seasonal mismatch** (`src/config.py`, `scripts/splitting.py`)
1. Removed raw `month`, `hour`, `day_of_week` from `FEATURE_COLS` (sin/cos already encode hour and weekday; `month` cannot be learned from one year). 17 features became 13.
2. Result: test/train error ratio fell from 11x to 4.5x (see the results table).
3. The remainder is the seasonal limit of a one-year series, which the split policy (F19) addresses. After retraining with it, Annika trains on the whole year and has no held-out period.

**F19 - per-building split policy** (`scripts/splitting.py`, `src/config.py`, `scripts/scaling.py`, `Model_Training.py`, `Model_Testing.py`)
1. New `decide_split`: **holdout** (train on the early part, test on the end) is used only when the training part still spans at least 365 days and covers all 12 calendar months; otherwise **full** (train on all data).
2. Settings in `src/config.py`: `split_strategy` ("auto" | "holdout" | "full"), `test_split`, `holdout_days` (for example 365 = "train on 2 years, test on year 3"), `min_train_days_for_holdout`, and a per-building `BUILDING_SPLIT` override.
3. Training writes `models/<building>_split.json`; testing reads it so both use the same boundary. If the CSV later gets newer rows than the model was trained on, those rows are scored as unseen "test" data - this is how future data for a full-history building is evaluated.
4. Full-history buildings print a clear message (no held-out period, flags are in-sample, future data will be evaluated) and the report's test line says "n/a".
5. Tested: all 5 real buildings -> full; synthetic 1.5-, 2- and 3-year buildings -> holdout; 3 years with `holdout_days=365` -> train 2 years / test last year; forced holdout on 1 year prints a missing-months warning; legacy models without the json fall back to 80/20; a full-history model scored on data extended by 30 days labels the new rows as test. The training script ran end to end with TensorFlow stubbed.
6. Still to confirm: retrain, then check each building prints its strategy at the end of training.
7. Limit: a full-history building has no held-out evaluation; see F17.

**F21 - validation blocks** (`scripts/splitting.py`, `Model_Training.py`)
1. Early stopping now validates on several 7-day blocks spread evenly over the whole training period (about 10% of the data) instead of the last 10%.
2. Windows that touch a block edge are dropped, so fit and validation windows never share a row (checked: 0 shared rows). Real-sized example: 7614 fit windows, 725 validation windows.
3. Side effect: about 5 weeks per building are used only for validation, not for updating the weights.

**F03 - pct_change** (`scripts/feature_engineering.py`, `src/config.py`): removed; non-finite values replaced and dropped. No NaN or inf in any building.

**F04 - daylight saving** (`scripts/feature_engineering.py`): index stays UTC on a strict hourly grid (`asfreq("1h")`); local time is used only for calendar features; lags and rolling windows computed after the grid is built. 0 duplicates and 0 gaps in all 5 buildings (before: 1 duplicate and 2 gaps in four of them). Timestamps in the CSVs and reports are now UTC.

**F05 - threshold stability** (`src/config.py`, `Model_Testing.py`): MAD floored at 10% of the rolling median, z-scores clipped to +/-100, first 168 hours are a warm-up with no flags. Confirmed by your run.

**F06 / F15 - helper scripts**: `pipeline.py` rewritten as a preprocessing-only step (output identical to the training data prep); `split_buildings.py` writes only the configured buildings.

**F07, F09, F12, F13, F14 - docs, config, labels**: diagram updated (window 24, 13 features, 32/16 encoder, latent 8, 16/32 decoder, patience 8, median + 6xMAD threshold); added root `README.md`, `requirements.txt` (unpinned) and `docs/dataset-notes.md`; removed the Windows `venv` check and unused config keys; fixed axis labels.

**F08 - training length** (`src/config.py`): `epochs` 25 -> 100 with `EarlyStopping(patience=8)`. Annika's curve confirmed (stopped near epoch 95).

**F10 - create_sequences** (`scripts/sequence_creations.py`): window count is `len(data) - window_size + 1`.

**F11 - notebooks**: rewritten for the configured buildings, shared scripts, relative paths; notebook 03 now uses the split policy. Every code cell ran on the real data. Old outputs were cleared; re-run them to regenerate plots.

**F16**: models retrained on 26 Oct with the 13-feature set.

---

## Change log - building list (26 Oct)

`BUILDINGS` in `src/config.py` changed from Annika, Cristina, Jolie, Jesus, Jaylin to Annika, Cristina, Jolie,
UnivDorm_Mitch, UnivLab_Brenna: Jesus (office) and Jaylin (school) duplicated types already covered, and the dataset's
College Laboratory, College Classroom and Dormitory buildings (almost half of the 507) were untested. All scripts
and notebooks read `BUILDINGS` from the config, so no code changed. The 26 Oct models, scalers and reports for Jesus
and Jaylin are still on disk (and not used by Model_Testing.py). The results table below is from that older run and
lists them for the record. Before relying on the new buildings: train them (`Model_Training.py`), then run
`Model_Testing.py` and `evaluation.py`. Evaluation is now saved per building and the combined report can be rebuilt
with `python scripts/evaluation.py --summarize-only`.

## F17 - injected-anomaly evaluation (`scripts/evaluation.py`)

The BDG1 buildings have no labelled anomalies and only one year of data, so the detector is tested by injecting
known anomalies (spike, dropout, flatline, level shift; settings in `EVAL` in `src/config.py`). What a building gets
follows its split policy:

- one year of data or less (full history): blocked testing plus injected anomalies, always. The year is cut into
  14-day blocks dealt to 5 folds; per fold a temporary model is trained on the other blocks (24 h purge before and
  168 h purge after each held-out block, so the look-back features cannot leak), anomalies are injected only into the
  held-out blocks, and the model must find them. Every season is in training for every fold; every block is tested once.
- more than a year (holdout): no blocked testing; the chronological test period is evaluated with the production
  model. Injection is optional (`--inject`, or `inject_on_holdout` in `EVAL`).

Realism (added after your first `--quick` run): dropout / flatline / level_shift are placed where the building's load
is active (mean clean load >= 30% of its 90th percentile), and every event's real effect is measured. Events that
changed almost nothing (mean change < 0.5 std, e.g. a dropout in a school that is idle at night) are marked `inert`:
they stay in the data and in `<building>_events.csv`, but are not scored. In the first run, the missed school-building
dropouts happened at about 12-17% of peak load, against about 85% for the detected ones, so those misses were an
artefact of the test, not of the model. Settings: `min_active_frac`, `min_effect_std`, `placement_tries` in `EVAL`.

Metrics: event recall, detection delay, false-alarm rate on clean windows, window precision, ROC-AUC, and a
"same hour last week" baseline. Windows up to 7 days after an event are ignored (lag features carry the event forward).
Run: `python scripts/evaluation.py` (add `--quick` for 3 folds / 30 epochs, `--backend pca` to run without TensorFlow).

Checked so far (in the cloud workspace, TensorFlow not available): all 5 buildings with the PCA stand-in backend,
and the holdout path on synthetic 3-year data. Invariants verified: each row held out exactly once, injected events
lie wholly inside held-out blocks, no evaluated window touches a training row. PCA results (5 folds, 20-25 events per
building): recall 52-88%, false alarms 0.6-1.9% of clean windows. Spikes and dropouts are found most reliably;
level shifts are the weakest (the drift logic deliberately stops flagging sustained shifts, so `alerted` in the
events CSV also counts drift flags). Those PCA numbers are a stand-in; the LSTM numbers come from running the script.

## Demo UI (done, 26 Oct)

A local Streamlit demo, `streamlit run app/app.py` (steps in the README, section "Running the demo"). Logic is in
`scripts/demo_core.py` (no Streamlit, tested by `tests/test_demo_core.py` with the PCA stand-in scorer); `app/` is layout
only. It reuses `score_series`, `rebuild_features`, `inject`, `decide_split` and the training pipeline's own
preprocessing and feature functions; it does not retrain, does not change the model, features or thresholds, and writes
nothing to `data/`, `models/` or `scalers/`. Pages: building view (timeline, flagged periods, split-policy box), inject and
detect, new-building onboarding with timed steps, evaluation (reads `output_plots/evaluation/`), replay.

What was checked against the real models (all five in `models/`): the flags and drift flags produced by the demo's
scoring are identical to the existing `output_plots/<building>_anomaly_report.csv` for all five buildings (the only change
is a larger predict batch size, which changes the errors by about 1e-8). Warm scoring takes about 2.5 s per building and an
injection about 2.5 s; the first score of each model in a process takes about 5 s (graph tracing), which the warm-up pays.

Observations made while building it (not fixed, and not counted in the fault table above):

- **Echo window in `evaluation.py`:** windows ending up to 168 h after an event are ignored, but a window that still
  contains an echoed row (`lag_168`) can end up to 23 h later. In the demo, a 2 h spike produced 22 flagged hours exactly
  7 days later that a clean run does not have. The demo counts that extra day as echo; `evaluation.py` would count those
  hours as false alarms, so its false-alarm rate may be slightly too high.
- **Combined evaluation summary:** `write_combined` lists only buildings that have a `<building>_summary.json`, so after a
  single-building run `evaluation_summary.csv` and `evaluation_report.md` can shrink to that building (seen on 26 Oct after
  the UnivDorm_Mitch run, while the other buildings' events / windows / png files were still there). The demo rebuilds the
  missing rows in memory from those files; `python scripts/evaluation.py --summarize-only` rebuilds the report on disk.
- **Models without a split file:** none of the five models has a `*_split.json` yet, so every building shows the legacy
  80/20 split in the demo until the models are retrained.

## Open faults - what to do

| ID | Next step |
|---|---|
| F18 | Rename to `Model_Architecture.py` and update the import in `Model_Training.py` in one commit (needs a rename on your computer). |
| F20 | Either store the data on shared storage / a download script, or stop committing models and regenerate them from the pipeline. |

## Checklist after the next retraining (confirms F02, F19, F21) and the first LSTM evaluation (F17)

- [ ] Training finishes for all 5 buildings and prints a "Split strategy per building" summary (all 5 should say FULL history).
- [ ] Each `models/<building>_split.json` exists.
- [ ] Loss curves flatten before epoch 100 for all 5 (look at every `output_plots/*_loss.png`).
- [ ] `Model_Testing.py` runs without any `[SKIP]` line and prints the full-history message for each building.
- [ ] Reports have no `test` rows; anomaly rate per building is roughly 1-5%.
- [ ] Annika's error no longer climbs to 10x its summer level in December (compare `01_error_timeline.png`).
- [ ] `python scripts/evaluation.py` finishes for all 5 buildings (about 5 temporary models each; `--quick` is faster); read `output_plots/evaluation/evaluation_report.md`.
- [ ] LSTM beats the PCA stand-in and the "same hour last week" baseline on AUC; recall and false-alarm rate are acceptable per event type.
- [ ] Update the summary table above.

## Future scope

- **Evaluation (F17 follow-ups):** first `--quick` LSTM run (before the realism fix): recall 48-84%, AUC only 0.00-0.05 above the "same hour last week" baseline (Cristina +0.05, Jesus -0.03); flatline weakest (about 40%); false alarms mostly on holiday-like days; threshold stays raised for days after a large spike because of the 168 h lag features. Re-run after the fix for final numbers; then try dropping `lag_168` / `rolling_mean_168`, freezing the threshold while a point is flagged, a rule-based flatline check, and holiday features. Also: tune the threshold from the injected-anomaly results; improve level-shift detection (drift currently suppresses it); add realistic anomaly types (gradual ramp, wrong schedule, missing data); evaluate on a labelled set such as LEAD1.0 or a 2-year dataset (BDG2) for real holdout testing.
- **Onboarding and scoring scripts:** one script to train a new building from a plain CSV (timestamp, energy, timezone), and one to score newly arrived rows using the saved scaler and model plus the previous 7 days as context.
- **Retraining policy:** retrain periodically or when sustained drift is flagged; re-decide the split strategy as a building accumulates more history (for example move from full to holdout after 2+ years).
- **Shared models:** train one model across many buildings of the same type (offices, schools) so a new building works with little history; compare with per-building models. This is where the 507-building dataset would help.
- Explainability: per-feature reconstruction error (SHAP was mentioned in the diagram's next steps).
- Model variants: GRU or VAE autoencoder, window-size and latent-size sweeps; energy relative to the recent level instead of absolute.
- Calendar awareness: holidays, school terms, occupancy schedules; confirm whether Annika's 24-26 December error spike matches holidays.
- Reporting in building-local time. (The flags-per-building dashboard is done: see "Demo UI" above.)
- Fusion with the log-anomaly pipeline and graded (low / medium / high) alerting, as planned in the diagram.
- Engineering: pin the environment, unit tests for windowing and the split policy, a single `run_all` entry point, CI check, rename `Model_Architechture.py`.
- Scale beyond 5 buildings (the metadata lists 507).
