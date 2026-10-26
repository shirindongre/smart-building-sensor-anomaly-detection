# Evaluation report (injected anomalies)

Backend: `lstm`. Event types: spike, dropout, flatline, level_shift. Seed 7.

| Building | Mode | Events | Recall | Median delay (h) | False alarms | Precision | AUC (error) | AUC baseline |
|---|---|---|---|---|---|---|---|---|
| UnivDorm_Mitch | blocked (5 folds, max 30 epochs) | 19/24 | 79% | 3 | 2.7% | 81% | 0.88 | 0.86 |
| UnivLab_Brenna | blocked (5 folds, max 30 epochs) | 19/25 | 76% | 2 | 3.3% | 75% | 0.92 | 0.85 |

Recall by event type:

| Building | spike | dropout | flatline | level_shift |
|---|---|---|---|---|
| UnivDorm_Mitch | 6/7 | 5/5 | 1/5 | 7/7 |
| UnivLab_Brenna | 4/7 | 5/5 | 3/6 | 7/7 |

Injected / inert events per building: UnivDorm_Mitch 25 / 1, UnivLab_Brenna 25 / 0. Inert = changed (almost) nothing, so they are listed in the events CSV but not scored.

False alarm = a flagged window with no injected event nearby. Event recall counts an event as found if at least one of its windows is flagged as an anomaly (drift flags are counted separately in the CSV).
Held-out windows up to 7 days after an event are ignored in every metric (the lag features echo the event).
