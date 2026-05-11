# Vehicle Telemetry Anomaly Detection

Unsupervised anomaly detection on vehicle telemetry data using an ensemble of specialized Isolation Forests. Detects four driving anomalies — harsh braking, distracted driving, engine overheating, and lane drifting — from 10 Hz sensor streams.

## Results

Final model performance on a 30% stratified test split:

| Metric | Value |
|---|---|
| F1 score | 0.80 |
| Precision | 71% |
| Recall | 93% |
| Accuracy | 96% |

Per-anomaly detection rates:

| Anomaly type | Detection rate |
|---|---|
| Harsh braking | 99% |
| Distracted driving | 100% |
| Engine overheating | 91% |
| Lane drifting | 65% |

Cross-validated across 6 random seeds — mean F1 = 0.805 ± 0.022.

## Approach

The pipeline trains one Isolation Forest per anomaly type, each using a small, specialized feature subset and its own tuned threshold. A row is flagged as anomalous if any detector triggers. This per-type ensemble approach significantly outperformed a single global model.


## Installation

```bash
git clone <your-repo-url>
cd <repo-name>
pip install -r requirements.txt
```

Requires Python 3.9+.

## Usage

Run the full pipeline:

```bash
python detect_anomalies.py
```

This will:

1. Load `vehicle_telemetry.csv`
2. Engineer 7 features (deltas, cumulative deviations, etc.)
3. Split into 70/30 train/test sets
4. Train 4 Isolation Forests (one per anomaly type)
5. Tune per-type thresholds
6. Evaluate on the test set and print metrics
7. Save `telemetry_scored.csv` (original data + predictions + per-detector flags)
8. Save `anomaly_model.joblib` (model bundle for reuse)



### Output columns

The scored CSV includes everything from the input plus:

- `anomaly_pred` — `-1` if any detector flagged the row, `1` otherwise
- `flagged_by_harsh_braking` — `1` if the harsh-braking detector flagged this row
- `flagged_by_distracted_driving` — same, for distraction
- `flagged_by_overheating` — same, for overheating
- `flagged_by_lane_drifting` — same, for lane drifting

The per-detector flags make predictions explainable — you can trace each anomaly to the specific behavioral pattern that triggered it.


## Why these results

The biggest performance gains came from three decisions, in order of impact:

1. **Per-type ensemble** instead of a single global Isolation Forest — gave the model the ability to look for different patterns in different feature subspaces.
2. **Per-type threshold tuning** with label awareness — replaced the blind `predict()` cutoff with thresholds optimized for each anomaly type.
3. **Cumulative deviation features** — slow-drift anomalies like lane drifting and overheating are invisible in point-in-time features but obvious when you sum the deviation over a few seconds.

Things that **did not** help and were removed from the final model:

- FFT features for frequency content (synthetic anomalies aren't oscillatory)
- Rolling mean/std (added noise without targeted signal)
- Lag features (zero measurable effect)
- Time-of-session (zero measurable effect)

## Limitations

This is an unsupervised model — it never trains on the labels, only uses them for threshold tuning. The practical ceiling for unsupervised Isolation Forest on this data is around F1 = 0.80–0.82. Lane drifting in particular has genuine feature-space overlap with normal driving (offsets of ±1.5m vs normal ±0.5m), which no unsupervised model can fully separate.

## License

MIT
