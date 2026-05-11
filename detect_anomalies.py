
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib, warnings
warnings.filterwarnings('ignore')

# ── Configuration ──
INPUT_CSV = '/home/salma/Documents/telemetry/vehicle_telemetry.csv'
OUTPUT_CSV = '/home/salma/Documents/telemetry/telemetry_scored_v3.csv'
MODEL_OUTPUT = '/home/salma/Documents/telemetry/anomaly_model_v3.joblib'
TEST_SIZE = 0.3
RANDOM_STATE = 42

# Per-type detector hyperparameters
TYPE_N_ESTIMATORS = 200
TYPE_CONTAMINATION = 0.05
TYPE_PRECISION_FLOOR = 0.30


def engineer_features(df):
    df = df.copy()

    # Rate of change (deltas) — critical for harsh braking
    df['speed_delta'] = df['speed_kmh'].diff().fillna(0)
    df['temp_delta'] = df['engine_temp_c'].diff().fillna(0)
    df['steering_delta'] = df['steering_angle_deg'].diff().abs().fillna(0)

    # Interaction — captures high-speed harsh stops
    df['speed_x_brake'] = df['speed_kmh'] * df['brake_pressure_pct']

    # Cumulative deviations — critical for slow-drift anomalies
    df['temp_cumdev_10s'] = (df['engine_temp_c'] - 70).clip(lower=0).rolling(window=100, min_periods=1).sum()
    df['attention_low_count_5s'] = (df['driver_attention'] < 0.5).rolling(window=50, min_periods=1).sum()

    # Threshold-count for lane (catches sustained drift better than cumulative sum)
    df['lane_above_05_count'] = (df['lane_offset_m'].abs() > 0.5).rolling(window=50, min_periods=1).sum()

    return df


TYPE_FEATURES = {
    'harsh_braking': ['brake_pressure_pct', 'acceleration_ms2', 'speed_delta', 'speed_x_brake'],
    'distracted_driving': ['driver_attention', 'attention_low_count_5s', 'steering_delta'],
    'overheating': ['engine_temp_c', 'temp_cumdev_10s', 'temp_delta'],
    'lane_drifting': ['lane_offset_m', 'lane_above_05_count'],
}


def find_type_threshold(model, X_train_scaled, y_type_train, precision_floor=0.3):
    """Find threshold optimized for one specific anomaly type."""
    train_scores = -model.decision_function(X_train_scaled)
    best_thresh, best_score = None, -1

    for pct in np.arange(85, 99, 0.25):
        thresh = np.percentile(train_scores, pct)
        preds = np.where(train_scores >= thresh, -1, 1)
        tp = ((preds == -1) & (y_type_train == -1)).sum()
        fp = ((preds == -1) & (y_type_train == 1)).sum()
        fn = ((preds == 1) & (y_type_train == -1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        if prec < precision_floor:
            continue
        if prec + rec > 0:
            f1 = 2 * prec * rec / (prec + rec)
            if f1 > best_score:
                best_score, best_thresh = f1, thresh

    if best_thresh is None:
        best_thresh = np.percentile(train_scores, 95)
    return best_thresh


# ── Load & engineer features ──
print("Loading data...")
df = pd.read_csv(INPUT_CSV, parse_dates=['timestamp'])
print(f"Loaded {len(df)} rows")

print("Engineering features...")
df = engineer_features(df)
all_features = [c for c in df.columns if c not in ['timestamp', 'label']]
print(f"Total features: {len(all_features)}")

y_true = np.where(df['label'] == 'normal', 1, -1)

# ── Train/test split ──
_, _, y_train, y_test, idx_train, idx_test = train_test_split(
    df.index.values, y_true, df.index,
    test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df['label'])
df_test = df.iloc[idx_test]

print(f"\nTrain: {len(idx_train)} rows ({(y_train==-1).sum()} anomalies)")
print(f"Test:  {len(idx_test)} rows ({(y_test==-1).sum()} anomalies)")

# ── Train one detector per anomaly type ──
print("\nTraining per-type detectors...")
type_models = {}

for atype, type_feats in TYPE_FEATURES.items():
    X_train_t = df[type_feats].iloc[idx_train].values

    s = StandardScaler()
    X_train_t_s = s.fit_transform(X_train_t)

    m = IsolationForest(
        n_estimators=TYPE_N_ESTIMATORS, max_samples=1.0,
        contamination=TYPE_CONTAMINATION, random_state=RANDOM_STATE, n_jobs=-1)
    m.fit(X_train_t_s)

    y_type_train = np.where(df['label'].iloc[idx_train].values == atype, -1, 1)
    thresh = find_type_threshold(m, X_train_t_s, y_type_train, TYPE_PRECISION_FLOOR)

    type_models[atype] = {
        'model': m, 'scaler': s, 'features': type_feats, 'threshold': thresh
    }
    print(f"  {atype}: {len(type_feats)} features, threshold={thresh:.4f}")


def predict_ensemble(df_subset, indices):
    """A row is anomaly if ANY type detector flags it."""
    n = len(indices)
    combined_preds = np.ones(n)
    per_type_flags = {}

    for atype, td in type_models.items():
        X_t = df.loc[indices, td['features']].values
        X_t_s = td['scaler'].transform(X_t)
        scores = -td['model'].decision_function(X_t_s)
        flags = scores >= td['threshold']
        per_type_flags[atype] = flags
        combined_preds[flags] = -1

    return combined_preds, per_type_flags


# ── Evaluate on test set ──
test_preds, per_type_test = predict_ensemble(df, idx_test)
cm = confusion_matrix(y_test, test_preds, labels=[-1, 1])
tp, fn, fp, tn = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

test_f1 = f1_score(y_test, test_preds, pos_label=-1)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
accuracy = (tp + tn) / (tp + fn + fp + tn)

print(f"\n{'='*50}")
print(f"TEST SET RESULTS")
print(f"{'='*50}")
print(f"F1 score:   {test_f1:.4f}")
print(f"Precision:  {precision:.4f}  (false positives: {fp})")
print(f"Recall:     {recall:.4f}  (missed anomalies: {fn})")
print(f"Accuracy:   {accuracy:.4f}")
print(f"Confusion:  TP={tp} FN={fn} FP={fp} TN={tn}")

print(f"\nPer-type detection rates:")
for t in ['harsh_braking', 'distracted_driving', 'overheating', 'lane_drifting']:
    mask = df_test['label'] == t
    r = (test_preds[mask.values] == -1).sum() / mask.sum() if mask.sum() > 0 else 0
    print(f"  {t}: {r:.1%}")

print(f"\nDetector activations:")
for atype, flags in per_type_test.items():
    n_flagged = int(flags.sum())
    n_correct = int(((flags) & (df_test['label'].values == atype)).sum())
    print(f"  {atype} detector: {n_flagged} flagged, {n_correct} were that type")

# ── Score full dataset ──
print("\nScoring full dataset...")
full_preds, full_per_type = predict_ensemble(df, df.index)
df['anomaly_pred'] = full_preds

for atype, flags in full_per_type.items():
    df[f'flagged_by_{atype}'] = flags.astype(int)

df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved scored data: {OUTPUT_CSV}")

joblib.dump({
    'type_models': type_models,
    'type_features': TYPE_FEATURES,
}, MODEL_OUTPUT)
print(f"Saved model: {MODEL_OUTPUT}")
print("Done.")
