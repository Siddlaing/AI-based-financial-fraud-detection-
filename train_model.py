"""
FraudShield AI — Model Training Pipeline
==========================================
Trains a Random Forest classifier on the Kaggle Credit Card Fraud dataset.

Prerequisites
-------------
1. Download creditcard.csv from:
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Place it in the same folder as this script.
3. (Optional but recommended) Install imbalanced-learn:
       pip install imbalanced-learn

Outputs
-------
fraud_model.pkl     — serialised scikit-learn model
model_config.json   — thresholds, metrics, feature importances
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

# ── FIX 1: check for dataset before doing anything ───────────────────────────
CSV_PATH = "creditcard.csv"
if not os.path.isfile(CSV_PATH):
    sys.exit(
        "\nERROR: 'creditcard.csv' not found in the current directory.\n"
        "Download it from:\n"
        "  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
        "Then re-run this script.\n"
    )

# ── SMOTE: import if available ────────────────────────────────────────────────
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠  imbalanced-learn not installed — skipping SMOTE oversampling.")
    print("   Install with:  pip install imbalanced-learn\n")


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("1. Loading dataset...")
data = pd.read_csv(CSV_PATH)
print(f"   Rows   : {data.shape[0]:,}")
print(f"   Columns: {data.shape[1]}")
fraud_count = int(data["Class"].sum())
print(f"   Fraud  : {fraud_count:,} cases ({data['Class'].mean()*100:.4f}%)")

# FIX 2: validate expected columns
expected = {"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}
missing_cols = expected - set(data.columns)
if missing_cols:
    sys.exit(f"\nERROR: Dataset is missing columns: {missing_cols}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESS
# ─────────────────────────────────────────────────────────────────────────────
print("\n2. Preprocessing...")
X = data.drop("Class", axis=1)
y = data["Class"]

# Stratified split — preserves class ratio in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train set : {len(X_train):,} rows  (fraud={int(y_train.sum()):,})")
print(f"   Test  set : {len(X_test):,}  rows  (fraud={int(y_test.sum()):,})")


# ─────────────────────────────────────────────────────────────────────────────
# 3. SMOTE OVERSAMPLING  (training set only — avoids data leakage)
# ─────────────────────────────────────────────────────────────────────────────
if SMOTE_AVAILABLE:
    print("\n3. Applying SMOTE oversampling on training data...")
    # sampling_strategy=0.1  →  fraud will be 10 % of training data after SMOTE
    sm = SMOTE(sampling_strategy=0.1, random_state=42, k_neighbors=5)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    before_g = int(y_train.value_counts()[0])
    before_f = int(y_train.value_counts()[1])
    after_g  = int(y_train_sm.value_counts()[0])
    after_f  = int(y_train_sm.value_counts()[1])
    print(f"   Before : genuine={before_g:,}  fraud={before_f:,}")
    print(f"   After  : genuine={after_g:,}  fraud={after_f:,}")
else:
    X_train_sm, y_train_sm = X_train, y_train
    print("\n3. Skipping SMOTE — relying on class_weight='balanced' only.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAIN
# ─────────────────────────────────────────────────────────────────────────────
print("\n4. Training Random Forest classifier...")
model = RandomForestClassifier(
    n_estimators=150,        # more trees → better generalisation
    max_depth=12,            # cap depth to avoid overfitting
    min_samples_leaf=2,      # smooths out synthetic SMOTE samples
    random_state=42,
    n_jobs=-1,               # use all CPU cores
    class_weight="balanced", # extra protection against class imbalance
)
model.fit(X_train_sm, y_train_sm)
print("   Training complete.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. EVALUATE  (always on the original non-SMOTE test set)
# ─────────────────────────────────────────────────────────────────────────────
print("\n5. Evaluating on original (non-SMOTE) test set...")
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc     = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
ap      = average_precision_score(y_test, y_proba)

print(f"\n   Accuracy          : {acc*100:.2f}%")
print(f"   ROC AUC           : {roc_auc:.4f}")
print(f"   Avg Precision (AP): {ap:.4f}   ← key metric for imbalanced data")

print("\n   Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Genuine", "Fraud"]))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print("   Confusion Matrix:")
print(f"     True Negatives  (Genuine → Genuine) : {tn:>6,}  — correctly cleared")
print(f"     False Positives (Genuine → Fraud)   : {fp:>6,}  — false alarms")
print(f"     False Negatives (Fraud   → Genuine) : {fn:>6,}  — missed fraud !")
print(f"     True Positives  (Fraud   → Fraud)   : {tp:>6,}  — caught ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 6. OPTIMAL DECISION THRESHOLD  (Youden J — maximises TPR - FPR on ROC curve)
# ─────────────────────────────────────────────────────────────────────────────
print("\n6. Finding optimal decision threshold...")
fpr_arr, tpr_arr, thresholds_roc = roc_curve(y_test, y_proba)
opt_idx       = int(np.argmax(tpr_arr - fpr_arr))
opt_threshold = float(thresholds_roc[opt_idx])

# Suspicious band: transactions between (opt * 0.6) and opt_threshold
# will be labelled "Suspicious" — flagged for manual review
suspicious_threshold = round(opt_threshold * 0.6, 4)

print(f"   Suspicious threshold : {suspicious_threshold:.4f}  (amber zone lower bound)")
print(f"   Fraud threshold      : {opt_threshold:.4f}  (red zone lower bound)")


# ─────────────────────────────────────────────────────────────────────────────
# 7. BEST F1 THRESHOLD  (from Precision-Recall curve)
# ─────────────────────────────────────────────────────────────────────────────
prec_arr, rec_arr, thresholds_pr = precision_recall_curve(y_test, y_proba)
# FIX 3: safe index — precision_recall_curve returns len(thresholds) = len(prec)-1
f1_arr      = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9)
pr_opt_idx  = int(np.argmax(f1_arr))
# Clamp to valid range before indexing thresholds_pr
pr_opt_idx  = min(pr_opt_idx, len(thresholds_pr) - 1)
pr_threshold = float(thresholds_pr[pr_opt_idx])
print(f"\n   Best F1 threshold (PR curve): {pr_threshold:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. FEATURE IMPORTANCES
# ─────────────────────────────────────────────────────────────────────────────
feat_imp = dict(zip(
    model.feature_names_in_,
    model.feature_importances_.round(6).tolist(),
))
top10 = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:10]
print("\n   Top-10 feature importances:")
for fname, fimp in top10:
    bar = "█" * int(fimp * 300)
    print(f"     {fname:<8} {fimp:.4f}  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. SAVE ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n7. Saving model and configuration...")
joblib.dump(model, "fraud_model.pkl")
print("   ✓ fraud_model.pkl saved")

config = {
    "threshold":            round(opt_threshold,    4),
    "suspicious_threshold": suspicious_threshold,
    "pr_threshold":         round(pr_threshold,     4),
    "smote_applied":        SMOTE_AVAILABLE,
    "feature_names":        list(model.feature_names_in_),
    "top_features":         [[k, v] for k, v in top10],
    "model_metrics": {
        "accuracy":         round(acc * 100, 4),
        "roc_auc":          round(roc_auc,   4),
        "avg_precision":    round(ap,         4),
        "true_positives":   int(tp),
        "false_positives":  int(fp),
        "false_negatives":  int(fn),
        "true_negatives":   int(tn),
    },
}
with open("model_config.json", "w") as f:
    json.dump(config, f, indent=2)
print("   ✓ model_config.json saved")

print("\n" + "=" * 60)
print("Training complete!")
print(f"  Thresholds → suspicious ≥ {suspicious_threshold:.4f} | fraud ≥ {opt_threshold:.4f}")
print(f"  ROC AUC   : {roc_auc:.4f}   |   AP : {ap:.4f}")
print("\nNext step:  python app.py")
print("=" * 60)