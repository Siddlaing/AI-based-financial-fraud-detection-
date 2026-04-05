import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve)
import joblib
import json

print("1. Loading the dataset...")
data = pd.read_csv('creditcard.csv')
print(f"   Dataset shape: {data.shape}")
print(f"   Fraud cases: {data['Class'].sum()} ({data['Class'].mean()*100:.3f}%)")

print("2. Preprocessing data...")
X = data.drop('Class', axis=1)
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("3. Training the Random Forest Model...")
# FIX 1: class_weight='balanced' — handles the 0.17% fraud imbalance
# FIX 2: n_estimators=100 — more reliable predictions than 50
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'   # <-- KEY FIX for imbalanced dataset
)
model.fit(X_train, y_train)

print("4. Evaluating the Model...")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"\nAccuracy : {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"ROC AUC  : {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Genuine', 'Fraud']))

# FIX 3: Print confusion matrix so you can see false positives/negatives
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix:")
print(f"  True Negatives  (Genuine correctly identified): {tn}")
print(f"  False Positives (Genuine flagged as Fraud):     {fp}")
print(f"  False Negatives (Fraud missed):                 {fn}")
print(f"  True Positives  (Fraud correctly caught):       {tp}")

# FIX 4: Find the optimal threshold using ROC curve
# Default 0.5 is wrong for imbalanced data — this finds the better threshold
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"\nOptimal Decision Threshold: {optimal_threshold:.4f}")
print("(Saved to model_config.json — use this in app.py instead of 0.5)")

print("\n5. Saving model and config...")
joblib.dump(model, 'fraud_model.pkl')
print("   fraud_model.pkl saved!")

# FIX 5: Save feature importances + threshold for the dashboard
feature_importance = dict(zip(
    model.feature_names_in_,
    model.feature_importances_.round(6).tolist()
))
config = {
    "threshold": round(float(optimal_threshold), 4),
    "feature_names": list(model.feature_names_in_),
    "top_features": sorted(feature_importance.items(),
                           key=lambda x: x[1], reverse=True)[:10],
    "model_metrics": {
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn)
    }
}
with open('model_config.json', 'w') as f:
    json.dump(config, f, indent=2)
print("   model_config.json saved (threshold + feature importances)!")
print("\nDone! Run app.py to start the server.")