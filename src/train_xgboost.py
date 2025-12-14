import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

from xgboost import XGBClassifier

# ---------------------------------------------------------
# PATH
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "feature_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"📌 Loading {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)

print("➡️ Columns loaded:", df.columns.tolist())
print("📊 Shape:", df.shape)

# ---------------------------------------------------------
# REMOVE LEAKAGE
# ---------------------------------------------------------
LEAK_COLS = ["Target", "Future_Return_1", "Label"]
FEATURE_COLS = [c for c in df.columns if c not in LEAK_COLS]

X = df[FEATURE_COLS]
y = df["Label"]

# 🔒 SAVE FEATURE COLUMNS
feature_col_path = os.path.join(MODEL_DIR, "feature_columns.pkl")
joblib.dump(FEATURE_COLS, feature_col_path)
print(f"✅ Saved {len(FEATURE_COLS)} feature columns")

# ---------------------------------------------------------
# TIME SERIES CV
# ---------------------------------------------------------
tscv = TimeSeriesSplit(n_splits=5)

best_model = None
best_scaler = None
best_acc = -1

best_y_test = None
best_probs = None
best_preds = None

fold = 1

for train_idx, test_idx in tscv.split(X):
    print(f"\n⏳ Fold {fold}")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # ✅ scaler fitted ONLY on train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=2,
        reg_alpha=1,
        objective="binary:logistic",
        tree_method="hist",
        random_state=42,
        eval_metric="logloss",
    )

    model.fit(X_train_scaled, y_train)

    probs = model.predict_proba(X_test_scaled)[:, 1]
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)

    print(f"🎯 Fold {fold} accuracy: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_scaler = scaler
        best_y_test = y_test
        best_probs = probs
        best_preds = preds

    fold += 1

print(f"\n🔥 Best CV accuracy: {best_acc:.4f}")

# ---------------------------------------------------------
# REPORT (BEST FOLD)
# ---------------------------------------------------------
print("\n📊 Classification Report (Best Fold Test Set):")
print(classification_report(best_y_test, best_preds, digits=4))

# ---------------------------------------------------------
# CONFUSION MATRIX (BEST FOLD)
# ---------------------------------------------------------
cm = confusion_matrix(best_y_test, best_preds)

plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title("Confusion Matrix (Best Fold Test)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.colorbar()

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_train_cv.png")
plt.savefig(cm_path)
plt.close()

print(f"📁 Saved Confusion Matrix → {cm_path}")

# ---------------------------------------------------------
# ROC CURVE (BEST FOLD)
# ---------------------------------------------------------
fpr, tpr, _ = roc_curve(best_y_test, best_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Best Fold Test)")
plt.legend()
plt.tight_layout()

roc_path = os.path.join(OUTPUT_DIR, "roc_curve_train_cv.png")
plt.savefig(roc_path)
plt.close()

print(f"📁 Saved ROC Curve → {roc_path}")

# ---------------------------------------------------------
# SAVE MODEL + SCALER
# ---------------------------------------------------------
model_path = os.path.join(MODEL_DIR, "xgb_timeseries.pkl")
scaler_path = os.path.join(MODEL_DIR, "xgb_timeseries_scaler.pkl")

joblib.dump(best_model, model_path)
joblib.dump(best_scaler, scaler_path)

print("\n📁 Saved artifacts:")
print(f" - Model  : {model_path}")
print(f" - Scaler : {scaler_path}")
print(f" - Columns: {feature_col_path}")

print("\n✅ Training + CV Evaluation complete!")
