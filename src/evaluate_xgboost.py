import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

# ---------------------------------------------------------
# PATH
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "data", "feature_dataset.csv")

MODEL_DIR = os.path.join(BASE_DIR, "..", "models_xgboost")
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_timeseries.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "xgb_timeseries_scaler.pkl")
FEATURE_COL_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")

OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# EVALUATE
# ---------------------------------------------------------
def evaluate_model(test_ratio=0.2):
    print("🔍 Loading data...")
    df = pd.read_csv(DATA_PATH)

    feature_cols = joblib.load(FEATURE_COL_PATH)

    X = df[feature_cols]
    y = df["Label"]

    # -------------------------------
    # TIME SERIES SPLIT
    # -------------------------------
    split_idx = int(len(df) * (1 - test_ratio))

    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    print(f"📌 Train size: {split_idx} | Test size: {len(X_test)}")

    # -------------------------------
    # LOAD SCALER
    # -------------------------------
    scaler = joblib.load(SCALER_PATH)
    X_test_scaled = scaler.transform(X_test)

    # -------------------------------
    # LOAD MODEL
    # -------------------------------
    model = joblib.load(MODEL_PATH)

    # -------------------------------
    # PREDICT
    # -------------------------------
    probs = model.predict_proba(X_test_scaled)[:, 1]
    preds = (probs >= 0.5).astype(int)

    # -------------------------------
    # METRICS
    # -------------------------------
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print("\n==============================")
    print("📊 MODEL EVALUATION (TEST SET)")
    print("==============================")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-score  : {f1:.4f}")

    print("\n📌 Classification Report:")
    print(classification_report(y_test, preds, digits=4))

    # =====================================================
    # CONFUSION MATRIX (TEST)
    # =====================================================
    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Confusion Matrix (Test Set)")
    plt.colorbar()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_test.png")
    plt.savefig(cm_path)
    plt.close()

    print(f"📁 Saved Confusion Matrix → {cm_path}")

    # =====================================================
    # ROC CURVE (TEST)
    # =====================================================
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test Set)")
    plt.legend()
    plt.tight_layout()

    roc_path = os.path.join(OUTPUT_DIR, "roc_curve_test.png")
    plt.savefig(roc_path)
    plt.close()

    print(f"📁 Saved ROC Curve → {roc_path}")
    print("==============================\n")


if __name__ == "__main__":
    evaluate_model()
