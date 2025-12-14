import os
import joblib
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# PATH
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "data", "feature_dataset.csv")

MODEL_DIR = os.path.join(BASE_DIR, "..", "models_xgboost")
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_timeseries.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "xgb_timeseries_scaler.pkl")
FEATURE_COL_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------
# PREDICT NEXT DAY DIRECTION
# ---------------------------------------------------------
def predict_next_day():
    print("🔍 Loading data...")
    df = pd.read_csv(DATA_PATH)

    # ===== Load feature columns (GUARANTEED SAME AS TRAIN) =====
    if not os.path.exists(FEATURE_COL_PATH):
        raise FileNotFoundError("❌ feature_columns.pkl not found. Train model first.")

    feature_cols = joblib.load(FEATURE_COL_PATH)

    # ===== Take last row =====
    X_last = df[feature_cols].iloc[-1:].copy()

    print(f"📌 Using {len(feature_cols)} features")
    print("\n📌 Last row used for prediction:")
    print(X_last)

    # ===== Load scaler =====
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("❌ Scaler not found.")

    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(X_last)

    # ===== Load model =====
    model = load_model()

    # ===== Predict =====
    prob = model.predict_proba(X_scaled)[0, 1]
    pred = int(prob >= 0.5)

    label = "📈 TĂNG" if pred == 1 else "📉 GIẢM"

    print("\n==============================")
    print("🔮 DỰ ĐOÁN NGÀY TIẾP THEO")
    print("==============================")
    print(f"Kết quả        : {label}")
    print(f"Xác suất tăng  : {prob:.4f}")
    print(f"Xác suất giảm : {1 - prob:.4f}")
    print("==============================\n")


if __name__ == "__main__":
    predict_next_day()
