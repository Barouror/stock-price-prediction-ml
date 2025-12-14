import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

print("📌 Loading data/feature_dataset.csv ...")
df = pd.read_csv("data/feature_dataset.csv")

print("➡️ Columns loaded:", list(df.columns))

# =========================================================
# 🔥 REMOVE LEAKAGE COLUMNS
# =========================================================
LEAKAGE_COLS = [
    "Future_Return_1",  # chứa return của ngày tương lai → leakage
    "Target",  # cũng chứa thông tin tương lai
]

# Chỉ drop các cột có tồn tại
cols_to_drop = [c for c in LEAKAGE_COLS if c in df.columns]

# =========================================================
# 🔥 BUILD X, y
# =========================================================
if "Label" not in df.columns:
    raise ValueError("❌ ERROR: Column 'Label' not found! Check feature generation.")

y = df["Label"]
X = df.drop(columns=cols_to_drop + ["Label"])

print(f"📌 Features shape: {X.shape}")
print("📌 Target distribution:")
print(y.value_counts())

# =========================================================
# 🔥 TRAIN/TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # time-series → KHÔNG shuffle
)

# =========================================================
# 🔥 SCALE FEATURES
# =========================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# 🔥 TRAIN MODEL
# =========================================================
print("🚀 Training RandomForestClassifier ...")
model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

# =========================================================
# 🔥 EVALUATE
# =========================================================
pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, pred)

print(f"\n🎯 Accuracy: {acc:.4f}\n")
print("📊 Classification Report:")
print(classification_report(y_test, pred))

# =========================================================
# 🔥 SAVE ARTIFACTS
# =========================================================
joblib.dump(model, "models/stock_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\n✅ Training complete!")
print("📁 Saved model → models/stock_model.pkl")
print("📁 Saved scaler → models/scaler.pkl")
