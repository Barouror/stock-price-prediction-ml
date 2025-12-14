# ==========================================
# features_engineering.py (ĐÃ TỐI ƯU)
# Tạo features cho mô hình dự đoán tăng/giảm ngày mai
# ==========================================

import pandas as pd
import numpy as np


# ---------- RSI ----------
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ---------- ADD FEATURES ----------
def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ===== Basic checks =====
    needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Thiếu cột bắt buộc: {c}")

    # ===== 1. Returns =====
    df["Return_1"] = df["Close"].pct_change()
    df["Return_3"] = df["Close"].pct_change(3)

    # ===== 2. Lag basic =====
    df["Lag_Close_1"] = df["Close"].shift(1)
    df["Lag_Close_3"] = df["Close"].shift(3)
    df["Lag_Return_1"] = df["Return_1"].shift(1)

    # ===== 3. Candle =====
    df["Body"] = df["Close"] - df["Open"]
    df["Range"] = (df["High"] - df["Low"]).replace(0, np.nan)
    df["Body_Range"] = df["Body"] / df["Range"]

    # ===== 4. Moving Average (ngắn) =====
    df["MA3"] = df["Close"].rolling(3).mean().shift(1)
    df["MA5"] = df["Close"].rolling(5).mean().shift(1)

    # ===== 5. Volatility =====
    df["Volatility_3"] = df["Return_1"].rolling(3).std().shift(1)
    df["Volatility_5"] = df["Return_1"].rolling(5).std().shift(1)

    # ===== 6. RSI =====
    df["RSI14"] = compute_rsi(df["Close"], 14).shift(1)

    # ===== 7. Momentum =====
    df["Momentum_3"] = df["Close"].diff(3).shift(1)
    df["Momentum_7"] = df["Close"].diff(7).shift(1)

    # ===== 8. Sentiment (QUAN TRỌNG) =====
    if "SentimentMean" in df.columns:
        df["Sentiment_Lag_1"] = df["SentimentMean"].shift(1)
        df["Sentiment_Lag_3"] = df["SentimentMean"].rolling(3).mean().shift(1)

    if "NewsCount" in df.columns:
        df["News_Lag_1"] = df["NewsCount"].shift(1)
        df["News_Lag_3"] = df["NewsCount"].rolling(3).mean().shift(1)

    # ===== CLEAN =====
    df = df.dropna().reset_index(drop=True)

    return df


# ---------- LABEL ----------
def add_labels(df: pd.DataFrame, up_th=0.002, down_th=-0.002):
    df = df.copy()

    fut = df["Close"].shift(-1) / df["Close"] - 1
    df["Future_Return_1"] = fut

    df["Label"] = np.where(fut > up_th, 1, np.where(fut < down_th, 0, np.nan))

    df = df.dropna().reset_index(drop=True)
    df["Label"] = df["Label"].astype(int)

    return df


# ---------- FULL PIPELINE ----------
def build_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = add_technical_features(raw_df)
    df = add_labels(df)

    # Xóa NA ở các cột numeric
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df.dropna(subset=numeric_cols).reset_index(drop=True)

    # Convert numeric columns only
    df[numeric_cols] = df[numeric_cols].astype(float)

    # ❗ Bỏ cột Date vì là string
    if "Date" in df.columns:
        df = df.drop(columns=["Date"])

    return df


# ---------- MAIN ----------
if __name__ == "__main__":
    print("📌 Loading data/merged_data_clean.csv ...")
    raw = pd.read_csv("data/merged_data_clean.csv")

    print("⚙️ Building final dataset ...")
    final = build_dataset(raw)

    final.to_csv("data/feature_dataset.csv", index=False)
    print("✅ Saved data/feature_dataset.csv | shape:", final.shape)
