import pandas as pd
from datetime import timedelta
import numpy as np

# === 1. Đọc dữ liệu ===
news_df = pd.read_csv("data/news_sentiment.csv")
price_df = pd.read_csv("data/googl_price.csv")

# Loại bỏ dòng lỗi ngày trong price (nếu có)
price_df = price_df[price_df["Date"].notna()]

# Chuẩn hóa format ngày
news_df["date"] = pd.to_datetime(news_df["date"]).dt.date
price_df["Date"] = pd.to_datetime(price_df["Date"]).dt.date

# Ép toàn bộ cột giá về dạng số
numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
for col in numeric_cols:
    if col in price_df.columns:
        price_df[col] = pd.to_numeric(price_df[col], errors="coerce")

# Fill giá (ffill)
price_df = price_df.fillna(method="ffill")

# === 2. Chuyển sentiment chữ → số ===
sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
news_df["SentimentScore"] = news_df["Sentiment"].map(sentiment_map)
news_df = news_df.dropna(subset=["SentimentScore"])


# ============================================================================
# 3. FEATURE ENGINEERING TỪ NEWS
# ============================================================================

# 3.1. Số lượng news mỗi ngày
daily_news_count = news_df.groupby("date").size().reset_index(name="NewsCount")

# 3.2. Mean sentiment mỗi ngày
daily_sentiment = (
    news_df.groupby("date")["SentimentScore"].mean().reset_index(name="SentimentMean")
)

# 3.3. Tổng sentiment mỗi ngày
daily_sent_sum = (
    news_df.groupby("date")["SentimentScore"].sum().reset_index(name="SentimentSum")
)

# 3.4. Độ phân tán sentiment (STD) — fillna để tránh NaN
daily_sent_std = (
    news_df.groupby("date")["SentimentScore"].std().reset_index(name="SentimentSTD")
)
daily_sent_std["SentimentSTD"] = daily_sent_std["SentimentSTD"].fillna(0)

# 3.5. % news positive / negative / neutral
daily_sent_ratio = (
    news_df.groupby("date")["Sentiment"]
    .value_counts(normalize=True)
    .unstack()
    .fillna(0)
    .reset_index()
    .rename(
        columns={
            "positive": "PositiveRatio",
            "negative": "NegativeRatio",
            "neutral": "NeutralRatio",
        }
    )
)

# --- Gộp tất cả các feature của news ---
daily_features = (
    daily_news_count.merge(daily_sentiment, on="date", how="left")
    .merge(daily_sent_sum, on="date", how="left")
    .merge(daily_sent_std, on="date", how="left")
    .merge(daily_sent_ratio, on="date", how="left")
)

# ============================================================================
# 4. Ghép news → ngày trading gần nhất
# ============================================================================

trading_dates = set(price_df["Date"])


def map_to_trading_day(news_date, trading_dates):
    """Lùi về ngày giao dịch gần nhất."""
    while news_date not in trading_dates:
        news_date -= timedelta(days=1)
    return news_date


daily_features["mapped_date"] = daily_features["date"].apply(
    lambda d: map_to_trading_day(d, trading_dates)
)

# ============================================================================
# 5. Merge với bảng giá — mỗi ngày chỉ 1 dòng
# ============================================================================

merged = pd.merge(
    price_df, daily_features, left_on="Date", right_on="mapped_date", how="left"
)

# === 6. Fill missing feature (sạch) ===
for col in [
    "NewsCount",
    "SentimentMean",
    "SentimentSum",
    "SentimentSTD",
    "PositiveRatio",
    "NegativeRatio",
    "NeutralRatio",
]:
    merged[col] = merged[col].fillna(0)

# ============================================================================
# 7. FEATURE ENGINEERING TỪ PRICE
# ============================================================================

merged["Return"] = merged["Close"].pct_change()

merged["Volatility_3d"] = merged["Return"].rolling(3).std()
merged["Volatility_7d"] = merged["Return"].rolling(7).std()

merged["MA_3"] = merged["Close"].rolling(3).mean()
merged["MA_7"] = merged["Close"].rolling(7).mean()

merged["Momentum_3"] = merged["Close"] - merged["Close"].shift(3)
merged["Momentum_7"] = merged["Close"] - merged["Close"].shift(7)

merged = merged.fillna(0)

# ============================================================================
# 8. Tạo nhãn Target
# ============================================================================

merged["Target"] = (merged["Close"].shift(-1) > merged["Close"]).astype(int)

# ============================================================================
# 9. Lưu file
# ============================================================================
merged = merged.drop(columns=["mapped_date", "date"])
merged.to_csv("data/merged_data_all.csv", index=False)

print("✅ MERGE + FEATURE ENGINEERING DONE!")
print(merged.head(10))
