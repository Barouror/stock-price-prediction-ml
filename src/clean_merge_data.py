import pandas as pd
import os

# Lấy đường dẫn tuyệt đối tới thư mục src (nơi file .py đang chạy)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn đến file CSV trong folder data
csv_path = os.path.join(BASE_DIR, "..", "data", "merged_data_all.csv")

df = pd.read_csv(csv_path)

# Bước 1: Chuyển Date về dạng datetime
df["Date"] = pd.to_datetime(df["Date"])

# Bước 2: Group theo Date
df_clean = (
    df.groupby("Date")
    .agg(
        {
            "Close": "first",
            "High": "first",
            "Low": "first",
            "Open": "first",
            "Volume": "first",
            # News features
            "NewsCount": "sum",
            "SentimentMean": "mean",
            "SentimentSum": "sum",
            "SentimentSTD": "mean",
            "NegativeRatio": "mean",
            "NeutralRatio": "mean",
            "PositiveRatio": "mean",
            # Market features
            "Return": "first",
            "Volatility_3d": "first",
            "Volatility_7d": "first",
            "MA_3": "first",
            "MA_7": "first",
            "Momentum_3": "first",
            "Momentum_7": "first",
            "Target": "first",
        }
    )
    .reset_index()
)

# Lưu file sạch lại vào folder data
output_path = os.path.join(BASE_DIR, "..", "data", "merged_data_clean.csv")
df_clean.to_csv(output_path, index=False)

print("Done! File saved to:", output_path)
