# src/download_price.py
import yfinance as yf


def download_price():
    print("📈 Downloading GOOGL stock price data...")
    data = yf.download("GOOGL", start="2024-10-10", end="2025-10-10")
    data.reset_index(inplace=True)
    data.to_csv("data/googl_price.csv", index=False)
    print(f"✅ Saved {len(data)} rows → data/googl_price.csv")
    print(data.head())


if __name__ == "__main__":
    download_price()
