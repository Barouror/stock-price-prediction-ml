# =====================================
# 📄 crawl_gdelt.py
# Crawl tin tức về Google từ GDELT
# =====================================

import pandas as pd
import requests
import io
import os
from datetime import datetime, timedelta


def crawl_gdelt(keyword="Google", days=365, output_path="../data/gdelt_news.csv"):
    print(f"🚀 Crawling GDELT for keyword: {keyword} (last {days} days)")
    base_url = "http://api.gdeltproject.org/api/v2/doc/doc"

    # Tạo danh sách ngày chia nhỏ để tránh timeout
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    delta = timedelta(days=7)  # mỗi lần crawl 7 ngày

    all_data = []

    while start_date < end_date:
        next_date = min(start_date + delta, end_date)
        start_str = start_date.strftime("%Y%m%d%H%M%S")
        end_str = next_date.strftime("%Y%m%d%H%M%S")

        params = {
            "query": f'"{keyword}"',
            "mode": "ArtList",
            "maxrecords": 250,
            "format": "CSV",
            "startdatetime": start_str,
            "enddatetime": end_str,
        }

        try:
            r = requests.get(base_url, params=params, timeout=30)
            if r.status_code == 200 and len(r.text) > 0:
                df = pd.read_csv(io.StringIO(r.text))
                if not df.empty:
                    all_data.append(df)
                    print(f"✅ {len(df)} articles from {start_str[:8]} → {end_str[:8]}")
            else:
                print(f"⚠️ No data from {start_str[:8]} → {end_str[:8]}")
        except Exception as e:
            print(f"❌ Error for {start_str[:8]}: {e}")

        start_date = next_date

    # Hợp nhất toàn bộ dữ liệu
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"\n🎯 Done! Saved {len(result)} articles → {output_path}")
    else:
        print("\n❌ No articles found!")


if __name__ == "__main__":
    crawl_gdelt(keyword="Google", days=365)
