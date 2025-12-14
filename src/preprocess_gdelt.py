# =====================================
# 📄 preprocess_gdelt.py (fixed)
# Làm sạch dữ liệu GDELT, tự động xử lý khi thiếu cột
# =====================================

import pandas as pd
import os
import re


def preprocess_gdelt(
    input_path="data/gdelt_news.csv", output_path="data/news_clean.csv"
):
    print("🧹 Starting GDELT preprocessing...")

    # Đọc file
    df = pd.read_csv(input_path, low_memory=False)

    print(f"🔍 Columns detected: {list(df.columns)}")

    # --- 1️⃣ Chỉ giữ cột có liên quan (nếu tồn tại) ---
    possible_cols = [
        "DATE",
        "Date",
        "DocumentIdentifier",
        "DocumentIdentifierURL",
        "MobileURL",
        "SourceCommonName",
        "Language",
        "Tone",
        "Themes",
        "Title",
    ]
    keep_cols = [c for c in possible_cols if c in df.columns]
    df = df[keep_cols]

    # --- 2️⃣ Nếu không có cột Language thì bỏ qua ---
    if "Language" in df.columns:
        df = df[df["Language"].str.lower() == "english"]

    # --- 3️⃣ Loại trùng URL (nếu có) ---
    url_col = (
        "DocumentIdentifier" if "DocumentIdentifier" in df.columns else "MobileURL"
    )
    df = df.drop_duplicates(subset=[url_col])

    # --- 4️⃣ Loại bỏ tin rỗng hoặc không có tiêu đề ---
    if "Title" in df.columns:
        df = df.dropna(subset=["Title"])
    else:
        print("⚠️ Warning: No 'Title' column found!")

    # --- 5️⃣ Giữ lại tin có chứa Google hoặc Alphabet ---
    df = df[
        df.apply(
            lambda row: bool(
                re.search(
                    r"\b(Google|Alphabet)\b", str(row.get("Title", "")), re.IGNORECASE
                )
            ),
            axis=1,
        )
    ]

    # --- 6️⃣ Đổi tên cột cho dễ đọc ---
    rename_map = {
        "DocumentIdentifier": "url",
        "MobileURL": "url",
        "SourceCommonName": "source",
        "Tone": "tone",
        "DATE": "date",
        "Date": "date",
    }
    df = df.rename(columns=rename_map)

    # --- 7️⃣ Chuẩn hóa dữ liệu ---
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values(by="date", ascending=False)

    # --- 8️⃣ Lưu kết quả ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"✅ Cleaned {len(df)} articles → {output_path}")
    print("\n🧠 Preview:")
    print(df.head(5))


if __name__ == "__main__":
    preprocess_gdelt()
