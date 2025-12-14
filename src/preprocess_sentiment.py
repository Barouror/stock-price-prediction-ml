# src/preprocess_reduce.py
import pandas as pd
import re, os
from langdetect import detect, LangDetectException

# --- Cấu hình ---
input_path = "data/news_clean.csv"
output_path = "data/news_reduced.csv"

# --- Đọc dữ liệu ---
if not os.path.exists(input_path):
    raise FileNotFoundError(f"Không tìm thấy file {input_path}")
df = pd.read_csv(input_path)

print(f"📦 Đọc {len(df)} dòng ban đầu")


# --- Làm sạch cơ bản (nếu chưa) ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-ZÀ-ỹ\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


if "clean_text" not in df.columns:
    possible_cols = [c for c in df.columns if c.lower() in ["content", "title", "text"]]
    if not possible_cols:
        raise KeyError("Không tìm thấy cột văn bản ('title', 'content', 'text')")
    df["clean_text"] = df[possible_cols[0]].apply(clean_text)

# --- 1️⃣ Bỏ trùng ---
df = df.drop_duplicates(subset="clean_text")
print(f"🧹 Sau khi bỏ trùng: {len(df)} dòng")

# --- 2️⃣ Lọc bài đủ dài ---
df = df[df["clean_text"].str.len() > 40]
print(f"✂️ Sau khi bỏ bài ngắn: {len(df)} dòng")


# --- 3️⃣ Giữ lại bài tiếng Anh ---
def is_english(text):
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


df = df[df["clean_text"].apply(is_english)]
print(f"🇬🇧 Sau khi lọc tiếng Anh: {len(df)} dòng")

# --- 4️⃣ Giới hạn thời gian (nếu có cột Date) ---
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"] >= "2025-03-01"]  # chỉ giữ tin 6 tháng gần nhất
    print(f"🕓 Sau khi lọc theo thời gian: {len(df)} dòng")

# --- 5️⃣ Lấy mẫu ngẫu nhiên còn khoảng 2000 dòng ---
if len(df) > 2000:
    df = df.sample(n=2000, random_state=42)
    print(f"🎯 Lấy mẫu ngẫu nhiên 2000 dòng cuối cùng")

# --- 6️⃣ Lưu ---
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"✅ Đã lưu file: {output_path}")
