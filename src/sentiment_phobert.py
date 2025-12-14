from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from tqdm import tqdm


def predict_sentiment(
    input_path="data/news_clean.csv", output_path="data/news_sentiment.csv"
):
    # --- 1. Load model FinBERT ---
    print("🧠 Loading FinBERT model...")
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    # --- 2. Đọc dữ liệu ---
    print(f"📂 Reading data from {input_path} ...")
    df = pd.read_csv(input_path)
    if "Title" not in df.columns:
        raise ValueError("❌ Không tìm thấy cột 'Title' trong file CSV!")

    # --- 3. Dự đoán sentiment ---
    sentiments = []
    for text in tqdm(df["Title"].fillna(""), desc="🔍 Predicting sentiment"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            label = torch.argmax(probs, dim=1).item()

        if label == 0:
            sentiments.append("negative")
        elif label == 1:
            sentiments.append("neutral")
        else:
            sentiments.append("positive")

    # --- 4. Gắn nhãn và lưu ---
    df["Sentiment"] = sentiments
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ Saved sentiment results to {output_path}")
    print(df.head())


if __name__ == "__main__":
    predict_sentiment()
