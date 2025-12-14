# GOOGLE STOCK SENTIMENT AI

📈 **GOOGLE_STOCK_AI**
Dự đoán xu hướng giá cổ phiếu **Google (GOOGL)** bằng **Machine Learning** kết hợp **Sentiment Analysis** từ tin tức.
---

Link Github: https://github.com/Barouror/stock-price-prediction-ml

Link report: https://docs.google.com/document/d/1JCZnQrZV_0i0pjwkaRj5WVA4El9jzdQqJUeI3qvWwLs/edit?usp=sharing

Link slide: 	https://www.canva.com/design/DAG7aEkvzfQ/C7Nw8SsxubG0INmZFs2-nQ/edit?utm_content=DAG7aEkvzfQ&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

Link thuyết trình: 	
https://youtu.be/Y8TenXPGybM

---

## 🎯 Mục tiêu đề tài

Đề tài xây dựng một **pipeline Machine Learning hoàn chỉnh** nhằm:

* Thu thập dữ liệu giá cổ phiếu Google (GOOGL)
* Crawl và xử lý tin tức tài chính từ **GDELT**
* Phân tích cảm xúc (Sentiment Analysis)
* Kết hợp dữ liệu **giá + tin tức**
* Trích xuất đặc trưng (Feature Engineering)
* Huấn luyện mô hình Machine Learning
* Đánh giá và dự đoán xu hướng giá cổ phiếu

---

## 🧠 Công nghệ sử dụng

* **Python 3.10**
* **Pandas**, **NumPy**
* **XGBoost**, **LightGBM**
* **Scikit-learn**
* **Matplotlib**
* **GDELT API**
* **PhoBERT** (Sentiment model)
* GPU support (CUDA – nếu có)

---

## 📂 Cấu trúc thư mục dự án

Cấu trúc project được tổ chức theo pipeline: **thu thập dữ liệu → tiền xử lý → huấn luyện → đánh giá → dự đoán**.

```
GOOGLE_STOCK_AI/
│
├── data/                       # Dữ liệu CSV qua từng bước xử lý
│   ├── googl_price.csv
│   ├── gdelt_news.csv
│   ├── news_clean.csv
│   ├── news_reduced.csv
│   ├── news_sentiment.csv
│   ├── merged_data_all.csv
│   ├── merged_data_clean.csv
│   └── feature_dataset.csv
│
├── models_xgboost/             # Model XGBoost & scaler
│   ├── xgb_model.pkl
│   ├── xgb_model.json
│   ├── xgb_scaler.pkl
│   ├── feature_columns.pkl
│   └── feature_importance.png
│
├── model_lgbm/                 # Model LightGBM (baseline)
│   └── lgbm_stock_model.pkl
│
├── output/                     # Output trung gian / feature importance
│   └── feature_importance.csv
│
├── reports/                    # Báo cáo & hình ảnh đánh giá
│
├── src/                        # Source code chính
│   ├── download_price.py       # Tải dữ liệu giá cổ phiếu
│   ├── crawl_gdelt.py          # Crawl tin tức từ GDELT
│   ├── preprocess_gdelt.py     # Làm sạch dữ liệu news
│   ├── preprocess_sentiment.py # Chuẩn bị dữ liệu sentiment
│   ├── sentiment_phobert.py    # Phân tích cảm xúc
│   ├── merge_data.py           # Merge news + price
│   ├── clean_merge_data.py     # Clean & aggregate theo ngày
│   ├── features_engineering.py # Trích xuất feature kỹ thuật
│   ├── train_xgboost.py        # Huấn luyện XGBoost
│   ├── evaluate_xgboost.py     # Đánh giá mô hình
│   └── predict_price.py        # Dự đoán xu hướng giá
│
├── README.md
└── requirements.txt
```

---

## 🗺️ Pipeline xử lý dữ liệu

### 1. Thu thập dữ liệu

* Dữ liệu giá cổ phiếu Google
* Tin tức tài chính từ GDELT

### 2. Tiền xử lý

* Làm sạch dữ liệu tin tức
* Chuẩn hóa và tổng hợp sentiment theo ngày

### 3. Feature Engineering

* Chỉ báo kỹ thuật: MA, RSI, MACD, Volatility, Momentum, Lag features
* Đặc trưng sentiment: mean, sum, ratio, rolling statistics

### 4. Huấn luyện mô hình

* **XGBoost** (mô hình chính)
* **LightGBM** (baseline so sánh)

### 5. Đánh giá & dự đoán

* Accuracy, Precision, Recall, F1-score
* Confusion Matrix
* Feature Importance

---

## ▶️ Cách chạy project

### 1️⃣ Tạo môi trường

```bash
conda create -n google_stock python=3.10
conda activate google_stock
pip install -r requirements.txt
```

### 2️⃣ Chạy pipeline

```bash
python src/download_price.py
python src/crawl_gdelt.py
python src/preprocess_gdelt.py
python src/preprocess_sentiment.py
python src/merge_data.py
python src/clean_merge_data.py
python src/features_engineering.py
python src/train_xgboost.py
python src/evaluate_xgboost.py
```

---

## 📊 Kết quả

* Mô hình **XGBoost** đạt hiệu năng tốt trên tập test
* Các feature quan trọng nhất:

  * Return
  * Volatility
  * Momentum
  * Sentiment-based features

⚠️ **Lưu ý**: Dataset có kích thước nhỏ (theo ngày), cần cẩn trọng với hiện tượng overfitting.

---

## 📌 Ghi chú

* Các file trung gian được giữ lại để minh họa pipeline, đúng chuẩn đồ án học thuật
* Code sử dụng **relative path**, đảm bảo có thể chạy trên máy khác

---

## 👤 Tác giả

**Phan Gia Bảo**
Đồ án mô
