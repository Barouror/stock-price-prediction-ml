# GOOGLE STOCK SENTIMENT AI

📈 GOOGLE_STOCK_AI

Dự đoán xu hướng giá cổ phiếu Google (GOOGL) bằng Machine Learning kết hợp Sentiment Analysis từ tin tức

🎯 Mục tiêu đề tài

Đồ án nhằm xây dựng một pipeline hoàn chỉnh để:

Thu thập dữ liệu giá cổ phiếu Google

Crawl và xử lý tin tức (GDELT)

Phân tích cảm xúc (sentiment analysis)

Kết hợp dữ liệu giá + tin tức

Trích xuất đặc trưng (feature engineering)

Huấn luyện mô hình Machine Learning (XGBoost, LightGBM)

Đánh giá và dự đoán xu hướng giá cổ phiếu

🧠 Công nghệ sử dụng

Python 3.10

Pandas, NumPy

XGBoost, LightGBM

Scikit-learn

Matplotlib

GDELT API

PhoBERT / Sentiment model

GPU support (CUDA – nếu có)

📂 Cấu trúc thư mục dự án

Cấu trúc hiện tại của project được tổ chức theo pipeline xử lý dữ liệu → huấn luyện → đánh giá → lưu model.

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
│   ├── google_xgb_model.pkl
│   ├── xgb_model.pkl
│   ├── xgb_model.json
│   ├── xgb_model_overfit_protected.json
│   ├── xgb_scaler.pkl
│   ├── xgb_timeseries_scaler.pkl
│   ├── feature_columns.pkl
│   └── feature_importance.png
│
├── model_lgbm/                 # Model LightGBM
│   ├── best_model.pkl
│   └── lgbm_stock_model.pkl
│
├── output/                     # Output trung gian / feature importance
│   └── feature_importance.csv
│
├── reports/                    # Báo cáo, hình ảnh, kết quả đánh giá
│
├── src/                        # Source code chính
│   ├── crawl_gdelt.py          # Crawl tin tức từ GDELT
│   ├── download_price.py       # Tải dữ liệu giá cổ phiếu
│   ├── preprocess_gdelt.py     # Làm sạch dữ liệu news
│   ├── preprocess_sentiment.py # Chuẩn bị dữ liệu sentiment
│   ├── sentiment_phobert.py    # Phân tích cảm xúc
│   ├── merge_data.py           # Merge news + price
│   ├── clean_merge_data.py     # Clean & aggregate theo ngày
│   ├── features_engineering.py # Trích xuất feature kỹ thuật
│   ├── train_xgboost.py        # Huấn luyện XGBoost
│   ├── train_model.py          # Pipeline huấn luyện
│   ├── evaluate_xgboost.py     # Đánh giá mô hình
│   └── predict_price.py        # Dự đoán giá / xu hướng
│
├── .vscode/
│   └── settings.json
│
├── README.md
└── requirements.txt

🗺️ Pipeline xử lý dữ liệu

Thu thập dữ liệu

Giá cổ phiếu Google

Tin tức từ GDELT

Tiền xử lý

Làm sạch dữ liệu news

Chuẩn hóa sentiment

Feature Engineering

Technical indicators (MA, RSI, MACD, volatility, momentum,…)

Sentiment-based features (mean, sum, ratio, rolling)

Huấn luyện mô hình

XGBoost (GPU support)

LightGBM (baseline)

Đánh giá & dự đoán

Accuracy, Precision, Recall, F1-score

Confusion Matrix

Feature Importance

▶️ Cách chạy project
1️⃣ Tạo môi trường
conda create -n google_stock python=3.10
conda activate google_stock
pip install -r requirements.txt

2️⃣ Chạy pipeline
python src/download_price.py
python src/crawl_gdelt.py
python src/preprocess_gdelt.py
python src/preprocess_sentiment.py
python src/merge_data.py
python src/clean_merge_data.py
python src/features_engineering.py
python src/train_xgboost.py
python src/evaluate_xgboost.py

📊 Kết quả

Mô hình XGBoost đạt độ chính xác cao trên tập test

Feature quan trọng nhất:

Return

Volatility

Momentum

Sentiment-based features

Lưu ý: Dataset có kích thước nhỏ (theo ngày), cần cẩn trọng với overfitting.

📌 Ghi chú

Các file trung gian (news_clean.csv, merged_data_clean.csv, …) được giữ lại để minh họa pipeline, đúng chuẩn đồ án học thuật.

Đường dẫn trong code sử dụng relative path dựa trên vị trí file .py, đảm bảo chạy được trên máy khác.

👤 Tác giả

Phan Gia Bảo
Đồ án môn Trí tuệ Nhân tạo / Machine Learning
