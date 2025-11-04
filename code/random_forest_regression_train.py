import os
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor  # GPU
gpu_available = False

# ====== Cấu hình ======
INPUT_FILE = "train.csv"  # File đầu vào
MODEL_DIR = "models"             # Thư mục lưu mô hình
os.makedirs(MODEL_DIR, exist_ok=True)

# ====== Đọc dữ liệu ======
df = pd.read_csv(INPUT_FILE)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['ticker', 'Date'])

# ====== Xử lý theo từng ticker ======
for ticker, data in df.groupby('ticker'):
    print(f"🔹 Training model for {ticker}...")

    data['return'] = data['adjclose'].pct_change()
    data.dropna(inplace=True)

    # Tạo đặc trưng (lags)
    for lag in range(1, 4):
        data[f'return_lag{lag}'] = data['return'].shift(lag)
    data.dropna(inplace=True)

    X = data[[f'return_lag{i}' for i in range(1, 4)]]
    y = data['return']

    if len(data) < 20:
        print(f"⚠️ Skipping {ticker} — not enough data ({len(data)} rows)")
        continue

    # Huấn luyện mô hình
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42,
    )
    model.fit(X, y)

    # Lưu mô hình
    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.pkl")
    joblib.dump(model, model_path)

    print(f"✅ Saved model: {model_path}")

print("\n🎯 Training complete. All models saved to 'models/' directory.")
