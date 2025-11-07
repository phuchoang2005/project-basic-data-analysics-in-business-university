import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# --- 1. Cấu hình Đường dẫn ---
MODELS_DIR = 'models'
OUTPUT_CSV_FILE = 'expected_returns_prediction.csv'
DATA_FILE = 'quantumn_clean.csv'

# --- 2. Đọc dữ liệu lịch sử ---
df_history = pd.read_csv(DATA_FILE)
df_history['Date'] = pd.to_datetime(df_history['Date'])
current_date = df_history['Date'].max()

# --- 3. Hàm tạo features KHỚP VỚI MÔ HÌNH ---
def create_current_features(df_history, ticker, current_date):
    df_ticker = df_history[df_history['ticker'] == ticker].sort_values('Date')
    df_ticker['return'] = df_ticker['adjclose'].pct_change()
    df_ticker.dropna(inplace=True)

    # Lấy 3 ngày gần nhất trước current_date
    df_recent = df_ticker[df_ticker['Date'] <= current_date].tail(3)

    # Nếu không đủ dữ liệu, bỏ qua
    if len(df_recent) < 3:
        return None

    # Các feature giống lúc huấn luyện: return_lag1, lag2, lag3
    # return_lag1 = ngày gần nhất, lag2 = ngày trước đó, lag3 = ngày xa hơn
    lags = df_recent['return'].values[::-1]  # đảo ngược để lag1 là ngày gần nhất
    X_current = np.array([lags])  # shape (1,3)
    return X_current

# --- 4. Dự đoán ---
results = []

for model_file in os.listdir(MODELS_DIR):
    if not model_file.endswith('_model.pkl'):
        continue

    ticker = model_file.replace('_model.pkl', '')
    model_path = os.path.join(MODELS_DIR, model_file)

    print(f"🔹 Đang xử lý mô hình cho: {ticker}")

    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Lỗi khi tải {ticker}: {e}")
        continue

    X_current = create_current_features(df_history, ticker, current_date)

    if X_current is None or np.isnan(X_current).any():
        print(f"⚠️ Không đủ dữ liệu cho {ticker}, bỏ qua.")
        expected_return = np.nan
    else:
        try:
            expected_return = model.predict(X_current)[0]
        except Exception as e:
            print(f"❌ Lỗi khi dự đoán {ticker}: {e}")
            expected_return = np.nan

    results.append({
        'Ticker': ticker,
        'Expected_Return': expected_return,
        'Date_of_Prediction': current_date.strftime('%Y-%m-%d')
    })

# --- 5. Lưu kết quả ---
df_results = pd.DataFrame(results).sort_values('Ticker').reset_index(drop=True)
df_results.to_csv(OUTPUT_CSV_FILE, index=False)

print("\n✅ Dự đoán hoàn tất. File kết quả:", OUTPUT_CSV_FILE)
print(df_results.head())
