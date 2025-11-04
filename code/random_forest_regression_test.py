import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ====== Cấu hình ======
MODEL_DIR = "models"
TEST_FILE = "test.csv"
PRED_OUTPUT = "predicted_expected_return.csv"
BACKTEST_OUTPUT = "backtest_results.csv"

# ====== Đọc dữ liệu test ======
df = pd.read_csv(TEST_FILE)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['ticker', 'Date'])

# ====== Chuẩn bị kết quả lưu ======
predicted_results = []
backtest_results = []

# ====== Xử lý theo từng ticker ======
for ticker, data in df.groupby('ticker'):
    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.pkl")
    if not os.path.exists(model_path):
        print(f"⚠️ Model for {ticker} not found, skipping.")
        continue

    print(f"🔹 Predicting for {ticker}...")

    # Tính daily return
    data['return'] = data['adjclose'].pct_change()

    # Tạo lag features
    for lag in range(1, 4):
        data[f'return_lag{lag}'] = data['return'].shift(lag)
    data.dropna(inplace=True)

    if len(data) == 0:
        print(f"⚠️ Not enough data to predict for {ticker}")
        continue

    X = data[[f'return_lag{i}' for i in range(1, 4)]].astype(np.float32)
    y_true = data['return']

    # Load model
    model = joblib.load(model_path)

    # Predict
    y_pred = model.predict(X)

    # Tính expected return (daily)
    daily_expected_return = np.mean(y_pred)

    # Ghi kết quả dự đoán chi tiết
    temp_df = pd.DataFrame({
        "Date": data['Date'],
        "Ticker": ticker,
        "ActualReturn": y_true,
        "PredictedReturn": y_pred
    })
    predicted_results.append(temp_df)

    # Backtest metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]

    backtest_results.append({
        "Ticker": ticker,
        "Samples": len(data),
        "DailyExpectedReturn": daily_expected_return,
        "MAE": mae,
        "MSE": mse,
        "R2": r2,
        "Correlation": corr
    })

    print(f"✅ {ticker}: Done ({len(data)} samples)")

# ====== Lưu kết quả dự đoán chi tiết ======
if predicted_results:
    predicted_df = pd.concat(predicted_results, ignore_index=True)
    predicted_df.to_csv(PRED_OUTPUT, index=False)
    print(f"\n📄 Saved detailed predictions to {PRED_OUTPUT}")

# ====== Lưu kết quả backtest ======
if backtest_results:
    backtest_df = pd.DataFrame(backtest_results)
    backtest_df['AnnualizedExpectedReturn'] = (1 + backtest_df['DailyExpectedReturn']) ** 252 - 1
    backtest_df.to_csv(BACKTEST_OUTPUT, index=False)
    print(f"📄 Saved backtest results to {BACKTEST_OUTPUT}")

print("\n🎯 Inference and backtest complete!")
