import pandas as pd
import numpy as np
from cuml.ensemble import RandomForestRegressor  # ⚡ GPU Random Forest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ===== 1️⃣ Đọc dữ liệu =====
df = pd.read_csv("quantumn_clean.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['ticker', 'Date'])

# ===== 2️⃣ Tính lợi suất hàng ngày =====
df['return'] = df.groupby('ticker')['adjclose'].pct_change()
df = df.dropna(subset=['return'])

# ===== 3️⃣ Tạo các đặc trưng lag =====
def create_lag_features(group, lags=5):
    for i in range(1, lags + 1):
        group[f'return_lag_{i}'] = group['return'].shift(i)
    return group.dropna()

df = df.groupby('ticker', group_keys=False).apply(create_lag_features)

# ===== 4️⃣ Hàm backtest theo Rolling Window =====
def rolling_backtest(data, window=200):
    X_cols = [c for c in data.columns if 'return_lag_' in c]
    X, y = data[X_cols], data['return']

    preds, reals = [], []

    for i in range(window, len(data)):
        X_train, y_train = X.iloc[i - window:i], y.iloc[i - window:i]
        X_test, y_test = X.iloc[i:i + 1], y.iloc[i]

        # ⚡ Mô hình chạy trên GPU (cuML)
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            random_state=42,
            n_streams=4,        # số luồng GPU song song
        )
        model.fit(X_train, y_train)
        y_pred = float(model.predict(X_test)[0])

        preds.append(y_pred)
        reals.append(y_test)

    if len(preds) == 0:
        return None

    mae = mean_absolute_error(reals, preds)
    rmse = np.sqrt(mean_squared_error(reals, preds))
    r2 = r2_score(reals, preds)
    direction_acc = np.mean(np.sign(reals) == np.sign(preds))
    expected_return = np.mean(preds)

    return mae, rmse, r2, direction_acc, expected_return

# ===== 5️⃣ Backtest cho tất cả các mã =====
results = []
for ticker, g in df.groupby('ticker'):
    if len(g) < 300:
        continue

    res = rolling_backtest(g, window=200)
    if res is None:
        continue

    mae, rmse, r2, direction_acc, expected_return = res
    results.append({
        "Ticker": ticker,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "DirectionalAccuracy": direction_acc,
        "ExpectedReturn_Daily": expected_return
    })

# ===== 6️⃣ Thêm Expected Return Annualized =====
TRADING_DAYS = 252
result_df = pd.DataFrame(results)
result_df['ExpectedReturn_Annualized'] = (1 + result_df['ExpectedReturn_Daily']) ** TRADING_DAYS - 1

# ===== 7️⃣ Xuất kết quả ra file CSV =====
result_df.to_csv("backtest_results.csv", index=False)
print("✅ Đã xuất file: backtest_results.csv")

expected_returns = result_df[['Ticker', 'ExpectedReturn_Daily', 'ExpectedReturn_Annualized']]
expected_returns.to_csv("expected_returns.csv", index=False)
print("✅ Đã xuất file: expected_returns.csv")

# ===== 8️⃣ In tóm tắt =====
print("\n📊 BACKTEST SUMMARY:")
print(result_df.sort_values(by="ExpectedReturn_Daily", ascending=False))
