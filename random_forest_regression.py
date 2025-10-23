import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# =========================
# 1️⃣ Đọc dữ liệu
# =========================
df = pd.read_csv("quantumn_clean.csv")  # file gồm Date, ticker, adjclose
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)

# =========================
# 2️⃣ Tính daily return
# =========================
df["return"] = df.groupby("ticker")["adjclose"].pct_change()

# =========================
# 3️⃣ Tạo các đặc trưng trễ (lag features)
# =========================
# Ví dụ: dùng 3 ngày trước làm đặc trưng để dự đoán ngày kế tiếp
lags = 3
for i in range(1, lags + 1):
    df[f"lag_{i}"] = df.groupby("ticker")["return"].shift(i)

# Bỏ NaN
df = df.dropna().reset_index(drop=True)

# =========================
# 4️⃣ Dự đoán cho từng ticker
# =========================
expected_next_day_returns = []

for ticker, group in df.groupby("ticker"):
    X = group[[f"lag_{i}" for i in range(1, lags + 1)]]
    y = group["return"]

    # Chia tập train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Huấn luyện mô hình
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Dự đoán tỷ suất lợi nhuận ngày tiếp theo
    X_last = X.iloc[-1:].values
    next_day_return_pred = model.predict(X_last)[0]

    # Đánh giá (tùy chọn)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    expected_next_day_returns.append({
        "ticker": ticker,
        "expected_next_day_return": next_day_return_pred,
        "rmse": rmse
    })

# =========================
# 5️⃣ Xuất kết quả ra CSV
# =========================
expected_df = pd.DataFrame(expected_next_day_returns)
expected_df["expected_annualized_return"] = expected_df["expected_next_day_return"] * 252  # ~252 ngày giao dịch/năm
expected_df.to_csv("expected-returns/expected_return_random_forest_regression.csv", index=False)

print("✅ Done! Kết quả được lưu trong expected_returns.csv")
print(expected_df)
