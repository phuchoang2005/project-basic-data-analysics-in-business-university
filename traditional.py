import pandas as pd
import numpy as np

# --- 1️⃣ Đọc dữ liệu ---
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Chuẩn hóa thời gian & xử lý trùng
train_df['Date'] = pd.to_datetime(train_df['Date'])
train_df = (train_df.groupby(['Date', 'ticker'], as_index=False).agg({'adjclose': 'mean'}))

test_df['Date'] = pd.to_datetime(test_df['Date'])
test_df = (test_df.groupby(['Date', 'ticker'], as_index=False).agg({'adjclose': 'mean'}))

train_price = train_df.pivot(index='Date', columns='ticker', values='adjclose').sort_index()
test_price  = test_df.pivot(index='Date', columns='ticker', values='adjclose').sort_index()

# --- 2️⃣ Tính daily returns ---
returns_train = train_price.pct_change().dropna()
returns_test  = test_price.pct_change().dropna()

# --- 3️⃣ Làm sạch dữ liệu ---
returns_train = returns_train.loc[:, returns_train.std() > 0]
returns_test  = returns_test[returns_train.columns.intersection(returns_test.columns)]
returns_test = returns_test.loc[:, returns_test.std() > 0]
returns_train = returns_train[returns_test.columns]

# --- 4️⃣ Tính μ và Σ ---
mean_returns = returns_train.mean()                    # vector lợi suất kỳ vọng
cov_matrix = returns_train.cov()                       # ma trận hiệp phương sai
cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6       # tránh singular

# --- 5️⃣ Tối ưu trọng số Markowitz ---
inv_cov = np.linalg.pinv(cov_matrix)
raw_weights = inv_cov.dot(mean_returns)
weights = raw_weights / np.sum(raw_weights)

# --- 6️⃣ Tính daily expected return (trên tập train) ---
daily_expected_returns = returns_train.dot(weights)

pd.DataFrame({
    'Date': daily_expected_returns.index,
    'daily_expected_return': daily_expected_returns.values
}).to_csv("daily_expected_returns.csv", index=False)

print("✅ Saved: daily_expected_returns.csv")

# --- 7️⃣ Backtest trên tập test ---
portfolio_returns_test = returns_test.dot(weights)

# Các chỉ số hiệu suất
mean_return = portfolio_returns_test.mean()
volatility = portfolio_returns_test.std()
sharpe = mean_return / volatility
cum_return = (1 + portfolio_returns_test).cumprod().iloc[-1] - 1

print("\n📊 Backtest Results:")
print(f"  Mean daily return: {mean_return:.6f}")
print(f"  Volatility: {volatility:.6f}")
print(f"  Sharpe ratio: {sharpe:.3f}")
print(f"  Cumulative return: {cum_return:.2%}")

# --- 8️⃣ Xuất file kết quả backtest ---
pd.DataFrame({
    'Date': portfolio_returns_test.index,
    'portfolio_return': portfolio_returns_test.values
}).to_csv("markowitz_backtest.csv", index=False)

print("✅ Saved: markowitz_backtest.csv")
