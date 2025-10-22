# requirements:
# pip install pandas numpy scikit-learn joblib

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import joblib

# ---------- CONFIG ----------
CSV_PATH = "quantumn_clean.csv"       # đổi thành đường dẫn file của bạn
OUTPUT_CSV = "expected-returns/expected_returns_decision_tree.csv"
MIN_ROWS_PER_TICKER = 60     # nếu ít hơn -> dùng historical mean fallback
LAG_WINDOWS = [1,2,3,5,10]   # tạo các lag returns
ROLL_WINDOW = 5              # rolling features
ANNUAL_TRADING_DAYS = 252
DT_PARAMS = {"max_depth": 6, "min_samples_leaf": 10, "random_state": 42}
# ----------------------------

# 1. Load
df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)

# 2. Compute returns (simple)
df["prev_adjclose"] = df.groupby("ticker")["adjclose"].shift(1)
df["ret"] = df["adjclose"] / df["prev_adjclose"] - 1
# Drop first row per ticker (NaN ret)
df = df.dropna(subset=["ret"]).reset_index(drop=True)

# 3. Create next-day target (what we want to predict)
df["next_ret"] = df.groupby("ticker")["ret"].shift(-1)
df = df.dropna(subset=["next_ret"]).reset_index(drop=True)

# 4. Feature engineering: lag returns, rolling mean/std, calendar features
for L in LAG_WINDOWS:
    df[f"lag_{L}"] = df.groupby("ticker")["ret"].shift(L)

df["rolling_mean_ret"] = df.groupby("ticker")["ret"].rolling(ROLL_WINDOW).mean().reset_index(level=0, drop=True)
df["rolling_std_ret"] = df.groupby("ticker")["ret"].rolling(ROLL_WINDOW).std().reset_index(level=0, drop=True)

# calendar features
df["dayofweek"] = df["Date"].dt.dayofweek
df["month"] = df["Date"].dt.month

# Drop rows with NaNs in features (due to lags)
feature_cols = [c for c in df.columns if c.startswith("lag_")] + ["rolling_mean_ret", "rolling_std_ret", "dayofweek", "month"]
df = df.dropna(subset=feature_cols + ["next_ret"]).reset_index(drop=True)

# 5. Option A: train one DecisionTree per ticker (recommended)
results = []
models = {}
for ticker, g in df.groupby("ticker"):
    g = g.sort_values("Date").reset_index(drop=True)
    n = len(g)
    if n < MIN_ROWS_PER_TICKER:
        # fallback: use historical mean next_ret
        expected_next = g["next_ret"].mean()
        expected_annual = (1 + expected_next) ** ANNUAL_TRADING_DAYS - 1
        results.append({"ticker": ticker,
                        "expected_next_day_return": expected_next,
                        "expected_annualized_return": expected_annual,
                        "method": "historical_mean",
                        "train_rows": n})
        continue

    X = g[feature_cols].values
    y = g["next_ret"].values

    # time-based train-test split: train on all but last day, predict last day
    # but to evaluate generalization, we can do a small time-series CV
    tscv = TimeSeriesSplit(n_splits=5)
    # simple: train final model on all available except last row, predict last row
    X_train, y_train = X[:-1], y[:-1]
    X_pred = X[-1].reshape(1, -1)   # features of the most recent date (predict next day)
    model = DecisionTreeRegressor(**DT_PARAMS)
    model.fit(X_train, y_train)

    # optionally evaluate via CV (compute mean RMSE across splits)
    rmses = []
    for train_idx, test_idx in tscv.split(X_train):
        m = DecisionTreeRegressor(**DT_PARAMS)
        m.fit(X_train[train_idx], y_train[train_idx])
        pred = m.predict(X_train[test_idx])
        rmses.append(np.sqrt(mean_squared_error(y_train[test_idx], pred)))
    cv_rmse = np.mean(rmses) if rmses else np.nan

    pred_next = model.predict(X_pred)[0]  # this is expected next-day simple return

    # annualize (geometric)
    expected_annual = (1 + pred_next) ** ANNUAL_TRADING_DAYS - 1

    results.append({"ticker": ticker,
                    "expected_next_day_return": float(pred_next),
                    "expected_annualized_return": float(expected_annual),
                    "method": "decision_tree",
                    "cv_rmse": float(cv_rmse),
                    "train_rows": n})
    models[ticker] = model

# 6. Save results
res_df = pd.DataFrame(results).sort_values("ticker").reset_index(drop=True)
res_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved expected returns to {OUTPUT_CSV}")

# 8. Quick display
print(res_df.head(30))
