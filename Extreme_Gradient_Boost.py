# predict_expected_returns_xgboost.py
"""
Input CSV format:
Date,ticker,adjclose
2020-01-01,AAPL,300.35
2020-01-01,MSFT,160.62
...

Output CSV:
ticker,last_date,expected_next_day_return,expected_annualized_return
"""

import pandas as pd
import numpy as np
import os
from datetime import timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import joblib

# Params
INPUT_CSV = "quantumn_clean.csv"            # đổi tên nếu cần
OUTPUT_CSV = "expected-returns/expected_returns_xgboost.csv"
MIN_DAYS = 60                       # số ngày tối thiểu để mô hình 1 ticker
LAGS = 5                             # số lag returns
ROLL_WINDOWS = [5, 10, 21]          # rolling windows
N_ESTIMATORS = 200
RANDOM_STATE = 42
TRADING_DAYS = 252

def prepare_features(df_ticker):
    """
    df_ticker: DataFrame with Date (datetime) và adjclose, sorted ascending
    Returns DataFrame with features and target (next-day return)
    """
    df = df_ticker.copy().sort_values("Date").reset_index(drop=True)
    # compute simple returns
    df["return"] = df["adjclose"].pct_change()  # NaN at first
    # lags
    for lag in range(1, LAGS + 1):
        df[f"lag_{lag}"] = df["return"].shift(lag)
    # rolling stats on returns (skip NaN)
    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = df["return"].rolling(window=w).mean()
        df[f"roll_std_{w}"] = df["return"].rolling(window=w).std()
    # technical-like features: momentum (close / close_n - 1)
    for w in ROLL_WINDOWS:
        df[f"mom_{w}"] = df["adjclose"] / df["adjclose"].shift(w) - 1
    # date features
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    # target: next-day return
    df["target_next_return"] = df["return"].shift(-1)
    # drop rows with NaN in features or target
    feature_cols = [c for c in df.columns if c.startswith("lag_") or c.startswith("roll_") or c.startswith("mom_")] + ["dayofweek", "month"]
    df = df.dropna(subset=feature_cols + ["target_next_return"]).reset_index(drop=True)
    return df, feature_cols

def train_predict_ticker(df_ticker, feature_cols):
    """
    Train XGB on historical rows and predict expected next day return
    Returns predicted value (float) and optionally model
    """
    # use all available data except last row (we can't use target of last row)
    X = df_ticker[feature_cols].values
    y = df_ticker["target_next_return"].values

    # if not enough data, skip
    if len(y) < 20:
        return None, None

    # time-series split for quick validation (not necessary but useful)
    tscv = TimeSeriesSplit(n_splits=3)
    preds_val = []
    rmses = []
    # We will train on full history and predict the last row's features (the most recent available features correspond to prediction for next day)
    last_feature = X[-1].reshape(1, -1)
    # Train final model on all data
    model = XGBRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, verbosity=0)
    model.fit(X, y)
    pred_next = model.predict(last_feature)[0]
    return pred_next, model

def annualize_return(daily_return):
    # geometric annualization: (1 + r_daily)^252 - 1
    return (1.0 + daily_return) ** TRADING_DAYS - 1.0

def main(input_csv=INPUT_CSV, output_csv=OUTPUT_CSV):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"{input_csv} not found")

    df = pd.read_csv(input_csv)
    # ensure Date column
    df['Date'] = pd.to_datetime(df['Date'])
    # keep needed columns
    df = df[['Date', 'ticker', 'adjclose']].dropna()
    # sort
    df = df.sort_values(['ticker', 'Date']).reset_index(drop=True)

    results = []
    models_dir = "models_xgb"
    os.makedirs(models_dir, exist_ok=True)

    tickers = df['ticker'].unique()
    for t in tickers:
        df_t = df[df['ticker'] == t].copy()
        if len(df_t) < MIN_DAYS:
            # skip tickers with too little history
            print(f"Skip {t}: only {len(df_t)} rows (<{MIN_DAYS})")
            continue

        df_feat, feature_cols = prepare_features(df_t)
        if df_feat.empty:
            print(f"No usable rows after feature creation for {t}; skip")
            continue

        pred_next, model = train_predict_ticker(df_feat, feature_cols)
        if pred_next is None:
            print(f"Not enough data to train model for {t}")
            continue

        # compute annualized
        ann = annualize_return(pred_next)
        last_date = df_feat['Date'].max()  # last date used for features => prediction is next trading day after this date

        results.append({
            "ticker": t,
            "last_date": last_date.strftime("%Y-%m-%d"),
            "expected_next_day_return": float(pred_next),
            "expected_annualized_return": float(ann)
        })

        # save model for future use
        model_path = os.path.join(models_dir, f"xgb_{t}.joblib")
        joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)

    # export results
    res_df = pd.DataFrame(results).sort_values("ticker").reset_index(drop=True)
    res_df.to_csv(output_csv, index=False)
    print(f"Saved expected returns to {output_csv}")
    return res_df

if __name__ == "__main__":
    out = main()
    print(out.head())
