# predict_expected_returns_ann.py
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
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# === Parameters ===
INPUT_CSV = "quantumn_clean.csv"
OUTPUT_CSV = "expected-returns/expected_return_ann.csv"
MIN_DAYS = 60
LAGS = 5
ROLL_WINDOWS = [5, 10, 21]
TRADING_DAYS = 252
EPOCHS = 100
BATCH_SIZE = 16
VERBOSE = 0
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# === Helper functions ===

def prepare_features(df_ticker):
    df = df_ticker.copy().sort_values("Date").reset_index(drop=True)
    df["return"] = df["adjclose"].pct_change()

    # Lags
    for lag in range(1, LAGS + 1):
        df[f"lag_{lag}"] = df["return"].shift(lag)

    # Rolling stats
    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = df["return"].rolling(window=w).mean()
        df[f"roll_std_{w}"] = df["return"].rolling(window=w).std()

    # Momentum
    for w in ROLL_WINDOWS:
        df[f"mom_{w}"] = df["adjclose"] / df["adjclose"].shift(w) - 1

    # Date features
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month

    # Target: next-day return
    df["target_next_return"] = df["return"].shift(-1)

    feature_cols = [
        c for c in df.columns
        if c.startswith("lag_") or c.startswith("roll_") or c.startswith("mom_")
    ] + ["dayofweek", "month"]

    df = df.dropna(subset=feature_cols + ["target_next_return"]).reset_index(drop=True)
    return df, feature_cols

def build_ann(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def annualize_return(daily_return):
    return (1.0 + daily_return) ** TRADING_DAYS - 1.0

def train_predict_ticker(df_feat, feature_cols, ticker):
    X = df_feat[feature_cols].values
    y = df_feat["target_next_return"].values

    if len(y) < 30:
        return None, None, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train ANN model
    model = build_ann(X_scaled.shape[1])
    es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X_scaled, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE, callbacks=[es])

    # Predict for last available day
    X_last = X_scaled[-1].reshape(1, -1)
    pred_next = float(model.predict(X_last, verbose=0)[0][0])

    return pred_next, scaler, model

def main(input_csv=INPUT_CSV, output_csv=OUTPUT_CSV):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"{input_csv} not found")

    df = pd.read_csv(input_csv)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', 'ticker', 'adjclose']].dropna()
    df = df.sort_values(['ticker', 'Date']).reset_index(drop=True)

    results = []

    for ticker in df['ticker'].unique():
        df_t = df[df['ticker'] == ticker].copy()
        if len(df_t) < MIN_DAYS:
            print(f"Skip {ticker}: only {len(df_t)} rows (<{MIN_DAYS})")
            continue

        df_feat, feature_cols = prepare_features(df_t)
        if df_feat.empty:
            print(f"No usable data for {ticker}")
            continue

        pred_next, scaler, model = train_predict_ticker(df_feat, feature_cols, ticker)
        if pred_next is None:
            print(f"Failed to train {ticker}")
            continue

        ann_return = annualize_return(pred_next)
        last_date = df_feat['Date'].max()

        results.append({
            "ticker": ticker,
            "last_date": last_date.strftime("%Y-%m-%d"),
            "expected_next_day_return": pred_next,
            "expected_annualized_return": ann_return
        })

        print(f"[{ticker}] next-day={pred_next:.5f}, annualized={ann_return:.5f}")

    # Save all results
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved expected returns to {output_csv}")
    return res_df

if __name__ == "__main__":
    df_out = main()
    print(df_out.head())
