# save as predict_expected_returns_dt.py
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

def prepare_features(df, n_lags=[1, 2], windows=[5, 10, 20]):
    """
    Chuẩn bị đặc trưng cho từng mã cổ phiếu.
    Dữ liệu đầu vào cần có: Date, ticker, adjclose
    """
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['ticker', 'Date'])

    frames = []
    for t, g in df.groupby('ticker'):
        g = g.copy().reset_index(drop=True)
        g['logp'] = np.log(g['adjclose'])
        g['logret'] = g['logp'].diff()

        # Tạo lag features
        for lag in n_lags:
            g[f'lag_{lag}'] = g['logret'].shift(lag)

        # Tạo rolling features
        for w in windows:
            g[f'roll_mean_{w}'] = g['logret'].rolling(w).mean().shift(1)
            g[f'roll_std_{w}'] = g['logret'].rolling(w).std().shift(1)
            g[f'mom_{w}'] = g['logp'] - g['logp'].shift(w)

        # Target: lợi nhuận ngày kế tiếp
        g['target_next_logret'] = g['logret'].shift(-1)

        features = [c for c in g.columns if c.startswith(('lag_', 'roll_', 'mom_'))]
        g = g.dropna(subset=features + ['target_next_logret'])
        g['ticker'] = t
        frames.append(g[['ticker'] + features + ['target_next_logret']])
    return pd.concat(frames, ignore_index=True)

def train_and_predict(df_feats, dt_params=None, annualize_factor=252):
    """
    Huấn luyện DecisionTree cho từng mã cổ phiếu và dự đoán return kỳ vọng.
    """
    dt_params = dt_params or {'max_depth': 5, 'min_samples_leaf': 10, 'random_state': 42}
    results = []

    for t, sub in df_feats.groupby('ticker'):
        X = sub.drop(columns=['ticker', 'target_next_logret'])
        y = sub['target_next_logret'].values

        if len(X) < 20:
            expected_next = np.nanmean(y)
            results.append({
                'ticker': t,
                'expected_next_day_logret': expected_next,
                'expected_annualized_logret': expected_next * annualize_factor
            })
            continue

        # Dùng dữ liệu cuối để dự đoán "ngày kế tiếp"
        X_train = X.iloc[:-1]
        y_train = y[:-1]
        X_pred = X.iloc[[-1]]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_pred = scaler.transform(X_pred)

        model = DecisionTreeRegressor(**dt_params)
        model.fit(X_train, y_train)

        pred_next = model.predict(X_pred)[0]
        results.append({
            'ticker': t,
            'expected_next_day_logret': float(pred_next),
            'expected_annualized_logret': float(pred_next * annualize_factor)
        })

    return pd.DataFrame(results)

def csv_to_expected_returns(csv_path, out_path='expected_returns.csv'):
    df = pd.read_csv(csv_path)
    required_cols = {'Date', 'ticker', 'adjclose'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV cần có các cột: {required_cols}")

    df_feats = prepare_features(df)
    df_res = train_and_predict(df_feats)

    # Chuyển log-return sang simple-return
    df_res['expected_next_day_return'] = np.exp(df_res['expected_next_day_logret']) - 1
    df_res['expected_annualized_return'] = np.exp(df_res['expected_annualized_logret']) - 1

    df_res.to_csv(out_path, index=False)
    print(f"✅ Kết quả đã lưu tại: {out_path}")
    print(df_res)
    return df_res

if __name__ == '__main__':
    csv_to_expected_returns("quantumn_clean.csv", "expected-returns/expected_rerturn_decision_tree.csv")
