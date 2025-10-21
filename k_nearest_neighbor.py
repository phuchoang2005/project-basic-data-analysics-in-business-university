# save as predict_expected_returns_knn.py
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

def prepare_features(df, n_lags=[1, 2], windows=[5, 10, 20]):
    """
    Chuẩn bị dữ liệu huấn luyện cho từng ticker.
    """
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['ticker', 'Date'])
    frames = []

    for t, g in df.groupby('ticker'):
        g = g.copy().reset_index(drop=True)
        g['logp'] = np.log(g['adjclose'])
        g['logret'] = g['logp'].diff()
        g['label'] = (g['logret'] > 0).astype(int)

        # tạo các feature lag
        for lag in n_lags:
            g[f'lag_{lag}'] = g['logret'].shift(lag)

        # rolling mean/std + momentum
        for w in windows:
            g[f'roll_mean_{w}'] = g['logret'].rolling(w).mean().shift(1)
            g[f'roll_std_{w}'] = g['logret'].rolling(w).std().shift(1)
            g[f'mom_{w}'] = g['logp'] - g['logp'].shift(w)

        g['target_next'] = g['label'].shift(-1)
        features = [c for c in g.columns if c.startswith(('lag_', 'roll_', 'mom_'))]
        g = g.dropna(subset=features + ['target_next'])
        frames.append(g[['ticker'] + features + ['logret', 'target_next']])
    return pd.concat(frames, ignore_index=True)

def train_and_predict_knn(df_feats, k=5, annualize_factor=252):
    """
    Huấn luyện KNN cho từng ticker và tính tỷ suất lợi nhuận kỳ vọng.
    """
    results = []
    for t, sub in df_feats.groupby('ticker'):
        X = sub.drop(columns=['ticker', 'target_next', 'logret'])
        y = sub['target_next'].astype(int)
        logrets = sub['logret']

        if len(X) < k + 1 or y.nunique() < 2:
            exp_ret = np.nanmean(logrets)
            results.append({
                'ticker': t,
                'expected_next_day_logret': exp_ret,
                'expected_annualized_logret': exp_ret * annualize_factor
            })
            continue

        X_train = X.iloc[:-1]
        y_train = y.iloc[:-1]
        X_pred = X.iloc[[-1]]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_pred = scaler.transform(X_pred)

        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)

        prob_up = model.predict_proba(X_pred)[0][1]

        pos_ret = logrets[y == 1].mean()
        neg_ret = logrets[y == 0].mean()

        expected_logret = prob_up * pos_ret + (1 - prob_up) * neg_ret

        results.append({
            'ticker': t,
            'expected_next_day_logret': expected_logret,
            'expected_annualized_logret': expected_logret * annualize_factor
        })

    return pd.DataFrame(results)

def csv_to_expected_returns(csv_path, out_path='expected_returns_knn.csv', k=5):
    df = pd.read_csv(csv_path)
    required_cols = {'Date', 'ticker', 'adjclose'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV cần có các cột: {required_cols}")

    df_feats = prepare_features(df)
    df_res = train_and_predict_knn(df_feats, k=k)

    df_res['expected_next_day_return'] = np.exp(df_res['expected_next_day_logret']) - 1
    df_res['expected_annualized_return'] = np.exp(df_res['expected_annualized_logret']) - 1

    df_res.to_csv(out_path, index=False)
    print(f"✅ Đã lưu kết quả tại: {out_path}")
    print(df_res)
    return df_res

if __name__ == '__main__':
    csv_to_expected_returns("quantumn_clean.csv", "expected-returns/expected_returns_KNN.csv", 5)
