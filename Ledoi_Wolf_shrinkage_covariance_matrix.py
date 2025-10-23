from sklearn.covariance import LedoitWolf
import numpy as np
import pandas as pd

# Giả sử file có cột Date, adjclose, ticker
df = pd.read_csv("quantumn_clean.csv")

# Nếu có thêm cột index tự sinh, bỏ đi
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# 🔥 Quan trọng: sort dữ liệu
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(by=["ticker", "Date"])

# ✅ Tính return (dùng log-return ổn định hơn)
df["return"] = np.log(df["adjclose"] / df.groupby("ticker")["adjclose"].shift(1))
df = df.dropna()


# Pivot sang dạng wide format: mỗi cột là 1 ticker
returns = df.pivot_table(index="Date", columns="ticker", values="return", aggfunc="mean")

# Bỏ các ngày bị NaN (mất dữ liệu)
returns = returns.dropna(how="any")

# Tính Ledoit–Wolf shrinkage covariance
lw = LedoitWolf()
lw.fit(returns)
cov_matrix = lw.covariance_

# Xuất ra file csv
cov_df = pd.DataFrame(cov_matrix, index=returns.columns, columns=returns.columns)
cov_df.to_csv("covariance-matrix/covariance_matrix_ledoitwolf.csv")

print("✅ Covariance matrix shape:", cov_df.shape)
print("✅ Ma trận hiệp phương sai Ledoit–Wolf đã lưu tại: covariance_matrix_ledoitwolf.csv")
