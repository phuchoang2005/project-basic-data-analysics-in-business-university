import pandas as pd
import numpy as np
from scipy.optimize import minimize

# 1. Tải và chuẩn bị dữ liệu
# Tải Ma trận Hiệp phương sai (Sigma) và Lợi nhuận Kỳ vọng (mu)
cov_matrix_df = pd.read_csv("covariance-matrix/Hiep_phuong_sai.csv")
returns_df = pd.read_csv("expected-returns/expected_returns_prediction.csv")

# Làm sạch và căn chỉnh dữ liệu (đảm bảo thứ tự các mã cổ phiếu là nhất quán)
cov_matrix_df = cov_matrix_df.rename(columns={'Unnamed: 0': 'Ticker'}).set_index('Ticker')
tickers = cov_matrix_df.index.tolist()
returns_df = returns_df.set_index('Ticker').loc[tickers]

Sigma = cov_matrix_df.values
mu = returns_df['Expected_Return'].values
num_assets = len(tickers)

# 2. Định nghĩa Hàm Mục tiêu (Objective Function)
# Hàm này tính Portfolio Variance: w^T * Sigma * w
def portfolio_variance(weights, Sigma):
    """Tính phương sai danh mục (rủi ro): w^T * Sigma * w"""
    weights = np.array(weights)
    return weights.T @ Sigma @ weights

# 3. Định nghĩa các Ràng buộc (Constraints)
# Ràng buộc Ngân sách: Tổng trọng số = 1
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

# Ràng buộc Không bán khống: 0 <= w_i <= 1
bounds = tuple((0, 1) for _ in range(num_assets))

# 4. Giải bài toán Tối ưu hóa (Sử dụng thuật toán SLSQP)
initial_weights = np.ones(num_assets) / num_assets
min_variance_result = minimize(
    portfolio_variance,
    initial_weights,
    args=(Sigma,),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

# 5. Tính toán và Trình bày Kết quả
optimal_weights = min_variance_result.x

# Tính Lợi nhuận Kỳ vọng Tối ưu: E(Rp) = w^T * mu
optimal_return = optimal_weights @ mu

# Tính Rủi ro Tối ưu (Độ lệch chuẩn): sqrt(w^T * Sigma * w)
optimal_risk = np.sqrt(min_variance_result.fun)

# Tạo DataFrame kết quả
optimal_portfolio = pd.DataFrame({
    'Ticker': tickers,
    'Optimal Weight (%)': optimal_weights * 100
})
optimal_portfolio = optimal_portfolio.sort_values(by='Optimal Weight (%)', ascending=False)
optimal_portfolio.to_csv("optimal_weights_math.csv")