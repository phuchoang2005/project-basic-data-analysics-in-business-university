import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ========== CẤU HÌNH ==========
PRED_FILE = "predicted_expected_return.csv"
BACKTEST_FILE = "backtest_results.csv"
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== ĐỌC DỮ LIỆU ==========
pred_df = pd.read_csv(PRED_FILE, parse_dates=["Date"])
backtest_df = pd.read_csv(BACKTEST_FILE)

# ========== 1️⃣ BIỂU ĐỒ CHUNG THEO TICKER ==========
metrics = ["MAE", "R2", "Correlation", "DailyExpectedReturn"]

plt.figure(figsize=(12, 6))
sns.barplot(data=backtest_df.melt(id_vars="Ticker", value_vars=metrics),
            x="Ticker", y="value", hue="variable")
plt.title("Backtest Metrics Comparison by Ticker")
plt.ylabel("Metric Value")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "backtest_metrics_comparison.png"))
plt.close()

print("📊 Saved: backtest_metrics_comparison.png")

# ========== 2️⃣ BIỂU ĐỒ TIME SERIES PREDICTION ==========
for ticker, data in pred_df.groupby("Ticker"):
    plt.figure(figsize=(12, 5))
    plt.plot(data["Date"], data["ActualReturn"], label="Actual Return", color="black", linewidth=1)
    plt.plot(data["Date"], data["PredictedReturn"], label="Predicted Return", color="orange", alpha=0.7)
    plt.title(f"{ticker} — Actual vs Predicted Daily Return")
    plt.xlabel("Date")
    plt.ylabel("Daily Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{ticker}_timeseries.png"))
    plt.close()
    print(f"📈 Saved: {ticker}_timeseries.png")

# ========== 3️⃣ BIỂU ĐỒ PHÂN TÁN (ACTUAL vs PREDICTED) ==========
for ticker, data in pred_df.groupby("Ticker"):
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=data["ActualReturn"], y=data["PredictedReturn"], alpha=0.6)
    plt.title(f"{ticker} — Actual vs Predicted Scatter")
    plt.xlabel("Actual Return")
    plt.ylabel("Predicted Return")
    # Đường y=x để dễ so sánh
    lims = [
        min(data["ActualReturn"].min(), data["PredictedReturn"].min()),
        max(data["ActualReturn"].max(), data["PredictedReturn"].max())
    ]
    plt.plot(lims, lims, 'r--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{ticker}_scatter.png"))
    plt.close()
    print(f"📊 Saved: {ticker}_scatter.png")

print("\n✅ Visualization complete. All plots saved to 'plots/' folder.")
