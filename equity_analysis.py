"""
Equity Risk & Return Analysis
Author: Adebola Awokoya

Pulls 5 years of daily price data for 9 stocks, calculates annualized
returns, volatility, Sharpe ratios, beta, and Jensen's alpha vs S&P 500.
Outputs a ranked summary table and two charts.

Requirements: pip install yfinance pandas matplotlib
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────

TICKERS        = ["VG", "OGN", "COUR", "JOB", "NVAX", "HD", "NVDA", "HOOD", "F"]
BENCHMARK      = "^GSPC"     
PERIOD         = "5y"
RISK_FREE      = 0.053       # 3-month T-bill rate — update as needed
ROLLING_WINDOW = 30          # days for rolling volatility chart

# ── DATA ──────────────────────────────────────────────────────────────────────

print("Pulling data from Yahoo Finance...")
raw = yf.download(
    TICKERS + [BENCHMARK], period=PERIOD,
    auto_adjust=True, progress=False
)["Close"]
raw.dropna(how="all", inplace=True)

bench_prices = raw[BENCHMARK]
stock_prices = raw[TICKERS]

stock_returns = stock_prices.pct_change().dropna()
bench_returns = bench_prices.pct_change().dropna()
stock_returns, bench_returns = stock_returns.align(bench_returns, join="inner", axis=0)

print(f"Trading days in sample: {len(stock_returns)}\n")

# ── METRICS ───────────────────────────────────────────────────────────────────

TRADING_DAYS = 252

results = []
for ticker in TICKERS:
    r = stock_returns[ticker].dropna()
    b = bench_returns.loc[r.index]

    ann_return = r.mean() * TRADING_DAYS
    ann_vol    = r.std()  * (TRADING_DAYS ** 0.5)
    sharpe     = (ann_return - RISK_FREE) / ann_vol if ann_vol > 0 else float("nan")

    cov_matrix = pd.concat([r, b], axis=1).cov()
    beta = cov_matrix.iloc[0, 1] / cov_matrix.iloc[1, 1]

    bench_ann = b.mean() * TRADING_DAYS
    alpha = ann_return - (RISK_FREE + beta * (bench_ann - RISK_FREE))

    # Max drawdown
    cum_ret    = (1 + r).cumprod()
    roll_max   = cum_ret.cummax()
    drawdown   = (cum_ret - roll_max) / roll_max
    max_dd     = drawdown.min()

    results.append({
        "Ticker":          ticker,
        "Ann. Return":     ann_return,
        "Ann. Volatility": ann_vol,
        "Sharpe Ratio":    sharpe,
        "Beta":            beta,
        "Jensen Alpha":    alpha,
        "Max Drawdown":    max_dd,
    })

df = pd.DataFrame(results).set_index("Ticker")
df_sorted = df.sort_values("Sharpe Ratio", ascending=False)

# ── SUMMARY TABLE ─────────────────────────────────────────────────────────────

print("── Equity Risk & Return Summary (ranked by Sharpe Ratio) ────────────────\n")
display = df_sorted.copy()
display["Ann. Return"]     = display["Ann. Return"].map("{:+.1%}".format)
display["Ann. Volatility"] = display["Ann. Volatility"].map("{:.1%}".format)
display["Sharpe Ratio"]    = display["Sharpe Ratio"].map("{:.2f}".format)
display["Beta"]            = display["Beta"].map("{:.2f}".format)
display["Jensen Alpha"]    = display["Jensen Alpha"].map("{:+.1%}".format)
display["Max Drawdown"]    = display["Max Drawdown"].map("{:.1%}".format)
print(display.to_string())

# Benchmarks
bench_ann_ret = bench_returns.mean() * TRADING_DAYS
bench_ann_vol = bench_returns.std()  * (TRADING_DAYS ** 0.5)
bench_sharpe  = (bench_ann_ret - RISK_FREE) / bench_ann_vol
print(f"\nS&P 500 (benchmark)  |  Ann. Return: {bench_ann_ret:+.1%}"
      f"  |  Ann. Vol: {bench_ann_vol:.1%}  |  Sharpe: {bench_sharpe:.2f}")

print(f"\nHighest Sharpe : {df['Sharpe Ratio'].idxmax()}  ({df['Sharpe Ratio'].max():.2f})")
print(f"Lowest Sharpe  : {df['Sharpe Ratio'].idxmin()}  ({df['Sharpe Ratio'].min():.2f})")
print(f"Best Alpha     : {df['Jensen Alpha'].idxmax()}  ({df['Jensen Alpha'].max():+.1%})")
print(f"Lowest Max DD  : {df['Max Drawdown'].idxmax()}  ({df['Max Drawdown'].max():.1%})")

# ── CHARTS ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Equity Risk & Return Analysis  |  5-Year Daily Data",
             fontsize=14, fontweight="bold")

# Chart 1: Risk vs Return scatter with Sharpe iso-lines
ax1 = axes[0]
cmap = plt.cm.RdYlGn
sharpe_vals = df["Sharpe Ratio"].values
sharpe_norm = (sharpe_vals - sharpe_vals.min()) / (sharpe_vals.max() - sharpe_vals.min())

for i, ticker in enumerate(df.index):
    x = df.loc[ticker, "Ann. Volatility"]
    y = df.loc[ticker, "Ann. Return"]
    ax1.scatter(x, y, s=100, color=cmap(sharpe_norm[i]), zorder=3, edgecolors="gray", linewidth=0.5)
    ax1.annotate(ticker, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)

# S&P 500 benchmark point
ax1.scatter(bench_ann_vol, bench_ann_ret, s=120, color="black",
            marker="D", zorder=4, label="S&P 500")
ax1.annotate("S&P 500", (bench_ann_vol, bench_ann_ret),
             textcoords="offset points", xytext=(6, 4), fontsize=8, fontweight="bold")

ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax1.axhline(RISK_FREE, color="navy", linewidth=0.5, linestyle=":", alpha=0.7)
ax1.text(df["Ann. Volatility"].max() * 0.9, RISK_FREE + 0.01,
         f"Rf = {RISK_FREE:.1%}", fontsize=7, color="navy")

ax1.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax1.set_xlabel("Annualized Volatility (Risk)")
ax1.set_ylabel("Annualized Return")
ax1.set_title("Risk vs. Return  (color = Sharpe Ratio)")
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

# Chart 2: Rolling 30-day volatility
ax2 = axes[1]
rolling_vol = stock_returns.rolling(ROLLING_WINDOW).std() * (TRADING_DAYS ** 0.5)
for ticker in TICKERS:
    ax2.plot(rolling_vol.index, rolling_vol[ticker],
             linewidth=0.9, label=ticker, alpha=0.85)

ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax2.set_xlabel("Date")
ax2.set_ylabel(f"Rolling {ROLLING_WINDOW}-Day Volatility (Ann.)")
ax2.set_title(f"Rolling {ROLLING_WINDOW}-Day Realized Volatility")
ax2.legend(fontsize=7, ncol=2, loc="upper right")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("equity_analysis_charts.png", dpi=150, bbox_inches="tight")
print("\nChart saved: equity_analysis_charts.png")
plt.show()
