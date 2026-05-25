"""
Equity Risk & Return Analysis
Author: Adebola Awokoya

Pulls 5 years of daily price data for 10 stocks, calculates
annualized returns, volatility, Sharpe ratios, and beta vs S&P 500.
Outputs a summary table and rolling volatility chart.

Requirements: pip install yfinance pandas matplotlib
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────

TICKERS = ["VG", "OGN", "COUR", "JOB", "NVAX", "HD", "NVDA", "HOOD", "F"]
BENCHMARK = "^GSPC"          # S&P 500
PERIOD = "5y"
RISK_FREE = 0.053            # ~current 3-month T-bill rate (update as needed)
ROLLING_WINDOW = 30          # days for rolling volatility

# ── DATA ──────────────────────────────────────────────────────────────────────

print("Pulling data...")
raw = yf.download(TICKERS + [BENCHMARK], period=PERIOD, auto_adjust=True, progress=False)["Close"]
raw.dropna(how="all", inplace=True)

# Split benchmark from equities
bench_prices = raw[BENCHMARK]
stock_prices = raw[TICKERS]

# Daily returns
stock_returns = stock_prices.pct_change().dropna()
bench_returns = bench_prices.pct_change().dropna()

# Align
stock_returns, bench_returns = stock_returns.align(bench_returns, join="inner", axis=0)

# ── METRICS ───────────────────────────────────────────────────────────────────

TRADING_DAYS = 252

results = []
for ticker in TICKERS:
    r = stock_returns[ticker].dropna()
    b = bench_returns.loc[r.index]

    ann_return = r.mean() * TRADING_DAYS
    ann_vol    = r.std() * (TRADING_DAYS ** 0.5)
    sharpe     = (ann_return - RISK_FREE) / ann_vol if ann_vol > 0 else float("nan")

    # Beta: cov(stock, bench) / var(bench)
    cov_matrix = pd.concat([r, b], axis=1).cov()
    beta = cov_matrix.iloc[0, 1] / cov_matrix.iloc[1, 1]

    # Alpha (Jensen's): actual return - (Rf + beta*(Rm - Rf))
    bench_ann = b.mean() * TRADING_DAYS
    alpha = ann_return - (RISK_FREE + beta * (bench_ann - RISK_FREE))

    results.append({
        "Ticker":          ticker,
        "Ann. Return":     ann_return,
        "Ann. Volatility": ann_vol,
        "Sharpe Ratio":    sharpe,
        "Beta":            beta,
        "Alpha":           alpha
    })

df = pd.DataFrame(results).set_index("Ticker")

# ── PRINT TABLE ───────────────────────────────────────────────────────────────

print("\n── Equity Risk & Return Summary ─────────────────────────────────────\n")
display = df.copy()
display["Ann. Return"]     = display["Ann. Return"].map("{:.1%}".format)
display["Ann. Volatility"] = display["Ann. Volatility"].map("{:.1%}".format)
display["Sharpe Ratio"]    = display["Sharpe Ratio"].map("{:.2f}".format)
display["Beta"]            = display["Beta"].map("{:.2f}".format)
display["Alpha"]           = display["Alpha"].map("{:.1%}".format)
print(display.to_string())

# Highlight best/worst
best_sharpe  = df["Sharpe Ratio"].idxmax()
worst_sharpe = df["Sharpe Ratio"].idxmin()
best_alpha   = df["Alpha"].idxmax()
print(f"\nHighest Sharpe: {best_sharpe} ({df.loc[best_sharpe, 'Sharpe Ratio']:.2f})")
print(f"Lowest Sharpe:  {worst_sharpe} ({df.loc[worst_sharpe, 'Sharpe Ratio']:.2f})")
print(f"Best Alpha:     {best_alpha} ({df.loc[best_alpha, 'Alpha']:.1%})")

# ── CHARTS ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Equity Risk & Return Analysis", fontsize=14, fontweight="bold", y=1.01)

# Chart 1: Risk vs Return scatter
ax1 = axes[0]
for ticker in TICKERS:
    x = df.loc[ticker, "Ann. Volatility"]
    y = df.loc[ticker, "Ann. Return"]
    ax1.scatter(x, y, s=80, zorder=3)
    ax1.annotate(ticker, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)

ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax1.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax1.set_xlabel("Annualized Volatility (Risk)")
ax1.set_ylabel("Annualized Return")
ax1.set_title("Risk vs. Return")
ax1.grid(True, alpha=0.3)

# Chart 2: Rolling 30-day volatility for all tickers
ax2 = axes[1]
rolling_vol = stock_returns.rolling(ROLLING_WINDOW).std() * (TRADING_DAYS ** 0.5)
for ticker in TICKERS:
    ax2.plot(rolling_vol.index, rolling_vol[ticker], linewidth=1, label=ticker, alpha=0.8)

ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax2.set_xlabel("Date")
ax2.set_ylabel(f"Rolling {ROLLING_WINDOW}-Day Volatility (Ann.)")
ax2.set_title(f"Rolling {ROLLING_WINDOW}-Day Volatility")
ax2.legend(fontsize=7, ncol=2, loc="upper right")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("equity_analysis_charts.png", dpi=150, bbox_inches="tight")
print("\nChart saved: equity_analysis_charts.png")
plt.show()
