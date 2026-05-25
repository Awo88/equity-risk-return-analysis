# Equity Risk & Return Analysis

Pulls 5 years of daily price data for a portfolio of stocks and computes key risk/return metrics vs the S&P 500. Outputs a ranked summary table and two charts.

---

## What It Does

1. **Pulls live data** from Yahoo Finance via `yfinance` — 5 years of daily adjusted closes
2. **Calculates per-stock metrics:** annualized return, volatility, Sharpe ratio, beta, Jensen's alpha, and max drawdown
3. **Benchmarks against the S&P 500** — every metric is computed relative to `^GSPC`
4. **Outputs** a ranked table (by Sharpe ratio) and two charts

---

## Metrics

| Metric | Formula |
|---|---|
| Annualized Return | Mean daily return × 252 |
| Annualized Volatility | Daily return std × √252 |
| Sharpe Ratio | (Ann. Return − Rf) / Ann. Vol |
| Beta | Cov(stock, market) / Var(market) |
| Jensen's Alpha | Ann. Return − [Rf + β × (Rm − Rf)] |
| Max Drawdown | Min((price − rolling max) / rolling max) |

Risk-free rate: 5.3% (3-month T-bill — update `RISK_FREE` at top of script).

---

## Stocks Analyzed

| Ticker | Company |
|---|---|
| NVDA | NVIDIA Corp |
| HD | Home Depot |
| HOOD | Robinhood Markets |
| F | Ford Motor Co |
| OGN | Organon & Co |
| COUR | Coursera |
| NVAX | Novavax |
| JOB | GEE Group |
| VG | Vonage Communications |

---

## Output

| File | Description |
|---|---|
| `equity_analysis_charts.png` | Risk vs Return scatter (Sharpe color-coded) + rolling volatility chart |

Console output includes the full ranked table plus best/worst Sharpe, best alpha, and shallowest drawdown.

---

## Installation & Usage

```bash
pip install yfinance pandas matplotlib
python equity_analysis.py
```

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `TICKERS` | 9 stocks | Edit to analyze any set of tickers |
| `PERIOD` | `"5y"` | yfinance lookback period |
| `RISK_FREE` | 0.053 | Risk-free rate (update to current T-bill) |
| `ROLLING_WINDOW` | 30 | Rolling volatility window (trading days) |

---

*Author: Adebola Awokoya — (2026)*
