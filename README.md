# SMA Crossover Efficacy in Indian-Markets

<img width="2500" height="1800" alt="nifty50_dual_ma_dashboard" src="dual_sma_dash_board.png" />

A comprehensive quantitative framework for backtesting and analyzing moving average strategies using 10 years of minute-level NIFTY 50 data. This systematic engine evaluates single and dual SMA crossover configurations with realistic transaction costs, providing deep statistical insights and professional visualizations for high-frequency trend following in Indian equity markets.

---

## 📊 Project Overview

- **Systematic Backtesting**: Comprehensive evaluation of single and dual SMA strategies with customizable parameters
- **Dataset**: 932,334 minute-bars (2015–2025) capturing NIFTY 50 market microstructure
- **Advanced Analytics**: Feature engineering, regime analysis, statistical diagnostics, and risk metrics
- **Realistic Simulation**: 0.015% transaction costs, no look-ahead bias, precise trade accounting
- **Professional Output**: Multi-panel dashboards, statistical summaries, and actionable insights
- **Production-Ready**: Modular, extensible, and fully reproducible codebase

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation & Setup](#installation--setup)
3. [Methodology](#methodology)
4. [Key Results & Insights](#key-results--insights)
5. [Statistical Diagnostics](#statistical-diagnostics)
6. [Visualization Dashboard](#visualization-dashboard)
7. [Limitations & Future Work](#limitations--future-work)
8. [Extending the Framework](#extending-the-framework)
9. [License](#license)
10. [Author & Contact](#author--contact)

---

## Introduction

This repository implements a rigorous quantitative pipeline for analyzing moving average strategies on high-frequency NIFTY 50 data. The framework evaluates 6 single SMA and 13 dual SMA configurations across a decade of intraday data, incorporating transaction costs and providing comprehensive risk-adjusted performance metrics. The analysis reveals critical insights about trend-following effectiveness in Indian equity markets at minute-level frequencies.

---

## Installation & Setup

### Requirements

- Python 3.7+ (Anaconda distribution recommended)
- Core libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`

### Data Preparation

- **Source File**: `NIFTY 50_minute_data.csv`
- **Structure**: DateTime index, Open, High, Low, Close, Volume columns
- **Note**: Volume data is zeroed in current dataset; analysis focuses on price-based strategies

### Execution

```bash
# Clone repository
git clone [https://github.com/shubh123a3/NIFTY50-Dual-MA-Momentum-Engine.git](https://github.com/manavsarvaiya/SMA-Crossover-Efficacy-in-Indian-Markets.git)
cd SMA-Crossover-Efficacy-in-Indian-Markets
```
# Install dependencies
```
pip install -r requirements.txt
```
# Run full analysis pipeline
```
python app.py
```

The complete analysis generates visualizations, performance metrics, and statistical diagnostics in the working directory.

# Methodology

## Data Processing Pipeline

**Data Audit:**  
Quality checks, missing value assessment, trading hour filtering (09:15–15:30 IST)

**Feature Engineering:**

- 6 single-period SMAs (5, 10, 20, 50, 100, 200)
- 13 dual-SMA pairs (fast < slow combinations)
- 48 auxiliary features for diagnostic analysis

**Strategy Implementation:**

- **Single SMA:** Long when price > SMA, flat otherwise
- **Dual SMA:** Long when fast SMA > slow SMA, flat otherwise
- **Execution:** Next-bar entry with 0.015% round-trip transaction costs
- **Performance Evaluation:** Comprehensive metrics with annualization and risk adjustment

## Key Assumptions

- No slippage or market impact  
- Immediate execution at quoted prices  
- Consistent 0.015% transaction cost per round-trip  
- No leverage or position sizing optimization  

---

# Key Results & Insights

## Dataset Characteristics

| Metric | Value | Description |
|--------|--------|-------------|
| Records | 932,334 | Minute-level bars |
| Date Range | 2015-01-09 to 2025-02-07 | Full decade coverage |
| Mean Price | 13,623.5 | Secular bull market trend |
| Data Integrity | 0 missing values | Complete dataset |
| Return Distribution | Highly non-normal | Skew: -12.24, Kurtosis: 2806.04 |

---

## Single SMA Performance

| SMA Period | Total Return | Sharpe Ratio | Max Drawdown | Win Rate | Trades |
|-----------|--------------|--------------|---------------|----------|--------|
| 5 | -100.0% | -7.770 | -100.0% | N/A | 119,880 |
| 10 | -100.0% | -7.269 | -100.0% | N/A | 80,197 |
| 20 | -100.0% | -6.412 | -100.0% | N/A | 55,057 |
| 50 | -100.0% | -4.855 | -100.0% | N/A | 33,522 |
| 100 | -99.4% | -3.653 | -99.4% | N/A | 23,024 |
| 200 | -93.5% | -2.160 | -93.5% | 48.6% | 15,044 |

**Critical Finding:**  
No single SMA configuration produced positive risk-adjusted returns, highlighting the challenge of high-frequency noise in Indian equity markets.

---

## Dual SMA Crossover Performance (Top 10)

| Strategy | Sharpe Ratio | Total Return | Calmar Ratio | Max DD | Win Rate | Trades |
|----------|--------------|--------------|--------------|--------|----------|--------|
| 50/200 | 1.090 | 200.9% | 0.653 | -17.93% | 12.06% | 2,737 |
| 20/200 | 0.579 | 83.7% | 0.300 | -21.03% | 10.97% | 3,857 |
| 50/100 | 0.052 | 5.8% | 0.017 | -34.16% | 11.46% | 5,123 |
| 10/200 | -0.046 | -4.9% | -0.012 | -41.19% | 8.77% | 5,228 |
| 20/100 | -0.194 | -18.9% | -0.050 | -41.81% | 10.97% | 6,227 |
| 5/200 | -0.555 | -46.4% | -0.109 | -55.65% | 6.54% | 7,186 |
| 10/100 | -0.764 | -57.7% | -0.139 | -59.55% | 8.56% | 8,147 |
| 5/100 | -1.611 | -85.3% | -0.204 | -86.10% | 6.49% | 11,101 |
| 20/50 | -1.615 | -85.9% | -0.206 | -86.74% | 10.46% | 10,527 |
| 10/50 | -2.078 | -93.1% | -0.252 | -93.43% | 8.56% | 12,624 |

**Optimal Strategy:**  
**50/200 crossover** emerges as the only configuration achieving **Sharpe > 1.0** with controlled drawdowns.

---

# Statistical Diagnostics

## 50/200 Strategy Deep Dive

### Performance Metrics

- **Annualized Return:** 11.71%  
- **Annualized Volatility:** 10.74%  
- **Sharpe Ratio:** 1.090  
- **Sortino Ratio:** 0.882  
- **Calmar Ratio:** 0.653  
- **Maximum Drawdown:** -17.93%  
- **Profit Factor:** 0.152  

---

### Return Distribution

- **Mean Trade Return:** -0.0184%  
- **Median Trade Return:** -0.0150%  
- **Standard Deviation:** 0.0520%  
- **Skewness:** -20.339 (extreme left skew)  
- **Kurtosis:** 782.262 (fat tails)  
- **Normality Test:** Rejected (Shapiro-Wilk p = 0.0000)

---

### Trade Return Percentiles

| Percentile | Return | Interpretation |
|------------|--------|----------------|
| 1st | -0.126% | Extreme losses |
| 25th | -0.018% | Typical loss |
| 50th | -0.015% | Median loss |
| 75th | -0.015% | Upper quartile |
| 95th | +0.020% | Strong gains |
| 99th | +0.064% | Tail-driven profits |

**Key Insight:**  
Profitability stems from **rare right-tail gains** (top 5% of trades) while taking frequent small losses.

---

# Market Regime Analysis

- **76/121 positive months** (62.8% hit-rate)  
- **Best Month:** +9.35%  
- **Worst Month:** -3.73%  
- **Monthly Volatility:** 2.56%  
- **Regime Dependency:** Outperformance concentrated in **high-volatility periods** (COVID crisis, 2022 spikes)

---

# Parameter Sensitivity

## Fast SMA Analysis

| Period | Sharpe | Total Return | Characteristic |
|--------|--------|--------------|----------------|
| 5 | -3.171 | -82.4% | Noise-dominated |
| 10 | -2.414 | -63.9% | High turnover |
| 20 | -0.416 | -7.0% | Transition zone |
| 50 | 0.596 | +103.3% | Optimal balance |
| 100 | -0.624 | -39.0% | Too slow |

## Slow SMA Analysis

| Period | Sharpe | Total Return | Characteristic |
|--------|--------|--------------|----------------|
| 20 | -6.826 | -100.0% | Too responsive |
| 50 | -2.482 | -92.3% | Insufficient smoothing |
| 100 | -0.624 | -39.0% | Moderate performance |
| 200 | 0.298 | +58.3% | Optimal smoothing |

---

# Visualization Dashboard

The framework generates comprehensive multi-panel dashboards:

### Strategy Performance Views

- Equity curves  
- Risk-return scatter plots  
- Drawdown profiles  
- Win rate & trade frequency comparisons  

### Statistical Diagnostics

- Histograms + KDE  
- QQ-plots  
- Autocorrelation & partial autocorrelation  
- Rolling volatility & statistics  

### Market Analysis

- Monthly heatmaps  
- Regime effectiveness  
- Time-of-day patterns  
- Seasonal effects  
- Volatility correlation  

---

# Limitations & Future Work

## Current Limitations

- Zeroed volume data  
- No slippage or market impact  
- Low win rate (12.06%)  
- Regime-dependent performance  
- Parameter instability  

## Research Extensions

- Volatility filters, correlation overlays  
- EMA, WMA, hybrid indicators  
- BANKNIFTY & multi-asset testing  
- ML-driven regime classification  
- VWAP/TWAP execution  
- Risk-parity portfolio integration  

---
