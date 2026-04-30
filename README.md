# Multi-Asset Portfolio Construction with Yahoo Finance ETF Proxies

A standalone Python notebook for **multi-asset portfolio construction** using **Yahoo Finance ETF proxies** to compare classical and robust allocators under realistic portfolio constraints, rolling walk-forward rebalancing, transaction costs, and multivariate stationary-bootstrap scenarios.

## Overview

This project builds a compact **multi-asset allocation research framework** around broad ETF sleeves downloaded directly from Yahoo Finance. It compares several portfolio construction approaches under:

- **long-only allocation bounds**
- **exposure constraints across defensive, credit, equity, and real-asset buckets**
- **rolling walk-forward backtests**
- **turnover-based transaction costs**
- **multivariate stationary-bootstrap scenarios**

The objective is not just to maximize an in-sample Sharpe ratio, but to compare how different allocators behave once **rebalancing, frictions, and scenario uncertainty** are introduced.

## ETF Universe

The asset universe is built from broad ETF proxies representing distinct sleeves of a diversified multi-asset portfolio:

- **Cash / T-Bills:** `BIL`
- **Intermediate Treasuries:** `IEF`
- **TIPS:** `TIP`
- **Investment Grade Credit:** `LQD`
- **High Yield Credit:** `HYG`
- **U.S. Equity:** `VTI`
- **Developed ex-U.S. Equity:** `EFA`
- **Emerging Markets Equity:** `VWO`
- **Gold:** `GLD`
- **Broad Commodities:** `DBC`

These are not perfectly mutually exclusive asset classes in a strict risk-factor sense, but they provide clean, interpretable ETF proxies for broad portfolio sleeves.

## Allocators Compared

The notebook compares the following portfolio construction methods:

- **Mean-Variance**
- **Constrained Mean-Variance**
- **Downside-Risk Minimization**
- **Sortino-based Allocation**
- **Equal Risk Contribution (ERC)**
- **Maximum Diversification**
- **Resampled Mean-Variance**
- **Endogenous Black-Litterman**

The **Black-Litterman block** uses internally generated views rather than fully manual discretionary inputs.

## Research Workflow

The notebook follows this sequence:

1. **Download ETF prices from Yahoo Finance**
2. Build **log-return series**
3. Inspect basic diagnostics:
   - annualized returns
   - volatility
   - correlation structure
4. Define:
   - allocation bounds
   - exposure constraints
   - endogenous Black-Litterman inputs
5. Solve the allocator horse race
6. Evaluate in sample with:
   - annualized return / volatility
   - downside volatility
   - historical and parametric VaR / ES
   - Max Drawdown
   - Sharpe / Sortino / Omega / Calmar
7. Run **rolling walk-forward backtests**
8. Adjust returns for **turnover-based transaction costs**
9. Compare allocators under **multivariate stationary-bootstrap scenarios**

## Key Features

- **Standalone notebook**
  - no project-root lookup
  - no external package structure required
- **Direct Yahoo Finance download**
  - one cell to refresh the full ETF universe
- **Optional local cache**
  - avoids repeated downloads when not needed
- **Allocator horse race**
  - compares both classical and more robust portfolio construction approaches
- **Walk-forward evaluation**
  - avoids relying only on static in-sample results
- **Bootstrap scenarios**
  - adds distributional stress testing beyond simple point estimates

## How to Run

### 1. Install dependencies

```bash
pip install yfinance pandas numpy matplotlib scipy openpyxl
```

### 2. Open the notebook

Run:

- the **parameter cell**
- the **Yahoo Finance download cell**

If you want fresh data, keep:

```python
REFRESH_DATA = True
```

If you want to reuse previously downloaded data, set:

```python
REFRESH_DATA = False
```

## Repository Structure

```text
.
├── qaa_multi_asset_portfolio.ipynb
└── README.md
```

## Why this project matters

Classical portfolio optimization often looks good on paper and weak in implementation because it is highly sensitive to:

- estimation error
- unstable weights
- rebalancing frictions
- changing dependence structure

This notebook is designed to make those issues explicit by moving from a static coursework-style allocation exercise toward a more realistic **allocator comparison framework**.

## Current Limitations

This is already a meaningful research notebook, but several extensions would improve it further:

- covariance shrinkage
- explicit turnover penalty in the optimizer
- stronger endogenous Black-Litterman signal design
- more formal stability diagnostics for portfolio weights
- regime-aware covariance or macro-conditioned priors

## Possible Next Steps

High-ROI extensions include:

1. **Covariance shrinkage**
2. **Turnover penalty directly in the objective**
3. **Weight stability diagnostics**
4. **Improved endogenous Black-Litterman signals**
5. **Stress testing in covariance / factor space**

## Notes

- The notebook uses **ETF proxies**, not total-return institutional indices.
- Results will vary depending on:
  - sample period
  - transaction-cost assumptions
  - rebalancing frequency
  - bootstrap settings
- The stationary-bootstrap section is intended to complement the walk-forward results, not replace them.
