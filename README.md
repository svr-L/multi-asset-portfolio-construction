# QAA Multi-Asset Portfolio Construction

Refactored and extended version of a Quantitative Asset Allocation course project originally developed as a single Python script.

The repository turns the assignment into a cleaner research workflow for **multi-asset portfolio construction, constrained allocation, Black-Litterman views, risk diagnostics, forward scenario analysis, and rolling out-of-sample evaluation**.

## What this repository does

The project studies a multi-asset allocation problem using daily index data and compares several allocation approaches:

- Mean-variance / Markowitz optimization
- Constrained mean-variance optimization
- Constrained downside-volatility minimization
- Equal Risk Contribution (ERC)
- Maximum Diversification
- Resampled mean-variance allocation
- Endogenous Black-Litterman allocation
- Constrained endogenous Black-Litterman allocation
- Sortino-ratio maximization
- Constrained Sortino-ratio maximization

It also includes:

- exploratory analysis of the investment universe,
- risk-return visualizations,
- proxy equilibrium returns,
- endogenous Black-Litterman views based on momentum, drawdown and correlation signals,
- historical portfolio comparison,
- multivariate block-bootstrap forward scenario simulation,
- rolling walk-forward out-of-sample backtest,
- turnover-based transaction-cost adjustment,
- VaR, Expected Shortfall, Max Drawdown, Sharpe, Sortino, Omega and Calmar diagnostics.

## Why this version is stronger than the original assignment

Compared with the original assignment, this version removes the weakest “toy” elements and adds a more defensible research layer:

1. **No hard-coded local paths**: data loading is centralized in `src/qaa/data.py`.
2. **Reusable project structure**: metrics, optimization, plotting and backtesting logic live in `src/qaa`.
3. **Broader allocator set**: ERC, Maximum Diversification and Resampled MV are added to the original Markowitz / BL / Sortino comparison.
4. **Endogenous BL views**: views are generated from cross-asset signals rather than being purely hand-coded.
5. **More realistic forward scenarios**: the old GBM-style simulation is replaced by a multivariate stationary block bootstrap.
6. **Walk-forward OOS layer**: allocation weights are re-estimated through time using only past data.
7. **Transaction costs**: net returns subtract turnover-based costs at each rebalance.

## Repository structure

```text
qaa-multi-asset-portfolio-repo/
├── README.md
├── PROJECT_AUDIT.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 01_qaa_multi_asset_portfolio.ipynb
├── src/
│   └── qaa/
│       ├── __init__.py
│       ├── backtesting.py
│       ├── config.py
│       ├── data.py
│       ├── metrics.py
│       ├── optimization.py
│       └── plotting.py
└── original/
    └── QAA Assignment_Saverio Lauriola.py
```

## Data requirements

The original assignment script expects two Excel files:

- `Database_gg3.xlsx`
- `Database_gg.xlsx`

Place them in:

```text
data/raw/
```

The notebook is written so that you can run the full workflow once those files are available.

## Setup

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

Then launch Jupyter:

```bash
jupyter lab
```

or

```bash
jupyter notebook
```

## How to run

Open:

```text
notebooks/01_qaa_multi_asset_portfolio.ipynb
```

and run the notebook top to bottom.

## Main limitations

This is still a compact research repo, not a production portfolio engine. Key limitations:

- the investment universe is built from assignment data rather than tradable live ETF proxies;
- transaction costs use a simple turnover-bps model;
- forward scenarios are empirical bootstrap scenarios, not a full macro-financial scenario generator;
- Black-Litterman view calibration is intentionally transparent and heuristic;
- no formal unit-test suite is included yet.

## Next upgrades worth doing

Potential extensions:

1. replace assignment indices with live investable ETF proxies,
2. add covariance shrinkage and regime-conditioned inputs,
3. introduce richer transaction-cost/slippage assumptions,
4. validate allocator stability across subperiods,
5. add unit tests for metrics, constraints and optimization outputs,
6. export final figures/tables automatically to `reports/`.

## Suggested CV description

> Built a Python multi-asset portfolio-construction framework comparing mean-variance, downside-risk, Black-Litterman, ERC, maximum-diversification, resampled mean-variance and Sortino-based allocations under long-only and exposure constraints; evaluated portfolios through historical analysis, rolling walk-forward backtests, transaction-cost-adjusted returns, multivariate bootstrap scenarios, VaR/ES, MaxDD and risk-adjusted metrics.
