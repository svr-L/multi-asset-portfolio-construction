# QAA Multi-Asset Portfolio Construction

Refactored version of a Quantitative Asset Allocation course project originally developed as a single Python script.

## What this repository does

The project studies a **multi-asset allocation problem** using daily index data and compares several portfolio construction approaches:

- Mean-variance (Markowitz)
- Constrained mean-variance optimization
- Constrained downside-volatility minimization
- Black-Litterman portfolio construction
- Constrained Black-Litterman optimization
- Sortino-ratio maximization
- Constrained Sortino-ratio maximization

It also includes:

- exploratory analysis of the investment universe,
- risk-return visualizations,
- proxy equilibrium returns,
- historical portfolio comparison,
- forward scenario simulation,
- risk metrics,
- risk-adjusted performance metrics.

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

## Main refactoring choices

Compared with the original script, this repository:

- removes hard-coded local Windows paths,
- isolates reusable code into `src/qaa`,
- reduces repeated function definitions,
- separates data loading, metrics, optimization, and plotting,
- makes the Black-Litterman block more standard and readable,
- adds a project audit documenting important issues found in the original version.

## Important note on faithfulness vs cleanup

This repo is a **professionalized refactor**, not a byte-for-byte transcription of the original assignment.  
Where the original script contained avoidable implementation issues or organizational problems, I preferred a cleaner research-repo structure over strict replication.

## Next upgrades worth doing

If you want to push this beyond a course-project repo, the next natural steps are:

1. add out-of-sample / rolling-window backtests,
2. replace static estimates with shrinkage or regime-aware inputs,
3. treat Black-Litterman views and confidence calibration more rigorously,
4. add tests for metrics and optimizer constraints,
5. export final figures/tables automatically to `reports/`.

