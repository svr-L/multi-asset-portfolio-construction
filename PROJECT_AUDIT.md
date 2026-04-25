# Project Audit

This file summarizes the most relevant issues found in the original assignment script and how the refactor handles them.

## Structural issues in the original script

1. **Single monolithic script**
   - Data loading, metrics, optimization, simulation, plotting, and reporting all lived in one file.
   - This made reuse and maintenance difficult.

2. **Hard-coded local path**
   - The original code reads data from a specific Windows desktop path.
   - This prevents reproducibility outside the original machine.

3. **Repeated function definitions**
   - Several functions are defined multiple times (`annualize_rets`, `annualize_vol`, `annualize_vol_dn`, etc.).
   - This increases the risk of silent inconsistencies.

4. **Global-state-heavy design**
   - Some helper functions implicitly depend on global variables rather than only their arguments.

## Logic / implementation issues worth flagging

1. **Constraint argument inconsistency**
   - In the original constrained optimization section, the high-yield constraint arguments are passed in an order that is easy to misread and potentially wrong in intent.

2. **Relative equity exposure helper**
   - The original EU-equity-relative exposure function uses a global flag internally rather than relying fully on its inputs.

3. **Historical equity-curve plotting bug**
   - The original script plots `P5` multiple times where `P6` and `P7` were likely intended.

4. **Monte Carlo diffusion scaling**
   - In the original simulation block, volatility appears inside both `dW` and the Euler update, effectively double-scaling the diffusion term.

5. **Label / reporting inconsistencies**
   - Some labels refer to assets that were previously dropped or renamed.
   - Some tables omit a portfolio from headers while still using its series in row calculations.

6. **Black-Litterman implementation style**
   - The original script mixes time-varying proxy market weights with row-wise posterior returns.
   - In the refactor I simplified this into a more standard, static posterior-mean implementation that is easier to understand and maintain.

## What the refactor preserves

- the overall research idea,
- the asset-universe engineering logic,
- the portfolio-construction menu,
- the key risk and performance metrics,
- the final comparison workflow.

## What the refactor changes intentionally

- cleaner package structure,
- explicit data-loading functions,
- clearer naming,
- more robust optimization helpers,
- more notebook-friendly workflow,
- better reproducibility.

