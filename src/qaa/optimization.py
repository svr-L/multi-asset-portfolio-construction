from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import scipy.optimize as spopt


def portfolio_series(returns: pd.DataFrame, weights: np.ndarray | pd.Series) -> pd.Series:
    w = np.asarray(weights, dtype=float)
    return returns @ w


def portfolio_mean_return(weights: np.ndarray, returns: pd.DataFrame, periods_per_year: int = 252) -> float:
    mu = returns.mean(axis=0).values
    return float(np.dot(weights, mu) * periods_per_year)


def portfolio_variance(weights: np.ndarray, returns: pd.DataFrame, periods_per_year: int = 252) -> float:
    sigma = returns.cov().values
    return float(weights @ sigma @ weights * periods_per_year)


def portfolio_volatility(weights: np.ndarray, returns: pd.DataFrame, periods_per_year: int = 252) -> float:
    return float(np.sqrt(portfolio_variance(weights, returns, periods_per_year)))


def portfolio_downside_volatility(weights: np.ndarray, returns: pd.DataFrame, periods_per_year: int = 252) -> float:
    rp = portfolio_series(returns, weights)
    return float(rp[rp < 0].std() * np.sqrt(periods_per_year))


def sortino_objective(weights: np.ndarray, returns: pd.DataFrame, risk_free_rate: float, periods_per_year: int = 252) -> float:
    mean_ret = portfolio_mean_return(weights, returns, periods_per_year)
    downside = portfolio_downside_volatility(weights, returns, periods_per_year)
    return -((mean_ret - risk_free_rate) / (downside + 1e-10))


def target_return_constraint(weights: np.ndarray, returns: pd.DataFrame, target: float, periods_per_year: int = 252) -> float:
    return portfolio_mean_return(weights, returns, periods_per_year) - target


def group_sum(weights: np.ndarray, flag: pd.Series) -> float:
    x = pd.Series(weights, index=flag.index)
    return float(x.loc[flag == 1].sum())


def group_lower_bound(weights: np.ndarray, flag: pd.Series, target: float) -> float:
    return group_sum(weights, flag) - target


def group_upper_bound(weights: np.ndarray, flag: pd.Series, target: float) -> float:
    return target - group_sum(weights, flag)


def relative_group_lower_bound(weights: np.ndarray, numerator_flag: pd.Series, denominator_flag: pd.Series, target: float) -> float:
    numerator = group_sum(weights, numerator_flag)
    denominator = group_sum(weights, denominator_flag)
    return numerator / (denominator + 1e-10) - target


def long_only_bounds(columns: pd.Index, max_weights: list[float] | None = None) -> list[tuple[float, float]]:
    if max_weights is None:
        max_weights = [1.0] * len(columns)
    return list(zip([0.0] * len(columns), max_weights))


def base_constraints(
    returns: pd.DataFrame,
    target_return: float | None = None,
    periods_per_year: int = 252,
) -> list[dict]:
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    if target_return is not None:
        constraints.append(
            {
                "type": "eq",
                "fun": target_return_constraint,
                "args": (returns, target_return, periods_per_year),
            }
        )
    return constraints


def solve_min_vol(
    returns: pd.DataFrame,
    bounds: list[tuple[float, float]],
    constraints: list[dict],
    periods_per_year: int = 252,
) -> spopt.OptimizeResult:
    w0 = np.repeat(1 / returns.shape[1], returns.shape[1])
    return spopt.minimize(
        portfolio_volatility,
        w0,
        args=(returns, periods_per_year),
        bounds=bounds,
        method="SLSQP",
        constraints=constraints,
        options={"disp": False, "ftol": 1e-10, "maxiter": 1000},
    )


def solve_min_downside_vol(
    returns: pd.DataFrame,
    bounds: list[tuple[float, float]],
    constraints: list[dict],
    periods_per_year: int = 252,
) -> spopt.OptimizeResult:
    w0 = np.repeat(1 / returns.shape[1], returns.shape[1])
    return spopt.minimize(
        portfolio_downside_volatility,
        w0,
        args=(returns, periods_per_year),
        bounds=bounds,
        method="SLSQP",
        constraints=constraints,
        options={"disp": False, "ftol": 1e-10, "maxiter": 1000},
    )


def solve_sortino(
    returns: pd.DataFrame,
    bounds: list[tuple[float, float]],
    constraints: list[dict],
    risk_free_rate: float,
    periods_per_year: int = 252,
) -> spopt.OptimizeResult:
    w0 = np.repeat(1 / returns.shape[1], returns.shape[1])
    return spopt.minimize(
        sortino_objective,
        w0,
        args=(returns, risk_free_rate, periods_per_year),
        bounds=bounds,
        method="SLSQP",
        constraints=constraints,
        options={"disp": False, "ftol": 1e-10, "maxiter": 1000},
    )


def implied_equilibrium_returns(
    sigma_annual: pd.DataFrame,
    equilibrium_weights: pd.Series,
    risk_free_rate: float,
    risk_aversion: float = 4.5,
) -> pd.Series:
    return pd.Series(
        risk_free_rate + risk_aversion * sigma_annual.values @ equilibrium_weights.values,
        index=sigma_annual.index,
        name="pi",
    )


def omega_from_confidence(confidences: pd.Series, sigma_annual: pd.DataFrame, P: pd.DataFrame, tau: float) -> pd.DataFrame:
    vals = []
    for i, conf in enumerate(confidences):
        p_i = P.iloc[i].values
        vals.append((1 / conf - 1) * (p_i @ (tau * sigma_annual.values) @ p_i.T))
    omega = np.diag(vals)
    return pd.DataFrame(omega, index=P.index, columns=P.index)


def black_litterman_posterior_mean(
    sigma_annual: pd.DataFrame,
    pi: pd.Series,
    P: pd.DataFrame,
    q: pd.Series,
    confidences: pd.Series,
    tau: float = 1 / 15,
) -> pd.Series:
    Sigma = sigma_annual.values
    Pi = pi.values.reshape(-1, 1)
    Pm = P.values
    Q = q.values.reshape(-1, 1)
    Omega = omega_from_confidence(confidences, sigma_annual, P, tau).values

    inv_tau_sigma = np.linalg.inv(tau * Sigma)
    middle = np.linalg.inv(inv_tau_sigma + Pm.T @ np.linalg.inv(Omega) @ Pm)
    rhs = inv_tau_sigma @ Pi + Pm.T @ np.linalg.inv(Omega) @ Q
    posterior = middle @ rhs
    return pd.Series(posterior.flatten(), index=sigma_annual.index, name="bl_posterior_mean")


def simulate_gbm_paths(
    x0: float,
    mu: float,
    sigma: float,
    n_years: int,
    periods_per_year: int,
    n_sims: int,
    seed: int = 2800,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_steps = n_years * periods_per_year
    dt = 1 / periods_per_year
    shocks = rng.normal(size=(n_sims, n_steps))
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
    log_paths = np.cumsum(increments, axis=1)
    paths = x0 * np.exp(np.column_stack([np.zeros(n_sims), log_paths]))
    return pd.DataFrame(paths.T)


def simulation_log_returns(paths: pd.DataFrame) -> pd.DataFrame:
    return np.log(paths).diff().dropna()


@dataclass
class ResampledMVResult:
    weights: pd.Series
    bootstrap_weights: pd.DataFrame
    success_ratio: float


@dataclass
class EndogenousBLViews:
    P: pd.DataFrame
    q: pd.Series
    confidences: pd.Series
    signal_table: pd.DataFrame


def _as_series_weights(x: np.ndarray, index: pd.Index, name: str | None = None) -> pd.Series:
    return pd.Series(np.asarray(x, dtype=float), index=index, name=name)


def risk_contributions(weights: np.ndarray, cov_annual: pd.DataFrame | np.ndarray) -> np.ndarray:
    cov = cov_annual.values if isinstance(cov_annual, pd.DataFrame) else np.asarray(cov_annual, dtype=float)
    port_var = float(weights @ cov @ weights)
    if port_var <= 0:
        return np.zeros_like(weights)
    marginal = cov @ weights
    return weights * marginal / port_var


def solve_equal_risk_contribution(
    returns: pd.DataFrame,
    bounds: list[tuple[float, float]],
    constraints: list[dict],
    periods_per_year: int = 252,
) -> spopt.OptimizeResult:
    cov = returns.cov() * periods_per_year
    n_assets = returns.shape[1]
    target = np.repeat(1 / n_assets, n_assets)

    def objective(w: np.ndarray) -> float:
        rc = risk_contributions(w, cov)
        return float(np.sum((rc - target) ** 2))

    w0 = np.repeat(1 / n_assets, n_assets)
    return spopt.minimize(
        objective,
        w0,
        bounds=bounds,
        method="SLSQP",
        constraints=constraints,
        options={"disp": False, "ftol": 1e-10, "maxiter": 1000},
    )


def diversification_ratio(weights: np.ndarray, returns: pd.DataFrame, periods_per_year: int = 252) -> float:
    cov = returns.cov().values * periods_per_year
    asset_vols = returns.std().values * np.sqrt(periods_per_year)
    numerator = float(weights @ asset_vols)
    denominator = float(np.sqrt(weights @ cov @ weights))
    return numerator / (denominator + 1e-12)


def solve_max_diversification(
    returns: pd.DataFrame,
    bounds: list[tuple[float, float]],
    constraints: list[dict],
    periods_per_year: int = 252,
) -> spopt.OptimizeResult:
    n_assets = returns.shape[1]

    def objective(w: np.ndarray) -> float:
        return -diversification_ratio(w, returns, periods_per_year)

    w0 = np.repeat(1 / n_assets, n_assets)
    return spopt.minimize(
        objective,
        w0,
        bounds=bounds,
        method="SLSQP",
        constraints=constraints,
        options={"disp": False, "ftol": 1e-10, "maxiter": 1000},
    )


def resampled_mean_variance_weights(
    returns: pd.DataFrame,
    bounds: list[tuple[float, float]],
    constraints_builder: Callable[[pd.DataFrame], list[dict]],
    periods_per_year: int = 252,
    n_bootstrap: int = 250,
    seed: int = 42,
) -> ResampledMVResult:
    rng = np.random.default_rng(seed)
    cols = returns.columns
    fitted_weights = []
    n_obs = len(returns)
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, n_obs, size=n_obs)
        sample = returns.iloc[sample_idx].reset_index(drop=True)
        sample.columns = cols
        res = solve_min_vol(sample, bounds, constraints_builder(sample), periods_per_year)
        if res.success and np.all(np.isfinite(res.x)):
            fitted_weights.append(res.x)

    if not fitted_weights:
        res = solve_min_vol(returns, bounds, constraints_builder(returns), periods_per_year)
        fitted_weights = [res.x]

    bw = pd.DataFrame(fitted_weights, columns=cols)
    avg = bw.mean(axis=0)
    avg = avg / avg.sum()
    return ResampledMVResult(
        weights=avg.rename("resampled_mv"),
        bootstrap_weights=bw,
        success_ratio=len(fitted_weights) / n_bootstrap,
    )


def _zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if std == 0 or not np.isfinite(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def build_endogenous_bl_views(
    returns: pd.DataFrame,
    periods_per_year: int = 252,
    n_views: int = 3,
    momentum_window: int = 126,
    drawdown_window: int = 126,
    corr_window: int = 63,
    view_scale: float = 0.35,
) -> EndogenousBLViews:
    trailing = returns.dropna().copy()
    momentum = trailing.tail(momentum_window).sum() * periods_per_year / momentum_window
    wealth = np.exp(trailing.tail(drawdown_window).cumsum())
    latest_drawdown = wealth.iloc[-1] / wealth.cummax().iloc[-1] - 1
    corr = trailing.tail(corr_window).corr().abs()
    avg_abs_corr = (corr.sum(axis=1) - 1) / (len(corr) - 1)
    vol_annual = trailing.tail(momentum_window).std() * np.sqrt(periods_per_year)

    score = _zscore(momentum) + _zscore(latest_drawdown) - 0.5 * _zscore(avg_abs_corr) - 0.25 * _zscore(vol_annual)
    signal_table = pd.DataFrame(
        {
            "momentum_6m": momentum,
            "latest_drawdown": latest_drawdown,
            "avg_abs_corr": avg_abs_corr,
            "vol_annual": vol_annual,
            "composite_score": score,
        }
    ).sort_values("composite_score", ascending=False)
    signal_table["rank"] = np.arange(1, len(signal_table) + 1)

    top = list(signal_table.head(n_views).index)
    bottom = list(signal_table.tail(n_views).index[::-1])
    rows = []
    q_vals = []
    conf_vals = []
    view_names = []
    for i, (winner, loser) in enumerate(zip(top, bottom), start=1):
        p = pd.Series(0.0, index=returns.columns)
        p[winner] = 1.0
        p[loser] = -1.0
        score_gap = signal_table.loc[winner, "composite_score"] - signal_table.loc[loser, "composite_score"]
        vol_gap = np.sqrt(vol_annual[winner] ** 2 + vol_annual[loser] ** 2)
        q = max(0.005, view_scale * score_gap * vol_gap / np.sqrt(periods_per_year))
        confidence = float(np.clip(0.35 + 0.10 * abs(score_gap), 0.35, 0.85))
        rows.append(p)
        q_vals.append(q)
        conf_vals.append(confidence)
        view_names.append(f"View {i}: {winner} > {loser}")

    P = pd.DataFrame(rows, index=view_names, columns=returns.columns)
    q = pd.Series(q_vals, index=view_names, name="q")
    confidences = pd.Series(conf_vals, index=view_names, name="confidence")
    return EndogenousBLViews(P=P, q=q, confidences=confidences, signal_table=signal_table)


def stationary_bootstrap_indices(
    n_obs: int,
    n_steps: int,
    n_sims: int,
    avg_block_length: int = 21,
    seed: int = 2800,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = 1.0 / avg_block_length
    idx = np.empty((n_steps, n_sims), dtype=int)
    idx[0] = rng.integers(0, n_obs, size=n_sims)
    starts = rng.integers(0, n_obs, size=(n_steps, n_sims))
    switches = rng.random((n_steps, n_sims)) < p
    for t in range(1, n_steps):
        idx[t] = np.where(switches[t], starts[t], (idx[t - 1] + 1) % n_obs)
    return idx


def simulate_block_bootstrap_portfolio_returns(
    asset_returns: pd.DataFrame,
    weights: pd.DataFrame | pd.Series,
    n_years: int = 22,
    periods_per_year: int = 252,
    n_sims: int = 1000,
    avg_block_length: int = 21,
    seed: int = 2800,
) -> dict[str, pd.DataFrame] | pd.DataFrame:
    n_steps = n_years * periods_per_year
    idx = stationary_bootstrap_indices(len(asset_returns), n_steps, n_sims, avg_block_length, seed)
    boot = asset_returns.values[idx]

    if isinstance(weights, pd.Series):
        wr = boot @ weights.reindex(asset_returns.columns).fillna(0.0).values
        return pd.DataFrame(wr)

    simulated = {}
    for name in weights.columns:
        w = weights[name].reindex(asset_returns.columns).fillna(0.0).values
        simulated[name] = pd.DataFrame(boot @ w)
    return simulated
