
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import scipy.optimize as spopt


@dataclass
class ResampledMVResult:
    weights: pd.Series
    bootstrap_weights: pd.DataFrame
    success_ratio: float


@dataclass
class EndogenousBLViews:
    signal_table: pd.DataFrame
    P: pd.DataFrame
    q: pd.Series
    confidences: pd.Series


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


def _risk_contribution_budget_objective(weights: np.ndarray, sigma_annual: np.ndarray) -> float:
    portfolio_var = float(weights @ sigma_annual @ weights)
    marginal = sigma_annual @ weights
    risk_contrib = weights * marginal / (portfolio_var + 1e-12)
    target = np.repeat(1 / len(weights), len(weights))
    return float(np.sum((risk_contrib - target) ** 2))


def solve_equal_risk_contribution(
    returns: pd.DataFrame,
    bounds: list[tuple[float, float]],
    constraints: list[dict],
    periods_per_year: int = 252,
) -> spopt.OptimizeResult:
    sigma_annual = (returns.cov() * periods_per_year).values
    w0 = np.repeat(1 / returns.shape[1], returns.shape[1])
    return spopt.minimize(
        _risk_contribution_budget_objective,
        w0,
        args=(sigma_annual,),
        bounds=bounds,
        method="SLSQP",
        constraints=constraints,
        options={"disp": False, "ftol": 1e-12, "maxiter": 2000},
    )


def _negative_diversification_ratio(weights: np.ndarray, returns: pd.DataFrame, periods_per_year: int = 252) -> float:
    asset_vols = returns.std().values * np.sqrt(periods_per_year)
    port_vol = portfolio_volatility(weights, returns, periods_per_year)
    diversification_ratio = float(np.dot(weights, asset_vols) / (port_vol + 1e-12))
    return -diversification_ratio


def solve_max_diversification(
    returns: pd.DataFrame,
    bounds: list[tuple[float, float]],
    constraints: list[dict],
    periods_per_year: int = 252,
) -> spopt.OptimizeResult:
    w0 = np.repeat(1 / returns.shape[1], returns.shape[1])
    return spopt.minimize(
        _negative_diversification_ratio,
        w0,
        args=(returns, periods_per_year),
        bounds=bounds,
        method="SLSQP",
        constraints=constraints,
        options={"disp": False, "ftol": 1e-12, "maxiter": 2000},
    )


def resampled_mean_variance_weights(
    returns: pd.DataFrame,
    bounds: list[tuple[float, float]],
    constraint_builder: Callable[[pd.DataFrame], list[dict]],
    periods_per_year: int = 252,
    n_bootstrap: int = 200,
    sample_size: int | None = None,
    seed: int = 42,
) -> ResampledMVResult:
    rng = np.random.default_rng(seed)
    n_obs = len(returns)
    sample_size = n_obs if sample_size is None else sample_size

    weight_list: list[np.ndarray] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_obs, size=sample_size)
        sample = returns.iloc[idx].reset_index(drop=True)
        constraints = constraint_builder(sample)
        res = solve_min_vol(sample, bounds, constraints, periods_per_year)
        if res.success and np.isfinite(res.fun):
            weight_list.append(res.x)

    if not weight_list:
        raise RuntimeError("No successful bootstrap optimizations in resampled_mean_variance_weights.")

    bootstrap_weights = pd.DataFrame(weight_list, columns=returns.columns)
    avg_weights = bootstrap_weights.mean(axis=0).clip(lower=0.0)
    avg_weights = avg_weights / avg_weights.sum()

    return ResampledMVResult(
        weights=avg_weights.rename("Resampled MV"),
        bootstrap_weights=bootstrap_weights,
        success_ratio=len(weight_list) / n_bootstrap,
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


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std < 1e-12:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def build_endogenous_bl_views(
    returns: pd.DataFrame,
    periods_per_year: int = 252,
    n_views: int = 3,
    momentum_window: int = 126,
    drawdown_window: int = 126,
    corr_window: int = 63,
    view_scale: float = 0.35,
    confidence_floor: float = 0.35,
    confidence_cap: float = 0.85,
) -> EndogenousBLViews:
    simple_returns = np.exp(returns) - 1.0

    momentum_sample = simple_returns.tail(momentum_window)
    momentum = (1.0 + momentum_sample).prod() - 1.0

    dd_sample = simple_returns.tail(drawdown_window)
    wealth = (1.0 + dd_sample).cumprod()
    latest_drawdown = wealth.div(wealth.cummax()).iloc[-1] - 1.0
    drawdown_score = -latest_drawdown

    corr_sample = simple_returns.tail(corr_window)
    avg_abs_corr = corr_sample.corr().abs().replace(1.0, np.nan).mean()
    diversification_score = -avg_abs_corr.fillna(avg_abs_corr.mean())

    vol_annual = returns.tail(momentum_window).std() * np.sqrt(periods_per_year)

    composite = (
        0.50 * _zscore(momentum)
        + 0.30 * _zscore(drawdown_score)
        + 0.20 * _zscore(diversification_score)
    )

    signal_table = pd.DataFrame(
        {
            "momentum_6m": momentum,
            "latest_drawdown": latest_drawdown,
            "avg_abs_corr": avg_abs_corr,
            "vol_annual": vol_annual,
            "composite_score": composite,
        }
    ).sort_values("composite_score", ascending=False)
    signal_table["rank"] = np.arange(1, len(signal_table) + 1)

    n_assets = len(signal_table)
    n_views = max(1, min(n_views, n_assets // 2))
    top_assets = signal_table.index[:n_views]
    bottom_assets = signal_table.index[-n_views:][::-1]

    spread_scores = []
    rows = []
    q_vals = []
    conf_vals = []
    view_names = []

    score_range = composite.max() - composite.min() + 1e-12
    for idx, (long_asset, short_asset) in enumerate(zip(top_assets, bottom_assets), start=1):
        row = pd.Series(0.0, index=returns.columns)
        row[long_asset] = 1.0
        row[short_asset] = -1.0

        spread_score = float(composite[long_asset] - composite[short_asset])
        spread_vol = float((vol_annual[long_asset] + vol_annual[short_asset]) / 2.0)
        q_val = view_scale * np.tanh(spread_score) * spread_vol
        confidence = confidence_floor + (confidence_cap - confidence_floor) * min(1.0, abs(spread_score) / score_range)

        rows.append(row.values)
        q_vals.append(q_val)
        conf_vals.append(confidence)
        spread_scores.append(spread_score)
        view_names.append(f"view_{idx}: {long_asset} > {short_asset}")

    P = pd.DataFrame(rows, index=view_names, columns=returns.columns)
    q = pd.Series(q_vals, index=view_names, name="q")
    confidences = pd.Series(conf_vals, index=view_names, name="confidence")

    signal_table["z_momentum"] = _zscore(signal_table["momentum_6m"])
    signal_table["z_drawdown"] = _zscore(drawdown_score.reindex(signal_table.index))
    signal_table["z_diversification"] = _zscore(diversification_score.reindex(signal_table.index))

    return EndogenousBLViews(
        signal_table=signal_table,
        P=P,
        q=q,
        confidences=confidences,
    )


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
