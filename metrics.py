from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def annualized_return(s: pd.Series | pd.DataFrame, periods_per_year: int) -> pd.Series | float:
    def _f(x: pd.Series) -> float:
        x = x.dropna()
        growth = (1 + x).prod()
        n_periods = x.shape[0]
        return growth ** (periods_per_year / n_periods) - 1

    if isinstance(s, pd.DataFrame):
        return s.aggregate(_f)
    return _f(s)


def annualized_volatility(s: pd.Series | pd.DataFrame, periods_per_year: int) -> pd.Series | float:
    def _f(x: pd.Series) -> float:
        x = x.dropna()
        return x.std() * np.sqrt(periods_per_year)

    if isinstance(s, pd.DataFrame):
        return s.aggregate(_f)
    return _f(s)


def annualized_downside_volatility(s: pd.Series | pd.DataFrame, periods_per_year: int) -> pd.Series | float:
    def _f(x: pd.Series) -> float:
        x = x.dropna()
        return x[x < 0].std() * np.sqrt(periods_per_year)

    if isinstance(s, pd.DataFrame):
        return s.aggregate(_f)
    return _f(s)


def historic_var(s: pd.Series | pd.DataFrame, periods_per_year: int, level: float = 0.05) -> pd.Series | float:
    def _f(x: pd.Series) -> float:
        x = x.dropna()
        return np.percentile(x, level * 100) * np.sqrt(periods_per_year)

    if isinstance(s, pd.DataFrame):
        return s.aggregate(_f)
    return _f(s)


def parametric_var(s: pd.Series | pd.DataFrame, periods_per_year: int, level: float = 0.05) -> pd.Series | float:
    def _f(x: pd.Series) -> float:
        x = x.dropna()
        mu = np.mean(x)
        sigma = np.std(x)
        z = norm.ppf(1 - level)
        return (mu - z * sigma) * np.sqrt(periods_per_year)

    if isinstance(s, pd.DataFrame):
        return s.aggregate(_f)
    return _f(s)


def historic_es(s: pd.Series | pd.DataFrame, periods_per_year: int, level: float = 0.05) -> pd.Series | float:
    def _f(x: pd.Series) -> float:
        x = x.dropna()
        cutoff = np.percentile(x, level * 100)
        return np.mean(x[x < cutoff]) * np.sqrt(periods_per_year)

    if isinstance(s, pd.DataFrame):
        return s.aggregate(_f)
    return _f(s)


def parametric_es(s: pd.Series | pd.DataFrame, periods_per_year: int, level: float = 0.05) -> pd.Series | float:
    def _f(x: pd.Series) -> float:
        x = x.dropna()
        mu = np.mean(x)
        sigma = np.std(x)
        z = norm.ppf(1 - level)
        return np.mean(x[x < mu - sigma * z]) * np.sqrt(periods_per_year)

    if isinstance(s, pd.DataFrame):
        return s.aggregate(_f)
    return _f(s)


def max_drawdown(s: pd.Series | pd.DataFrame, window: int) -> pd.Series | float:
    def _f(x: pd.Series) -> float:
        x = x.dropna()
        wealth = (1 + x).cumprod()
        rolling_peak = wealth.rolling(window, min_periods=1).max()
        drawdown = wealth / rolling_peak - 1.0
        return drawdown.rolling(window, min_periods=1).min().min()

    if isinstance(s, pd.DataFrame):
        return s.aggregate(_f)
    return _f(s)


def sharpe_ratio(s: pd.Series | pd.DataFrame, risk_free_rate: float, periods_per_year: int) -> pd.Series | float:
    def _f(x: pd.Series) -> float:
        x = x.dropna()
        rf_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        excess = x - rf_period
        return annualized_return(excess, periods_per_year) / annualized_volatility(x, periods_per_year)

    if isinstance(s, pd.DataFrame):
        return s.aggregate(_f)
    return _f(s)


def sortino_ratio(s: pd.Series | pd.DataFrame, risk_free_rate: float, periods_per_year: int) -> pd.Series | float:
    def _f(x: pd.Series) -> float:
        x = x.dropna()
        rf_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        excess = x - rf_period
        return annualized_return(excess, periods_per_year) / annualized_downside_volatility(x, periods_per_year)

    if isinstance(s, pd.DataFrame):
        return s.aggregate(_f)
    return _f(s)


def omega_ratio(s: pd.Series | pd.DataFrame, threshold: float, periods_per_year: int) -> pd.Series | float:
    def _f(x: pd.Series) -> float:
        x = x.dropna()
        threshold_period = (1 + threshold) ** (1 / periods_per_year) - 1
        gains = np.maximum(0, x - threshold_period)
        losses = np.maximum(0, threshold_period - x)
        return gains.sum() / (losses.sum() + 1e-10)

    if isinstance(s, pd.DataFrame):
        return s.aggregate(_f)
    return _f(s)


def calmar_ratio(s: pd.Series | pd.DataFrame, window: int, periods_per_year: int) -> pd.Series | float:
    def _f(x: pd.Series) -> float:
        x = x.dropna()
        return annualized_return(x, periods_per_year) / (abs(max_drawdown(x, window)) + 1e-10)

    if isinstance(s, pd.DataFrame):
        return s.aggregate(_f)
    return _f(s)
