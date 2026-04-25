from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class WalkForwardResult:
    returns_gross: pd.DataFrame
    returns_net: pd.DataFrame
    weights: dict[str, pd.DataFrame]
    turnover: pd.DataFrame
    diagnostics: pd.DataFrame


def turnover_from_weights(current: pd.Series, previous: pd.Series | None) -> float:
    if previous is None:
        previous = pd.Series(0.0, index=current.index)
    previous = previous.reindex(current.index).fillna(0.0)
    return float((current - previous).abs().sum())


def run_walk_forward_backtest(
    returns: pd.DataFrame,
    allocator_builders: dict[str, Callable[[pd.DataFrame], pd.Series]],
    estimation_window: int = 756,
    rebalance_frequency: int = 21,
    transaction_cost_bps: float = 10.0,
) -> WalkForwardResult:
    """Rolling out-of-sample allocator backtest with turnover-based transaction costs.

    Each allocator is estimated using only data available up to the rebalance date.
    The resulting weights are then held over the next OOS holding period. Trading cost
    is charged on the first OOS observation after each rebalance as
    ``cost_bps * turnover / 10_000``.
    """
    returns = returns.dropna().copy()
    cols = returns.columns
    dates = returns.index

    gross_parts = {name: [] for name in allocator_builders}
    net_parts = {name: [] for name in allocator_builders}
    weight_rows = {name: [] for name in allocator_builders}
    turnover_rows = []
    diagnostics = []
    prev_weights: dict[str, pd.Series | None] = {name: None for name in allocator_builders}

    for start in range(estimation_window, len(returns) - 1, rebalance_frequency):
        train = returns.iloc[start - estimation_window : start]
        test = returns.iloc[start : min(start + rebalance_frequency, len(returns))]
        rebalance_date = dates[start]
        turn_row = {}

        for name, builder in allocator_builders.items():
            try:
                w = builder(train).reindex(cols).fillna(0.0)
                w = w.clip(lower=0.0)
                if w.sum() <= 0 or not np.isfinite(w).all():
                    raise ValueError("invalid weights")
                w = w / w.sum()
                success = True
                error = ""
            except Exception as exc:  # fallback keeps the OOS panel usable
                w = pd.Series(1 / len(cols), index=cols)
                success = False
                error = str(exc)

            gross = test @ w
            turnover = turnover_from_weights(w, prev_weights[name])
            cost = transaction_cost_bps / 10_000 * turnover
            net = gross.copy()
            if len(net) > 0:
                net.iloc[0] -= cost

            gross_parts[name].append(gross)
            net_parts[name].append(net)
            weight_rows[name].append(pd.Series(w.values, index=cols, name=rebalance_date))
            turn_row[name] = turnover
            diagnostics.append(
                {
                    "date": rebalance_date,
                    "allocator": name,
                    "success": success,
                    "turnover": turnover,
                    "cost": cost,
                    "error": error,
                }
            )
            prev_weights[name] = w

        turnover_rows.append(pd.Series(turn_row, name=rebalance_date))

    gross_df = pd.concat({k: pd.concat(v) for k, v in gross_parts.items()}, axis=1).sort_index()
    net_df = pd.concat({k: pd.concat(v) for k, v in net_parts.items()}, axis=1).sort_index()
    weights_df = {k: pd.DataFrame(v) for k, v in weight_rows.items()}
    turnover_df = pd.DataFrame(turnover_rows)
    diagnostics_df = pd.DataFrame(diagnostics)

    return WalkForwardResult(
        returns_gross=gross_df,
        returns_net=net_df,
        weights=weights_df,
        turnover=turnover_df,
        diagnostics=diagnostics_df,
    )
