from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import (
    DATA_RAW_DIR,
    PRIMARY_DATA_FILE,
    SECONDARY_DATA_FILE,
    SECONDARY_KEEP_COLUMNS,
    DROP_COLUMNS,
    RENAMED_COLUMNS,
)


def _ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required data file not found: {path}. "
            "Place the original Excel files inside data/raw/."
        )


def load_assignment_data(data_dir: Path | None = None) -> pd.DataFrame:
    """Load and align the two Excel files used in the original assignment."""
    data_dir = DATA_RAW_DIR if data_dir is None else Path(data_dir)

    primary_path = data_dir / PRIMARY_DATA_FILE
    secondary_path = data_dir / SECONDARY_DATA_FILE

    _ensure_exists(primary_path)
    _ensure_exists(secondary_path)

    data = pd.read_excel(primary_path, index_col=0, parse_dates=True, na_values="n.e.")
    data.columns = [c.strip() for c in data.columns]

    data_2 = pd.read_excel(secondary_path, index_col=0, parse_dates=True, na_values="n.e.")
    data_2.columns = [c.strip() for c in data_2.columns]
    data_2 = data_2.loc[:, SECONDARY_KEEP_COLUMNS]
    data_2 = data_2.reindex(data.index)

    merged = pd.concat([data, data_2], axis=1)
    merged = merged.drop(DROP_COLUMNS, axis=1)
    merged.columns = RENAMED_COLUMNS
    return merged.sort_index()


def compute_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Convert price/index levels to log returns."""
    return np.log(price_df).diff().dropna()


def build_research_universe(data: pd.DataFrame) -> pd.DataFrame:
    """Replicate the asset engineering step from the original project."""
    df = data.copy()
    df["EM Bond"] = 0.5 * df["EM Bond"] + 0.5 * df["EM Foreing Currency Gov. Bond"]
    df["EU IG Bond"] = 0.5 * df["EU Bond Short Term"] + 0.5 * df["EU Gov. Bonds"]
    df["Opportunities"] = 0.5 * df["Small Cap Equity"] + 0.5 * df["Global Real Estates"]

    df = df.drop(
        [
            "EU Bond Short Term",
            "EU Gov. Bonds",
            "ITA Equity",
            "World Equity",
            "Small Cap Equity",
            "Global Real Estates",
            "EM Foreing Currency Gov. Bond",
        ],
        axis=1,
    )
    return df


def proxy_equilibrium_weights(levels: pd.DataFrame) -> pd.Series:
    """Use normalized index levels as a simple proxy for equilibrium weights.

    This mirrors the spirit of the original script, while acknowledging that
    these are not true market-cap weights.
    """
    weights_t = levels.div(levels.sum(axis=1), axis=0)
    return weights_t.mean(axis=0)
