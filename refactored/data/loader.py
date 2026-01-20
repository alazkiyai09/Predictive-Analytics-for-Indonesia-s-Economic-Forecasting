"""
Data loading module for Indonesia Economic Forecasting
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

from config.settings import data_config, path_config
from utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Load and manage economic data from various sources"""

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize data loader

        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = data_dir or path_config.RAW_DATA_DIR
        self.data_cache: Dict[str, pd.DataFrame] = {}

    def load_csv(
        self,
        filename: str,
        date_col: str = "Date",
        parse_dates: bool = True
    ) -> pd.DataFrame:
        """
        Load CSV file

        Args:
            filename: Name of CSV file
            date_col: Date column name
            parse_dates: Whether to parse date column

        Returns:
            DataFrame with data
        """
        filepath = self.data_dir / filename

        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return pd.DataFrame()

        try:
            df = pd.read_csv(filepath)

            if parse_dates and date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df = df.dropna(subset=[date_col])
                df = df.sort_values(date_col).reset_index(drop=True)

            logger.info(f"Loaded {len(df)} rows from {filename}")
            return df

        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
            return pd.DataFrame()

    def load_economic_indicators(self) -> Dict[str, pd.DataFrame]:
        """
        Load all economic indicator files

        Returns:
            Dictionary of DataFrames
        """
        indicators = {
            "inflation": data_config.INFLATION_FILE,
            "interest_rate": data_config.INTEREST_RATE_FILE,
            "exports": data_config.EXPORTS_FILE,
            "imports": data_config.IMPORTS_FILE,
            "gdp_current": data_config.GDP_CURRENT_FILE,
            "gdp_constant": data_config.GDP_CONSTANT_FILE,
            "outstanding_bond": data_config.OUTSTANDING_BOND_FILE
        }

        data = {}
        for name, filename in indicators.items():
            df = self.load_csv(filename)
            if not df.empty:
                data[name] = df
                self.data_cache[name] = df

        logger.info(f"Loaded {len(data)} economic indicator datasets")
        return data

    def load_market_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all market data files

        Returns:
            Dictionary of DataFrames
        """
        market_files = {
            "gold": data_config.GOLD_FILE,
            "usd_idr": data_config.USD_IDR_FILE,
            "usd_jpy": data_config.USD_JPY_FILE,
            "eur_usd": data_config.EUR_USD_FILE,
            "gbp_usd": data_config.GBP_USD_FILE,
            "dxy": data_config.DXY_FILE,
            "idx": data_config.IDX_FILE,
            "brent": data_config.BRENT_FILE,
            "wti": data_config.WTI_FILE,
            "spread_bond": data_config.SPREAD_BOND_FILE
        }

        data = {}
        for name, filename in market_files.items():
            df = self.load_csv(filename)
            if not df.empty:
                data[name] = df
                self.data_cache[name] = df

        logger.info(f"Loaded {len(data)} market datasets")
        return data

    def load_money_supply(self) -> Dict[str, pd.DataFrame]:
        """
        Load money supply data

        Returns:
            Dictionary of DataFrames
        """
        money_files = {
            "m1_m2_id": data_config.M1_M2_ID_FILE,
            "m1_m2_us": data_config.M1_M2_US_FILE,
            "m1_m2_eu": data_config.M1_M2_EU_FILE,
            "m1_m2_jp": data_config.M1_M2_JP_FILE,
            "m2_uk": data_config.M2_UK_FILE
        }

        data = {}
        for name, filename in money_files.items():
            df = self.load_csv(filename)
            if not df.empty:
                data[name] = df
                self.data_cache[name] = df

        logger.info(f"Loaded {len(data)} money supply datasets")
        return data

    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all available data

        Returns:
            Combined dictionary of all datasets
        """
        all_data = {}
        all_data.update(self.load_economic_indicators())
        all_data.update(self.load_market_data())
        all_data.update(self.load_money_supply())

        logger.info(f"Total datasets loaded: {len(all_data)}")
        return all_data

    def get_cached(self, name: str) -> Optional[pd.DataFrame]:
        """Get cached dataset by name"""
        return self.data_cache.get(name)

    def clear_cache(self):
        """Clear data cache"""
        self.data_cache.clear()
        logger.info("Data cache cleared")


def resample_to_monthly(
    df: pd.DataFrame,
    date_col: str = "Date",
    value_cols: Optional[List[str]] = None,
    agg_func: str = "last"
) -> pd.DataFrame:
    """
    Resample time series to monthly frequency

    Args:
        df: Input DataFrame
        date_col: Date column name
        value_cols: Columns to aggregate (None = all numeric)
        agg_func: Aggregation function ('last', 'mean', 'sum')

    Returns:
        Monthly resampled DataFrame
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)

    if value_cols is None:
        value_cols = df.select_dtypes(include=[float, int]).columns.tolist()

    if agg_func == "last":
        monthly = df[value_cols].resample('M').last()
    elif agg_func == "mean":
        monthly = df[value_cols].resample('M').mean()
    elif agg_func == "sum":
        monthly = df[value_cols].resample('M').sum()
    else:
        raise ValueError(f"Unknown aggregation function: {agg_func}")

    monthly = monthly.reset_index()
    monthly = monthly.rename(columns={date_col: "Date"})

    logger.debug(f"Resampled to {len(monthly)} monthly observations")
    return monthly


def resample_quarterly_to_monthly(
    df: pd.DataFrame,
    date_col: str = "Date",
    method: str = "interpolate"
) -> pd.DataFrame:
    """
    Convert quarterly data to monthly frequency

    Args:
        df: Quarterly DataFrame
        date_col: Date column name
        method: Interpolation method

    Returns:
        Monthly DataFrame
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)

    # Resample to monthly with forward fill then interpolate
    monthly = df.resample('M').asfreq()

    if method == "interpolate":
        monthly = monthly.interpolate(method='linear')
    elif method == "ffill":
        monthly = monthly.ffill()
    elif method == "bfill":
        monthly = monthly.bfill()

    monthly = monthly.reset_index()
    monthly = monthly.rename(columns={date_col: "Date"})

    logger.debug(f"Converted quarterly to {len(monthly)} monthly observations")
    return monthly


def merge_datasets(
    datasets: Dict[str, pd.DataFrame],
    date_col: str = "Date",
    how: str = "outer"
) -> pd.DataFrame:
    """
    Merge multiple datasets on date

    Args:
        datasets: Dictionary of DataFrames
        date_col: Date column name
        how: Merge type ('inner', 'outer', 'left')

    Returns:
        Merged DataFrame
    """
    if not datasets:
        return pd.DataFrame()

    # Start with first dataset
    names = list(datasets.keys())
    merged = datasets[names[0]].copy()

    # Add suffix to columns
    rename_cols = {
        col: f"{col}_{names[0]}" if col != date_col else col
        for col in merged.columns
    }
    merged = merged.rename(columns=rename_cols)

    # Merge remaining datasets
    for name in names[1:]:
        df = datasets[name].copy()
        rename_cols = {
            col: f"{col}_{name}" if col != date_col else col
            for col in df.columns
        }
        df = df.rename(columns=rename_cols)

        merged = pd.merge(merged, df, on=date_col, how=how)

    merged = merged.sort_values(date_col).reset_index(drop=True)
    logger.info(f"Merged {len(datasets)} datasets: {merged.shape}")

    return merged
