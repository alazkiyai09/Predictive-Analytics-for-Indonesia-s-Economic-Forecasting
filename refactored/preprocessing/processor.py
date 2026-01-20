"""
Data preprocessing module for Indonesia Economic Forecasting
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from datetime import datetime

from config.settings import preprocessing_config, model_config
from utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """Preprocess economic data for ML models"""

    def __init__(
        self,
        scaler_type: str = "minmax",
        use_pca: bool = True,
        pca_variance: float = 0.95
    ):
        """
        Initialize preprocessor

        Args:
            scaler_type: Type of scaler ('minmax', 'standard', 'robust')
            use_pca: Whether to apply PCA
            pca_variance: Variance to retain with PCA
        """
        self.scaler_type = scaler_type
        self.use_pca = use_pca
        self.pca_variance = pca_variance

        # Initialize scalers
        self.feature_scaler = self._create_scaler(scaler_type)
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.pca = None

        # State tracking
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.n_features_original: int = 0
        self.n_features_pca: int = 0

    def _create_scaler(self, scaler_type: str):
        """Create scaler based on type"""
        scalers = {
            "minmax": MinMaxScaler(feature_range=(0, 1)),
            "standard": StandardScaler(),
            "robust": RobustScaler()
        }
        return scalers.get(scaler_type, MinMaxScaler(feature_range=(0, 1)))

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series
    ) -> "DataPreprocessor":
        """
        Fit preprocessor on training data

        Args:
            features: Feature DataFrame
            target: Target Series

        Returns:
            self
        """
        self.feature_names = features.columns.tolist()
        self.n_features_original = len(self.feature_names)

        # Fit feature scaler
        self.feature_scaler.fit(features)

        # Fit target scaler
        self.target_scaler.fit(target.values.reshape(-1, 1))

        # Fit PCA if enabled
        if self.use_pca:
            scaled_features = self.feature_scaler.transform(features)
            self.pca = PCA(n_components=self.pca_variance)
            self.pca.fit(scaled_features)
            self.n_features_pca = self.pca.n_components_

            logger.info(
                f"PCA fitted: {self.n_features_original} features -> "
                f"{self.n_features_pca} components "
                f"({self.pca_variance * 100:.1f}% variance retained)"
            )

        self.is_fitted = True
        logger.info(f"Preprocessor fitted on {len(features)} samples")
        return self

    def transform(
        self,
        features: pd.DataFrame,
        target: Optional[pd.Series] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform features and target

        Args:
            features: Feature DataFrame
            target: Optional target Series

        Returns:
            Tuple of (transformed_features, transformed_target)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        # Scale features
        scaled_features = self.feature_scaler.transform(features)

        # Apply PCA if enabled
        if self.use_pca and self.pca is not None:
            scaled_features = self.pca.transform(scaled_features)

        # Scale target if provided
        scaled_target = None
        if target is not None:
            scaled_target = self.target_scaler.transform(
                target.values.reshape(-1, 1)
            ).flatten()

        return scaled_features, scaled_target

    def fit_transform(
        self,
        features: pd.DataFrame,
        target: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit and transform in one step

        Args:
            features: Feature DataFrame
            target: Target Series

        Returns:
            Tuple of (transformed_features, transformed_target)
        """
        self.fit(features, target)
        return self.transform(features, target)

    def inverse_transform_target(self, scaled_values: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled target values

        Args:
            scaled_values: Scaled values

        Returns:
            Original scale values
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        values = np.array(scaled_values).reshape(-1, 1)
        return self.target_scaler.inverse_transform(values).flatten()

    def get_pca_explained_variance(self) -> Optional[np.ndarray]:
        """Get PCA explained variance ratio"""
        if self.pca is not None:
            return self.pca.explained_variance_ratio_
        return None


def create_sequences(
    features: np.ndarray,
    target: np.ndarray,
    lookback: int,
    forecast_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series modeling

    Args:
        features: Feature array (n_samples, n_features)
        target: Target array (n_samples,)
        lookback: Number of past timesteps to use
        forecast_horizon: Number of future steps to predict

    Returns:
        Tuple of (X sequences, y targets)
    """
    X, y = [], []

    for i in range(lookback, len(features) - forecast_horizon + 1):
        X.append(features[i - lookback:i])
        y.append(target[i:i + forecast_horizon])

    X = np.array(X)
    y = np.array(y)

    if forecast_horizon == 1:
        y = y.flatten()

    logger.debug(f"Created {len(X)} sequences with lookback={lookback}")
    return X, y


def scale_data(
    train_data: pd.DataFrame,
    test_data: Optional[pd.DataFrame] = None,
    scaler_type: str = "minmax"
) -> Tuple[np.ndarray, Optional[np.ndarray], object]:
    """
    Scale training and test data

    Args:
        train_data: Training DataFrame
        test_data: Optional test DataFrame
        scaler_type: Type of scaler

    Returns:
        Tuple of (scaled_train, scaled_test, scaler)
    """
    scalers = {
        "minmax": MinMaxScaler(feature_range=(0, 1)),
        "standard": StandardScaler(),
        "robust": RobustScaler()
    }
    scaler = scalers.get(scaler_type, MinMaxScaler(feature_range=(0, 1)))

    scaled_train = scaler.fit_transform(train_data)

    scaled_test = None
    if test_data is not None:
        scaled_test = scaler.transform(test_data)

    logger.debug(f"Scaled data using {scaler_type} scaler")
    return scaled_train, scaled_test, scaler


def apply_pca(
    train_data: np.ndarray,
    test_data: Optional[np.ndarray] = None,
    n_components: Union[int, float] = 0.95
) -> Tuple[np.ndarray, Optional[np.ndarray], PCA]:
    """
    Apply PCA dimensionality reduction

    Args:
        train_data: Training data array
        test_data: Optional test data array
        n_components: Number of components or variance ratio

    Returns:
        Tuple of (transformed_train, transformed_test, pca)
    """
    pca = PCA(n_components=n_components)
    train_pca = pca.fit_transform(train_data)

    test_pca = None
    if test_data is not None:
        test_pca = pca.transform(test_data)

    logger.info(
        f"PCA: {train_data.shape[1]} -> {train_pca.shape[1]} features "
        f"({sum(pca.explained_variance_ratio_) * 100:.1f}% variance)"
    )

    return train_pca, test_pca, pca


def prepare_train_test_split(
    data: pd.DataFrame,
    target_col: str,
    train_ratio: float = 0.8,
    date_col: Optional[str] = "Date"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets (time-series aware)

    Args:
        data: Input DataFrame
        target_col: Target column name
        train_ratio: Ratio for training set
        date_col: Date column name for sorting

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    df = data.copy()

    # Sort by date if available
    if date_col and date_col in df.columns:
        df = df.sort_values(date_col).reset_index(drop=True)

    # Split point
    split_idx = int(len(df) * train_ratio)

    # Separate features and target
    feature_cols = [c for c in df.columns if c not in [target_col, date_col]]

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[target_col]
    y_test = test_df[target_col]

    logger.info(
        f"Train/test split: {len(X_train)} train, {len(X_test)} test "
        f"({train_ratio * 100:.0f}/{(1 - train_ratio) * 100:.0f})"
    )

    return X_train, X_test, y_train, y_test


def merge_and_align_data(
    datasets: Dict[str, pd.DataFrame],
    date_col: str = "Date",
    method: str = "ffill"
) -> pd.DataFrame:
    """
    Merge and align multiple datasets on date

    Args:
        datasets: Dictionary of DataFrames
        date_col: Date column name
        method: Fill method for missing values

    Returns:
        Merged and aligned DataFrame
    """
    if not datasets:
        return pd.DataFrame()

    # Start with first dataset
    names = list(datasets.keys())
    merged = datasets[names[0]].copy()

    # Ensure date column is datetime
    if date_col in merged.columns:
        merged[date_col] = pd.to_datetime(merged[date_col])

    # Add suffix to columns
    rename_cols = {
        col: f"{col}_{names[0]}" if col != date_col else col
        for col in merged.columns
    }
    merged = merged.rename(columns=rename_cols)

    # Merge remaining datasets
    for name in names[1:]:
        df = datasets[name].copy()

        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])

        rename_cols = {
            col: f"{col}_{name}" if col != date_col else col
            for col in df.columns
        }
        df = df.rename(columns=rename_cols)

        merged = pd.merge(merged, df, on=date_col, how="outer")

    # Sort by date
    merged = merged.sort_values(date_col).reset_index(drop=True)

    # Fill missing values
    if method == "ffill":
        merged = merged.ffill()
    elif method == "bfill":
        merged = merged.bfill()
    elif method == "interpolate":
        numeric_cols = merged.select_dtypes(include=[np.number]).columns
        merged[numeric_cols] = merged[numeric_cols].interpolate(method="linear")

    # Drop remaining NaNs
    merged = merged.dropna()

    logger.info(f"Merged {len(datasets)} datasets: {merged.shape}")
    return merged


def calculate_returns(
    df: pd.DataFrame,
    price_col: str,
    periods: List[int] = None
) -> pd.DataFrame:
    """
    Calculate returns for different periods

    Args:
        df: DataFrame with price data
        price_col: Price column name
        periods: List of periods for returns

    Returns:
        DataFrame with return columns added
    """
    if periods is None:
        periods = [1, 5, 21]  # Daily, weekly, monthly

    df = df.copy()

    for period in periods:
        # Simple return
        df[f"return_{period}d"] = df[price_col].pct_change(period)

        # Log return
        df[f"log_return_{period}d"] = np.log(
            df[price_col] / df[price_col].shift(period)
        )

    logger.debug(f"Added returns for periods: {periods}")
    return df


def create_lag_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int]
) -> pd.DataFrame:
    """
    Create lagged features

    Args:
        df: Input DataFrame
        columns: Columns to create lags for
        lags: Lag periods

    Returns:
        DataFrame with lag features
    """
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    logger.debug(f"Added {len(columns) * len(lags)} lag features")
    return df


def create_rolling_features(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int]
) -> pd.DataFrame:
    """
    Create rolling statistical features

    Args:
        df: Input DataFrame
        columns: Columns for rolling features
        windows: Window sizes

    Returns:
        DataFrame with rolling features
    """
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        for window in windows:
            # Mean
            df[f"{col}_roll_mean_{window}"] = df[col].rolling(window).mean()

            # Std
            df[f"{col}_roll_std_{window}"] = df[col].rolling(window).std()

            # Min/Max
            df[f"{col}_roll_min_{window}"] = df[col].rolling(window).min()
            df[f"{col}_roll_max_{window}"] = df[col].rolling(window).max()

    logger.debug(f"Added rolling features for windows: {windows}")
    return df


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "ffill",
    numeric_fill: Optional[float] = None
) -> pd.DataFrame:
    """
    Handle missing values in DataFrame

    Args:
        df: Input DataFrame
        strategy: Fill strategy ('ffill', 'bfill', 'interpolate', 'fill')
        numeric_fill: Value to fill numeric columns (if strategy='fill')

    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    initial_nulls = df.isnull().sum().sum()

    if strategy == "ffill":
        df = df.ffill()
    elif strategy == "bfill":
        df = df.bfill()
    elif strategy == "interpolate":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear")
    elif strategy == "fill" and numeric_fill is not None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(numeric_fill)

    # Final cleanup
    df = df.ffill().bfill()

    final_nulls = df.isnull().sum().sum()
    logger.info(f"Missing values: {initial_nulls} -> {final_nulls}")

    return df


def remove_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "iqr",
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    Remove or cap outliers

    Args:
        df: Input DataFrame
        columns: Columns to check (None = all numeric)
        method: Detection method ('iqr', 'zscore')
        threshold: Threshold for outlier detection

    Returns:
        DataFrame with outliers handled
    """
    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    outliers_removed = 0

    for col in columns:
        if col not in df.columns:
            continue

        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR

            mask = (df[col] >= lower) & (df[col] <= upper)
            outliers_removed += (~mask).sum()

            # Cap outliers instead of removing
            df[col] = df[col].clip(lower, upper)

        elif method == "zscore":
            mean = df[col].mean()
            std = df[col].std()
            z_scores = np.abs((df[col] - mean) / std)
            mask = z_scores <= threshold
            outliers_removed += (~mask).sum()

            # Cap at threshold std devs
            df[col] = df[col].clip(
                mean - threshold * std,
                mean + threshold * std
            )

    logger.info(f"Capped {outliers_removed} outlier values")
    return df
