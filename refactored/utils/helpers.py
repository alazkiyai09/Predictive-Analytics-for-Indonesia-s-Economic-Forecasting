"""
Helper utilities for Indonesia Economic Forecasting
"""
import os
import random
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional
import pandas as pd


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed value
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def create_sequences(
    data: np.ndarray,
    target: np.ndarray,
    lookback: int,
    horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series modeling

    Args:
        data: Feature array (n_samples, n_features)
        target: Target array (n_samples,)
        lookback: Number of past timesteps
        horizon: Number of future timesteps to predict

    Returns:
        Tuple of (X, y) sequences
    """
    X, y = [], []

    for i in range(lookback, len(data) - horizon + 1):
        X.append(data[i - lookback:i])
        y.append(target[i:i + horizon])

    return np.array(X), np.array(y)


def inverse_transform_predictions(
    predictions: np.ndarray,
    scaler,
    feature_idx: int = 0,
    n_features: int = 1
) -> np.ndarray:
    """
    Inverse transform scaled predictions

    Args:
        predictions: Scaled predictions
        scaler: Fitted scaler object
        feature_idx: Index of target feature in scaler
        n_features: Total number of features

    Returns:
        Unscaled predictions
    """
    # Create dummy array for inverse transform
    dummy = np.zeros((len(predictions), n_features))
    dummy[:, feature_idx] = predictions.flatten()
    inverted = scaler.inverse_transform(dummy)

    return inverted[:, feature_idx]


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray
) -> dict:
    """
    Calculate forecasting metrics

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score
    )

    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100
    r2 = r2_score(actual, predicted)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }


def format_currency(
    amount: float,
    currency: str = "IDR"
) -> str:
    """Format currency for display"""
    if currency == "IDR":
        return f"Rp {amount:,.0f}"
    return f"{currency} {amount:,.2f}"


def detect_frequency(df: pd.DataFrame, date_col: str = "Date") -> str:
    """
    Detect data frequency

    Args:
        df: DataFrame with datetime index or column
        date_col: Name of date column

    Returns:
        Frequency string ('D', 'M', 'Q', 'Y')
    """
    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col])
    else:
        dates = df.index

    diff = dates.diff().median()

    if diff <= pd.Timedelta(days=1):
        return 'D'
    elif diff <= pd.Timedelta(days=32):
        return 'M'
    elif diff <= pd.Timedelta(days=95):
        return 'Q'
    else:
        return 'Y'


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safe division with default for zero denominator"""
    return a / b if b != 0 else default
