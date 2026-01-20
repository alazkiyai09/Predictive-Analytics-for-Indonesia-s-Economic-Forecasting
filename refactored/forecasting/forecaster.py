"""
Forecasting engine for Indonesia Economic Forecasting
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from config.settings import forecast_config, path_config
from training.trainer import ModelTrainer
from preprocessing.processor import create_sequences
from utils.logger import get_logger
from utils.helpers import calculate_metrics

logger = get_logger(__name__)


class EconomicForecaster:
    """Generate economic forecasts using trained models"""

    def __init__(
        self,
        trainer: Optional[ModelTrainer] = None,
        model_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize forecaster

        Args:
            trainer: Trained ModelTrainer instance
            model_path: Path to load trained model
        """
        self.trainer = trainer

        if model_path is not None and trainer is None:
            self.trainer = ModelTrainer()
            self.trainer.load(model_path)

        self.forecasts: Dict[str, pd.DataFrame] = {}

    def forecast(
        self,
        features: np.ndarray,
        n_steps: int = 12,
        return_confidence: bool = False,
        confidence_level: float = 0.95
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate forecast

        Args:
            features: Input features (last lookback periods)
            n_steps: Number of steps to forecast
            return_confidence: Whether to return confidence intervals
            confidence_level: Confidence level for intervals

        Returns:
            Forecasts array, or tuple of (forecast, lower, upper) if confidence requested
        """
        if self.trainer is None or not self.trainer.is_trained:
            raise ValueError("No trained model available")

        predictions = []
        current_features = features.copy()

        for _ in range(n_steps):
            # Predict next step
            if len(current_features.shape) == 2:
                # Add batch dimension
                X = current_features[np.newaxis, :, :]
            else:
                X = current_features

            pred = self.trainer.predict(X)
            predictions.append(pred[0])

            # Update features for next prediction (rolling window)
            # Shift features and append new prediction
            if len(current_features.shape) == 2:
                current_features = np.roll(current_features, -1, axis=0)
                # Update last row with new prediction (simplified)
                current_features[-1, 0] = pred[0]

        predictions = np.array(predictions)

        # Inverse transform
        forecasts = self.trainer.preprocessor.inverse_transform_target(predictions)

        if return_confidence:
            # Simple confidence interval based on historical error
            std_error = np.std(forecasts) * 0.1  # Simplified
            z_score = 1.96 if confidence_level == 0.95 else 2.576

            lower = forecasts - z_score * std_error * np.arange(1, n_steps + 1) ** 0.5
            upper = forecasts + z_score * std_error * np.arange(1, n_steps + 1) ** 0.5

            return forecasts, lower, upper

        return forecasts

    def generate_date_range(
        self,
        last_date: datetime,
        n_periods: int,
        freq: str = "M"
    ) -> pd.DatetimeIndex:
        """
        Generate forecast date range

        Args:
            last_date: Last historical date
            n_periods: Number of periods
            freq: Frequency ('M' for monthly, 'Q' for quarterly)

        Returns:
            DatetimeIndex for forecast periods
        """
        if freq == "M":
            dates = [last_date + relativedelta(months=i + 1) for i in range(n_periods)]
        elif freq == "Q":
            dates = [last_date + relativedelta(months=(i + 1) * 3) for i in range(n_periods)]
        else:
            dates = pd.date_range(start=last_date, periods=n_periods + 1, freq=freq)[1:]

        return pd.DatetimeIndex(dates)

    def create_forecast_dataframe(
        self,
        forecasts: np.ndarray,
        dates: pd.DatetimeIndex,
        lower: Optional[np.ndarray] = None,
        upper: Optional[np.ndarray] = None,
        name: str = "forecast"
    ) -> pd.DataFrame:
        """
        Create forecast DataFrame

        Args:
            forecasts: Forecast values
            dates: Forecast dates
            lower: Lower confidence bound
            upper: Upper confidence bound
            name: Name for forecast column

        Returns:
            DataFrame with forecasts
        """
        df = pd.DataFrame({
            'Date': dates,
            name: forecasts
        })

        if lower is not None:
            df[f'{name}_lower'] = lower
        if upper is not None:
            df[f'{name}_upper'] = upper

        self.forecasts[name] = df
        return df

    def save_forecast(
        self,
        df: pd.DataFrame,
        filename: str,
        output_dir: Optional[Path] = None
    ):
        """Save forecast to CSV"""
        if output_dir is None:
            output_dir = path_config.REPORTS_DIR

        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / filename

        df.to_csv(filepath, index=False)
        logger.info(f"Forecast saved to {filepath}")


def generate_forecast(
    trainer: ModelTrainer,
    features: pd.DataFrame,
    target: pd.Series,
    n_steps: int = 12,
    last_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Generate forecast from trained model

    Args:
        trainer: Trained ModelTrainer
        features: Feature DataFrame
        target: Target Series (for preprocessing)
        n_steps: Number of steps to forecast
        last_date: Last historical date

    Returns:
        DataFrame with forecasts
    """
    # Prepare input features
    scaled_features, _ = trainer.preprocessor.transform(features, target)

    # Get last lookback periods
    lookback = trainer.lookback
    last_features = scaled_features[-lookback:]

    # Create forecaster and generate forecast
    forecaster = EconomicForecaster(trainer=trainer)
    forecasts, lower, upper = forecaster.forecast(
        last_features,
        n_steps=n_steps,
        return_confidence=True
    )

    # Generate dates
    if last_date is None:
        last_date = datetime.now()

    dates = forecaster.generate_date_range(last_date, n_steps)

    # Create DataFrame
    result = forecaster.create_forecast_dataframe(
        forecasts, dates, lower, upper, name="forecast"
    )

    return result


def walk_forward_forecast(
    trainer: ModelTrainer,
    features: pd.DataFrame,
    target: pd.Series,
    n_steps: int = 12,
    update_features: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Walk-forward forecasting with feature updates

    Args:
        trainer: Trained ModelTrainer
        features: Feature DataFrame
        target: Target Series
        n_steps: Number of steps
        update_features: Whether to update features with predictions

    Returns:
        Tuple of (forecasts, confidence_widths)
    """
    if not trainer.is_trained:
        raise ValueError("Model must be trained first")

    lookback = trainer.lookback
    scaled_features, scaled_target = trainer.preprocessor.transform(features, target)

    # Start with last lookback periods
    current_window = scaled_features[-lookback:].copy()

    forecasts = []
    confidence_widths = []

    for step in range(n_steps):
        # Reshape for model
        X = current_window[np.newaxis, :, :]

        # Predict
        pred = trainer.predict(X)[0]
        forecasts.append(pred)

        # Confidence width increases with forecast horizon
        width = 0.1 * (1 + step * 0.1)  # Simplified
        confidence_widths.append(width)

        if update_features:
            # Roll window and update
            current_window = np.roll(current_window, -1, axis=0)
            # Simple update: use prediction as first feature
            current_window[-1, 0] = pred

    # Inverse transform
    forecasts = trainer.preprocessor.inverse_transform_target(np.array(forecasts))

    return forecasts, np.array(confidence_widths)


def ensemble_forecast(
    trainers: List[ModelTrainer],
    features: pd.DataFrame,
    target: pd.Series,
    n_steps: int = 12,
    aggregation: str = "mean",
    weights: Optional[List[float]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate ensemble forecast from multiple models

    Args:
        trainers: List of trained ModelTrainers
        features: Feature DataFrame
        target: Target Series
        n_steps: Number of steps
        aggregation: Aggregation method ('mean', 'median', 'weighted')
        weights: Weights for weighted aggregation

    Returns:
        Tuple of (ensemble_forecast, lower_bound, upper_bound)
    """
    all_forecasts = []

    for trainer in trainers:
        forecasts, _ = walk_forward_forecast(
            trainer, features, target, n_steps, update_features=True
        )
        all_forecasts.append(forecasts)

    all_forecasts = np.array(all_forecasts)

    # Aggregate
    if aggregation == "mean":
        ensemble_pred = np.mean(all_forecasts, axis=0)
    elif aggregation == "median":
        ensemble_pred = np.median(all_forecasts, axis=0)
    elif aggregation == "weighted" and weights is not None:
        weights = np.array(weights)
        weights = weights / weights.sum()
        ensemble_pred = np.average(all_forecasts, axis=0, weights=weights)
    else:
        ensemble_pred = np.mean(all_forecasts, axis=0)

    # Confidence bounds from ensemble spread
    std = np.std(all_forecasts, axis=0)
    lower = ensemble_pred - 1.96 * std
    upper = ensemble_pred + 1.96 * std

    logger.info(f"Ensemble forecast from {len(trainers)} models")
    return ensemble_pred, lower, upper


def backtest_forecast(
    trainer: ModelTrainer,
    features: pd.DataFrame,
    target: pd.Series,
    test_periods: int = 24,
    forecast_horizon: int = 12
) -> Dict[str, float]:
    """
    Backtest forecasting performance

    Args:
        trainer: Trained ModelTrainer
        features: Feature DataFrame
        target: Target Series
        test_periods: Number of periods to test
        forecast_horizon: Forecast horizon for each test

    Returns:
        Dictionary of backtest metrics
    """
    lookback = trainer.lookback
    all_errors = []

    for i in range(test_periods):
        # Get historical data up to this point
        end_idx = len(features) - test_periods + i

        if end_idx < lookback + forecast_horizon:
            continue

        hist_features = features.iloc[:end_idx]
        hist_target = target.iloc[:end_idx]

        # Prepare data
        scaled_features, scaled_target = trainer.preprocessor.transform(
            hist_features, hist_target
        )

        # Get last lookback periods
        last_features = scaled_features[-lookback:]

        # Forecast
        forecaster = EconomicForecaster(trainer=trainer)
        forecasts = forecaster.forecast(last_features, n_steps=forecast_horizon)

        # Get actual values
        actual_start = end_idx
        actual_end = min(actual_start + forecast_horizon, len(target))
        actuals = target.iloc[actual_start:actual_end].values

        # Trim forecasts if needed
        forecasts = forecasts[:len(actuals)]

        # Calculate error
        error = np.abs(forecasts - actuals) / (np.abs(actuals) + 1e-8)
        all_errors.extend(error.tolist())

    # Aggregate metrics
    all_errors = np.array(all_errors)

    metrics = {
        'mape': np.mean(all_errors) * 100,
        'mape_std': np.std(all_errors) * 100,
        'max_error': np.max(all_errors) * 100,
        'min_error': np.min(all_errors) * 100,
        'n_forecasts': len(all_errors)
    }

    logger.info(f"Backtest results: MAPE={metrics['mape']:.2f}%")
    return metrics


class MultiTargetForecaster:
    """Forecast multiple economic indicators"""

    def __init__(self):
        """Initialize multi-target forecaster"""
        self.trainers: Dict[str, ModelTrainer] = {}
        self.forecasts: Dict[str, pd.DataFrame] = {}

    def add_trainer(self, target_name: str, trainer: ModelTrainer):
        """Add trained model for a target"""
        self.trainers[target_name] = trainer
        logger.info(f"Added trainer for {target_name}")

    def forecast_all(
        self,
        features_dict: Dict[str, pd.DataFrame],
        target_dict: Dict[str, pd.Series],
        n_steps: int = 12,
        last_date: Optional[datetime] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Forecast all targets

        Args:
            features_dict: Dictionary of features for each target
            target_dict: Dictionary of target series
            n_steps: Number of steps
            last_date: Last historical date

        Returns:
            Dictionary of forecast DataFrames
        """
        for name, trainer in self.trainers.items():
            if name not in features_dict or name not in target_dict:
                logger.warning(f"Missing data for {name}, skipping")
                continue

            try:
                forecast_df = generate_forecast(
                    trainer=trainer,
                    features=features_dict[name],
                    target=target_dict[name],
                    n_steps=n_steps,
                    last_date=last_date
                )
                self.forecasts[name] = forecast_df
                logger.info(f"Generated forecast for {name}")

            except Exception as e:
                logger.error(f"Error forecasting {name}: {e}")

        return self.forecasts

    def combine_forecasts(self) -> pd.DataFrame:
        """Combine all forecasts into single DataFrame"""
        if not self.forecasts:
            return pd.DataFrame()

        combined = None

        for name, df in self.forecasts.items():
            df_renamed = df.rename(columns={
                'forecast': f'{name}_forecast',
                'forecast_lower': f'{name}_lower',
                'forecast_upper': f'{name}_upper'
            })

            if combined is None:
                combined = df_renamed
            else:
                combined = pd.merge(combined, df_renamed, on='Date', how='outer')

        return combined.sort_values('Date').reset_index(drop=True)
