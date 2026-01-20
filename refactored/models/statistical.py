"""
Statistical models for Indonesia Economic Forecasting
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import warnings

# Statistical model imports
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from sklearn.base import BaseEstimator, RegressorMixin
from config.settings import model_config
from utils.logger import get_logger

logger = get_logger(__name__)


class SARIMAXWrapper(BaseEstimator, RegressorMixin):
    """Wrapper for SARIMAX model with sklearn interface"""

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12),
        trend: str = 'c',
        enforce_stationarity: bool = False,
        enforce_invertibility: bool = False
    ):
        """
        Initialize SARIMAX wrapper

        Args:
            order: (p, d, q) order
            seasonal_order: (P, D, Q, s) seasonal order
            trend: Trend component
            enforce_stationarity: Enforce stationarity
            enforce_invertibility: Enforce invertibility
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.model = None
        self.fitted_model = None

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        exog: Optional[Union[np.ndarray, pd.DataFrame]] = None
    ) -> "SARIMAXWrapper":
        """
        Fit SARIMAX model

        Args:
            y: Target time series
            exog: Optional exogenous variables

        Returns:
            self
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for SARIMAX")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.model = SARIMAX(
                y,
                exog=exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility
            )

            self.fitted_model = self.model.fit(disp=False, maxiter=200)

        logger.info(f"SARIMAX fitted with AIC: {self.fitted_model.aic:.2f}")
        return self

    def predict(
        self,
        steps: int = 1,
        exog: Optional[Union[np.ndarray, pd.DataFrame]] = None
    ) -> np.ndarray:
        """
        Forecast future values

        Args:
            steps: Number of steps to forecast
            exog: Exogenous variables for forecast period

        Returns:
            Forecast array
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        forecast = self.fitted_model.forecast(steps=steps, exog=exog)
        return np.array(forecast)

    def get_confidence_interval(
        self,
        steps: int = 1,
        alpha: float = 0.05,
        exog: Optional[Union[np.ndarray, pd.DataFrame]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get forecast confidence intervals

        Args:
            steps: Number of steps
            alpha: Significance level
            exog: Exogenous variables

        Returns:
            Tuple of (lower, upper) bounds
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        forecast = self.fitted_model.get_forecast(steps=steps, exog=exog)
        conf_int = forecast.conf_int(alpha=alpha)

        return conf_int.iloc[:, 0].values, conf_int.iloc[:, 1].values

    def get_summary(self) -> str:
        """Get model summary"""
        if self.fitted_model is None:
            return "Model not fitted"
        return str(self.fitted_model.summary())


class XGBoostWrapper(BaseEstimator, RegressorMixin):
    """Wrapper for XGBoost with time series features"""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 1,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: int = 42
    ):
        """
        Initialize XGBoost wrapper

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            subsample: Row subsampling ratio
            colsample_bytree: Column subsampling ratio
            min_child_weight: Minimum child weight
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            random_state: Random seed
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.model = None
        self.feature_names = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        eval_set: Optional[List[Tuple]] = None,
        early_stopping_rounds: Optional[int] = 10,
        verbose: bool = False
    ) -> "XGBoostWrapper":
        """
        Fit XGBoost model

        Args:
            X: Features
            y: Target
            eval_set: Evaluation set for early stopping
            early_stopping_rounds: Early stopping patience
            verbose: Verbosity

        Returns:
            self
        """
        if not XGB_AVAILABLE:
            raise ImportError("xgboost required")

        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()

        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            objective='reg:squarederror'
        )

        fit_params = {
            'eval_set': eval_set,
            'verbose': verbose
        }

        if early_stopping_rounds and eval_set:
            fit_params['early_stopping_rounds'] = early_stopping_rounds

        self.model.fit(X, y, **fit_params)

        logger.info(f"XGBoost fitted with {self.n_estimators} estimators")
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict using fitted model

        Args:
            X: Features

        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.model.predict(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if self.model is None:
            raise ValueError("Model not fitted")

        importance = self.model.feature_importances_

        if self.feature_names:
            return dict(zip(self.feature_names, importance))
        return {f"feature_{i}": v for i, v in enumerate(importance)}


def build_sarimax_model(
    order: Tuple[int, int, int] = None,
    seasonal_order: Tuple[int, int, int, int] = None
) -> SARIMAXWrapper:
    """
    Build SARIMAX model

    Args:
        order: ARIMA order (p, d, q)
        seasonal_order: Seasonal order (P, D, Q, s)

    Returns:
        SARIMAXWrapper instance
    """
    if order is None:
        order = model_config.SARIMAX_ORDER
    if seasonal_order is None:
        seasonal_order = model_config.SARIMAX_SEASONAL_ORDER

    return SARIMAXWrapper(
        order=order,
        seasonal_order=seasonal_order
    )


def build_xgboost_model(
    n_estimators: int = None,
    max_depth: int = None,
    learning_rate: float = None
) -> XGBoostWrapper:
    """
    Build XGBoost model

    Args:
        n_estimators: Number of estimators
        max_depth: Max tree depth
        learning_rate: Learning rate

    Returns:
        XGBoostWrapper instance
    """
    if n_estimators is None:
        n_estimators = model_config.XGB_N_ESTIMATORS
    if max_depth is None:
        max_depth = model_config.XGB_MAX_DEPTH
    if learning_rate is None:
        learning_rate = model_config.XGB_LEARNING_RATE

    return XGBoostWrapper(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate
    )


def auto_arima_order(
    series: pd.Series,
    max_p: int = 5,
    max_d: int = 2,
    max_q: int = 5,
    seasonal: bool = True,
    m: int = 12
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
    """
    Automatically determine ARIMA order using AIC

    Args:
        series: Time series data
        max_p: Maximum p value
        max_d: Maximum d value
        max_q: Maximum q value
        seasonal: Include seasonal component
        m: Seasonal period

    Returns:
        Tuple of (order, seasonal_order)
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels required")

    try:
        from pmdarima import auto_arima
        model = auto_arima(
            series,
            max_p=max_p,
            max_d=max_d,
            max_q=max_q,
            seasonal=seasonal,
            m=m,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore'
        )
        order = model.order
        seasonal_order = model.seasonal_order
        logger.info(f"Auto ARIMA: order={order}, seasonal={seasonal_order}")
        return order, seasonal_order

    except ImportError:
        # Fallback to default
        logger.warning("pmdarima not installed, using default orders")
        return (1, 1, 1), (1, 1, 1, m)


def create_time_features(
    df: pd.DataFrame,
    date_col: str = "Date"
) -> pd.DataFrame:
    """
    Create time-based features for XGBoost

    Args:
        df: Input DataFrame
        date_col: Date column name

    Returns:
        DataFrame with time features
    """
    df = df.copy()

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])

        # Extract time features
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['quarter'] = df[date_col].dt.quarter
        df['day_of_year'] = df[date_col].dt.dayofyear

        # Cyclical encoding for month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Cyclical encoding for quarter
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)

    logger.debug("Added time-based features")
    return df
