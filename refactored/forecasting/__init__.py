"""
Forecasting module for Indonesia Economic Forecasting
"""
from forecasting.forecaster import (
    EconomicForecaster,
    generate_forecast,
    walk_forward_forecast,
    ensemble_forecast
)

__all__ = [
    'EconomicForecaster',
    'generate_forecast',
    'walk_forward_forecast',
    'ensemble_forecast'
]
