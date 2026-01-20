"""
Data module for Indonesia Economic Forecasting
"""
from data.loader import (
    DataLoader,
    resample_to_monthly,
    resample_quarterly_to_monthly,
    merge_datasets
)

__all__ = [
    'DataLoader',
    'resample_to_monthly',
    'resample_quarterly_to_monthly',
    'merge_datasets'
]
