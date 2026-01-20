"""
Tests for preprocessing module
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.processor import (
    DataPreprocessor,
    create_sequences,
    scale_data,
    apply_pca,
    prepare_train_test_split,
    handle_missing_values,
    create_lag_features,
    create_rolling_features
)


class TestDataPreprocessor:
    """Test DataPreprocessor class"""

    def test_init(self):
        """Test preprocessor initialization"""
        preprocessor = DataPreprocessor()
        assert preprocessor.scaler_type == "minmax"
        assert preprocessor.use_pca is True
        assert not preprocessor.is_fitted

    def test_fit_transform(self):
        """Test fit and transform"""
        # Create sample data
        np.random.seed(42)
        features = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100) * 10,
            'feature3': np.random.randn(100) + 5
        })
        target = pd.Series(np.random.randn(100))

        preprocessor = DataPreprocessor(use_pca=False)
        scaled_features, scaled_target = preprocessor.fit_transform(features, target)

        assert preprocessor.is_fitted
        assert scaled_features.shape[0] == 100
        assert scaled_target.shape[0] == 100
        # Check scaling
        assert scaled_features.min() >= 0
        assert scaled_features.max() <= 1

    def test_inverse_transform_target(self):
        """Test inverse transform"""
        np.random.seed(42)
        target = pd.Series(np.random.randn(100) * 100 + 50)
        features = pd.DataFrame({'f1': np.random.randn(100)})

        preprocessor = DataPreprocessor(use_pca=False)
        _, scaled_target = preprocessor.fit_transform(features, target)

        # Inverse transform
        recovered = preprocessor.inverse_transform_target(scaled_target)

        # Check recovery
        np.testing.assert_array_almost_equal(recovered, target.values, decimal=5)

    def test_pca_reduction(self):
        """Test PCA dimensionality reduction"""
        np.random.seed(42)
        features = pd.DataFrame(np.random.randn(100, 20))
        target = pd.Series(np.random.randn(100))

        preprocessor = DataPreprocessor(use_pca=True, pca_variance=0.95)
        scaled_features, _ = preprocessor.fit_transform(features, target)

        # PCA should reduce dimensions
        assert scaled_features.shape[1] < 20


class TestCreateSequences:
    """Test create_sequences function"""

    def test_basic_sequences(self):
        """Test basic sequence creation"""
        features = np.random.randn(100, 5)
        target = np.random.randn(100)

        X, y = create_sequences(features, target, lookback=10)

        assert X.shape == (90, 10, 5)
        assert y.shape == (90,)

    def test_multi_step_forecast(self):
        """Test multi-step forecast sequences"""
        features = np.random.randn(100, 5)
        target = np.random.randn(100)

        X, y = create_sequences(features, target, lookback=10, forecast_horizon=3)

        assert X.shape[0] == y.shape[0]
        assert y.shape[1] == 3


class TestScaleData:
    """Test scale_data function"""

    def test_minmax_scaling(self):
        """Test MinMax scaling"""
        train = pd.DataFrame(np.random.randn(80, 5))
        test = pd.DataFrame(np.random.randn(20, 5))

        scaled_train, scaled_test, scaler = scale_data(train, test, "minmax")

        assert scaled_train.min() >= 0
        assert scaled_train.max() <= 1
        assert scaled_test is not None

    def test_standard_scaling(self):
        """Test Standard scaling"""
        train = pd.DataFrame(np.random.randn(80, 5) * 100)

        scaled_train, _, scaler = scale_data(train, scaler_type="standard")

        # Standard scaled data should have mean ~0, std ~1
        assert abs(scaled_train.mean()) < 0.1
        assert abs(scaled_train.std() - 1) < 0.1


class TestApplyPca:
    """Test apply_pca function"""

    def test_variance_retention(self):
        """Test PCA variance retention"""
        np.random.seed(42)
        train = np.random.randn(100, 20)

        pca_train, _, pca = apply_pca(train, n_components=0.95)

        # Check variance retained
        total_var = sum(pca.explained_variance_ratio_)
        assert total_var >= 0.95

    def test_dimension_reduction(self):
        """Test dimension reduction"""
        train = np.random.randn(100, 50)

        pca_train, _, pca = apply_pca(train, n_components=10)

        assert pca_train.shape[1] == 10


class TestPrepareTrainTestSplit:
    """Test prepare_train_test_split function"""

    def test_time_series_split(self):
        """Test time series aware split"""
        dates = pd.date_range(start='2020-01-01', periods=100, freq='M')
        df = pd.DataFrame({
            'Date': dates,
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randn(100)
        })

        X_train, X_test, y_train, y_test = prepare_train_test_split(
            df, target_col='target', train_ratio=0.8
        )

        assert len(X_train) == 80
        assert len(X_test) == 20


class TestHandleMissingValues:
    """Test handle_missing_values function"""

    def test_ffill(self):
        """Test forward fill"""
        df = pd.DataFrame({
            'a': [1, np.nan, 3, np.nan, 5],
            'b': [10, 20, np.nan, 40, 50]
        })

        result = handle_missing_values(df, strategy="ffill")

        assert result.isnull().sum().sum() == 0

    def test_interpolate(self):
        """Test interpolation"""
        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, np.nan, 5.0]
        })

        result = handle_missing_values(df, strategy="interpolate")

        assert result.isnull().sum().sum() == 0
        assert result['a'].iloc[1] == 2.0


class TestCreateLagFeatures:
    """Test create_lag_features function"""

    def test_lag_creation(self):
        """Test lag feature creation"""
        df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })

        result = create_lag_features(df, columns=['value'], lags=[1, 2, 3])

        assert 'value_lag_1' in result.columns
        assert 'value_lag_2' in result.columns
        assert 'value_lag_3' in result.columns
        assert result['value_lag_1'].iloc[3] == 3


class TestCreateRollingFeatures:
    """Test create_rolling_features function"""

    def test_rolling_creation(self):
        """Test rolling feature creation"""
        df = pd.DataFrame({
            'value': np.arange(20, dtype=float)
        })

        result = create_rolling_features(df, columns=['value'], windows=[3, 5])

        assert 'value_roll_mean_3' in result.columns
        assert 'value_roll_std_5' in result.columns
        assert 'value_roll_min_3' in result.columns
        assert 'value_roll_max_5' in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
