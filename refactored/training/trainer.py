"""
Training pipeline for Indonesia Economic Forecasting
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json
from datetime import datetime
import pickle

from sklearn.model_selection import TimeSeriesSplit

from config.settings import (
    model_config, training_config, path_config
)
from preprocessing.processor import (
    DataPreprocessor, create_sequences
)
from models.architectures import ModelFactory
from models.statistical import SARIMAXWrapper, XGBoostWrapper
from utils.logger import get_logger
from utils.helpers import calculate_metrics, set_random_seed

logger = get_logger(__name__)


class ModelTrainer:
    """Train and manage ML models for economic forecasting"""

    def __init__(
        self,
        model_type: str = "lstm",
        lookback: int = 12,
        forecast_horizon: int = 1,
        preprocessor: Optional[DataPreprocessor] = None
    ):
        """
        Initialize trainer

        Args:
            model_type: Type of model ('lstm', 'gru', 'cnn_lstm', 'ensemble', 'sarimax', 'xgboost')
            lookback: Number of past periods to use
            forecast_horizon: Number of future periods to predict
            preprocessor: Optional preprocessor instance
        """
        self.model_type = model_type.lower()
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.preprocessor = preprocessor or DataPreprocessor()

        self.model = None
        self.history = None
        self.metrics: Dict[str, float] = {}
        self.is_trained = False

    def prepare_data(
        self,
        features: pd.DataFrame,
        target: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training

        Args:
            features: Feature DataFrame
            target: Target Series

        Returns:
            Tuple of (X, y) arrays
        """
        # Fit and transform
        scaled_features, scaled_target = self.preprocessor.fit_transform(
            features, target
        )

        # Create sequences for deep learning models
        if self.model_type in ['lstm', 'gru', 'cnn_lstm', 'ensemble']:
            X, y = create_sequences(
                scaled_features,
                scaled_target,
                lookback=self.lookback,
                forecast_horizon=self.forecast_horizon
            )
        else:
            # For XGBoost, use flattened features
            X = scaled_features[self.lookback:]
            y = scaled_target[self.lookback:]

        logger.info(f"Prepared data: X={X.shape}, y={y.shape}")
        return X, y

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = None,
        batch_size: int = None,
        verbose: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train model

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of epochs
            batch_size: Batch size
            verbose: Verbosity level
            **kwargs: Additional training arguments

        Returns:
            Training history/results
        """
        if epochs is None:
            epochs = training_config.EPOCHS
        if batch_size is None:
            batch_size = training_config.BATCH_SIZE

        # Deep learning models
        if self.model_type in ['lstm', 'gru', 'cnn_lstm', 'ensemble']:
            return self._train_deep_learning(
                X_train, y_train, X_val, y_val,
                epochs, batch_size, verbose, **kwargs
            )

        # Statistical models
        elif self.model_type == 'sarimax':
            return self._train_sarimax(y_train, **kwargs)

        elif self.model_type == 'xgboost':
            return self._train_xgboost(
                X_train, y_train, X_val, y_val, **kwargs
            )

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _train_deep_learning(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        epochs: int,
        batch_size: int,
        verbose: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Train deep learning model"""
        input_shape = (X_train.shape[1], X_train.shape[2])
        output_size = self.forecast_horizon if len(y_train.shape) > 1 else 1

        # Build model
        self.model = ModelFactory.create(
            self.model_type,
            input_shape=input_shape,
            output_size=output_size,
            **kwargs
        )

        # Get callbacks
        callbacks = ModelFactory.get_callbacks(
            patience=training_config.EARLY_STOPPING_PATIENCE,
            lr_patience=training_config.LR_PATIENCE,
            lr_factor=training_config.LR_FACTOR
        )

        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        # Train
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )

        self.is_trained = True
        logger.info(f"Training complete. Final loss: {self.history.history['loss'][-1]:.6f}")

        return self.history.history

    def _train_sarimax(
        self,
        y_train: np.ndarray,
        exog: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train SARIMAX model"""
        self.model = SARIMAXWrapper(
            order=model_config.SARIMAX_ORDER,
            seasonal_order=model_config.SARIMAX_SEASONAL_ORDER
        )
        self.model.fit(y_train, exog=exog)
        self.is_trained = True

        return {"aic": self.model.fitted_model.aic}

    def _train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        **kwargs
    ) -> Dict[str, Any]:
        """Train XGBoost model"""
        self.model = XGBoostWrapper(
            n_estimators=model_config.XGB_N_ESTIMATORS,
            max_depth=model_config.XGB_MAX_DEPTH,
            learning_rate=model_config.XGB_LEARNING_RATE
        )

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        self.model.fit(X_train, y_train, eval_set=eval_set)
        self.is_trained = True

        return {"feature_importance": self.model.get_feature_importance()}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Input features

        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        predictions = self.model.predict(X)
        return np.array(predictions).flatten()

    def predict_inverse_scaled(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions and inverse transform to original scale

        Args:
            X: Input features

        Returns:
            Predictions in original scale
        """
        scaled_preds = self.predict(X)
        return self.preprocessor.inverse_transform_target(scaled_preds)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        inverse_scale: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model on test data

        Args:
            X_test: Test features
            y_test: Test targets
            inverse_scale: Whether to inverse transform before evaluation

        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(X_test)

        if inverse_scale:
            predictions = self.preprocessor.inverse_transform_target(predictions)
            y_actual = self.preprocessor.inverse_transform_target(y_test)
        else:
            y_actual = y_test

        self.metrics = calculate_metrics(y_actual, predictions)
        logger.info(f"Evaluation metrics: {self.metrics}")

        return self.metrics

    def save(self, filepath: Union[str, Path]):
        """Save model and preprocessor"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        if self.model_type in ['lstm', 'gru', 'cnn_lstm', 'ensemble']:
            self.model.save(str(filepath.with_suffix('.keras')))
        else:
            with open(filepath.with_suffix('.pkl'), 'wb') as f:
                pickle.dump(self.model, f)

        # Save preprocessor
        with open(filepath.with_suffix('.preprocessor.pkl'), 'wb') as f:
            pickle.dump(self.preprocessor, f)

        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'lookback': self.lookback,
            'forecast_horizon': self.forecast_horizon,
            'metrics': self.metrics,
            'trained_at': datetime.now().isoformat()
        }
        with open(filepath.with_suffix('.meta.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: Union[str, Path]):
        """Load model and preprocessor"""
        filepath = Path(filepath)

        # Load metadata
        with open(filepath.with_suffix('.meta.json'), 'r') as f:
            metadata = json.load(f)

        self.model_type = metadata['model_type']
        self.lookback = metadata['lookback']
        self.forecast_horizon = metadata['forecast_horizon']
        self.metrics = metadata.get('metrics', {})

        # Load model
        if self.model_type in ['lstm', 'gru', 'cnn_lstm', 'ensemble']:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(
                str(filepath.with_suffix('.keras'))
            )
        else:
            with open(filepath.with_suffix('.pkl'), 'rb') as f:
                self.model = pickle.load(f)

        # Load preprocessor
        with open(filepath.with_suffix('.preprocessor.pkl'), 'rb') as f:
            self.preprocessor = pickle.load(f)

        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    model_type: str = "lstm",
    lookback: int = 12,
    test_ratio: float = 0.2,
    seed: int = 42,
    **kwargs
) -> Tuple[ModelTrainer, Dict[str, float]]:
    """
    Train a single model

    Args:
        features: Feature DataFrame
        target: Target Series
        model_type: Type of model
        lookback: Lookback period
        test_ratio: Test set ratio
        seed: Random seed
        **kwargs: Additional training arguments

    Returns:
        Tuple of (trainer, metrics)
    """
    set_random_seed(seed)

    trainer = ModelTrainer(
        model_type=model_type,
        lookback=lookback
    )

    # Prepare data
    X, y = trainer.prepare_data(features, target)

    # Split
    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Further split training for validation
    val_split = int(len(X_train) * 0.9)
    X_train_actual = X_train[:val_split]
    y_train_actual = y_train[:val_split]
    X_val = X_train[val_split:]
    y_val = y_train[val_split:]

    # Train
    trainer.train(
        X_train_actual, y_train_actual,
        X_val, y_val,
        **kwargs
    )

    # Evaluate
    metrics = trainer.evaluate(X_test, y_test)

    return trainer, metrics


def train_ensemble(
    features: pd.DataFrame,
    target: pd.Series,
    model_types: List[str] = None,
    lookback: int = 12,
    n_seeds: int = 3,
    test_ratio: float = 0.2,
    **kwargs
) -> Tuple[List[ModelTrainer], Dict[str, float]]:
    """
    Train ensemble of models with multiple seeds

    Args:
        features: Feature DataFrame
        target: Target Series
        model_types: List of model types
        lookback: Lookback period
        n_seeds: Number of seeds per model
        test_ratio: Test set ratio
        **kwargs: Additional training arguments

    Returns:
        Tuple of (trainers, aggregated_metrics)
    """
    if model_types is None:
        model_types = ['lstm', 'gru', 'cnn_lstm']

    trainers = []
    all_predictions = []

    for model_type in model_types:
        logger.info(f"Training {model_type} ensemble with {n_seeds} seeds...")

        for seed in range(n_seeds):
            trainer, _ = train_model(
                features=features,
                target=target,
                model_type=model_type,
                lookback=lookback,
                test_ratio=test_ratio,
                seed=42 + seed,
                **kwargs
            )
            trainers.append(trainer)

    # Calculate ensemble metrics on test set
    preprocessor = trainers[0].preprocessor
    X, y = trainers[0].prepare_data(features, target)
    split_idx = int(len(X) * (1 - test_ratio))
    X_test = X[split_idx:]
    y_test = y[split_idx:]

    # Aggregate predictions
    for trainer in trainers:
        preds = trainer.predict(X_test)
        all_predictions.append(preds)

    ensemble_preds = np.mean(all_predictions, axis=0)

    # Inverse transform
    ensemble_preds_orig = preprocessor.inverse_transform_target(ensemble_preds)
    y_test_orig = preprocessor.inverse_transform_target(y_test)

    metrics = calculate_metrics(y_test_orig, ensemble_preds_orig)
    logger.info(f"Ensemble metrics: {metrics}")

    return trainers, metrics


def cross_validate(
    features: pd.DataFrame,
    target: pd.Series,
    model_type: str = "lstm",
    lookback: int = 12,
    n_splits: int = 5,
    **kwargs
) -> Dict[str, List[float]]:
    """
    Time series cross-validation

    Args:
        features: Feature DataFrame
        target: Target Series
        model_type: Type of model
        lookback: Lookback period
        n_splits: Number of CV splits
        **kwargs: Additional training arguments

    Returns:
        Dictionary of metric lists across folds
    """
    preprocessor = DataPreprocessor()
    scaled_features, scaled_target = preprocessor.fit_transform(features, target)

    if model_type in ['lstm', 'gru', 'cnn_lstm', 'ensemble']:
        X, y = create_sequences(
            scaled_features, scaled_target, lookback=lookback
        )
    else:
        X = scaled_features[lookback:]
        y = scaled_target[lookback:]

    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_results = {
        'mae': [], 'mse': [], 'rmse': [], 'mape': [], 'r2': []
    }

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        logger.info(f"CV Fold {fold + 1}/{n_splits}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Split training for validation
        val_split = int(len(X_train) * 0.9)

        trainer = ModelTrainer(model_type=model_type, lookback=lookback)
        trainer.preprocessor = preprocessor
        trainer.is_trained = False

        trainer.train(
            X_train[:val_split], y_train[:val_split],
            X_train[val_split:], y_train[val_split:],
            verbose=0,
            **kwargs
        )

        metrics = trainer.evaluate(X_test, y_test)

        for key in cv_results:
            if key in metrics:
                cv_results[key].append(metrics[key])

    # Log summary
    logger.info("Cross-validation results:")
    for key, values in cv_results.items():
        logger.info(f"  {key}: {np.mean(values):.4f} +/- {np.std(values):.4f}")

    return cv_results
