"""
Deep Learning model architectures for Indonesia Economic Forecasting
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

# TensorFlow imports with error handling
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        LSTM, GRU, Dense, Dropout, Input, Bidirectional,
        Conv1D, MaxPooling1D, Flatten, BatchNormalization,
        Concatenate, Add, LayerNormalization
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l2
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from config.settings import model_config
from utils.logger import get_logger

logger = get_logger(__name__)


def check_tf_available():
    """Check if TensorFlow is available"""
    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is required for deep learning models. "
            "Install with: pip install tensorflow"
        )


def build_lstm_model(
    input_shape: Tuple[int, int],
    output_size: int = 1,
    units: List[int] = None,
    dropout_rate: float = 0.2,
    bidirectional: bool = True,
    l2_reg: float = 0.001
) -> "Sequential":
    """
    Build LSTM model for time series forecasting

    Args:
        input_shape: Shape of input (timesteps, features)
        output_size: Number of output predictions
        units: List of LSTM units per layer
        dropout_rate: Dropout rate
        bidirectional: Whether to use bidirectional LSTM
        l2_reg: L2 regularization factor

    Returns:
        Compiled Keras model
    """
    check_tf_available()

    if units is None:
        units = model_config.LSTM_UNITS

    model = Sequential(name="LSTM_Model")

    for i, unit in enumerate(units):
        return_sequences = i < len(units) - 1

        lstm_layer = LSTM(
            unit,
            return_sequences=return_sequences,
            kernel_regularizer=l2(l2_reg),
            recurrent_regularizer=l2(l2_reg)
        )

        if bidirectional:
            lstm_layer = Bidirectional(lstm_layer)

        if i == 0:
            model.add(lstm_layer)
            model.add(BatchNormalization())
        else:
            model.add(lstm_layer)

        model.add(Dropout(dropout_rate))

    # Output layers
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropout_rate / 2))
    model.add(Dense(output_size, activation='linear'))

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=model_config.LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )

    # Build model to show summary
    model.build(input_shape=(None,) + input_shape)
    logger.info(f"Built LSTM model: {model.count_params()} parameters")

    return model


def build_gru_model(
    input_shape: Tuple[int, int],
    output_size: int = 1,
    units: List[int] = None,
    dropout_rate: float = 0.2,
    bidirectional: bool = True,
    l2_reg: float = 0.001
) -> "Sequential":
    """
    Build GRU model for time series forecasting

    Args:
        input_shape: Shape of input (timesteps, features)
        output_size: Number of output predictions
        units: List of GRU units per layer
        dropout_rate: Dropout rate
        bidirectional: Whether to use bidirectional GRU
        l2_reg: L2 regularization factor

    Returns:
        Compiled Keras model
    """
    check_tf_available()

    if units is None:
        units = model_config.GRU_UNITS

    model = Sequential(name="GRU_Model")

    for i, unit in enumerate(units):
        return_sequences = i < len(units) - 1

        gru_layer = GRU(
            unit,
            return_sequences=return_sequences,
            kernel_regularizer=l2(l2_reg),
            recurrent_regularizer=l2(l2_reg)
        )

        if bidirectional:
            gru_layer = Bidirectional(gru_layer)

        if i == 0:
            model.add(gru_layer)
            model.add(BatchNormalization())
        else:
            model.add(gru_layer)

        model.add(Dropout(dropout_rate))

    # Output layers
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropout_rate / 2))
    model.add(Dense(output_size, activation='linear'))

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=model_config.LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )

    model.build(input_shape=(None,) + input_shape)
    logger.info(f"Built GRU model: {model.count_params()} parameters")

    return model


def build_cnn_lstm_model(
    input_shape: Tuple[int, int],
    output_size: int = 1,
    cnn_filters: List[int] = None,
    lstm_units: List[int] = None,
    kernel_size: int = 3,
    dropout_rate: float = 0.2,
    l2_reg: float = 0.001
) -> "Sequential":
    """
    Build CNN-LSTM hybrid model

    Args:
        input_shape: Shape of input (timesteps, features)
        output_size: Number of output predictions
        cnn_filters: List of CNN filter sizes
        lstm_units: List of LSTM units
        kernel_size: CNN kernel size
        dropout_rate: Dropout rate
        l2_reg: L2 regularization factor

    Returns:
        Compiled Keras model
    """
    check_tf_available()

    if cnn_filters is None:
        cnn_filters = model_config.CNN_FILTERS
    if lstm_units is None:
        lstm_units = model_config.CNN_LSTM_UNITS

    model = Sequential(name="CNN_LSTM_Model")

    # CNN layers
    for i, filters in enumerate(cnn_filters):
        if i == 0:
            model.add(Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                padding='same',
                kernel_regularizer=l2(l2_reg),
                input_shape=input_shape
            ))
        else:
            model.add(Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                padding='same',
                kernel_regularizer=l2(l2_reg)
            ))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate / 2))

    # LSTM layers
    for i, unit in enumerate(lstm_units):
        return_sequences = i < len(lstm_units) - 1
        model.add(LSTM(
            unit,
            return_sequences=return_sequences,
            kernel_regularizer=l2(l2_reg)
        ))
        model.add(Dropout(dropout_rate))

    # Output layers
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Dense(output_size, activation='linear'))

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=model_config.LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )

    logger.info(f"Built CNN-LSTM model: {model.count_params()} parameters")
    return model


def build_ensemble_model(
    input_shape: Tuple[int, int],
    output_size: int = 1,
    n_models: int = 3
) -> "Model":
    """
    Build ensemble of different architectures

    Args:
        input_shape: Shape of input (timesteps, features)
        output_size: Number of output predictions
        n_models: Number of models in ensemble

    Returns:
        Compiled Keras ensemble model
    """
    check_tf_available()

    inputs = Input(shape=input_shape, name="ensemble_input")

    # LSTM branch
    lstm_branch = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    lstm_branch = LSTM(32)(lstm_branch)
    lstm_branch = Dense(16, activation='relu')(lstm_branch)

    # GRU branch
    gru_branch = Bidirectional(GRU(64, return_sequences=True))(inputs)
    gru_branch = GRU(32)(gru_branch)
    gru_branch = Dense(16, activation='relu')(gru_branch)

    # CNN branch
    cnn_branch = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    cnn_branch = Conv1D(32, kernel_size=3, activation='relu', padding='same')(cnn_branch)
    cnn_branch = Flatten()(cnn_branch)
    cnn_branch = Dense(16, activation='relu')(cnn_branch)

    # Merge branches
    merged = Concatenate()([lstm_branch, gru_branch, cnn_branch])
    merged = Dense(32, activation='relu')(merged)
    merged = Dropout(0.2)(merged)
    outputs = Dense(output_size, activation='linear')(merged)

    model = Model(inputs=inputs, outputs=outputs, name="Ensemble_Model")

    model.compile(
        optimizer=Adam(learning_rate=model_config.LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )

    logger.info(f"Built Ensemble model: {model.count_params()} parameters")
    return model


class ModelFactory:
    """Factory for creating and managing models"""

    AVAILABLE_MODELS = ['lstm', 'gru', 'cnn_lstm', 'ensemble']

    @staticmethod
    def create(
        model_type: str,
        input_shape: Tuple[int, int],
        output_size: int = 1,
        **kwargs
    ) -> "Model":
        """
        Create model by type

        Args:
            model_type: Type of model ('lstm', 'gru', 'cnn_lstm', 'ensemble')
            input_shape: Shape of input
            output_size: Number of outputs
            **kwargs: Additional model arguments

        Returns:
            Compiled Keras model
        """
        model_type = model_type.lower()

        builders = {
            'lstm': build_lstm_model,
            'gru': build_gru_model,
            'cnn_lstm': build_cnn_lstm_model,
            'ensemble': build_ensemble_model
        }

        if model_type not in builders:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {ModelFactory.AVAILABLE_MODELS}"
            )

        return builders[model_type](
            input_shape=input_shape,
            output_size=output_size,
            **kwargs
        )

    @staticmethod
    def get_callbacks(
        patience: int = 15,
        min_delta: float = 0.0001,
        lr_patience: int = 5,
        lr_factor: float = 0.5
    ) -> List:
        """
        Get standard training callbacks

        Args:
            patience: Early stopping patience
            min_delta: Minimum improvement
            lr_patience: LR reduction patience
            lr_factor: LR reduction factor

        Returns:
            List of Keras callbacks
        """
        check_tf_available()

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                min_delta=min_delta,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=lr_factor,
                patience=lr_patience,
                min_lr=1e-7,
                verbose=1
            )
        ]

        return callbacks

    @staticmethod
    def save_model(model: "Model", filepath: str):
        """Save model to file"""
        check_tf_available()
        model.save(filepath)
        logger.info(f"Model saved to {filepath}")

    @staticmethod
    def load_model(filepath: str) -> "Model":
        """Load model from file"""
        check_tf_available()
        model = tf.keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model
