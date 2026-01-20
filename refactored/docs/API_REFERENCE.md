# API Reference

Complete API documentation for the Indonesia Economic Forecasting System.

---

## Table of Contents

- [Data Module](#data-module)
- [Preprocessing Module](#preprocessing-module)
- [Models Module](#models-module)
- [Training Module](#training-module)
- [Forecasting Module](#forecasting-module)
- [Visualization Module](#visualization-module)
- [Utilities Module](#utilities-module)

---

## Data Module

### `data.loader.DataLoader`

Main class for loading economic data from CSV files.

```python
from data.loader import DataLoader

loader = DataLoader(data_dir=None)
```

**Parameters:**
- `data_dir` (Path, optional): Directory containing data files. Defaults to `data/raw/`.

#### Methods

##### `load_csv(filename, date_col='Date', parse_dates=True)`

Load a single CSV file.

```python
df = loader.load_csv('Inflation_ID.csv')
```

**Parameters:**
- `filename` (str): Name of CSV file
- `date_col` (str): Date column name
- `parse_dates` (bool): Whether to parse dates

**Returns:** `pd.DataFrame`

##### `load_economic_indicators()`

Load all economic indicator files.

```python
data = loader.load_economic_indicators()
# Returns: {'inflation': df, 'interest_rate': df, ...}
```

**Returns:** `Dict[str, pd.DataFrame]`

##### `load_market_data()`

Load all market data files.

```python
market = loader.load_market_data()
# Returns: {'gold': df, 'usd_idr': df, 'brent': df, ...}
```

**Returns:** `Dict[str, pd.DataFrame]`

##### `load_money_supply()`

Load money supply data.

```python
money = loader.load_money_supply()
```

**Returns:** `Dict[str, pd.DataFrame]`

##### `load_all_data()`

Load all available data.

```python
all_data = loader.load_all_data()
```

**Returns:** `Dict[str, pd.DataFrame]`

##### `get_cached(name)`

Get cached dataset by name.

```python
df = loader.get_cached('inflation')
```

**Returns:** `Optional[pd.DataFrame]`

##### `clear_cache()`

Clear the data cache.

```python
loader.clear_cache()
```

---

### Standalone Functions

#### `resample_to_monthly(df, date_col, value_cols, agg_func)`

Resample time series to monthly frequency.

```python
from data.loader import resample_to_monthly

monthly_df = resample_to_monthly(
    df,
    date_col='Date',
    value_cols=['Close', 'Volume'],
    agg_func='last'  # 'last', 'mean', 'sum'
)
```

#### `resample_quarterly_to_monthly(df, date_col, method)`

Convert quarterly data to monthly.

```python
from data.loader import resample_quarterly_to_monthly

monthly = resample_quarterly_to_monthly(
    quarterly_df,
    method='interpolate'  # 'interpolate', 'ffill', 'bfill'
)
```

#### `merge_datasets(datasets, date_col, how)`

Merge multiple datasets on date.

```python
from data.loader import merge_datasets

merged = merge_datasets(
    {'inflation': df1, 'gdp': df2},
    date_col='Date',
    how='outer'  # 'inner', 'outer', 'left'
)
```

---

## Preprocessing Module

### `preprocessing.processor.DataPreprocessor`

Main class for data preprocessing.

```python
from preprocessing.processor import DataPreprocessor

preprocessor = DataPreprocessor(
    scaler_type='minmax',    # 'minmax', 'standard', 'robust'
    use_pca=True,
    pca_variance=0.95
)
```

**Parameters:**
- `scaler_type` (str): Type of scaler
- `use_pca` (bool): Whether to apply PCA
- `pca_variance` (float): Variance to retain with PCA

#### Methods

##### `fit(features, target)`

Fit preprocessor on training data.

```python
preprocessor.fit(features_df, target_series)
```

##### `transform(features, target)`

Transform features and target.

```python
scaled_X, scaled_y = preprocessor.transform(features, target)
```

**Returns:** `Tuple[np.ndarray, Optional[np.ndarray]]`

##### `fit_transform(features, target)`

Fit and transform in one step.

```python
scaled_X, scaled_y = preprocessor.fit_transform(features, target)
```

##### `inverse_transform_target(scaled_values)`

Convert scaled values back to original scale.

```python
original_values = preprocessor.inverse_transform_target(scaled_predictions)
```

##### `get_pca_explained_variance()`

Get PCA explained variance ratio.

```python
variance = preprocessor.get_pca_explained_variance()
```

---

### Standalone Functions

#### `create_sequences(features, target, lookback, forecast_horizon)`

Create sequences for time series modeling.

```python
from preprocessing.processor import create_sequences

X, y = create_sequences(
    features,           # np.ndarray (n_samples, n_features)
    target,            # np.ndarray (n_samples,)
    lookback=12,       # Past periods to use
    forecast_horizon=1 # Future periods to predict
)
# X shape: (n_samples - lookback, lookback, n_features)
# y shape: (n_samples - lookback,) or (n_samples - lookback, forecast_horizon)
```

#### `scale_data(train_data, test_data, scaler_type)`

Scale training and test data.

```python
from preprocessing.processor import scale_data

scaled_train, scaled_test, scaler = scale_data(
    train_df,
    test_df,
    scaler_type='minmax'
)
```

#### `apply_pca(train_data, test_data, n_components)`

Apply PCA dimensionality reduction.

```python
from preprocessing.processor import apply_pca

train_pca, test_pca, pca = apply_pca(
    train_data,
    test_data,
    n_components=0.95  # int or float (variance ratio)
)
```

#### `prepare_train_test_split(data, target_col, train_ratio, date_col)`

Split data for time series.

```python
from preprocessing.processor import prepare_train_test_split

X_train, X_test, y_train, y_test = prepare_train_test_split(
    data_df,
    target_col='Inflation',
    train_ratio=0.8,
    date_col='Date'
)
```

#### `handle_missing_values(df, strategy, numeric_fill)`

Handle missing values.

```python
from preprocessing.processor import handle_missing_values

clean_df = handle_missing_values(
    df,
    strategy='ffill'  # 'ffill', 'bfill', 'interpolate', 'fill'
)
```

#### `create_lag_features(df, columns, lags)`

Create lagged features.

```python
from preprocessing.processor import create_lag_features

df_with_lags = create_lag_features(
    df,
    columns=['inflation', 'gdp'],
    lags=[1, 3, 6, 12]
)
```

#### `create_rolling_features(df, columns, windows)`

Create rolling statistical features.

```python
from preprocessing.processor import create_rolling_features

df_with_rolling = create_rolling_features(
    df,
    columns=['inflation'],
    windows=[3, 6, 12]
)
# Creates: inflation_roll_mean_3, inflation_roll_std_3, etc.
```

---

## Models Module

### Deep Learning Models

#### `models.architectures.build_lstm_model()`

Build LSTM model.

```python
from models.architectures import build_lstm_model

model = build_lstm_model(
    input_shape=(12, 20),     # (lookback, features)
    output_size=1,
    units=[128, 64, 32],
    dropout_rate=0.2,
    bidirectional=True,
    l2_reg=0.001
)
```

**Returns:** `tf.keras.Model`

#### `models.architectures.build_gru_model()`

Build GRU model.

```python
from models.architectures import build_gru_model

model = build_gru_model(
    input_shape=(12, 20),
    units=[128, 64],
    bidirectional=True
)
```

#### `models.architectures.build_cnn_lstm_model()`

Build CNN-LSTM hybrid model.

```python
from models.architectures import build_cnn_lstm_model

model = build_cnn_lstm_model(
    input_shape=(12, 20),
    cnn_filters=[64, 32],
    lstm_units=[64, 32],
    kernel_size=3,
    dropout_rate=0.2
)
```

#### `models.architectures.build_ensemble_model()`

Build ensemble of multiple architectures.

```python
from models.architectures import build_ensemble_model

model = build_ensemble_model(
    input_shape=(12, 20),
    output_size=1,
    n_models=3
)
```

### ModelFactory

Factory class for creating models.

```python
from models.architectures import ModelFactory

# Create model by type
model = ModelFactory.create(
    model_type='lstm',
    input_shape=(12, 20),
    output_size=1
)

# Get training callbacks
callbacks = ModelFactory.get_callbacks(
    patience=15,
    min_delta=0.0001,
    lr_patience=5,
    lr_factor=0.5
)

# Save/Load
ModelFactory.save_model(model, 'path/to/model.keras')
model = ModelFactory.load_model('path/to/model.keras')
```

---

### Statistical Models

#### `models.statistical.SARIMAXWrapper`

SARIMAX model wrapper with sklearn interface.

```python
from models.statistical import SARIMAXWrapper

model = SARIMAXWrapper(
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    trend='c'
)

# Fit
model.fit(y_train, exog=X_train)

# Predict
forecast = model.predict(steps=12, exog=X_future)

# Confidence intervals
lower, upper = model.get_confidence_interval(steps=12, alpha=0.05)

# Summary
print(model.get_summary())
```

#### `models.statistical.XGBoostWrapper`

XGBoost wrapper for time series.

```python
from models.statistical import XGBoostWrapper

model = XGBoostWrapper(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8
)

# Fit with early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=10
)

# Predict
predictions = model.predict(X_test)

# Feature importance
importance = model.get_feature_importance()
```

---

## Training Module

### `training.trainer.ModelTrainer`

Main class for training models.

```python
from training.trainer import ModelTrainer

trainer = ModelTrainer(
    model_type='lstm',
    lookback=12,
    forecast_horizon=1,
    preprocessor=None  # Optional DataPreprocessor
)
```

#### Methods

##### `prepare_data(features, target)`

Prepare data for training.

```python
X, y = trainer.prepare_data(features_df, target_series)
```

##### `train(X_train, y_train, X_val, y_val, epochs, batch_size)`

Train the model.

```python
history = trainer.train(
    X_train, y_train,
    X_val, y_val,
    epochs=100,
    batch_size=32,
    verbose=1
)
```

**Returns:** Training history dict

##### `predict(X)`

Make predictions.

```python
predictions = trainer.predict(X_test)
```

##### `predict_inverse_scaled(X)`

Predict and inverse transform to original scale.

```python
predictions = trainer.predict_inverse_scaled(X_test)
```

##### `evaluate(X_test, y_test, inverse_scale)`

Evaluate model on test data.

```python
metrics = trainer.evaluate(X_test, y_test, inverse_scale=True)
# Returns: {'mae': 0.12, 'rmse': 0.18, 'mape': 4.2, 'r2': 0.92}
```

##### `save(filepath)`

Save model and preprocessor.

```python
trainer.save('artifacts/my_model')
```

##### `load(filepath)`

Load model and preprocessor.

```python
trainer.load('artifacts/my_model')
```

---

### Standalone Functions

#### `train_model(features, target, model_type, ...)`

Train a single model (convenience function).

```python
from training.trainer import train_model

trainer, metrics = train_model(
    features=features_df,
    target=target_series,
    model_type='lstm',
    lookback=12,
    test_ratio=0.2,
    seed=42,
    epochs=100,
    batch_size=32
)
```

**Returns:** `Tuple[ModelTrainer, Dict[str, float]]`

#### `train_ensemble(features, target, model_types, n_seeds, ...)`

Train ensemble of models.

```python
from training.trainer import train_ensemble

trainers, metrics = train_ensemble(
    features=features_df,
    target=target_series,
    model_types=['lstm', 'gru', 'cnn_lstm'],
    n_seeds=3,
    lookback=12,
    test_ratio=0.2
)
```

**Returns:** `Tuple[List[ModelTrainer], Dict[str, float]]`

#### `cross_validate(features, target, model_type, n_splits, ...)`

Time series cross-validation.

```python
from training.trainer import cross_validate

cv_results = cross_validate(
    features=features_df,
    target=target_series,
    model_type='lstm',
    lookback=12,
    n_splits=5
)
# Returns: {'mae': [0.1, 0.12, ...], 'rmse': [...], ...}
```

---

## Forecasting Module

### `forecasting.forecaster.EconomicForecaster`

Main class for generating forecasts.

```python
from forecasting.forecaster import EconomicForecaster

forecaster = EconomicForecaster(
    trainer=trained_trainer,     # OR
    model_path='artifacts/model' # Load from file
)
```

#### Methods

##### `forecast(features, n_steps, return_confidence)`

Generate forecast.

```python
# Simple forecast
forecasts = forecaster.forecast(features, n_steps=12)

# With confidence intervals
forecasts, lower, upper = forecaster.forecast(
    features,
    n_steps=12,
    return_confidence=True,
    confidence_level=0.95
)
```

##### `generate_date_range(last_date, n_periods, freq)`

Generate forecast dates.

```python
dates = forecaster.generate_date_range(
    last_date=datetime(2026, 1, 1),
    n_periods=12,
    freq='M'  # 'M', 'Q'
)
```

##### `create_forecast_dataframe(forecasts, dates, lower, upper, name)`

Create forecast DataFrame.

```python
df = forecaster.create_forecast_dataframe(
    forecasts,
    dates,
    lower,
    upper,
    name='inflation_forecast'
)
```

##### `save_forecast(df, filename, output_dir)`

Save forecast to CSV.

```python
forecaster.save_forecast(df, 'forecast.csv')
```

---

### Standalone Functions

#### `generate_forecast(trainer, features, target, n_steps, last_date)`

Generate forecast from trained model.

```python
from forecasting.forecaster import generate_forecast

forecast_df = generate_forecast(
    trainer=trained_trainer,
    features=features_df,
    target=target_series,
    n_steps=12,
    last_date=datetime(2026, 1, 1)
)
```

#### `walk_forward_forecast(trainer, features, target, n_steps)`

Walk-forward forecasting.

```python
from forecasting.forecaster import walk_forward_forecast

forecasts, confidence_widths = walk_forward_forecast(
    trainer=trained_trainer,
    features=features_df,
    target=target_series,
    n_steps=12,
    update_features=True
)
```

#### `ensemble_forecast(trainers, features, target, n_steps, aggregation)`

Generate ensemble forecast.

```python
from forecasting.forecaster import ensemble_forecast

forecast, lower, upper = ensemble_forecast(
    trainers=list_of_trainers,
    features=features_df,
    target=target_series,
    n_steps=12,
    aggregation='mean',  # 'mean', 'median', 'weighted'
    weights=[0.4, 0.3, 0.3]  # For weighted
)
```

---

## Visualization Module

### Plotting Functions

#### `plot_forecast(historical, forecast, ...)`

Plot forecast with confidence intervals.

```python
from visualization.plots import plot_forecast

fig = plot_forecast(
    historical=historical_df,
    forecast=forecast_df,
    date_col='Date',
    actual_col='actual',
    forecast_col='forecast',
    lower_col='forecast_lower',
    upper_col='forecast_upper',
    title='Inflation Forecast',
    ylabel='Inflation (%)',
    figsize=(14, 6),
    save_path='reports/forecast.png'
)
```

#### `plot_comparison(df, ...)`

Plot actual vs predicted comparison.

```python
from visualization.plots import plot_comparison

fig = plot_comparison(
    df,
    date_col='Date',
    actual_col='actual',
    predicted_col='predicted',
    title='Actual vs Predicted',
    save_path='reports/comparison.png'
)
```

#### `plot_metrics(metrics, ...)`

Plot performance metrics as bar chart.

```python
from visualization.plots import plot_metrics

fig = plot_metrics(
    {'MAE': 0.12, 'RMSE': 0.18, 'MAPE': 4.2, 'R2': 0.92},
    title='Model Performance',
    save_path='reports/metrics.png'
)
```

#### `plot_feature_importance(importance, top_n, ...)`

Plot feature importance.

```python
from visualization.plots import plot_feature_importance

fig = plot_feature_importance(
    importance_dict,
    top_n=20,
    title='Feature Importance',
    save_path='reports/importance.png'
)
```

#### `plot_correlation_matrix(df, columns, ...)`

Plot correlation heatmap.

```python
from visualization.plots import plot_correlation_matrix

fig = plot_correlation_matrix(
    df,
    columns=['inflation', 'gdp', 'interest_rate'],
    title='Correlation Matrix'
)
```

#### `plot_historical_trend(df, value_cols, normalize, ...)`

Plot historical trends.

```python
from visualization.plots import plot_historical_trend

fig = plot_historical_trend(
    df,
    date_col='Date',
    value_cols=['inflation', 'gdp'],
    normalize=True,  # Scale all to 0-1
    title='Historical Trends'
)
```

#### `create_dashboard(historical_df, forecast_df, metrics, ...)`

Create comprehensive dashboard.

```python
from visualization.plots import create_dashboard

fig = create_dashboard(
    historical_df=historical,
    forecast_df=forecast,
    metrics={'mae': 0.12, 'rmse': 0.18},
    title='Economic Forecast Dashboard',
    save_path='reports/dashboard.png'
)
```

---

## Utilities Module

### Logger Functions

```python
from utils.logger import get_logger, setup_logger

# Get logger for module
logger = get_logger(__name__)
logger.info("Processing data...")
logger.warning("Missing values detected")
logger.error("Failed to load file")

# Setup with custom file
logger = setup_logger(
    name='my_logger',
    log_file='logs/custom.log',
    level='DEBUG'
)
```

### Helper Functions

#### `set_random_seed(seed)`

Set random seed for reproducibility.

```python
from utils.helpers import set_random_seed

set_random_seed(42)
```

#### `calculate_metrics(actual, predicted)`

Calculate evaluation metrics.

```python
from utils.helpers import calculate_metrics

metrics = calculate_metrics(y_actual, y_predicted)
# Returns: {'mae': ..., 'mse': ..., 'rmse': ..., 'mape': ..., 'r2': ...}
```

#### `create_sequences(data, lookback)`

Create sequences for RNN models.

```python
from utils.helpers import create_sequences

X, y = create_sequences(data_array, lookback=12)
```

#### `inverse_transform_predictions(predictions, scaler)`

Inverse transform scaled predictions.

```python
from utils.helpers import inverse_transform_predictions

original = inverse_transform_predictions(scaled_preds, scaler)
```

---

## Configuration

### Config Classes

```python
from config.settings import (
    path_config,
    data_config,
    preprocessing_config,
    model_config,
    training_config,
    forecast_config
)

# Access settings
print(path_config.ARTIFACTS_DIR)
print(model_config.LSTM_UNITS)
print(training_config.EPOCHS)
```

### Key Configuration Options

| Config | Attribute | Default | Description |
|--------|-----------|---------|-------------|
| `path_config` | `RAW_DATA_DIR` | `data/raw` | Raw data directory |
| `path_config` | `ARTIFACTS_DIR` | `artifacts` | Model storage |
| `model_config` | `LSTM_UNITS` | `[128, 64, 32]` | LSTM layer sizes |
| `model_config` | `LEARNING_RATE` | `0.001` | Learning rate |
| `training_config` | `EPOCHS` | `100` | Training epochs |
| `training_config` | `BATCH_SIZE` | `32` | Batch size |

---

## Error Handling

Most functions raise standard Python exceptions:

- `ValueError`: Invalid parameters
- `FileNotFoundError`: Missing data files
- `ImportError`: Missing dependencies

Example:
```python
try:
    trainer, metrics = train_model(features, target, model_type='invalid')
except ValueError as e:
    print(f"Invalid model type: {e}")
```

---

## Type Hints

All public functions include type hints:

```python
def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    model_type: str = "lstm",
    lookback: int = 12,
    test_ratio: float = 0.2,
    seed: int = 42,
    **kwargs
) -> Tuple[ModelTrainer, Dict[str, float]]:
    ...
```

---

**Version:** 2.0.0 | **Last Updated:** January 2026
