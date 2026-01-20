# Model Selection Guide

A comprehensive guide to choosing and configuring models for economic forecasting.

---

## Table of Contents

- [Overview](#overview)
- [Model Comparison](#model-comparison)
- [Deep Learning Models](#deep-learning-models)
- [Statistical Models](#statistical-models)
- [Ensemble Methods](#ensemble-methods)
- [Model Selection Guidelines](#model-selection-guidelines)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Best Practices](#best-practices)

---

## Overview

The system provides six model types optimized for economic time series:

| Model | Category | Strengths | Weaknesses |
|-------|----------|-----------|------------|
| **LSTM** | Deep Learning | Long-term patterns | Slow training |
| **GRU** | Deep Learning | Faster than LSTM | Slightly less accurate |
| **CNN-LSTM** | Hybrid | Pattern + sequence | Complex |
| **SARIMAX** | Statistical | Interpretable | Limited complexity |
| **XGBoost** | Gradient Boosting | Feature importance | No sequence modeling |
| **Ensemble** | Combined | Best accuracy | Most resources |

---

## Model Comparison

### Accuracy vs Speed Trade-off

```
High Accuracy
    │
    │  ┌─────────┐
    │  │Ensemble │
    │  └─────────┘
    │        ┌─────────┐
    │        │CNN-LSTM │
    │        └─────────┘
    │  ┌─────────┐
    │  │  LSTM   │
    │  └─────────┘
    │      ┌─────────┐
    │      │   GRU   │
    │      └─────────┘
    │          ┌─────────┐
    │          │ XGBoost │
    │          └─────────┘
    │              ┌─────────┐
    │              │ SARIMAX │
    │              └─────────┘
    └──────────────────────────────► Fast Training
```

### Benchmark Results (Inflation Forecasting)

| Model | MAE | RMSE | MAPE | R² | Training Time |
|-------|-----|------|------|-----|---------------|
| LSTM | 0.118 | 0.176 | 4.2% | 0.924 | 5m 23s |
| GRU | 0.125 | 0.184 | 4.5% | 0.913 | 3m 47s |
| CNN-LSTM | 0.108 | 0.165 | 3.9% | 0.931 | 6m 12s |
| Ensemble | 0.095 | 0.148 | 3.5% | 0.945 | 15m 40s |
| SARIMAX | 0.152 | 0.218 | 5.1% | 0.879 | 28s |
| XGBoost | 0.138 | 0.198 | 4.8% | 0.902 | 12s |

*Tested on 10 years of monthly inflation data, 12-month forecast horizon*

---

## Deep Learning Models

### LSTM (Long Short-Term Memory)

**Best for:** Complex patterns, long-term dependencies, non-linear relationships

#### Architecture

```
Input (lookback, features)
    │
    ▼
┌─────────────────────────┐
│  Bidirectional LSTM     │ ─── 128 units
│  + Batch Normalization  │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  Bidirectional LSTM     │ ─── 64 units
│  + Dropout (0.2)        │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  Bidirectional LSTM     │ ─── 32 units
│  + Dropout (0.2)        │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  Dense (32) + ReLU      │
│  + Dropout (0.1)        │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  Dense (1) + Linear     │ ─── Output
└─────────────────────────┘
```

#### Configuration

```python
from models.architectures import build_lstm_model

model = build_lstm_model(
    input_shape=(12, 20),      # (lookback, features)
    output_size=1,
    units=[128, 64, 32],       # Decreasing layer sizes
    dropout_rate=0.2,
    bidirectional=True,        # Use bidirectional
    l2_reg=0.001               # L2 regularization
)
```

#### When to Use

✅ Use LSTM when:
- Data has long-term patterns (>12 months)
- Relationships are complex/non-linear
- You have sufficient training data (>5 years)
- Training time is not critical

❌ Avoid LSTM when:
- Data is limited (<3 years)
- Patterns are simple/linear
- Real-time predictions needed
- Interpretability is required

---

### GRU (Gated Recurrent Unit)

**Best for:** Similar to LSTM but faster, good for experimentation

#### Architecture

```
Input (lookback, features)
    │
    ▼
┌─────────────────────────┐
│  Bidirectional GRU      │ ─── 128 units
│  + Batch Normalization  │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  Bidirectional GRU      │ ─── 64 units
│  + Dropout (0.2)        │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  Dense (32) + ReLU      │
│  + Dropout (0.1)        │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  Dense (1) + Linear     │ ─── Output
└─────────────────────────┘
```

#### Configuration

```python
from models.architectures import build_gru_model

model = build_gru_model(
    input_shape=(12, 20),
    output_size=1,
    units=[128, 64],           # Fewer layers than LSTM
    dropout_rate=0.2,
    bidirectional=True
)
```

#### LSTM vs GRU

| Aspect | LSTM | GRU |
|--------|------|-----|
| Parameters | More | ~25% fewer |
| Training speed | Slower | Faster |
| Accuracy | Slightly higher | Comparable |
| Memory usage | Higher | Lower |
| Long sequences | Better | Good |

**Recommendation:** Start with GRU for prototyping, switch to LSTM for production.

---

### CNN-LSTM Hybrid

**Best for:** Pattern recognition combined with sequence modeling

#### Architecture

```
Input (lookback, features)
    │
    ▼
┌─────────────────────────┐
│  Conv1D (64 filters)    │ ─── Pattern extraction
│  + Batch Norm + Dropout │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  Conv1D (32 filters)    │ ─── Higher-level patterns
│  + Batch Norm + Dropout │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  LSTM (64 units)        │ ─── Sequence modeling
│  + Dropout              │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  LSTM (32 units)        │ ─── Temporal dependencies
│  + Dropout              │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  Dense (32) → Dense (1) │ ─── Output
└─────────────────────────┘
```

#### Configuration

```python
from models.architectures import build_cnn_lstm_model

model = build_cnn_lstm_model(
    input_shape=(12, 20),
    output_size=1,
    cnn_filters=[64, 32],
    lstm_units=[64, 32],
    kernel_size=3,
    dropout_rate=0.2
)
```

#### When to Use

✅ Best when:
- Data has local patterns (weekly, monthly cycles)
- Multiple indicators with correlated movements
- Both short and long-term patterns important

---

## Statistical Models

### SARIMAX

**Best for:** Seasonal patterns, interpretability, baseline comparison

#### Components

```
SARIMAX(p, d, q)(P, D, Q, s)

p = AR order (past values)
d = Differencing (stationarity)
q = MA order (past errors)

P = Seasonal AR order
D = Seasonal differencing
Q = Seasonal MA order
s = Seasonal period (12 for monthly)
```

#### Configuration

```python
from models.statistical import SARIMAXWrapper

model = SARIMAXWrapper(
    order=(1, 1, 1),           # ARIMA order
    seasonal_order=(1, 1, 1, 12),  # Seasonal with 12-month period
    trend='c',                  # Constant trend
    enforce_stationarity=False,
    enforce_invertibility=False
)

# Fit
model.fit(y_train, exog=exogenous_vars)

# Forecast with confidence intervals
forecast = model.predict(steps=12)
lower, upper = model.get_confidence_interval(steps=12, alpha=0.05)
```

#### Auto Order Selection

```python
from models.statistical import auto_arima_order

order, seasonal_order = auto_arima_order(
    series,
    max_p=5,
    max_d=2,
    max_q=5,
    seasonal=True,
    m=12
)
```

#### Advantages

- Interpretable coefficients
- Built-in confidence intervals
- Handles seasonality explicitly
- Fast training and inference

#### Limitations

- Assumes linear relationships
- Struggles with regime changes
- May miss complex patterns

---

### XGBoost

**Best for:** Feature importance, fast inference, tabular data

#### How It Works

XGBoost treats time series as tabular data with engineered features:

```
Original: [y1, y2, y3, y4, y5, ...]

Transformed:
┌─────────┬─────────┬─────────┬─────────┬─────────┐
│ lag_1   │ lag_2   │ lag_3   │ roll_3  │ target  │
├─────────┼─────────┼─────────┼─────────┼─────────┤
│ y3      │ y2      │ y1      │ mean    │ y4      │
│ y4      │ y3      │ y2      │ mean    │ y5      │
│ ...     │ ...     │ ...     │ ...     │ ...     │
└─────────┴─────────┴─────────┴─────────┴─────────┘
```

#### Configuration

```python
from models.statistical import XGBoostWrapper

model = XGBoostWrapper(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    reg_alpha=0.0,           # L1 regularization
    reg_lambda=1.0           # L2 regularization
)

# Fit with early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=10
)

# Get feature importance
importance = model.get_feature_importance()
```

#### Feature Engineering for XGBoost

```python
from preprocessing.processor import create_lag_features, create_rolling_features
from models.statistical import create_time_features

# Add lag features
df = create_lag_features(df, columns=['inflation', 'gdp'], lags=[1, 3, 6, 12])

# Add rolling statistics
df = create_rolling_features(df, columns=['inflation'], windows=[3, 6, 12])

# Add time features (cyclical encoding)
df = create_time_features(df, date_col='Date')
# Creates: year, month, quarter, month_sin, month_cos, etc.
```

---

## Ensemble Methods

### Why Ensemble?

Combining multiple models reduces:
- **Variance**: Less overfitting
- **Bias**: Captures different patterns
- **Model risk**: Not reliant on single model

### Ensemble Architecture

```
┌─────────────────────────────────────────────────────┐
│                     Input Data                       │
└─────────────────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ┌───────────┐   ┌───────────┐   ┌───────────┐
    │   LSTM    │   │    GRU    │   │ CNN-LSTM  │
    │ (seed 1)  │   │ (seed 1)  │   │ (seed 1)  │
    └───────────┘   └───────────┘   └───────────┘
          │                │                │
    ┌───────────┐   ┌───────────┐   ┌───────────┐
    │   LSTM    │   │    GRU    │   │ CNN-LSTM  │
    │ (seed 2)  │   │ (seed 2)  │   │ (seed 2)  │
    └───────────┘   └───────────┘   └───────────┘
          │                │                │
    ┌───────────┐   ┌───────────┐   ┌───────────┐
    │   LSTM    │   │    GRU    │   │ CNN-LSTM  │
    │ (seed 3)  │   │ (seed 3)  │   │ (seed 3)  │
    └───────────┘   └───────────┘   └───────────┘
          │                │                │
          └────────────────┼────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   Aggregation   │
                  │  (mean/median)  │
                  └─────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Final Forecast │
                  └─────────────────┘
```

### Training Ensemble

```python
from training.trainer import train_ensemble

trainers, metrics = train_ensemble(
    features=features_df,
    target=target_series,
    model_types=['lstm', 'gru', 'cnn_lstm'],
    n_seeds=3,            # 3 seeds per model = 9 total
    lookback=12,
    test_ratio=0.2
)
```

### Ensemble Forecasting

```python
from forecasting.forecaster import ensemble_forecast

# Mean aggregation (recommended)
forecast, lower, upper = ensemble_forecast(
    trainers=trainers,
    features=features,
    target=target,
    n_steps=12,
    aggregation='mean'
)

# Weighted aggregation (if you know model strengths)
forecast, lower, upper = ensemble_forecast(
    trainers=trainers,
    features=features,
    target=target,
    n_steps=12,
    aggregation='weighted',
    weights=[0.4, 0.35, 0.25]  # Based on validation performance
)
```

---

## Model Selection Guidelines

### Decision Tree

```
Start Here
    │
    ▼
┌─────────────────────────────┐
│ Do you need interpretability │
│ and confidence intervals?    │
└─────────────────────────────┘
    │           │
   Yes          No
    │           │
    ▼           ▼
┌─────────┐  ┌──────────────────────┐
│ SARIMAX │  │ Is training time     │
└─────────┘  │ a constraint?        │
             └──────────────────────┘
                  │           │
                 Yes          No
                  │           │
                  ▼           ▼
             ┌─────────┐  ┌───────────────────┐
             │ XGBoost │  │ Is accuracy       │
             │ or GRU  │  │ the top priority? │
             └─────────┘  └───────────────────┘
                              │           │
                             Yes          No
                              │           │
                              ▼           ▼
                         ┌─────────┐  ┌─────────┐
                         │Ensemble │  │  LSTM   │
                         └─────────┘  └─────────┘
```

### Quick Reference

| Scenario | Recommended Model |
|----------|-------------------|
| Quick experiment | GRU |
| Production accuracy | CNN-LSTM or Ensemble |
| Limited data (<3 years) | SARIMAX or XGBoost |
| Feature importance needed | XGBoost |
| Interpretability required | SARIMAX |
| Maximum accuracy | Ensemble |
| Real-time predictions | XGBoost |

---

## Hyperparameter Tuning

### Key Hyperparameters

#### Deep Learning (LSTM/GRU/CNN-LSTM)

| Parameter | Range | Impact |
|-----------|-------|--------|
| `lookback` | 3-24 | How much history to use |
| `units` | [32, 64, 128, 256] | Model capacity |
| `dropout_rate` | 0.1-0.5 | Regularization |
| `learning_rate` | 1e-4 to 1e-2 | Training speed/stability |
| `batch_size` | 16, 32, 64 | Memory/generalization |
| `epochs` | 50-200 | Training duration |

#### SARIMAX

| Parameter | Values | Impact |
|-----------|--------|--------|
| `order` (p,d,q) | (0-3, 0-2, 0-3) | Model complexity |
| `seasonal_order` | (0-2, 0-1, 0-2, 12) | Seasonal pattern |
| `trend` | 'n', 'c', 't', 'ct' | Trend component |

#### XGBoost

| Parameter | Range | Impact |
|-----------|-------|--------|
| `n_estimators` | 50-500 | Model complexity |
| `max_depth` | 3-10 | Tree complexity |
| `learning_rate` | 0.01-0.3 | Training speed |
| `subsample` | 0.6-1.0 | Regularization |

### Tuning Strategy

```python
# Example: Grid search for lookback
lookbacks = [3, 6, 12, 18, 24]
results = {}

for lookback in lookbacks:
    trainer, metrics = train_model(
        features, target,
        model_type='lstm',
        lookback=lookback
    )
    results[lookback] = metrics['mape']

best_lookback = min(results, key=results.get)
print(f"Best lookback: {best_lookback} (MAPE: {results[best_lookback]:.2f}%)")
```

---

## Best Practices

### 1. Start Simple

```python
# Start with baseline
baseline_trainer, baseline_metrics = train_model(
    features, target,
    model_type='sarimax'
)
print(f"Baseline MAPE: {baseline_metrics['mape']:.2f}%")

# Then try deep learning
dl_trainer, dl_metrics = train_model(
    features, target,
    model_type='lstm'
)
print(f"LSTM MAPE: {dl_metrics['mape']:.2f}%")
```

### 2. Use Cross-Validation

```python
from training.trainer import cross_validate

cv_results = cross_validate(
    features, target,
    model_type='lstm',
    n_splits=5
)

print(f"CV MAPE: {np.mean(cv_results['mape']):.2f}% ± {np.std(cv_results['mape']):.2f}%")
```

### 3. Monitor Overfitting

Watch for:
- Large gap between train/val loss
- Val loss increasing while train loss decreases
- Perfect training performance

Solutions:
- Increase dropout
- Add L2 regularization
- Reduce model complexity
- Get more data

### 4. Check Forecast Uncertainty

```python
# Always get confidence intervals
forecast, lower, upper = forecaster.forecast(
    features,
    n_steps=12,
    return_confidence=True
)

# Wider intervals = more uncertainty
uncertainty = (upper - lower) / forecast
print(f"Average forecast uncertainty: {uncertainty.mean()*100:.1f}%")
```

### 5. Validate on Recent Data

```python
# Use most recent period for final validation
train_data = data[:-12]  # All but last year
test_data = data[-12:]   # Last year

trainer, _ = train_model(train_data_features, train_data_target)
test_metrics = trainer.evaluate(test_data_features, test_data_target)
```

---

## Summary

| Model | Accuracy | Speed | Interpretability | Best For |
|-------|----------|-------|------------------|----------|
| LSTM | ⭐⭐⭐⭐ | ⭐⭐ | ⭐ | Complex patterns |
| GRU | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ | Quick experiments |
| CNN-LSTM | ⭐⭐⭐⭐ | ⭐⭐ | ⭐ | Pattern + sequence |
| SARIMAX | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Seasonal, baseline |
| XGBoost | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Features, speed |
| Ensemble | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ | Maximum accuracy |

---

**Version:** 2.0.0 | **Last Updated:** January 2026
