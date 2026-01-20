# Jupyter Notebook Guide

Complete guide to using the `Predictive Analytics for Indonesia's Economic Forecasting.ipynb` notebook.

---

## Table of Contents

- [Overview](#overview)
- [Notebook Structure](#notebook-structure)
- [Getting Started](#getting-started)
- [Section-by-Section Guide](#section-by-section-guide)
- [Key Functions](#key-functions)
- [Model Configuration](#model-configuration)
- [Running Experiments](#running-experiments)
- [Interpreting Results](#interpreting-results)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)

---

## Overview

The Jupyter notebook contains the complete research workflow for economic forecasting, including:

- Data loading and preprocessing
- Exploratory data analysis
- Multiple deep learning models (LSTM, GRU, CNN-LSTM)
- Cross-validation and ensemble methods
- Visualization and analysis

### Notebook Stats

| Metric | Value |
|--------|-------|
| Total Cells | 82 |
| Code Cells | ~70 |
| Markdown Cells | ~12 |
| Lines of Code | ~27,000 |
| Estimated Runtime | 30-60 minutes |

---

## Notebook Structure

```
Notebook Organization
│
├── 📊 Section 1: Data Loading (Cells 1-10)
│   ├── Import libraries
│   ├── Load CSV files
│   ├── Resample to monthly frequency
│   └── Merge datasets
│
├── 📈 Section 2: Exploratory Data Analysis (Cells 11-20)
│   ├── Correlation analysis
│   ├── Time series plots
│   └── Feature distributions
│
├── 💹 Section 3: USD/IDR Forecasting (Cells 21-60)
│   ├── LSTM models (with/without PCA)
│   ├── GRU models (with/without PCA)
│   ├── CNN-LSTM models (with/without PCA)
│   └── Cross-validation
│
├── 📊 Section 4: Ensemble Methods (Cells 61-70)
│   ├── Model aggregation
│   ├── Weighted averaging
│   └── Neural network stacking
│
├── 💰 Section 5: Inflation Forecasting (Cells 71-80)
│   ├── LSTM for inflation
│   ├── GRU for inflation
│   └── CNN-LSTM for inflation
│
└── 📋 Section 6: Results & Conclusions (Cells 81-82)
    ├── Model comparison
    └── Final insights
```

---

## Getting Started

### Prerequisites

```bash
# Required packages
pip install jupyter pandas numpy tensorflow scikit-learn matplotlib seaborn
```

### Launch Notebook

```bash
# Navigate to repository
cd Predictive-Analytics-for-Indonesia-s-Economic-Forecasting

# Start Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab
```

### Open the Notebook

1. Click on `Predictive Analytics for Indonesia's Economic Forecasting.ipynb`
2. Wait for kernel to start
3. Run cells sequentially or selectively

---

## Section-by-Section Guide

### Section 1: Data Loading

**Cells 1-10** | **Runtime: ~1 minute**

This section loads and preprocesses all data files.

```python
# Cell 1: Import libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
# Cell 2-5: Load CSV files
# The notebook loads 22+ CSV files including:
# - Inflation_ID.csv
# - BI_Rate.csv
# - USD_IDR.csv
# - Gold_Price.csv
# - GDP data, exports, imports, etc.
```

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `daily_monthly(data)` | Resample daily data to monthly (last value) |
| `quarterly_monthly(data)` | Interpolate quarterly to monthly |
| `monthly_period(data)` | Standardize monthly date format |

**Output:** Merged DataFrame with all indicators aligned to monthly frequency.

---

### Section 2: Exploratory Data Analysis

**Cells 11-20** | **Runtime: ~2 minutes**

Analyze relationships between economic indicators.

```python
# Correlation analysis
correlation_matrix = data.corr()

# Heatmap visualization
plt.figure(figsize=(20, 16))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Economic Indicators')
plt.show()
```

**What to Look For:**

- Strong correlations (|r| > 0.7) between indicators
- Potential multicollinearity issues
- Features most correlated with target variables

---

### Section 3: USD/IDR Forecasting

**Cells 21-60** | **Runtime: ~20-30 minutes**

Train and evaluate models for exchange rate prediction.

#### LSTM Model

```python
def LSTM_process(data, PCA_Enable, n_prev_days):
    """
    Train LSTM model for USD/IDR forecasting.

    Parameters:
    -----------
    data : DataFrame
        Input data with features and target
    PCA_Enable : bool
        Whether to apply PCA dimensionality reduction
    n_prev_days : int
        Lookback period (number of previous time steps)

    Returns:
    --------
    model : Keras model
        Trained LSTM model
    predictions : array
        Model predictions
    metrics : dict
        Performance metrics (MAE, RMSE, MAPE, R²)
    """
```

**Model Architecture:**
```
Input Shape: (n_prev_days, n_features)
    │
    ▼
Bidirectional LSTM (128 units, return_sequences=True)
    │
    ▼
Dropout (0.2)
    │
    ▼
Bidirectional LSTM (64 units, return_sequences=True)
    │
    ▼
Dropout (0.2)
    │
    ▼
Bidirectional LSTM (32 units)
    │
    ▼
Dense (16, activation='relu')
    │
    ▼
Dense (1, activation='linear')
```

#### GRU Model

```python
def GRU_process(data, PCA_Enable, n_prev_days):
    """
    Train GRU model for USD/IDR forecasting.
    Similar to LSTM but uses GRU layers.
    """
```

#### CNN-LSTM Model

```python
def CNNLSTM_process(data, PCA_Enable, n_prev_days):
    """
    Train CNN-LSTM hybrid model.
    Uses Conv1D layers for feature extraction
    followed by LSTM for sequence modeling.
    """
```

**Model Architecture:**
```
Input Shape: (n_prev_days, n_features)
    │
    ▼
Conv1D (64 filters, kernel_size=3)
    │
    ▼
BatchNormalization + Dropout
    │
    ▼
Conv1D (32 filters, kernel_size=3)
    │
    ▼
LSTM (64 units, return_sequences=True)
    │
    ▼
LSTM (32 units)
    │
    ▼
Dense (1)
```

---

### Section 4: Ensemble Methods

**Cells 61-70** | **Runtime: ~5 minutes**

Combine predictions from multiple models.

```python
# Collect predictions from all models
predictions_lstm = model_lstm.predict(X_test)
predictions_gru = model_gru.predict(X_test)
predictions_cnn_lstm = model_cnn_lstm.predict(X_test)

# Stack predictions
all_predictions = np.stack([
    predictions_lstm,
    predictions_gru,
    predictions_cnn_lstm
], axis=1)

# Aggregation methods
ensemble_mean = np.mean(all_predictions, axis=1)
ensemble_median = np.median(all_predictions, axis=1)

# Weighted average (based on validation performance)
weights = [0.4, 0.3, 0.3]  # LSTM, GRU, CNN-LSTM
ensemble_weighted = np.average(all_predictions, axis=1, weights=weights)
```

**Neural Network Stacking:**

```python
# Use predictions as features for meta-learner
X_meta = np.column_stack([predictions_lstm, predictions_gru, predictions_cnn_lstm])

# Train neural network meta-learner
meta_model = Sequential([
    Dense(16, activation='relu', input_shape=(3,)),
    Dropout(0.2),
    Dense(8, activation='relu'),
    Dense(1)
])

meta_model.compile(optimizer='adam', loss='mse')
meta_model.fit(X_meta_train, y_train, epochs=100, validation_split=0.2)
```

---

### Section 5: Inflation Forecasting

**Cells 71-80** | **Runtime: ~15-20 minutes**

Apply the same models to inflation prediction.

```python
def LSTM_process_inflation(data, PCA_Enable, n_prev_days):
    """
    Train LSTM for inflation forecasting.
    Similar architecture but different target variable.
    """

def GRU_process_Inflation(data, PCA_Enable, n_prev_days):
    """Train GRU for inflation."""

def CNNLSTM_process_Inflation(data, PCA_Enable, n_prev_days):
    """Train CNN-LSTM for inflation."""
```

**Key Differences from USD/IDR:**

| Aspect | USD/IDR | Inflation |
|--------|---------|-----------|
| Target Range | 10,000-16,000 | 0-10% |
| Volatility | Higher | Lower |
| Seasonality | Less | More (monthly patterns) |
| Best Model | CNN-LSTM | LSTM |

---

## Key Functions

### Data Processing Functions

```python
def daily_monthly(data):
    """
    Resample daily data to monthly frequency.

    Uses last available value for each month.
    Handles missing months with forward fill.

    Parameters:
    -----------
    data : DataFrame
        Daily data with DatetimeIndex

    Returns:
    --------
    DataFrame with monthly frequency
    """
    data.index = pd.to_datetime(data.index)
    monthly = data.resample('M').last()
    return monthly.ffill()


def quarterly_monthly(data):
    """
    Convert quarterly data to monthly using interpolation.

    Parameters:
    -----------
    data : DataFrame
        Quarterly data

    Returns:
    --------
    DataFrame with monthly frequency (interpolated)
    """
    data.index = pd.to_datetime(data.index)
    monthly = data.resample('M').asfreq()
    return monthly.interpolate(method='linear')


def monthly_period(data):
    """
    Standardize monthly data format.

    Ensures consistent date format (end of month).
    """
    data.index = pd.to_datetime(data.index)
    data.index = data.index.to_period('M').to_timestamp('M')
    return data
```

### Model Training Functions

```python
def create_sequences(data, n_prev_days):
    """
    Create input sequences for LSTM/GRU models.

    Parameters:
    -----------
    data : array
        Scaled feature data
    n_prev_days : int
        Number of previous timesteps to use

    Returns:
    --------
    X : array of shape (samples, n_prev_days, features)
    y : array of shape (samples,)
    """
    X, y = [], []
    for i in range(n_prev_days, len(data)):
        X.append(data[i-n_prev_days:i])
        y.append(data[i, 0])  # Target is first column
    return np.array(X), np.array(y)
```

### Evaluation Functions

```python
def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics.

    Returns:
    --------
    dict with MAE, MSE, RMSE, MAPE, R²
    """
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }
```

---

## Model Configuration

### Hyperparameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `n_prev_days` | 12 | 3-24 | Lookback period (months) |
| `PCA_Enable` | True | True/False | Use PCA dimensionality reduction |
| `pca_variance` | 0.95 | 0.9-0.99 | Variance retained by PCA |
| `lstm_units` | [128, 64, 32] | - | LSTM layer sizes |
| `dropout_rate` | 0.2 | 0.1-0.5 | Dropout rate |
| `learning_rate` | 0.001 | 1e-4 to 1e-2 | Adam learning rate |
| `epochs` | 100 | 50-200 | Training epochs |
| `batch_size` | 32 | 16-64 | Batch size |

### Modifying Parameters

```python
# Example: Change lookback period
n_prev_days = 18  # Use 18 months of history

# Example: Disable PCA
PCA_Enable = False

# Example: Modify model architecture
# Find and modify the model definition in the LSTM_process function:
model = Sequential([
    Bidirectional(LSTM(256, return_sequences=True)),  # Changed from 128
    Dropout(0.3),  # Changed from 0.2
    Bidirectional(LSTM(128, return_sequences=True)),  # Changed from 64
    Dropout(0.3),
    Bidirectional(LSTM(64)),  # Changed from 32
    Dense(32, activation='relu'),  # Changed from 16
    Dense(1)
])
```

---

## Running Experiments

### Quick Experiment (10 minutes)

Run only essential cells:

1. Cells 1-5: Load data
2. Cells 6-10: Preprocess
3. Cell 21: Train single LSTM
4. Cell 22: Evaluate

```python
# Skip cross-validation for quick test
# Set smaller epochs
epochs = 20

# Use smaller lookback
n_prev_days = 6
```

### Full Experiment (60 minutes)

Run all cells sequentially:

1. **Kernel > Restart & Run All**
2. Monitor progress in output cells
3. Results saved in variables

### Custom Experiment

```python
# Test different lookback periods
lookbacks = [3, 6, 12, 18, 24]
results = {}

for n_prev in lookbacks:
    _, preds, metrics = LSTM_process(data, PCA_Enable=True, n_prev_days=n_prev)
    results[n_prev] = metrics
    print(f"Lookback {n_prev}: MAPE = {metrics['MAPE']:.2f}%")
```

---

## Interpreting Results

### Model Performance Comparison

After running all models, compare metrics:

```python
# Results comparison table
results_df = pd.DataFrame({
    'Model': ['LSTM', 'GRU', 'CNN-LSTM', 'Ensemble'],
    'MAE': [mae_lstm, mae_gru, mae_cnn, mae_ensemble],
    'RMSE': [rmse_lstm, rmse_gru, rmse_cnn, rmse_ensemble],
    'MAPE': [mape_lstm, mape_gru, mape_cnn, mape_ensemble],
    'R2': [r2_lstm, r2_gru, r2_cnn, r2_ensemble]
})

print(results_df.sort_values('MAPE'))
```

### Visualization

```python
# Actual vs Predicted plot
plt.figure(figsize=(14, 6))
plt.plot(dates_test, y_actual, label='Actual', linewidth=2)
plt.plot(dates_test, y_predicted, label='Predicted', linewidth=2, alpha=0.8)
plt.fill_between(dates_test, y_lower, y_upper, alpha=0.2, label='95% CI')
plt.legend()
plt.title('USD/IDR: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Exchange Rate')
plt.show()
```

### What Good Results Look Like

| Metric | Excellent | Good | Acceptable | Poor |
|--------|-----------|------|------------|------|
| MAPE | < 2% | 2-5% | 5-10% | > 10% |
| R² | > 0.95 | 0.9-0.95 | 0.8-0.9 | < 0.8 |
| RMSE | Context-dependent | - | - | - |

---

## Customization

### Adding New Features

```python
# Add new feature to data
data['new_feature'] = calculate_new_feature(...)

# Update feature list
features = ['USD_IDR', 'Gold', 'Brent', ..., 'new_feature']

# Re-run preprocessing and training
```

### Adding New Models

```python
def Transformer_process(data, PCA_Enable, n_prev_days):
    """
    Add Transformer model for comparison.
    """
    # Preprocess data (same as LSTM)
    X_train, X_test, y_train, y_test = preprocess_data(data, PCA_Enable, n_prev_days)

    # Build Transformer model
    model = build_transformer_model(input_shape=(n_prev_days, n_features))

    # Train
    model.fit(X_train, y_train, epochs=100, validation_split=0.2)

    # Evaluate
    predictions = model.predict(X_test)
    metrics = calculate_metrics(y_test, predictions)

    return model, predictions, metrics
```

### Changing Target Variable

```python
# Change from USD/IDR to Gold price
target_column = 'Gold_Price'  # Instead of 'USD_IDR'

# Update in data preparation section
y = data[target_column].values
```

---

## Troubleshooting

### Common Issues

#### 1. Memory Error

```python
# Reduce batch size
batch_size = 16

# Reduce model complexity
lstm_units = [64, 32]  # Instead of [128, 64, 32]

# Use fewer epochs
epochs = 50
```

#### 2. Slow Training

```python
# Enable GPU
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Reduce lookback period
n_prev_days = 6

# Use early stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(patience=10, restore_best_weights=True)
```

#### 3. Poor Model Performance

```python
# Try different lookback periods
for n in [3, 6, 12, 18]:
    test_model(n_prev_days=n)

# Try with/without PCA
test_model(PCA_Enable=False)

# Check for data issues
print(data.isnull().sum())
print(data.describe())
```

#### 4. Data Loading Errors

```python
# Check file paths
import os
print(os.listdir('data/'))

# Check file format
df = pd.read_csv('data/file.csv')
print(df.head())
print(df.dtypes)
```

#### 5. Kernel Crashes

```python
# Restart kernel and run cells one by one
# Clear outputs: Cell > All Output > Clear

# Check memory usage
import psutil
print(f"Memory: {psutil.Process().memory_info().rss / 1e9:.1f} GB")
```

---

## Tips for Best Results

### 1. Data Quality

- Ensure no missing values in critical columns
- Check for outliers and handle appropriately
- Verify date alignment across datasets

### 2. Model Selection

- Start with LSTM as baseline
- Try GRU if training is too slow
- Use CNN-LSTM for pattern-heavy data
- Always use ensemble for final predictions

### 3. Hyperparameter Tuning

- Test lookback periods: 3, 6, 12, 18, 24
- Compare with/without PCA
- Adjust dropout if overfitting

### 4. Validation

- Use time-series cross-validation
- Never shuffle time series data
- Keep most recent data for final test

---

## Next Steps

After mastering the notebook:

1. **Try the Refactored System** - For production use
2. **Add New Data Sources** - More indicators = better predictions
3. **Experiment with New Models** - Transformers, attention mechanisms
4. **Build a Dashboard** - Visualize forecasts in real-time

---

**Version:** 2.0.0 | **Last Updated:** January 2026
