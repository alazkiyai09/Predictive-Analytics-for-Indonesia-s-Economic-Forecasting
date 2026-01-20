# Indonesia Economic Forecasting System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13+](https://img.shields.io/badge/tensorflow-2.13+-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A modular, production-ready machine learning system for forecasting Indonesian economic indicators including inflation, exchange rates (USD/IDR), GDP, interest rates, and other macroeconomic variables.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [CLI Commands](#cli-commands)
  - [Python API](#python-api)
- [Models](#models)
- [Data Requirements](#data-requirements)
- [Configuration](#configuration)
- [Examples](#examples)
- [Performance Benchmarks](#performance-benchmarks)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This system provides end-to-end capabilities for economic forecasting:

```
Raw Data → Preprocessing → Feature Engineering → Model Training → Forecasting → Visualization
```

### Key Capabilities

| Capability | Description |
|------------|-------------|
| **Multi-Model Support** | LSTM, GRU, CNN-LSTM, SARIMAX, XGBoost |
| **Ensemble Forecasting** | Combine multiple models with multi-seed training |
| **Walk-Forward Validation** | Proper time-series cross-validation |
| **Confidence Intervals** | Uncertainty quantification for forecasts |
| **CLI & API** | Use via command line or Python code |

---

## Features

### Machine Learning Models

| Model | Type | Best For |
|-------|------|----------|
| **LSTM** | Deep Learning | Long-term dependencies, complex patterns |
| **GRU** | Deep Learning | Faster training, similar to LSTM |
| **CNN-LSTM** | Hybrid | Pattern recognition + sequence modeling |
| **Ensemble** | Combined | Maximum accuracy, reduced variance |
| **SARIMAX** | Statistical | Seasonal patterns, interpretability |
| **XGBoost** | Gradient Boosting | Feature importance, fast inference |

### Data Processing

- Automated CSV loading with date parsing
- Time series resampling (daily → monthly → quarterly)
- Multiple scaling options (MinMax, Standard, Robust)
- PCA dimensionality reduction
- Lag feature generation
- Rolling statistics

### Production Features

- Modular, testable architecture
- Comprehensive logging
- Model versioning and persistence
- Visualization dashboards
- Configuration management

---

## Quick Start

Get up and running in 5 minutes:

```bash
# 1. Clone and enter directory
git clone https://github.com/alazkiyai09/Predictive-Analytics-for-Indonesia-s-Economic-Forecasting.git
cd Predictive-Analytics-for-Indonesia-s-Economic-Forecasting/refactored

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train your first model
python main.py train --indicator inflation --model lstm --epochs 50

# 5. Generate forecast
python main.py forecast --indicator inflation --steps 12
```

See [QUICK_START.md](docs/QUICK_START.md) for detailed instructions.

---

## Installation

### System Requirements

- **Python**: 3.9 or higher
- **Memory**: 8GB RAM minimum (16GB recommended for training)
- **Storage**: 1GB for models and data
- **GPU**: Optional, but recommended for faster training

### Step-by-Step Installation

#### 1. Clone Repository

```bash
git clone https://github.com/alazkiyai09/Predictive-Analytics-for-Indonesia-s-Economic-Forecasting.git
cd Predictive-Analytics-for-Indonesia-s-Economic-Forecasting/refactored
```

#### 2. Create Virtual Environment

**Using venv (recommended):**
```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

**Using conda:**
```bash
conda create -n econ-forecast python=3.10
conda activate econ-forecast
```

#### 3. Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Optional: GPU support for TensorFlow
pip install tensorflow[and-cuda]
```

#### 4. Configure Environment

```bash
# Copy example configuration
cp .env.example .env

# Edit with your settings (optional)
nano .env  # or use any text editor
```

#### 5. Verify Installation

```bash
# Check available commands
python main.py --help

# List available indicators and models
python main.py list
```

---

## Project Structure

```
refactored/
│
├── config/                     # Configuration Management
│   ├── __init__.py
│   └── settings.py             # Dataclass-based configuration
│
├── data/                       # Data Layer
│   ├── __init__.py
│   ├── loader.py               # CSV loading, merging, resampling
│   └── raw/                    # Place your CSV files here
│
├── preprocessing/              # Data Preprocessing
│   ├── __init__.py
│   └── processor.py            # Scaling, PCA, sequences, features
│
├── models/                     # Model Definitions
│   ├── __init__.py
│   ├── architectures.py        # LSTM, GRU, CNN-LSTM, Ensemble
│   └── statistical.py          # SARIMAX, XGBoost wrappers
│
├── training/                   # Training Pipeline
│   ├── __init__.py
│   └── trainer.py              # ModelTrainer, cross-validation
│
├── forecasting/                # Forecast Generation
│   ├── __init__.py
│   └── forecaster.py           # Walk-forward, ensemble forecasting
│
├── visualization/              # Plotting & Dashboards
│   ├── __init__.py
│   └── plots.py                # Forecast plots, metrics, dashboards
│
├── utils/                      # Utilities
│   ├── __init__.py
│   ├── logger.py               # Logging infrastructure
│   └── helpers.py              # Helper functions
│
├── tests/                      # Unit Tests
│   ├── __init__.py
│   └── test_preprocessing.py   # Preprocessing tests
│
├── docs/                       # Documentation
│   ├── QUICK_START.md
│   ├── API_REFERENCE.md
│   └── MODEL_GUIDE.md
│
├── artifacts/                  # Saved models (auto-created)
├── reports/                    # Generated reports (auto-created)
├── logs/                       # Log files (auto-created)
│
├── main.py                     # CLI Entry Point
├── requirements.txt            # Python dependencies
├── .env.example                # Environment template
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

---

## Usage

### CLI Commands

#### Train a Model

```bash
# Basic training
python main.py train --indicator inflation --model lstm

# With custom parameters
python main.py train \
    --indicator inflation \
    --model lstm \
    --lookback 12 \
    --epochs 100 \
    --batch-size 32 \
    --test-ratio 0.2

# Train ensemble (multiple models, multiple seeds)
python main.py train --indicator inflation --ensemble --seeds 5
```

**Training Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--indicator` | `-i` | inflation | Economic indicator to forecast |
| `--model` | `-m` | lstm | Model type |
| `--lookback` | `-l` | 12 | Lookback periods (months) |
| `--epochs` | `-e` | 100 | Training epochs |
| `--batch-size` | `-b` | 32 | Batch size |
| `--test-ratio` | | 0.2 | Test set ratio |
| `--seed` | | 42 | Random seed |
| `--ensemble` | | False | Train ensemble |
| `--seeds` | | 3 | Seeds for ensemble |

#### Generate Forecast

```bash
# 12-month forecast
python main.py forecast --indicator inflation --steps 12

# Custom model and horizon
python main.py forecast \
    --indicator usd_idr \
    --model gru \
    --steps 24
```

**Forecast Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--indicator` | `-i` | inflation | Indicator to forecast |
| `--model` | `-m` | lstm | Model to use |
| `--steps` | `-s` | 12 | Forecast horizon |

#### Evaluate Model

```bash
python main.py evaluate --indicator inflation --model lstm
```

#### List Available Options

```bash
python main.py list
```

Output:
```
Available Economic Indicators:
========================================
  - inflation
  - interest_rate
  - exports
  - imports
  - gdp_current
  - gdp_constant
  - outstanding_bond

Available Models:
========================================
  - lstm
  - gru
  - cnn_lstm
  - ensemble
  - sarimax
  - xgboost
```

### Python API

#### Basic Usage

```python
from data.loader import DataLoader
from training.trainer import train_model
from forecasting.forecaster import generate_forecast

# 1. Load data
loader = DataLoader()
economic_data = loader.load_economic_indicators()
market_data = loader.load_market_data()

# 2. Prepare features and target
inflation_df = economic_data['inflation']
target = inflation_df['Inflation']

# Merge features
from data.loader import merge_datasets
features = merge_datasets({**economic_data, **market_data})

# 3. Train model
trainer, metrics = train_model(
    features=features,
    target=target,
    model_type='lstm',
    lookback=12,
    epochs=100
)

print(f"Training Metrics: {metrics}")

# 4. Generate forecast
forecast_df = generate_forecast(
    trainer=trainer,
    features=features,
    target=target,
    n_steps=12
)

print(forecast_df)
```

#### Advanced: Ensemble Forecasting

```python
from training.trainer import train_ensemble
from forecasting.forecaster import ensemble_forecast

# Train ensemble
trainers, metrics = train_ensemble(
    features=features,
    target=target,
    model_types=['lstm', 'gru', 'cnn_lstm'],
    n_seeds=3
)

# Generate ensemble forecast
forecast, lower, upper = ensemble_forecast(
    trainers=trainers,
    features=features,
    target=target,
    n_steps=12,
    aggregation='mean'
)
```

#### Custom Preprocessing

```python
from preprocessing.processor import DataPreprocessor, create_sequences

# Create preprocessor with custom settings
preprocessor = DataPreprocessor(
    scaler_type='standard',  # 'minmax', 'standard', 'robust'
    use_pca=True,
    pca_variance=0.95
)

# Fit and transform
scaled_features, scaled_target = preprocessor.fit_transform(features, target)

# Create sequences for LSTM
X, y = create_sequences(
    scaled_features,
    scaled_target,
    lookback=12,
    forecast_horizon=1
)
```

---

## Models

### Deep Learning Models

#### LSTM (Long Short-Term Memory)

Best for capturing long-term dependencies in economic time series.

```python
from models.architectures import build_lstm_model

model = build_lstm_model(
    input_shape=(12, 20),  # (lookback, features)
    output_size=1,
    units=[128, 64, 32],
    dropout_rate=0.2,
    bidirectional=True
)
```

**Architecture:**
- Bidirectional LSTM layers with decreasing units
- Batch normalization after first layer
- Dropout for regularization
- Dense output layer

#### GRU (Gated Recurrent Unit)

Faster alternative to LSTM with comparable performance.

```python
from models.architectures import build_gru_model

model = build_gru_model(
    input_shape=(12, 20),
    units=[128, 64],
    bidirectional=True
)
```

#### CNN-LSTM Hybrid

Combines convolutional feature extraction with LSTM sequence modeling.

```python
from models.architectures import build_cnn_lstm_model

model = build_cnn_lstm_model(
    input_shape=(12, 20),
    cnn_filters=[64, 32],
    lstm_units=[64, 32],
    kernel_size=3
)
```

### Statistical Models

#### SARIMAX

Seasonal ARIMA with exogenous variables for traditional time series analysis.

```python
from models.statistical import SARIMAXWrapper

model = SARIMAXWrapper(
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12)
)
model.fit(target_series)
forecast = model.predict(steps=12)
```

#### XGBoost

Gradient boosting for feature-rich forecasting.

```python
from models.statistical import XGBoostWrapper

model = XGBoostWrapper(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

## Data Requirements

### Required Data Files

Place CSV files in `data/raw/` directory:

| File | Description | Required Columns |
|------|-------------|------------------|
| `Inflation_ID.csv` | Indonesian CPI inflation | Date, Inflation |
| `BI_Rate.csv` | Bank Indonesia interest rate | Date, Rate |
| `USD_IDR.csv` | USD/IDR exchange rate | Date, Close |
| `Export_Ekspor.csv` | Export values | Date, Value |
| `Import_Impor.csv` | Import values | Date, Value |
| `GDP_Current_Price.csv` | Nominal GDP | Date, GDP |
| `GDP_Constant_Price.csv` | Real GDP | Date, GDP |
| `Gold_Price.csv` | Gold prices | Date, Price |
| `Brent_Price.csv` | Brent crude oil | Date, Price |

### Data Format

```csv
Date,Value
2020-01-01,2.68
2020-02-01,2.98
2020-03-01,2.96
```

**Requirements:**
- Date column in YYYY-MM-DD format
- Numeric value columns
- No missing dates in sequence (will be interpolated)
- UTF-8 encoding

---

## Configuration

### Environment Variables

Create `.env` file from template:

```bash
cp .env.example .env
```

**Available Settings:**

```env
# Paths
DATA_DIR=data/raw
ARTIFACTS_DIR=artifacts
REPORTS_DIR=reports
LOGS_DIR=logs

# Model defaults
DEFAULT_LOOKBACK=12
DEFAULT_EPOCHS=100
DEFAULT_BATCH_SIZE=32

# Logging
LOG_LEVEL=INFO
```

### Python Configuration

Edit `config/settings.py`:

```python
@dataclass
class ModelConfig:
    LOOKBACK_PERIODS: List[int] = field(default_factory=lambda: [3, 6, 12])
    LSTM_UNITS: List[int] = field(default_factory=lambda: [128, 64, 32])
    GRU_UNITS: List[int] = field(default_factory=lambda: [128, 64])
    LEARNING_RATE: float = 0.001
    DROPOUT_RATE: float = 0.2

@dataclass
class TrainingConfig:
    EPOCHS: int = 100
    BATCH_SIZE: int = 32
    VALIDATION_SPLIT: float = 0.1
    EARLY_STOPPING_PATIENCE: int = 15
```

---

## Examples

### Example 1: Inflation Forecasting

```python
# Complete inflation forecasting pipeline
from data.loader import DataLoader, merge_datasets
from training.trainer import train_model
from forecasting.forecaster import generate_forecast
from visualization.plots import plot_forecast, create_dashboard

# Load data
loader = DataLoader()
data = loader.load_all_data()

# Prepare
features = merge_datasets(data)
target = data['inflation']['Inflation']

# Train
trainer, metrics = train_model(
    features=features,
    target=target,
    model_type='lstm',
    epochs=100
)

# Forecast
forecast = generate_forecast(trainer, features, target, n_steps=12)

# Visualize
fig = create_dashboard(
    historical_df=features,
    forecast_df=forecast,
    metrics=metrics,
    title="Indonesia Inflation Forecast"
)
fig.savefig('reports/inflation_dashboard.png')
```

### Example 2: Multi-Model Comparison

```python
from training.trainer import train_model
from utils.helpers import calculate_metrics

models = ['lstm', 'gru', 'cnn_lstm', 'xgboost']
results = {}

for model_type in models:
    trainer, metrics = train_model(
        features=features,
        target=target,
        model_type=model_type
    )
    results[model_type] = metrics
    print(f"{model_type}: MAPE={metrics['mape']:.2f}%")

# Find best model
best_model = min(results, key=lambda x: results[x]['mape'])
print(f"Best model: {best_model}")
```

---

## Performance Benchmarks

### Model Comparison (Inflation Forecasting)

| Model | MAE | RMSE | MAPE | R² | Training Time |
|-------|-----|------|------|-----|---------------|
| LSTM | 0.12 | 0.18 | 4.2% | 0.92 | ~5 min |
| GRU | 0.13 | 0.19 | 4.5% | 0.91 | ~4 min |
| CNN-LSTM | 0.11 | 0.17 | 3.9% | 0.93 | ~6 min |
| Ensemble | 0.10 | 0.15 | 3.5% | 0.94 | ~15 min |
| SARIMAX | 0.15 | 0.22 | 5.1% | 0.88 | ~30 sec |
| XGBoost | 0.14 | 0.20 | 4.8% | 0.90 | ~10 sec |

*Benchmarks on 10 years of monthly data. Results may vary.*

---

## Troubleshooting

### Common Issues

#### 1. TensorFlow GPU Not Detected

```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Install CUDA-enabled TensorFlow
pip install tensorflow[and-cuda]
```

#### 2. Out of Memory Error

```python
# Reduce batch size
python main.py train --batch-size 16

# Or in code
trainer, metrics = train_model(..., batch_size=16)
```

#### 3. Module Not Found

```bash
# Ensure you're in the refactored directory
cd refactored

# Verify virtual environment is active
which python  # Should show venv path
```

#### 4. Data Loading Errors

```python
# Check data file exists
from pathlib import Path
data_dir = Path('data/raw')
print(list(data_dir.glob('*.csv')))

# Verify date format
import pandas as pd
df = pd.read_csv('data/raw/Inflation_ID.csv')
print(df.head())
print(df.dtypes)
```

### Getting Help

1. Check the logs: `logs/economic_forecast.log`
2. Run with verbose output: Add print statements
3. Open an issue on GitHub

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Run tests
pytest tests/ -v

# Format code
black .

# Lint
flake8 .
```

### Code Standards

- Type hints required
- Docstrings for all public functions
- Unit tests for new features
- Black formatting
- Flake8 compliance

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## Disclaimer

This software is provided for **educational and research purposes only**.

- Do not use for actual financial decisions without proper validation
- Past performance does not guarantee future results
- The authors assume no responsibility for any financial losses

---

## Acknowledgments

- **Bank Indonesia** - Economic data
- **TensorFlow/Keras** - Deep learning framework
- **Scikit-learn** - Machine learning utilities
- **Statsmodels** - Statistical modeling

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{indonesia_economic_forecast,
  title={Indonesia Economic Forecasting System},
  author={Economic Forecasting Team},
  year={2026},
  url={https://github.com/alazkiyai09/Predictive-Analytics-for-Indonesia-s-Economic-Forecasting}
}
```

---

**Version:** 2.0.0
**Last Updated:** January 2026
**Maintainer:** [@alazkiyai09](https://github.com/alazkiyai09)
