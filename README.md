# Predictive Analytics for Indonesia's Economic Forecasting

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-ff6f00.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive machine learning system for forecasting Indonesian economic indicators, including USD/IDR exchange rate and inflation, using deep learning models (LSTM, GRU, CNN-LSTM) and ensemble methods.

---

## Project Overview

This project provides two versions of the economic forecasting system:

| Version | Description | Best For |
|---------|-------------|----------|
| **[Jupyter Notebook](#jupyter-notebook)** | Original research notebook with full analysis | Learning, experimentation, visualization |
| **[Refactored System](#refactored-system)** | Production-ready modular codebase | Production deployment, integration |

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Jupyter Notebook](#jupyter-notebook)
- [Refactored System](#refactored-system)
- [Data Sources](#data-sources)
- [Models](#models)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## Features

### Forecasting Capabilities

- **USD/IDR Exchange Rate** - Forecast Indonesian Rupiah against US Dollar
- **Inflation Rate** - Predict Consumer Price Index changes
- **GDP Growth** - Economic growth projections

### Machine Learning Models

| Model | Description | Use Case |
|-------|-------------|----------|
| **LSTM** | Long Short-Term Memory networks | Long-term dependencies |
| **GRU** | Gated Recurrent Units | Faster training |
| **CNN-LSTM** | Convolutional + Recurrent hybrid | Pattern recognition |
| **Ensemble** | Multi-model combination | Maximum accuracy |
| **SARIMAX** | Statistical time series | Baseline, interpretability |

### Data Processing

- 22+ economic indicators from multiple sources
- Automated data merging and alignment
- PCA dimensionality reduction
- Multiple lookback period analysis

---

## Quick Start

### Option 1: Jupyter Notebook (Research/Learning)

```bash
# Clone repository
git clone https://github.com/alazkiyai09/Predictive-Analytics-for-Indonesia-s-Economic-Forecasting.git
cd Predictive-Analytics-for-Indonesia-s-Economic-Forecasting

# Install dependencies
pip install jupyter pandas numpy tensorflow scikit-learn matplotlib seaborn

# Launch notebook
jupyter notebook "Predictive Analytics for Indonesia's Economic Forecasting.ipynb"
```

### Option 2: Refactored System (Production)

```bash
# Navigate to refactored system
cd refactored

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Train model
python main.py train --indicator inflation --model lstm

# Generate forecast
python main.py forecast --indicator inflation --steps 12
```

---

## Jupyter Notebook

### Overview

The Jupyter notebook (`Predictive Analytics for Indonesia's Economic Forecasting.ipynb`) contains the complete research and development workflow:

```
Notebook Structure (82 cells, ~27,000 lines)
├── Data Loading & Preprocessing
│   ├── Load 22+ CSV data files
│   ├── Resample daily/quarterly to monthly
│   ├── Merge and align datasets
│   └── Handle missing values
│
├── Exploratory Data Analysis
│   ├── Correlation analysis
│   ├── Time series visualization
│   └── Feature distributions
│
├── USD/IDR Exchange Rate Forecasting
│   ├── LSTM model (with/without PCA)
│   ├── GRU model (with/without PCA)
│   ├── CNN-LSTM model (with/without PCA)
│   ├── Cross-validation
│   └── Ensemble predictions
│
├── Inflation Forecasting
│   ├── LSTM model
│   ├── GRU model
│   ├── CNN-LSTM model
│   └── Model comparison
│
└── Results & Visualization
    ├── Actual vs Predicted plots
    ├── Error analysis
    └── Model performance comparison
```

### Key Functions in Notebook

| Function | Description |
|----------|-------------|
| `daily_monthly(data)` | Convert daily data to monthly frequency |
| `quarterly_monthly(data)` | Convert quarterly data to monthly |
| `monthly_period(data)` | Standardize monthly data format |
| `LSTM_process(data, PCA_Enable, n_prev_days)` | Train LSTM for USD/IDR |
| `GRU_process(data, PCA_Enable, n_prev_days)` | Train GRU for USD/IDR |
| `CNNLSTM_process(data, PCA_Enable, n_prev_days)` | Train CNN-LSTM for USD/IDR |
| `LSTM_process_inflation(data, PCA_Enable, n_prev_days)` | Train LSTM for inflation |
| `GRU_process_Inflation(data, PCA_Enable, n_prev_days)` | Train GRU for inflation |
| `CNNLSTM_process_Inflation(data, PCA_Enable, n_prev_days)` | Train CNN-LSTM for inflation |

### Running the Notebook

1. **Full Execution**: Run all cells sequentially (may take 30-60 minutes)
2. **Selective Execution**: Run specific sections based on your needs

```python
# To use specific models:
# Section 1: Data Loading (cells 1-15)
# Section 2: USD/IDR LSTM (cells 16-30)
# Section 3: USD/IDR GRU (cells 31-45)
# Section 4: USD/IDR CNN-LSTM (cells 46-60)
# Section 5: Ensemble (cells 61-70)
# Section 6: Inflation models (cells 71-82)
```

### Notebook Dependencies

```python
# Core
import pandas as pd
import numpy as np

# Machine Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D

# Preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
```

See [NOTEBOOK_GUIDE.md](docs/NOTEBOOK_GUIDE.md) for detailed documentation.

---

## Refactored System

The refactored version provides a production-ready, modular codebase.

### Structure

```
refactored/
├── config/settings.py      # Configuration management
├── data/loader.py          # Data loading utilities
├── preprocessing/          # Data preprocessing
├── models/                 # Model architectures
├── training/               # Training pipeline
├── forecasting/            # Forecast generation
├── visualization/          # Plotting utilities
├── utils/                  # Helper functions
├── tests/                  # Unit tests
├── main.py                 # CLI entry point
└── docs/                   # Documentation
    ├── QUICK_START.md
    ├── API_REFERENCE.md
    ├── MODEL_GUIDE.md
    └── CONTRIBUTING.md
```

### CLI Commands

```bash
# Train models
python main.py train --indicator inflation --model lstm --epochs 100

# Generate forecasts
python main.py forecast --indicator inflation --steps 12

# Evaluate models
python main.py evaluate --indicator inflation --model lstm

# List options
python main.py list
```

See [refactored/README.md](refactored/README.md) for complete documentation.

---

## Data Sources

### Economic Indicators

| Category | Indicator | File | Frequency |
|----------|-----------|------|-----------|
| **Monetary** | Inflation | `Inflation_ID.csv` | Monthly |
| **Monetary** | BI Rate | `BI_Rate.csv` | Monthly |
| **Trade** | Exports | `Export_Ekspor.csv` | Monthly |
| **Trade** | Imports | `Import_Impor.csv` | Monthly |
| **GDP** | Current Price | `GDP_Current_Price.csv` | Quarterly |
| **GDP** | Constant Price | `GDP_Constant_Price.csv` | Quarterly |
| **Bonds** | Outstanding | `Outstanding_bond.csv` | Monthly |

### Market Data

| Category | Indicator | File | Frequency |
|----------|-----------|------|-----------|
| **Forex** | USD/IDR | `USD_IDR.csv` | Daily |
| **Forex** | USD/JPY | `USD_JPY.csv` | Daily |
| **Forex** | EUR/USD | `EUR_USD.csv` | Daily |
| **Forex** | GBP/USD | `GBP_USD.csv` | Daily |
| **Forex** | DXY Index | `DXY.csv` | Daily |
| **Commodities** | Gold | `Gold_Price.csv` | Daily |
| **Commodities** | Brent Oil | `Brent_Price.csv` | Daily |
| **Commodities** | WTI Oil | `WTI_Price.csv` | Daily |
| **Equities** | IDX Composite | `IDX.csv` | Daily |
| **Bonds** | Spread | `Spread_bond.csv` | Daily |

### Money Supply

| Region | Indicator | File | Frequency |
|--------|-----------|------|-----------|
| Indonesia | M1, M2 | `M1_M2_ID.csv` | Monthly |
| USA | M1, M2 | `M1_M2_US.csv` | Monthly |
| EU | M1, M2 | `M1_M2_EU.csv` | Monthly |
| Japan | M1, M2 | `M1_M2_JP.csv` | Monthly |
| UK | M2 | `M2_UK.csv` | Monthly |

### Data Format

All CSV files should follow this format:

```csv
Date,Value
2020-01-01,14500.50
2020-01-02,14525.75
2020-01-03,14480.25
```

---

## Models

### Architecture Comparison

```
LSTM                          GRU                           CNN-LSTM
────────────────────          ────────────────────          ────────────────────
Input (lookback, features)    Input (lookback, features)    Input (lookback, features)
        │                             │                             │
        ▼                             ▼                             ▼
┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐
│  Bidirectional  │           │  Bidirectional  │           │    Conv1D       │
│     LSTM        │           │     GRU         │           │   (64 filters)  │
└─────────────────┘           └─────────────────┘           └─────────────────┘
        │                             │                             │
        ▼                             ▼                             ▼
┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐
│  Bidirectional  │           │  Bidirectional  │           │    Conv1D       │
│     LSTM        │           │     GRU         │           │   (32 filters)  │
└─────────────────┘           └─────────────────┘           └─────────────────┘
        │                             │                             │
        ▼                             ▼                             ▼
┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐
│     Dense       │           │     Dense       │           │      LSTM       │
│    (Output)     │           │    (Output)     │           │   + Dense       │
└─────────────────┘           └─────────────────┘           └─────────────────┘
```

### Ensemble Method

The ensemble combines predictions from multiple models:

1. Train LSTM, GRU, CNN-LSTM with different random seeds
2. Generate predictions from each model
3. Aggregate using mean/median/weighted average
4. Calculate confidence intervals from prediction spread

---

## Results

### USD/IDR Exchange Rate Forecasting

| Model | MAE | RMSE | MAPE | R² |
|-------|-----|------|------|-----|
| LSTM | 285.4 | 412.8 | 1.92% | 0.918 |
| GRU | 298.7 | 428.3 | 2.01% | 0.911 |
| CNN-LSTM | 271.2 | 395.6 | 1.83% | 0.924 |
| Ensemble | 254.8 | 372.1 | 1.72% | 0.933 |

### Inflation Forecasting

| Model | MAE | RMSE | MAPE | R² |
|-------|-----|------|------|-----|
| LSTM | 0.118 | 0.176 | 4.2% | 0.924 |
| GRU | 0.125 | 0.184 | 4.5% | 0.913 |
| CNN-LSTM | 0.108 | 0.165 | 3.9% | 0.931 |
| Ensemble | 0.095 | 0.148 | 3.5% | 0.945 |

### Key Findings

1. **PCA Impact**: Using PCA (95% variance) improves model stability without significant accuracy loss
2. **Lookback Period**: 12-month lookback performs best for most indicators
3. **Ensemble Advantage**: Combining models reduces MAPE by 15-20% vs single models
4. **Cross-Validation**: 5-fold time series CV ensures robust evaluation

---

## Installation

### Requirements

- Python 3.9+
- 8GB RAM minimum (16GB recommended)
- GPU optional but recommended for training

### Dependencies

```bash
# Core
pip install pandas numpy scipy

# Machine Learning
pip install tensorflow scikit-learn xgboost statsmodels

# Visualization
pip install matplotlib seaborn plotly

# Jupyter (for notebook)
pip install jupyter notebook

# Full installation
pip install -r refactored/requirements.txt
```

### GPU Setup (Optional)

```bash
# For NVIDIA GPUs
pip install tensorflow[and-cuda]

# Verify GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## Usage

### Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Open the notebook file and run cells
```

### Python API

```python
# Using the refactored system
from data.loader import DataLoader
from training.trainer import train_model
from forecasting.forecaster import generate_forecast

# Load data
loader = DataLoader()
data = loader.load_all_data()

# Train model
trainer, metrics = train_model(
    features=data['features'],
    target=data['inflation'],
    model_type='lstm'
)

# Generate forecast
forecast = generate_forecast(trainer, data['features'], data['inflation'], n_steps=12)
```

### Command Line

```bash
cd refactored

# Train
python main.py train -i inflation -m lstm -e 100

# Forecast
python main.py forecast -i inflation -s 12

# Evaluate
python main.py evaluate -i inflation -m lstm
```

---

## Project Structure

```
Predictive-Analytics-for-Indonesia-s-Economic-Forecasting/
│
├── Predictive Analytics for Indonesia's Economic Forecasting.ipynb
│   └── Original Jupyter notebook with full analysis
│
├── README.md                    # This file
│
├── docs/                        # Root-level documentation
│   ├── NOTEBOOK_GUIDE.md        # Jupyter notebook guide
│   └── DATA_DICTIONARY.md       # Data documentation
│
├── refactored/                  # Production codebase
│   ├── config/                  # Configuration
│   ├── data/                    # Data loading
│   ├── preprocessing/           # Data preprocessing
│   ├── models/                  # Model architectures
│   ├── training/                # Training pipeline
│   ├── forecasting/             # Forecast generation
│   ├── visualization/           # Plotting
│   ├── utils/                   # Utilities
│   ├── tests/                   # Unit tests
│   ├── docs/                    # Module documentation
│   ├── main.py                  # CLI entry point
│   ├── requirements.txt         # Dependencies
│   └── README.md                # Refactored system docs
│
└── data/                        # Data files (not tracked)
    └── raw/                     # CSV files
```

---

## Contributing

We welcome contributions! Please see:

- [refactored/docs/CONTRIBUTING.md](refactored/docs/CONTRIBUTING.md) for guidelines
- Open an issue for bugs or feature requests
- Submit pull requests with tests

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

## Citation

If you use this project in your research:

```bibtex
@software{indonesia_economic_forecast,
  title={Predictive Analytics for Indonesia's Economic Forecasting},
  author={alazkiyai09},
  year={2026},
  url={https://github.com/alazkiyai09/Predictive-Analytics-for-Indonesia-s-Economic-Forecasting}
}
```

---

## Acknowledgments

- **Bank Indonesia** - Economic data and publications
- **Yahoo Finance** - Market data
- **TensorFlow/Keras** - Deep learning framework
- **Scikit-learn** - Machine learning utilities

---

**Maintainer:** [@alazkiyai09](https://github.com/alazkiyai09)
**Version:** 2.0.0
**Last Updated:** January 2026
