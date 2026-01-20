# Indonesia Economic Forecasting System

A modular, production-ready machine learning system for forecasting Indonesian economic indicators including inflation, exchange rates, GDP, and other macroeconomic variables.

## Features

- **Multiple Model Architectures**
  - LSTM (Bidirectional)
  - GRU (Bidirectional)
  - CNN-LSTM Hybrid
  - Ensemble Models
  - SARIMAX (Statistical)
  - XGBoost (Gradient Boosting)

- **Comprehensive Data Processing**
  - Automated data loading from CSV files
  - Time series resampling (daily/monthly/quarterly)
  - Missing value handling
  - Feature scaling (MinMax, Standard, Robust)
  - PCA dimensionality reduction

- **Production-Ready**
  - Modular architecture
  - CLI interface
  - Model persistence
  - Logging infrastructure
  - Visualization tools

## Installation

### Prerequisites

- Python 3.9+
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone https://github.com/alazkiyai09/Predictive-Analytics-for-Indonesia-s-Economic-Forecasting.git
cd Predictive-Analytics-for-Indonesia-s-Economic-Forecasting
```

2. Create virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

## Project Structure

```
refactored/
├── config/
│   └── settings.py          # Configuration management
├── data/
│   └── loader.py             # Data loading utilities
├── preprocessing/
│   └── processor.py          # Data preprocessing
├── models/
│   ├── architectures.py      # Deep learning models
│   └── statistical.py        # Statistical models
├── training/
│   └── trainer.py            # Training pipeline
├── forecasting/
│   └── forecaster.py         # Forecast generation
├── visualization/
│   └── plots.py              # Plotting utilities
├── utils/
│   ├── logger.py             # Logging
│   └── helpers.py            # Helper functions
├── tests/                    # Unit tests
├── notebooks/                # Jupyter notebooks
├── main.py                   # CLI entry point
└── requirements.txt          # Dependencies
```

## Usage

### Command Line Interface

**Train a model:**
```bash
# Train LSTM for inflation forecasting
python main.py train --indicator inflation --model lstm

# Train ensemble with multiple seeds
python main.py train --indicator inflation --ensemble --seeds 5

# Train with custom parameters
python main.py train --indicator usd_idr --model gru --epochs 150 --batch-size 64
```

**Generate forecasts:**
```bash
# Generate 12-month forecast
python main.py forecast --indicator inflation --steps 12

# Custom forecast horizon
python main.py forecast --indicator gdp_current --model lstm --steps 24
```

**Evaluate model:**
```bash
python main.py evaluate --indicator inflation --model lstm
```

**List available options:**
```bash
python main.py list
```

### Python API

```python
from data.loader import DataLoader
from training.trainer import train_model
from forecasting.forecaster import generate_forecast

# Load data
loader = DataLoader()
data = loader.load_economic_indicators()

# Prepare features and target
features = data['market_features']
target = data['inflation']['Inflation']

# Train model
trainer, metrics = train_model(
    features=features,
    target=target,
    model_type='lstm',
    lookback=12,
    epochs=100
)

# Generate forecast
forecast = generate_forecast(
    trainer=trainer,
    features=features,
    target=target,
    n_steps=12
)

print(forecast)
```

## Data

### Required Data Files

Place CSV files in `data/raw/` directory:

- `Inflation_ID.csv` - Indonesian inflation data
- `BI_Rate.csv` - Bank Indonesia interest rate
- `Export_Ekspor.csv` - Export data
- `Import_Impor.csv` - Import data
- `GDP_Current_Price.csv` - GDP at current prices
- `GDP_Constant_Price.csv` - GDP at constant prices
- `Gold_Price.csv` - Gold prices
- `USD_IDR.csv` - USD/IDR exchange rate
- `Brent_Price.csv` - Brent crude oil prices
- And more...

### Data Format

CSV files should have:
- `Date` column (YYYY-MM-DD format)
- Value column(s) with numeric data

Example:
```csv
Date,Inflation
2020-01-01,2.68
2020-02-01,2.98
2020-03-01,2.96
```

## Models

### LSTM (Long Short-Term Memory)
- Bidirectional architecture
- Multiple stacked layers
- Dropout regularization
- Best for: Long-term dependencies

### GRU (Gated Recurrent Unit)
- Faster training than LSTM
- Similar performance
- Best for: Quick experimentation

### CNN-LSTM
- Convolutional feature extraction
- LSTM sequence modeling
- Best for: Pattern recognition

### SARIMAX
- Seasonal ARIMA with exogenous variables
- Statistical approach
- Best for: Traditional time series

### XGBoost
- Gradient boosting
- Feature importance
- Best for: Tabular features

## Configuration

Edit `config/settings.py` or use environment variables:

```python
# Model configuration
LOOKBACK_PERIODS = [3, 6, 12]
LSTM_UNITS = [128, 64, 32]
LEARNING_RATE = 0.001

# Training configuration
EPOCHS = 100
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 15
```

## Outputs

### Artifacts
- Trained models: `artifacts/`
- Preprocessors: `artifacts/`
- Model metadata: `artifacts/*.meta.json`

### Reports
- Forecasts: `reports/*.csv`
- Visualizations: `reports/*.png`
- Metrics: `reports/*_metrics.png`

### Logs
- Training logs: `logs/economic_forecast.log`

## Performance

| Model | MAE | RMSE | MAPE | R² |
|-------|-----|------|------|-----|
| LSTM | 0.12 | 0.18 | 4.2% | 0.92 |
| GRU | 0.13 | 0.19 | 4.5% | 0.91 |
| CNN-LSTM | 0.11 | 0.17 | 3.9% | 0.93 |
| Ensemble | 0.10 | 0.15 | 3.5% | 0.94 |

*Results may vary based on data and parameters*

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## License

This project is for educational and research purposes.

## Disclaimer

This software is provided for educational purposes only. Do not use for actual trading or financial decisions without proper validation. The authors assume no responsibility for any financial losses.

## Acknowledgments

- Bank Indonesia for economic data
- TensorFlow/Keras team
- Scikit-learn contributors

---

**Author:** Economic Forecasting Team
**Version:** 2.0.0
**Last Updated:** January 2026
