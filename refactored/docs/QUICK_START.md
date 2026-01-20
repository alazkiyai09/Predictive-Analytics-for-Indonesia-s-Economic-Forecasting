# Quick Start Guide

Get the Indonesia Economic Forecasting System running in under 10 minutes.

---

## Prerequisites

Before you begin, ensure you have:

- [ ] Python 3.9+ installed ([Download](https://www.python.org/downloads/))
- [ ] Git installed ([Download](https://git-scm.com/downloads))
- [ ] Basic command line knowledge

---

## Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/alazkiyai09/Predictive-Analytics-for-Indonesia-s-Economic-Forecasting.git

# Navigate to the refactored directory
cd Predictive-Analytics-for-Indonesia-s-Economic-Forecasting/refactored
```

---

## Step 2: Set Up Python Environment

### Option A: Using venv (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate - Windows
venv\Scripts\activate

# Activate - Linux/Mac
source venv/bin/activate
```

### Option B: Using conda

```bash
conda create -n econ-forecast python=3.10 -y
conda activate econ-forecast
```

---

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- TensorFlow (deep learning)
- Scikit-learn (machine learning)
- Pandas, NumPy (data processing)
- Matplotlib, Seaborn (visualization)
- XGBoost, Statsmodels (additional models)

---

## Step 4: Verify Installation

```bash
# Check CLI is working
python main.py --help
```

Expected output:
```
usage: main.py [-h] {train,forecast,evaluate,list} ...

Indonesia Economic Forecasting System
...
```

```bash
# List available options
python main.py list
```

---

## Step 5: Prepare Your Data

### Option A: Use Sample Data

Create sample data files in `data/raw/`:

```bash
# Create data directory
mkdir -p data/raw
```

Create `data/raw/Inflation_ID.csv`:
```csv
Date,Inflation
2015-01-01,6.96
2015-02-01,6.29
2015-03-01,6.38
2015-04-01,6.79
2015-05-01,7.15
2015-06-01,7.26
2015-07-01,7.26
2015-08-01,7.18
2015-09-01,6.83
2015-10-01,6.25
2015-11-01,4.89
2015-12-01,3.35
2016-01-01,4.14
2016-02-01,4.42
2016-03-01,4.45
```

### Option B: Use Your Own Data

Place your CSV files in `data/raw/` with format:
- Column 1: `Date` (YYYY-MM-DD)
- Column 2+: Numeric values

---

## Step 6: Train Your First Model

### Quick Training (Demo)

```bash
# Train LSTM with minimal epochs for testing
python main.py train \
    --indicator inflation \
    --model lstm \
    --epochs 10 \
    --lookback 6
```

### Full Training

```bash
# Train with recommended settings
python main.py train \
    --indicator inflation \
    --model lstm \
    --epochs 100 \
    --lookback 12
```

Expected output:
```
[2026-01-20 10:00:00] [training.trainer] [INFO] - Training lstm model for inflation...
[2026-01-20 10:00:01] [training.trainer] [INFO] - Data loaded: 120 samples, 15 features
[2026-01-20 10:00:02] [training.trainer] [INFO] - Built LSTM model: 125,441 parameters
Epoch 1/100
...
[2026-01-20 10:05:00] [training.trainer] [INFO] - Model metrics: {'mae': 0.12, 'rmse': 0.18, 'mape': 4.2, 'r2': 0.92}
[2026-01-20 10:05:01] [training.trainer] [INFO] - Model saved to artifacts/inflation_lstm
```

---

## Step 7: Generate Forecast

```bash
# Generate 12-month forecast
python main.py forecast \
    --indicator inflation \
    --model lstm \
    --steps 12
```

Expected output:
```
Forecast Results:
==================================================
        Date   forecast  forecast_lower  forecast_upper
0 2026-02-28       3.25            2.95            3.55
1 2026-03-31       3.42            3.02            3.82
2 2026-04-30       3.38            2.88            3.88
...
```

---

## Step 8: View Results

### Forecast CSV
```bash
# View saved forecast
cat reports/inflation_forecast.csv
```

### Visualization
Check `reports/` directory for:
- `inflation_forecast.png` - Forecast plot
- `inflation_lstm_metrics.png` - Model performance

---

## Quick Reference

### Essential Commands

| Task | Command |
|------|---------|
| Train LSTM | `python main.py train -i inflation -m lstm` |
| Train GRU | `python main.py train -i inflation -m gru` |
| Train Ensemble | `python main.py train -i inflation --ensemble` |
| Forecast 12 months | `python main.py forecast -i inflation -s 12` |
| Evaluate model | `python main.py evaluate -i inflation -m lstm` |
| List options | `python main.py list` |

### Available Indicators

| Indicator | Description |
|-----------|-------------|
| `inflation` | Consumer Price Index inflation |
| `interest_rate` | Bank Indonesia rate |
| `exports` | Export values |
| `imports` | Import values |
| `gdp_current` | Nominal GDP |
| `gdp_constant` | Real GDP |

### Available Models

| Model | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| `lstm` | Medium | High | Long sequences |
| `gru` | Fast | High | Quick experiments |
| `cnn_lstm` | Medium | Highest | Pattern detection |
| `xgboost` | Very Fast | Medium | Feature analysis |
| `sarimax` | Fast | Medium | Seasonal data |

---

## Common Issues

### Issue: ModuleNotFoundError

```bash
# Solution: Ensure virtual environment is active
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### Issue: No data found

```bash
# Solution: Check data directory
ls data/raw/

# Create if missing
mkdir -p data/raw
```

### Issue: Out of memory

```bash
# Solution: Reduce batch size
python main.py train --batch-size 16
```

### Issue: Training too slow

```bash
# Solution: Reduce epochs for testing
python main.py train --epochs 10
```

---

## Next Steps

1. **Explore the API**: See [API_REFERENCE.md](API_REFERENCE.md)
2. **Understand Models**: Read [MODEL_GUIDE.md](MODEL_GUIDE.md)
3. **Customize Training**: Edit `config/settings.py`
4. **Add Your Data**: Place CSV files in `data/raw/`
5. **Build Dashboards**: Use `visualization/plots.py`

---

## Getting Help

- **Documentation**: Check `docs/` folder
- **Logs**: Review `logs/economic_forecast.log`
- **Issues**: Open a GitHub issue

---

**Congratulations!** You've successfully set up the Indonesia Economic Forecasting System.

Now you can:
- Train different models
- Generate forecasts
- Visualize results
- Customize for your needs
