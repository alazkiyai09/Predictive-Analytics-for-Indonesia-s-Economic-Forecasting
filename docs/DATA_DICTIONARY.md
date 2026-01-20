# Data Dictionary

Complete documentation of all data sources, variables, and formats used in the Indonesia Economic Forecasting System.

---

## Table of Contents

- [Overview](#overview)
- [Data Sources](#data-sources)
- [Economic Indicators](#economic-indicators)
- [Market Data](#market-data)
- [Money Supply Data](#money-supply-data)
- [Data Format Standards](#data-format-standards)
- [Data Quality](#data-quality)
- [Data Transformations](#data-transformations)

---

## Overview

The forecasting system uses 22+ economic and financial indicators from multiple sources, covering:

| Category | Count | Frequency | Coverage |
|----------|-------|-----------|----------|
| Economic Indicators | 7 | Monthly/Quarterly | 2010-2025 |
| Forex & Currencies | 6 | Daily | 2010-2025 |
| Commodities | 3 | Daily | 2010-2025 |
| Money Supply | 5 | Monthly | 2010-2025 |
| Market Indices | 1 | Daily | 2010-2025 |

---

## Data Sources

### Primary Sources

| Source | Data Types | Website |
|--------|------------|---------|
| Bank Indonesia | Inflation, BI Rate, Money Supply | [bi.go.id](https://www.bi.go.id) |
| BPS Statistics Indonesia | GDP, Trade Data | [bps.go.id](https://www.bps.go.id) |
| Yahoo Finance | Forex, Commodities, Indices | [finance.yahoo.com](https://finance.yahoo.com) |
| FRED | US Economic Data | [fred.stlouisfed.org](https://fred.stlouisfed.org) |
| ECB | European Economic Data | [ecb.europa.eu](https://www.ecb.europa.eu) |

### Data Collection

- **Historical Range**: 2010-01-01 to present
- **Update Frequency**: Monthly (economic), Daily (market)
- **Format**: CSV files with standardized column names

---

## Economic Indicators

### Inflation (Inflation_ID.csv)

Consumer Price Index (CPI) based inflation rate for Indonesia.

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `Date` | datetime | YYYY-MM-DD | End of month date |
| `Inflation` | float | Percentage (%) | Year-over-year inflation rate |

**Sample Data:**
```csv
Date,Inflation
2020-01-01,2.68
2020-02-01,2.98
2020-03-01,2.96
```

**Statistics:**
- Range: -0.5% to 12%
- Mean: ~4%
- Typical volatility: ±2%

**Notes:**
- Published by Bank Indonesia monthly
- Headline inflation (includes food and energy)
- Base year changes periodically

---

### Bank Indonesia Rate (BI_Rate.csv)

Central bank policy interest rate.

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `Date` | datetime | YYYY-MM-DD | Effective date |
| `BI_Rate` | float | Percentage (%) | Policy rate |

**Sample Data:**
```csv
Date,BI_Rate
2020-01-01,5.00
2020-02-01,4.75
2020-03-01,4.50
```

**Statistics:**
- Range: 3.5% to 7.75%
- Changes: Typically 0.25% increments

**Notes:**
- Changed from BI Rate to BI 7-Day Reverse Repo Rate in 2016
- Key monetary policy tool

---

### Exports (Export_Ekspor.csv)

Total Indonesian exports in USD.

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `Date` | datetime | YYYY-MM-DD | Month end |
| `Export` | float | Million USD | Total export value |

**Sample Data:**
```csv
Date,Export
2020-01-01,13620.5
2020-02-01,12480.3
2020-03-01,14050.8
```

**Major Components:**
- Palm oil
- Coal
- Natural gas
- Manufactured goods

---

### Imports (Import_Impor.csv)

Total Indonesian imports in USD.

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `Date` | datetime | YYYY-MM-DD | Month end |
| `Import` | float | Million USD | Total import value |

**Sample Data:**
```csv
Date,Import
2020-01-01,14250.2
2020-02-01,12890.5
2020-03-01,13420.1
```

**Major Components:**
- Machinery
- Oil & gas
- Electronics
- Raw materials

---

### GDP Current Price (GDP_Current_Price.csv)

Gross Domestic Product at current market prices.

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `Date` | datetime | YYYY-MM-DD | Quarter end |
| `GDP_Current` | float | Trillion IDR | Nominal GDP |

**Sample Data:**
```csv
Date,GDP_Current
2020-03-31,3980.5
2020-06-30,3680.2
2020-09-30,3950.8
2020-12-31,4150.3
```

**Notes:**
- Quarterly data (requires interpolation to monthly)
- Includes inflation effects
- BPS releases ~2 months after quarter end

---

### GDP Constant Price (GDP_Constant_Price.csv)

Gross Domestic Product at constant 2010 prices.

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `Date` | datetime | YYYY-MM-DD | Quarter end |
| `GDP_Constant` | float | Trillion IDR | Real GDP |

**Sample Data:**
```csv
Date,GDP_Constant
2020-03-31,2680.5
2020-06-30,2450.2
2020-09-30,2620.8
2020-12-31,2750.3
```

**Notes:**
- Removes inflation effects
- Better for growth comparison
- Base year: 2010

---

### Outstanding Bonds (Outstanding_bond.csv)

Indonesian government bonds outstanding.

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `Date` | datetime | YYYY-MM-DD | Month end |
| `Outstanding` | float | Trillion IDR | Total bonds |

**Sample Data:**
```csv
Date,Outstanding
2020-01-01,3250.5
2020-02-01,3280.2
2020-03-01,3350.8
```

---

## Market Data

### USD/IDR Exchange Rate (USD_IDR.csv)

US Dollar to Indonesian Rupiah exchange rate.

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `Date` | datetime | YYYY-MM-DD | Trading date |
| `Open` | float | IDR | Opening rate |
| `High` | float | IDR | Daily high |
| `Low` | float | IDR | Daily low |
| `Close` | float | IDR | Closing rate |
| `Adj Close` | float | IDR | Adjusted close |
| `Volume` | int | - | Trading volume |

**Sample Data:**
```csv
Date,Open,High,Low,Close,Adj Close,Volume
2020-01-02,13880,13920,13850,13900,13900,0
2020-01-03,13900,13950,13880,13925,13925,0
```

**Statistics:**
- Range: 8,500 - 16,500 IDR
- Typical daily change: ±0.5%
- High volatility during crises

---

### USD/JPY Exchange Rate (USD_JPY.csv)

US Dollar to Japanese Yen exchange rate.

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `Date` | datetime | YYYY-MM-DD | Trading date |
| `Close` | float | JPY | Closing rate |

**Relevance:** Japan is major trading partner; JPY movements affect regional currencies.

---

### EUR/USD Exchange Rate (EUR_USD.csv)

Euro to US Dollar exchange rate.

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `Date` | datetime | YYYY-MM-DD | Trading date |
| `Close` | float | USD | Closing rate |

**Relevance:** Global risk sentiment indicator; affects emerging market currencies.

---

### GBP/USD Exchange Rate (GBP_USD.csv)

British Pound to US Dollar exchange rate.

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `Date` | datetime | YYYY-MM-DD | Trading date |
| `Close` | float | USD | Closing rate |

---

### DXY Index (DXY.csv)

US Dollar Index - weighted average of USD against major currencies.

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `Date` | datetime | YYYY-MM-DD | Trading date |
| `Close` | float | Index | DXY value |

**Composition:**
- EUR: 57.6%
- JPY: 13.6%
- GBP: 11.9%
- CAD: 9.1%
- SEK: 4.2%
- CHF: 3.6%

**Relevance:** Strong inverse correlation with emerging market currencies including IDR.

---

### Gold Price (Gold_Price.csv)

Gold spot price in USD per troy ounce.

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `Date` | datetime | YYYY-MM-DD | Trading date |
| `Close` | float | USD/oz | Closing price |

**Sample Data:**
```csv
Date,Close
2020-01-02,1528.50
2020-01-03,1545.75
```

**Relevance:**
- Safe haven asset
- Inverse correlation with USD
- Inflation hedge

---

### Brent Crude Oil (Brent_Price.csv)

Brent crude oil futures price.

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `Date` | datetime | YYYY-MM-DD | Trading date |
| `Close` | float | USD/barrel | Closing price |

**Sample Data:**
```csv
Date,Close
2020-01-02,66.25
2020-01-03,68.50
```

**Relevance:**
- Indonesia is net oil importer
- Affects trade balance
- Impacts transportation costs and inflation

---

### WTI Crude Oil (WTI_Price.csv)

West Texas Intermediate crude oil futures price.

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `Date` | datetime | YYYY-MM-DD | Trading date |
| `Close` | float | USD/barrel | Closing price |

**Relevance:** Global oil benchmark, highly correlated with Brent.

---

### IDX Composite (IDX.csv)

Indonesia Stock Exchange Composite Index (IHSG).

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `Date` | datetime | YYYY-MM-DD | Trading date |
| `Open` | float | Points | Opening value |
| `High` | float | Points | Daily high |
| `Low` | float | Points | Daily low |
| `Close` | float | Points | Closing value |
| `Volume` | int | Shares | Trading volume |

**Sample Data:**
```csv
Date,Open,High,Low,Close,Volume
2020-01-02,6280,6320,6250,6300,8500000000
```

**Relevance:**
- Leading economic indicator
- Reflects investor sentiment
- Correlated with foreign capital flows

---

### Bond Spread (Spread_bond.csv)

Spread between Indonesian government bonds and US Treasuries.

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `Date` | datetime | YYYY-MM-DD | Date |
| `Spread` | float | Basis points | Yield spread |

**Relevance:**
- Measure of country risk
- Affects capital flows
- Higher spread = higher perceived risk

---

## Money Supply Data

### Indonesia M1 & M2 (M1_M2_ID.csv)

Indonesian money supply measures.

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `Date` | datetime | YYYY-MM-DD | Month end |
| `M1` | float | Trillion IDR | Narrow money (currency + demand deposits) |
| `M2` | float | Trillion IDR | Broad money (M1 + savings + time deposits) |

**Sample Data:**
```csv
Date,M1,M2
2020-01-01,1580.5,6250.8
2020-02-01,1592.3,6320.5
```

**Relevance:**
- Monetary policy indicator
- Inflation predictor
- Economic activity gauge

---

### US M1 & M2 (M1_M2_US.csv)

United States money supply.

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `Date` | datetime | YYYY-MM-DD | Month end |
| `M1` | float | Billion USD | Narrow money |
| `M2` | float | Billion USD | Broad money |

**Relevance:** US monetary policy affects global capital flows and emerging markets.

---

### EU M1 & M2 (M1_M2_EU.csv)

European Union money supply.

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `Date` | datetime | YYYY-MM-DD | Month end |
| `M1` | float | Billion EUR | Narrow money |
| `M2` | float | Billion EUR | Broad money |

---

### Japan M1 & M2 (M1_M2_JP.csv)

Japanese money supply.

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `Date` | datetime | YYYY-MM-DD | Month end |
| `M1` | float | Trillion JPY | Narrow money |
| `M2` | float | Trillion JPY | Broad money |

---

### UK M2 (M2_UK.csv)

United Kingdom broad money supply.

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `Date` | datetime | YYYY-MM-DD | Month end |
| `M2` | float | Billion GBP | Broad money |

---

## Data Format Standards

### Date Format

All dates should follow ISO 8601 format:

```
YYYY-MM-DD

Examples:
2020-01-01
2020-12-31
```

### Numeric Format

- Decimal separator: Period (.)
- Thousands separator: None
- Precision: 2-4 decimal places

```csv
# Correct
14523.50
0.0425

# Incorrect
14,523.50
14523,50
```

### File Encoding

- Encoding: UTF-8
- Line endings: LF or CRLF
- No BOM (Byte Order Mark)

### Column Naming

- Use descriptive names
- No spaces (use underscore)
- Consistent capitalization

```csv
# Correct
Date,Close_Price,Volume

# Incorrect
date,Close Price,vol
```

---

## Data Quality

### Quality Checks

Before using data, verify:

| Check | Method | Action |
|-------|--------|--------|
| Missing values | `df.isnull().sum()` | Interpolate or forward fill |
| Duplicates | `df.duplicated().sum()` | Remove duplicates |
| Date gaps | Check date sequence | Interpolate missing dates |
| Outliers | Z-score > 3 | Investigate or cap |
| Type errors | `df.dtypes` | Convert to correct type |

### Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Missing weekends | Gaps every 5 days | Use business day frequency |
| Holiday gaps | Irregular missing dates | Forward fill |
| Currency format | Commas in numbers | Remove commas before parsing |
| Date format | Mixed formats | Standardize to YYYY-MM-DD |

### Quality Validation Script

```python
def validate_data(df, date_col='Date'):
    """Validate data quality."""
    issues = []

    # Check missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        issues.append(f"Missing values: {missing[missing > 0].to_dict()}")

    # Check date continuity
    df[date_col] = pd.to_datetime(df[date_col])
    date_diff = df[date_col].diff().dt.days
    gaps = date_diff[date_diff > 35]  # More than 1 month gap
    if len(gaps) > 0:
        issues.append(f"Date gaps found: {len(gaps)} gaps")

    # Check for duplicates
    dups = df.duplicated(subset=[date_col]).sum()
    if dups > 0:
        issues.append(f"Duplicate dates: {dups}")

    return issues
```

---

## Data Transformations

### Daily to Monthly Resampling

```python
def daily_to_monthly(df, date_col='Date', method='last'):
    """
    Resample daily data to monthly.

    Parameters:
    -----------
    df : DataFrame
    date_col : str
    method : str
        'last' - Last value of month
        'mean' - Average of month
        'sum' - Sum of month

    Returns:
    --------
    Monthly DataFrame
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)

    if method == 'last':
        monthly = df.resample('M').last()
    elif method == 'mean':
        monthly = df.resample('M').mean()
    elif method == 'sum':
        monthly = df.resample('M').sum()

    return monthly.reset_index()
```

### Quarterly to Monthly Interpolation

```python
def quarterly_to_monthly(df, date_col='Date'):
    """
    Interpolate quarterly data to monthly.

    Uses linear interpolation between quarterly values.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)

    # Resample to monthly frequency
    monthly = df.resample('M').asfreq()

    # Linear interpolation
    monthly = monthly.interpolate(method='linear')

    return monthly.reset_index()
```

### Log Returns

```python
def calculate_log_returns(df, price_col='Close'):
    """
    Calculate log returns.

    log_return = ln(P_t / P_{t-1})
    """
    df = df.copy()
    df['log_return'] = np.log(df[price_col] / df[price_col].shift(1))
    return df
```

### Percentage Change

```python
def calculate_pct_change(df, col, periods=1):
    """
    Calculate percentage change.

    pct_change = (P_t - P_{t-n}) / P_{t-n} * 100
    """
    df = df.copy()
    df[f'{col}_pct_{periods}'] = df[col].pct_change(periods) * 100
    return df
```

---

## Variable Correlations

### Expected Correlations with USD/IDR

| Variable | Expected Correlation | Rationale |
|----------|---------------------|-----------|
| DXY | Positive (+) | Strong USD strengthens vs IDR |
| Gold | Negative (-) | Safe haven, inverse to USD |
| Brent Oil | Negative (-) | Indonesia net importer |
| BI Rate | Mixed | Higher rates attract capital |
| Inflation | Positive (+) | Inflation weakens currency |
| IDX | Negative (-) | Foreign capital flows |
| US M2 | Positive (+) | USD liquidity affects EM |

### Actual Correlations (Sample)

```
USD_IDR Correlations:
├── DXY:        +0.72
├── Gold:       -0.45
├── Brent:      -0.38
├── BI_Rate:    +0.15
├── Inflation:  +0.52
├── IDX:        -0.61
└── US_M2:      +0.58
```

---

## Data Update Procedure

### Manual Update

1. Download latest data from sources
2. Append to existing CSV files
3. Run validation checks
4. Update documentation if schema changes

### Automated Update (Future)

```python
# Example automation script
def update_all_data():
    """Update all data sources."""

    # Update market data (Yahoo Finance)
    update_yahoo_data(['USD_IDR', 'Gold', 'Brent'])

    # Update economic data (manual for now)
    print("Please update economic data manually from bi.go.id and bps.go.id")

    # Validate
    validate_all_data()
```

---

**Version:** 2.0.0 | **Last Updated:** January 2026
