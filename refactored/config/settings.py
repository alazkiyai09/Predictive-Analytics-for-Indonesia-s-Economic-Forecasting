"""
Centralized Configuration for Indonesia Economic Forecasting System
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class DataConfig:
    """Data source configuration"""
    # Data directory
    DATA_DIR: Path = BASE_DIR / "data" / "raw"
    PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"

    # Economic indicators
    INFLATION_FILE: str = "Inflation_ID.csv"
    INTEREST_RATE_FILE: str = "Interest_Rate_ID.csv"
    EXPORTS_FILE: str = "Data_Export_ID.csv"
    IMPORTS_FILE: str = "Data_Import_ID.csv"
    GDP_CURRENT_FILE: str = "GDP_ID_Current_Price.csv"
    GDP_CONSTANT_FILE: str = "GDP_ID_Constant_Price(2010).csv"
    OUTSTANDING_BOND_FILE: str = "Data_Outstanding_Bond_ID.csv"

    # Market data
    GOLD_FILE: str = "Data_XAU_USD.csv"
    USD_IDR_FILE: str = "Data_USD_IDR.csv"
    USD_JPY_FILE: str = "Data_USD_JPY.csv"
    EUR_USD_FILE: str = "Data_EUR_USD.csv"
    GBP_USD_FILE: str = "Data_GBP_USD.csv"
    DXY_FILE: str = "Data_DXY.csv"
    IDX_FILE: str = "Data_IDX_Composite.csv"
    BRENT_FILE: str = "Data_Brent_USD.csv"
    WTI_FILE: str = "Data_WTI_USD.csv"
    SPREAD_BOND_FILE: str = "Data_Spread_Bond_ID.csv"

    # Money supply
    M1_M2_ID_FILE: str = "Data_M1&M2_ID.csv"
    M1_M2_US_FILE: str = "Data_M1&M2_US.csv"
    M1_M2_EU_FILE: str = "Data_M1&M2_EU.csv"
    M1_M2_JP_FILE: str = "Data_M1&M2_JP.csv"
    M2_UK_FILE: str = "Data_M2_UK.csv"

    # Date column name
    DATE_COLUMN: str = "Date"

    # Frequency conversion
    DAILY_TO_MONTHLY: bool = True
    QUARTERLY_TO_MONTHLY: bool = True


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration"""
    # Scaling
    SCALER_TYPE: str = "minmax"  # "minmax" or "standard"
    SCALE_RANGE: Tuple[int, int] = (0, 1)

    # Missing values
    FILL_METHOD: str = "ffill"  # "ffill", "bfill", "interpolate"

    # PCA
    PCA_ENABLED: bool = True
    PCA_VARIANCE_RATIO: float = 0.95  # Keep 95% of variance
    PCA_N_COMPONENTS: Optional[int] = None  # Or specify exact number

    # Stationarity
    DIFFERENCING_ORDER: int = 1
    ADF_SIGNIFICANCE: float = 0.05

    # Outliers
    OUTLIER_METHOD: str = "iqr"  # "iqr", "zscore"
    OUTLIER_THRESHOLD: float = 1.5  # IQR multiplier


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Sequence parameters
    LOOKBACK_PERIODS: List[int] = field(default_factory=lambda: [3, 6, 12])
    DEFAULT_LOOKBACK: int = 12  # months
    FORECAST_HORIZON: int = 12  # months ahead

    # LSTM configuration
    LSTM_UNITS: List[int] = field(default_factory=lambda: [128, 64, 32])
    LSTM_DROPOUT: float = 0.2
    LSTM_RECURRENT_DROPOUT: float = 0.2
    LSTM_BIDIRECTIONAL: bool = True
    LSTM_L2_REG: float = 0.001

    # GRU configuration
    GRU_UNITS: List[int] = field(default_factory=lambda: [128, 64, 32])
    GRU_DROPOUT: float = 0.2
    GRU_RECURRENT_DROPOUT: float = 0.2

    # CNN-LSTM configuration
    CNN_FILTERS: int = 64
    CNN_KERNEL_SIZE: int = 3
    CNN_POOL_SIZE: int = 2
    CNN_LSTM_UNITS: int = 64

    # SARIMAX configuration
    SARIMAX_ORDER: Tuple[int, int, int] = (1, 1, 1)
    SARIMAX_SEASONAL_ORDER: Tuple[int, int, int, int] = (1, 1, 1, 12)
    SARIMAX_AUTO: bool = True  # Use auto_arima

    # XGBoost configuration
    XGB_N_ESTIMATORS: int = 100
    XGB_MAX_DEPTH: int = 6
    XGB_LEARNING_RATE: float = 0.1
    XGB_SUBSAMPLE: float = 0.8

    # Ensemble configuration
    ENSEMBLE_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "lstm": 0.3,
        "gru": 0.2,
        "cnn_lstm": 0.2,
        "sarimax": 0.15,
        "xgboost": 0.15
    })


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Training parameters
    EPOCHS: int = 100
    BATCH_SIZE: int = 32
    VALIDATION_SPLIT: float = 0.2
    TEST_SPLIT: float = 0.1

    # Time series CV
    N_SPLITS: int = 5
    GAP: int = 1  # Embargo period

    # Callbacks
    EARLY_STOPPING_PATIENCE: int = 15
    EARLY_STOPPING_MIN_DELTA: float = 0.0001
    REDUCE_LR_PATIENCE: int = 7
    REDUCE_LR_FACTOR: float = 0.5
    REDUCE_LR_MIN: float = 1e-6

    # Optimizer
    LEARNING_RATE: float = 0.001
    OPTIMIZER: str = "adam"

    # Loss function
    LOSS: str = "huber"  # "mse", "mae", "huber"

    # Random seeds for reproducibility
    SEEDS: List[int] = field(default_factory=lambda: [42, 73, 101])

    # Multi-seed ensemble
    USE_MULTI_SEED: bool = True


@dataclass
class ForecastConfig:
    """Forecasting configuration"""
    # Forecast targets
    PRIMARY_TARGETS: List[str] = field(default_factory=lambda: [
        "USD_IDR",
        "Inflation"
    ])

    # Forecast horizons (months)
    SHORT_TERM: int = 3
    MEDIUM_TERM: int = 6
    LONG_TERM: int = 12

    # Confidence intervals
    CONFIDENCE_LEVELS: List[float] = field(default_factory=lambda: [0.80, 0.95])

    # Monte Carlo simulation
    MC_SIMULATIONS: int = 1000

    # Scenario analysis
    SCENARIOS: List[str] = field(default_factory=lambda: [
        "baseline",
        "optimistic",
        "pessimistic"
    ])


@dataclass
class VisualizationConfig:
    """Visualization configuration"""
    FIGURE_SIZE: Tuple[int, int] = (14, 7)
    STYLE: str = "seaborn-v0_8-whitegrid"
    DPI: int = 100
    SAVE_FIGURES: bool = True
    FIGURES_DIR: Path = BASE_DIR / "reports" / "figures"

    # Colors
    ACTUAL_COLOR: str = "#1f77b4"
    PREDICTED_COLOR: str = "#ff7f0e"
    CONFIDENCE_COLOR: str = "#2ca02c"


@dataclass
class PathConfig:
    """Path configuration"""
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = BASE_DIR / "data" / "raw"
    PROCESSED_DATA_DIR: Path = BASE_DIR / "data" / "processed"
    MODELS_DIR: Path = BASE_DIR / "models"
    ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
    REPORTS_DIR: Path = BASE_DIR / "reports"
    LOGS_DIR: Path = BASE_DIR / "logs"

    def create_directories(self):
        """Create all necessary directories"""
        for path in [
            self.DATA_DIR, self.RAW_DATA_DIR, self.PROCESSED_DATA_DIR,
            self.MODELS_DIR, self.ARTIFACTS_DIR, self.REPORTS_DIR, self.LOGS_DIR
        ]:
            path.mkdir(parents=True, exist_ok=True)


# Create default instances
data_config = DataConfig()
preprocessing_config = PreprocessingConfig()
model_config = ModelConfig()
training_config = TrainingConfig()
forecast_config = ForecastConfig()
visualization_config = VisualizationConfig()
path_config = PathConfig()

# Ensure directories exist
path_config.create_directories()
