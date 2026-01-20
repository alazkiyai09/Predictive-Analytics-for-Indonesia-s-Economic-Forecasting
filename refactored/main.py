#!/usr/bin/env python
"""
Indonesia Economic Forecasting System
Main entry point and CLI
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    path_config, data_config, model_config, training_config, forecast_config
)
from data.loader import DataLoader, merge_datasets, resample_to_monthly
from preprocessing.processor import (
    DataPreprocessor, prepare_train_test_split, handle_missing_values
)
from models.architectures import ModelFactory
from training.trainer import ModelTrainer, train_model, train_ensemble
from forecasting.forecaster import (
    EconomicForecaster, generate_forecast, ensemble_forecast
)
from visualization.plots import (
    plot_forecast, plot_comparison, plot_metrics, create_dashboard
)
from utils.logger import get_logger

logger = get_logger(__name__)


def load_data(indicator: str = "inflation") -> tuple:
    """
    Load and prepare data for forecasting

    Args:
        indicator: Economic indicator to forecast

    Returns:
        Tuple of (features_df, target_series)
    """
    logger.info(f"Loading data for {indicator}...")

    loader = DataLoader()

    # Load all data
    economic_data = loader.load_economic_indicators()
    market_data = loader.load_market_data()
    money_supply = loader.load_money_supply()

    # Get target data
    target_df = economic_data.get(indicator)
    if target_df is None:
        raise ValueError(f"Indicator '{indicator}' not found in data")

    # Prepare target
    value_cols = [c for c in target_df.columns if c not in ['Date', 'date']]
    target_col = value_cols[0] if value_cols else target_df.columns[1]

    target = target_df[target_col].copy()

    # Merge features
    all_data = {}
    all_data.update(economic_data)
    all_data.update(market_data)
    all_data.update(money_supply)

    # Remove target from features
    if indicator in all_data:
        del all_data[indicator]

    # Merge and align
    features = merge_datasets(all_data)

    # Handle missing values
    features = handle_missing_values(features, strategy="ffill")

    # Align target with features
    if 'Date' in features.columns and 'Date' in target_df.columns:
        merged = features.merge(
            target_df[['Date', target_col]],
            on='Date',
            how='inner'
        )
        features = merged.drop(columns=[target_col])
        target = merged[target_col]

    logger.info(f"Data loaded: {features.shape[0]} samples, {features.shape[1]} features")
    return features, target


def train_command(args):
    """Handle train command"""
    logger.info(f"Training {args.model} model for {args.indicator}...")

    # Load data
    features, target = load_data(args.indicator)

    # Remove date column for training
    feature_cols = [c for c in features.columns if c != 'Date']
    X = features[feature_cols]
    y = target

    if args.ensemble:
        # Train ensemble
        trainers, metrics = train_ensemble(
            features=X,
            target=y,
            model_types=['lstm', 'gru', 'cnn_lstm'],
            lookback=args.lookback,
            n_seeds=args.seeds,
            test_ratio=args.test_ratio,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        logger.info(f"Ensemble metrics: {metrics}")

        # Save ensemble models
        for i, trainer in enumerate(trainers):
            model_path = path_config.ARTIFACTS_DIR / f"{args.indicator}_ensemble_{i}"
            trainer.save(model_path)

    else:
        # Train single model
        trainer, metrics = train_model(
            features=X,
            target=y,
            model_type=args.model,
            lookback=args.lookback,
            test_ratio=args.test_ratio,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        logger.info(f"Model metrics: {metrics}")

        # Save model
        model_path = path_config.ARTIFACTS_DIR / f"{args.indicator}_{args.model}"
        trainer.save(model_path)

        # Plot metrics
        fig = plot_metrics(
            metrics,
            title=f"{args.indicator.upper()} - {args.model.upper()} Performance",
            save_path=path_config.REPORTS_DIR / f"{args.indicator}_{args.model}_metrics.png"
        )

    logger.info("Training complete!")


def forecast_command(args):
    """Handle forecast command"""
    logger.info(f"Generating {args.steps}-step forecast for {args.indicator}...")

    # Load data
    features, target = load_data(args.indicator)

    # Remove date column for prediction
    feature_cols = [c for c in features.columns if c != 'Date']
    X = features[feature_cols]
    y = target

    # Load model
    model_path = path_config.ARTIFACTS_DIR / f"{args.indicator}_{args.model}"

    if not model_path.with_suffix('.meta.json').exists():
        logger.error(f"Model not found at {model_path}")
        logger.error("Please train a model first: python main.py train")
        return

    trainer = ModelTrainer()
    trainer.load(model_path)

    # Get last date
    last_date = features['Date'].max() if 'Date' in features.columns else datetime.now()

    # Generate forecast
    forecast_df = generate_forecast(
        trainer=trainer,
        features=X,
        target=y,
        n_steps=args.steps,
        last_date=last_date
    )

    # Save forecast
    forecast_path = path_config.REPORTS_DIR / f"{args.indicator}_forecast.csv"
    forecast_df.to_csv(forecast_path, index=False)
    logger.info(f"Forecast saved to {forecast_path}")

    # Print forecast
    print("\nForecast Results:")
    print("=" * 50)
    print(forecast_df.to_string(index=False))

    # Create visualization
    if 'Date' in features.columns:
        historical_df = features[['Date']].copy()
        historical_df['actual'] = target.values

        fig = plot_forecast(
            historical=historical_df,
            forecast=forecast_df,
            title=f"{args.indicator.upper()} Forecast",
            save_path=path_config.REPORTS_DIR / f"{args.indicator}_forecast.png"
        )

    logger.info("Forecast complete!")


def evaluate_command(args):
    """Handle evaluate command"""
    logger.info(f"Evaluating model for {args.indicator}...")

    # Load data
    features, target = load_data(args.indicator)

    feature_cols = [c for c in features.columns if c != 'Date']
    X = features[feature_cols]
    y = target

    # Load model
    model_path = path_config.ARTIFACTS_DIR / f"{args.indicator}_{args.model}"

    if not model_path.with_suffix('.meta.json').exists():
        logger.error(f"Model not found at {model_path}")
        return

    trainer = ModelTrainer()
    trainer.load(model_path)

    # Prepare data
    scaled_X, scaled_y = trainer.preprocessor.transform(X, y)

    # Get test split
    from preprocessing.processor import create_sequences
    X_seq, y_seq = create_sequences(scaled_X, scaled_y, lookback=trainer.lookback)

    test_split = int(len(X_seq) * (1 - args.test_ratio))
    X_test = X_seq[test_split:]
    y_test = y_seq[test_split:]

    # Evaluate
    metrics = trainer.evaluate(X_test, y_test)

    print("\nEvaluation Results:")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"{metric.upper():>10}: {value:.4f}")

    # Plot comparison
    predictions = trainer.predict_inverse_scaled(X_test)
    y_actual = trainer.preprocessor.inverse_transform_target(y_test)

    import pandas as pd
    comparison_df = pd.DataFrame({
        'Date': range(len(predictions)),
        'actual': y_actual,
        'predicted': predictions
    })

    fig = plot_comparison(
        comparison_df,
        title=f"{args.indicator.upper()} - Actual vs Predicted",
        save_path=path_config.REPORTS_DIR / f"{args.indicator}_{args.model}_comparison.png"
    )

    logger.info("Evaluation complete!")


def list_command(args):
    """List available models and data"""
    print("\nAvailable Economic Indicators:")
    print("=" * 40)
    indicators = [
        "inflation", "interest_rate", "exports", "imports",
        "gdp_current", "gdp_constant", "outstanding_bond"
    ]
    for ind in indicators:
        print(f"  - {ind}")

    print("\nAvailable Models:")
    print("=" * 40)
    models = ['lstm', 'gru', 'cnn_lstm', 'ensemble', 'sarimax', 'xgboost']
    for model in models:
        print(f"  - {model}")

    print("\nTrained Models:")
    print("=" * 40)
    if path_config.ARTIFACTS_DIR.exists():
        meta_files = list(path_config.ARTIFACTS_DIR.glob("*.meta.json"))
        if meta_files:
            for f in meta_files:
                print(f"  - {f.stem.replace('.meta', '')}")
        else:
            print("  No trained models found")
    else:
        print("  Artifacts directory not found")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Indonesia Economic Forecasting System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train LSTM model for inflation:
    python main.py train --indicator inflation --model lstm

  Train ensemble for USD/IDR:
    python main.py train --indicator usd_idr --ensemble

  Generate 12-month forecast:
    python main.py forecast --indicator inflation --steps 12

  Evaluate model:
    python main.py evaluate --indicator inflation --model lstm

  List available options:
    python main.py list
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train forecasting model")
    train_parser.add_argument(
        "--indicator", "-i",
        type=str,
        default="inflation",
        help="Economic indicator to forecast"
    )
    train_parser.add_argument(
        "--model", "-m",
        type=str,
        default="lstm",
        choices=['lstm', 'gru', 'cnn_lstm', 'ensemble', 'sarimax', 'xgboost'],
        help="Model type"
    )
    train_parser.add_argument(
        "--lookback", "-l",
        type=int,
        default=12,
        help="Lookback period (months)"
    )
    train_parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    train_parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Batch size"
    )
    train_parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Test set ratio"
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    train_parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Train ensemble of models"
    )
    train_parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of seeds for ensemble"
    )
    train_parser.set_defaults(func=train_command)

    # Forecast command
    forecast_parser = subparsers.add_parser("forecast", help="Generate forecast")
    forecast_parser.add_argument(
        "--indicator", "-i",
        type=str,
        default="inflation",
        help="Economic indicator"
    )
    forecast_parser.add_argument(
        "--model", "-m",
        type=str,
        default="lstm",
        help="Model to use"
    )
    forecast_parser.add_argument(
        "--steps", "-s",
        type=int,
        default=12,
        help="Number of forecast steps"
    )
    forecast_parser.set_defaults(func=forecast_command)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model")
    eval_parser.add_argument(
        "--indicator", "-i",
        type=str,
        default="inflation",
        help="Economic indicator"
    )
    eval_parser.add_argument(
        "--model", "-m",
        type=str,
        default="lstm",
        help="Model to evaluate"
    )
    eval_parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Test set ratio"
    )
    eval_parser.set_defaults(func=evaluate_command)

    # List command
    list_parser = subparsers.add_parser("list", help="List available options")
    list_parser.set_defaults(func=list_command)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Ensure directories exist
    path_config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path_config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path_config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()
