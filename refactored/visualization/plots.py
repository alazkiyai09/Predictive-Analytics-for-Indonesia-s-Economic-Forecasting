"""
Visualization functions for Indonesia Economic Forecasting
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from config.settings import path_config
from utils.logger import get_logger

logger = get_logger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_forecast(
    historical: pd.DataFrame,
    forecast: pd.DataFrame,
    date_col: str = "Date",
    actual_col: str = "actual",
    forecast_col: str = "forecast",
    lower_col: Optional[str] = "forecast_lower",
    upper_col: Optional[str] = "forecast_upper",
    title: str = "Economic Forecast",
    ylabel: str = "Value",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot forecast with confidence intervals

    Args:
        historical: Historical data DataFrame
        forecast: Forecast DataFrame
        date_col: Date column name
        actual_col: Actual values column
        forecast_col: Forecast values column
        lower_col: Lower bound column
        upper_col: Upper bound column
        title: Plot title
        ylabel: Y-axis label
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot historical
    if actual_col in historical.columns:
        ax.plot(
            historical[date_col],
            historical[actual_col],
            label='Historical',
            color='#2C3E50',
            linewidth=2
        )

    # Plot forecast
    ax.plot(
        forecast[date_col],
        forecast[forecast_col],
        label='Forecast',
        color='#E74C3C',
        linewidth=2,
        linestyle='--'
    )

    # Plot confidence interval
    if lower_col in forecast.columns and upper_col in forecast.columns:
        ax.fill_between(
            forecast[date_col],
            forecast[lower_col],
            forecast[upper_col],
            alpha=0.3,
            color='#E74C3C',
            label='95% CI'
        )

    # Formatting
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(loc='best')

    # Date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")

    return fig


def plot_comparison(
    df: pd.DataFrame,
    date_col: str = "Date",
    actual_col: str = "actual",
    predicted_col: str = "predicted",
    title: str = "Actual vs Predicted",
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot actual vs predicted comparison

    Args:
        df: DataFrame with actual and predicted values
        date_col: Date column name
        actual_col: Actual values column
        predicted_col: Predicted values column
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # Time series comparison
    ax1 = axes[0]
    ax1.plot(df[date_col], df[actual_col], label='Actual', color='#2C3E50', linewidth=2)
    ax1.plot(df[date_col], df[predicted_col], label='Predicted', color='#E74C3C', linewidth=2, alpha=0.8)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Scatter plot
    ax2 = axes[1]
    ax2.scatter(df[actual_col], df[predicted_col], alpha=0.5, color='#3498DB')

    # Perfect prediction line
    min_val = min(df[actual_col].min(), df[predicted_col].min())
    max_val = max(df[actual_col].max(), df[predicted_col].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax2.set_title('Scatter: Actual vs Predicted')
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")

    return fig


def plot_metrics(
    metrics: Dict[str, float],
    title: str = "Model Performance Metrics",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot performance metrics as bar chart

    Args:
        metrics: Dictionary of metric names and values
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    names = list(metrics.keys())
    values = list(metrics.values())

    colors = sns.color_palette("husl", len(names))
    bars = ax.bar(names, values, color=colors)

    # Add value labels
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01 * max(values),
            f'{value:.4f}',
            ha='center',
            va='bottom',
            fontsize=10
        )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Value')
    ax.set_ylim(0, max(values) * 1.15)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")

    return fig


def plot_feature_importance(
    importance: Dict[str, float],
    top_n: int = 20,
    title: str = "Feature Importance",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot feature importance

    Args:
        importance: Dictionary of feature importance scores
        top_n: Number of top features to show
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    # Sort and get top N
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]

    features = [f[0] for f in sorted_features]
    scores = [f[1] for f in sorted_features]

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(features))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(features)))

    ax.barh(y_pos, scores, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score')
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")

    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    title: str = "Correlation Matrix",
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot correlation matrix heatmap

    Args:
        df: DataFrame with numeric columns
        columns: Specific columns to include
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    if columns:
        df = df[columns]

    # Get numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()

    fig, ax = plt.subplots(figsize=figsize)

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdYlBu_r',
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={'shrink': 0.8}
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")

    return fig


def plot_historical_trend(
    df: pd.DataFrame,
    date_col: str = "Date",
    value_cols: List[str] = None,
    title: str = "Historical Trends",
    figsize: Tuple[int, int] = (14, 8),
    normalize: bool = False,
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot historical trends for multiple series

    Args:
        df: DataFrame with time series data
        date_col: Date column name
        value_cols: Columns to plot
        title: Plot title
        figsize: Figure size
        normalize: Whether to normalize values
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    if value_cols is None:
        value_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(value_cols)))

    for col, color in zip(value_cols, colors):
        if col in df.columns:
            values = df[col].values

            if normalize:
                values = (values - values.min()) / (values.max() - values.min() + 1e-8)

            ax.plot(df[date_col], values, label=col, color=color, linewidth=1.5)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value' + (' (Normalized)' if normalize else ''))
    ax.legend(loc='best', fontsize=8)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")

    return fig


def plot_residuals(
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str = "Residual Analysis",
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot residual analysis

    Args:
        actual: Actual values
        predicted: Predicted values
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    residuals = actual - predicted

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Residuals over time
    ax1 = axes[0, 0]
    ax1.plot(residuals, color='#3498DB', alpha=0.7)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_title('Residuals Over Time')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Residual')

    # Residual histogram
    ax2 = axes[0, 1]
    ax2.hist(residuals, bins=30, color='#3498DB', edgecolor='white', alpha=0.7)
    ax2.axvline(x=0, color='r', linestyle='--')
    ax2.set_title('Residual Distribution')
    ax2.set_xlabel('Residual')
    ax2.set_ylabel('Frequency')

    # Q-Q plot
    ax3 = axes[1, 0]
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot')

    # Residuals vs Predicted
    ax4 = axes[1, 1]
    ax4.scatter(predicted, residuals, alpha=0.5, color='#3498DB')
    ax4.axhline(y=0, color='r', linestyle='--')
    ax4.set_title('Residuals vs Predicted')
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('Residual')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")

    return fig


def create_dashboard(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    metrics: Dict[str, float],
    title: str = "Economic Forecast Dashboard",
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Create comprehensive forecast dashboard

    Args:
        historical_df: Historical data
        forecast_df: Forecast data
        metrics: Performance metrics
        title: Dashboard title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)

    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Main forecast plot (top spanning 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])

    if 'actual' in historical_df.columns:
        ax1.plot(
            historical_df['Date'],
            historical_df['actual'],
            label='Historical',
            color='#2C3E50',
            linewidth=2
        )

    ax1.plot(
        forecast_df['Date'],
        forecast_df['forecast'],
        label='Forecast',
        color='#E74C3C',
        linewidth=2,
        linestyle='--'
    )

    if 'forecast_lower' in forecast_df.columns:
        ax1.fill_between(
            forecast_df['Date'],
            forecast_df['forecast_lower'],
            forecast_df['forecast_upper'],
            alpha=0.3,
            color='#E74C3C'
        )

    ax1.set_title('Forecast', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Metrics (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    metric_names = list(metrics.keys())[:5]
    metric_values = [metrics[k] for k in metric_names]

    colors = sns.color_palette("husl", len(metric_names))
    bars = ax2.barh(metric_names, metric_values, color=colors)
    ax2.set_title('Metrics', fontsize=12, fontweight='bold')

    for bar, val in zip(bars, metric_values):
        ax2.text(val, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                 va='center', fontsize=9)

    # Historical trend (middle left)
    ax3 = fig.add_subplot(gs[1, :2])
    if 'actual' in historical_df.columns:
        ax3.plot(historical_df['Date'], historical_df['actual'], color='#3498DB')
    ax3.set_title('Historical Trend', fontsize=12, fontweight='bold')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    # Forecast values table (middle right)
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')

    table_data = forecast_df[['Date', 'forecast']].head(6).copy()
    table_data['Date'] = table_data['Date'].dt.strftime('%Y-%m')
    table_data['forecast'] = table_data['forecast'].round(2)

    table = ax4.table(
        cellText=table_data.values,
        colLabels=['Date', 'Forecast'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax4.set_title('Forecast Values', fontsize=12, fontweight='bold')

    # Summary statistics (bottom)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')

    summary_text = f"""
    Model Performance Summary
    ─────────────────────────
    MAE: {metrics.get('mae', 'N/A'):.4f}     RMSE: {metrics.get('rmse', 'N/A'):.4f}     MAPE: {metrics.get('mape', 'N/A'):.2f}%     R²: {metrics.get('r2', 'N/A'):.4f}

    Forecast Period: {forecast_df['Date'].min().strftime('%Y-%m')} to {forecast_df['Date'].max().strftime('%Y-%m')}
    Number of Forecast Points: {len(forecast_df)}
    """

    ax5.text(0.5, 0.5, summary_text, transform=ax5.transAxes,
             fontsize=11, verticalalignment='center', horizontalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Dashboard saved to {save_path}")

    return fig
