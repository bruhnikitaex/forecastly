"""
Type definitions for Forecastly project.

Provides type hints and type aliases for better type safety.
"""

from typing import TypedDict, Union, Optional, Dict, List, Any
from pathlib import Path
import pandas as pd
from datetime import datetime


# ==============================================================================
# Path Types
# ==============================================================================

PathLike = Union[str, Path]


# ==============================================================================
# Data Types
# ==============================================================================

class ForecastRow(TypedDict):
    """Single forecast row structure."""
    date: str
    sku_id: str
    prophet: float
    xgb: float
    ensemble: float
    p_low: Optional[float]
    p_high: Optional[float]


class MetricRow(TypedDict):
    """Single metric row structure."""
    sku_id: str
    mape_prophet: float
    mape_xgboost: float
    mape_naive: float
    mape_ens: float
    best_model: str


class SKUData(TypedDict):
    """SKU data structure."""
    sku_id: str
    total_sales: float
    avg_daily_sales: float
    min_sales: float
    max_sales: float
    date_range_days: int


# ==============================================================================
# API Response Types
# ==============================================================================

class HealthCheckResponse(TypedDict):
    """Health check response structure."""
    status: str
    service: str
    version: str
    timestamp: str
    database_mode: bool
    database_connected: Optional[bool]


class SKUListResponse(TypedDict):
    """SKU list response structure."""
    skus: List[str]
    count: int


class PredictionResponse(TypedDict):
    """Prediction response structure."""
    sku_id: str
    horizon: int
    count: int
    source: str
    predictions: List[ForecastRow]


class MetricsResponse(TypedDict):
    """Metrics response structure."""
    count: int
    source: str
    metrics: List[MetricRow]


class ErrorResponse(TypedDict):
    """Error response structure."""
    code: str
    message: str
    details: Dict[str, Any]


class APIErrorResponse(TypedDict):
    """Full API error response."""
    error: ErrorResponse
    path: str
    timestamp: Optional[str]


# ==============================================================================
# Model Types
# ==============================================================================

ModelDict = Dict[str, Any]  # Generic model dictionary
ProphetModelDict = Dict[str, 'Prophet']  # SKU -> Prophet model
XGBoostModelDict = Dict[str, 'XGBRegressor']  # SKU -> XGBoost model


class ModelConfig(TypedDict, total=False):
    """Model configuration structure."""
    n_estimators: int
    learning_rate: float
    max_depth: int
    min_child_weight: int
    subsample: float
    colsample_bytree: float


class ProphetConfig(TypedDict, total=False):
    """Prophet model configuration."""
    changepoint_prior_scale: float
    seasonality_prior_scale: float
    holidays_prior_scale: float
    seasonality_mode: str
    daily_seasonality: bool
    weekly_seasonality: bool
    yearly_seasonality: bool


# ==============================================================================
# Database Types
# ==============================================================================

class UserDict(TypedDict):
    """User data structure."""
    id: int
    username: str
    full_name: str
    is_active: bool
    is_admin: bool
    created_at: str


class ForecastRunDict(TypedDict):
    """Forecast run data structure."""
    run_id: str
    horizon: int
    model_type: str
    status: str
    started_at: Optional[str]
    completed_at: Optional[str]
    records_count: Optional[int]
    error_message: Optional[str]


class DatabaseStats(TypedDict):
    """Database statistics structure."""
    total_skus: int
    total_predictions: int
    total_forecast_runs: int
    total_users: int


# ==============================================================================
# Configuration Types
# ==============================================================================

class PathsConfig(TypedDict):
    """Paths configuration structure."""
    data: Dict[str, str]


class ModelParamsConfig(TypedDict):
    """Model parameters configuration structure."""
    model: Dict[str, ModelConfig]


# ==============================================================================
# Utility Types
# ==============================================================================

JSONSerializable = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]
DateLike = Union[str, datetime, pd.Timestamp]


# ==============================================================================
# Export all types
# ==============================================================================

__all__ = [
    # Path types
    'PathLike',
    # Data types
    'ForecastRow',
    'MetricRow',
    'SKUData',
    # API types
    'HealthCheckResponse',
    'SKUListResponse',
    'PredictionResponse',
    'MetricsResponse',
    'ErrorResponse',
    'APIErrorResponse',
    # Model types
    'ModelDict',
    'ProphetModelDict',
    'XGBoostModelDict',
    'ModelConfig',
    'ProphetConfig',
    # Database types
    'UserDict',
    'ForecastRunDict',
    'DatabaseStats',
    # Config types
    'PathsConfig',
    'ModelParamsConfig',
    # Utility types
    'JSONSerializable',
    'DateLike',
]
