"""
Validators for data quality checks.

Provides reusable validation functions.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime


def validate_schema(
    df: pd.DataFrame,
    required_columns: List[str],
    optional_columns: Optional[List[str]] = None
) -> Tuple[bool, List[str]]:
    """
    Validate DataFrame schema.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        optional_columns: List of optional column names

    Returns:
        Tuple of (is_valid, missing_columns)
    """
    missing = [col for col in required_columns if col not in df.columns]
    return len(missing) == 0, missing


def validate_date_range(
    df: pd.DataFrame,
    date_column: str = 'date',
    min_date: Optional[datetime] = None,
    max_date: Optional[datetime] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate date range.

    Args:
        df: DataFrame to validate
        date_column: Name of date column
        min_date: Minimum allowed date
        max_date: Maximum allowed date

    Returns:
        Tuple of (is_valid, details)
    """
    if date_column not in df.columns:
        return False, {'error': f'Column {date_column} not found'}

    try:
        dates = pd.to_datetime(df[date_column])
    except Exception as e:
        return False, {'error': f'Cannot parse dates: {str(e)}'}

    issues = {}

    if min_date:
        before_min = (dates < min_date).sum()
        if before_min > 0:
            issues['before_min'] = int(before_min)

    if max_date:
        after_max = (dates > max_date).sum()
        if after_max > 0:
            issues['after_max'] = int(after_max)

    # Check for future dates
    future = (dates > pd.Timestamp.now()).sum()
    if future > 0:
        issues['future_dates'] = int(future)

    return len(issues) == 0, issues


def validate_numeric_range(
    df: pd.DataFrame,
    column: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_negative: bool = False
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate numeric column range.

    Args:
        df: DataFrame to validate
        column: Column name
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        allow_negative: Whether negative values are allowed

    Returns:
        Tuple of (is_valid, details)
    """
    if column not in df.columns:
        return False, {'error': f'Column {column} not found'}

    if not pd.api.types.is_numeric_dtype(df[column]):
        return False, {'error': f'Column {column} is not numeric'}

    issues = {}

    if not allow_negative:
        negative_count = (df[column] < 0).sum()
        if negative_count > 0:
            issues['negative_values'] = int(negative_count)

    if min_value is not None:
        below_min = (df[column] < min_value).sum()
        if below_min > 0:
            issues['below_minimum'] = int(below_min)

    if max_value is not None:
        above_max = (df[column] > max_value).sum()
        if above_max > 0:
            issues['above_maximum'] = int(above_max)

    return len(issues) == 0, issues


def check_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None
) -> Tuple[int, pd.DataFrame]:
    """
    Check for duplicate rows.

    Args:
        df: DataFrame to check
        subset: Columns to consider for duplication check

    Returns:
        Tuple of (duplicate_count, duplicate_rows)
    """
    duplicates = df.duplicated(subset=subset, keep='first')
    duplicate_count = duplicates.sum()
    duplicate_rows = df[duplicates] if duplicate_count > 0 else pd.DataFrame()

    return int(duplicate_count), duplicate_rows


def check_missing_values(
    df: pd.DataFrame,
    threshold: float = 0.1
) -> Dict[str, Dict[str, Any]]:
    """
    Check for missing values in all columns.

    Args:
        df: DataFrame to check
        threshold: Threshold for missing value percentage (0.1 = 10%)

    Returns:
        Dictionary with missing value information per column
    """
    result = {}

    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            missing_pct = missing_count / len(df)
            result[col] = {
                'count': int(missing_count),
                'percentage': float(missing_pct),
                'exceeds_threshold': missing_pct > threshold
            }

    return result


def detect_outliers_zscore(
    df: pd.DataFrame,
    column: str,
    threshold: float = 3.0
) -> Tuple[int, pd.Series]:
    """
    Detect outliers using Z-score method.

    Args:
        df: DataFrame
        column: Column to check
        threshold: Z-score threshold (default: 3 standard deviations)

    Returns:
        Tuple of (outlier_count, outlier_mask)
    """
    if column not in df.columns:
        return 0, pd.Series([False] * len(df))

    values = df[column].dropna()
    if len(values) == 0:
        return 0, pd.Series([False] * len(df))

    mean = values.mean()
    std = values.std()

    if std == 0:
        return 0, pd.Series([False] * len(df))

    z_scores = np.abs((df[column] - mean) / std)
    outlier_mask = z_scores > threshold

    return int(outlier_mask.sum()), outlier_mask


def detect_outliers_iqr(
    df: pd.DataFrame,
    column: str,
    multiplier: float = 1.5
) -> Tuple[int, pd.Series]:
    """
    Detect outliers using IQR (Interquartile Range) method.

    Args:
        df: DataFrame
        column: Column to check
        multiplier: IQR multiplier (default: 1.5)

    Returns:
        Tuple of (outlier_count, outlier_mask)
    """
    if column not in df.columns:
        return 0, pd.Series([False] * len(df))

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)

    return int(outlier_mask.sum()), outlier_mask


def check_data_types(
    df: pd.DataFrame,
    expected_types: Dict[str, str]
) -> Dict[str, Dict[str, Any]]:
    """
    Check if columns have expected data types.

    Args:
        df: DataFrame to check
        expected_types: Dictionary mapping column names to expected types
                       (e.g., {'date': 'datetime', 'units': 'int', 'price': 'float'})

    Returns:
        Dictionary with type mismatch information
    """
    mismatches = {}

    type_mapping = {
        'int': pd.api.types.is_integer_dtype,
        'float': pd.api.types.is_float_dtype,
        'numeric': pd.api.types.is_numeric_dtype,
        'string': pd.api.types.is_string_dtype,
        'datetime': pd.api.types.is_datetime64_any_dtype,
        'bool': pd.api.types.is_bool_dtype,
    }

    for col, expected_type in expected_types.items():
        if col not in df.columns:
            mismatches[col] = {
                'error': 'column_not_found',
                'expected': expected_type,
                'actual': None
            }
            continue

        checker = type_mapping.get(expected_type)
        if checker is None:
            continue

        if not checker(df[col]):
            mismatches[col] = {
                'expected': expected_type,
                'actual': str(df[col].dtype)
            }

    return mismatches


def validate_categorical_values(
    df: pd.DataFrame,
    column: str,
    allowed_values: List[Any]
) -> Tuple[bool, pd.Series]:
    """
    Validate that categorical column contains only allowed values.

    Args:
        df: DataFrame
        column: Column name
        allowed_values: List of allowed values

    Returns:
        Tuple of (is_valid, invalid_rows)
    """
    if column not in df.columns:
        return False, pd.Series([], dtype=bool)

    invalid_mask = ~df[column].isin(allowed_values)
    return not invalid_mask.any(), df[invalid_mask]


def check_referential_integrity(
    df: pd.DataFrame,
    foreign_key: str,
    reference_df: pd.DataFrame,
    reference_key: str
) -> Tuple[int, pd.DataFrame]:
    """
    Check referential integrity between two DataFrames.

    Args:
        df: DataFrame with foreign key
        foreign_key: Foreign key column name
        reference_df: Reference DataFrame
        reference_key: Reference key column name

    Returns:
        Tuple of (orphan_count, orphan_rows)
    """
    valid_values = reference_df[reference_key].unique()
    orphans = ~df[foreign_key].isin(valid_values)
    orphan_count = orphans.sum()

    return int(orphan_count), df[orphans]


def validate_time_series_continuity(
    df: pd.DataFrame,
    date_column: str = 'date',
    group_by: Optional[List[str]] = None,
    frequency: str = 'D'
) -> Dict[str, Any]:
    """
    Check for gaps in time series data.

    Args:
        df: DataFrame
        date_column: Date column name
        group_by: Columns to group by (e.g., ['sku_id', 'store_id'])
        frequency: Expected frequency ('D' for daily, 'W' for weekly, etc.)

    Returns:
        Dictionary with gap information
    """
    if date_column not in df.columns:
        return {'error': f'Column {date_column} not found'}

    df_sorted = df.sort_values(date_column)
    dates = pd.to_datetime(df_sorted[date_column])

    if group_by:
        gaps = []
        for name, group in df_sorted.groupby(group_by):
            group_dates = pd.to_datetime(group[date_column])
            expected_range = pd.date_range(
                start=group_dates.min(),
                end=group_dates.max(),
                freq=frequency
            )
            missing_dates = expected_range.difference(group_dates)
            if len(missing_dates) > 0:
                gaps.append({
                    'group': name if isinstance(name, tuple) else (name,),
                    'missing_dates_count': len(missing_dates),
                    'first_missing': str(missing_dates[0]) if len(missing_dates) > 0 else None
                })
        return {'gaps': gaps, 'total_gaps': len(gaps)}
    else:
        expected_range = pd.date_range(
            start=dates.min(),
            end=dates.max(),
            freq=frequency
        )
        missing_dates = expected_range.difference(dates)
        return {
            'missing_dates_count': len(missing_dates),
            'has_gaps': len(missing_dates) > 0
        }
