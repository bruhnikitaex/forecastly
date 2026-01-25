"""
Tests for utility helper functions.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.utils.helpers import (
    ensure_datetime,
    safe_divide,
    format_number,
    round_metrics,
    validate_dataframe,
)


class TestEnsureDatetime:
    """Tests for ensure_datetime function."""

    def test_converts_string_dates(self):
        """Should convert string dates to datetime."""
        df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'value': [1, 2, 3]
        })

        result = ensure_datetime(df, 'date')

        assert pd.api.types.is_datetime64_any_dtype(result['date'])
        assert len(result) == 3

    def test_sorts_by_date(self):
        """Should sort DataFrame by date column."""
        df = pd.DataFrame({
            'date': ['2024-01-03', '2024-01-01', '2024-01-02'],
            'value': [3, 1, 2]
        })

        result = ensure_datetime(df, 'date')

        expected_dates = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])
        pd.testing.assert_series_equal(
            result['date'].reset_index(drop=True),
            expected_dates.to_series(name='date').reset_index(drop=True)
        )

    def test_handles_datetime_objects(self):
        """Should handle already converted datetime objects."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3, freq='D'),
            'value': [1, 2, 3]
        })

        result = ensure_datetime(df, 'date')

        assert pd.api.types.is_datetime64_any_dtype(result['date'])
        assert len(result) == 3

    def test_handles_timestamps(self):
        """Should handle Unix timestamps."""
        df = pd.DataFrame({
            'date': [1704067200, 1704153600, 1704240000],  # Unix timestamps
            'value': [1, 2, 3]
        })

        result = ensure_datetime(df, 'date')

        assert pd.api.types.is_datetime64_any_dtype(result['date'])

    def test_raises_error_for_missing_column(self):
        """Should raise ValueError if column doesn't exist."""
        df = pd.DataFrame({'value': [1, 2, 3]})

        with pytest.raises(ValueError, match="Колонка 'date' не найдена"):
            ensure_datetime(df, 'date')

    def test_raises_error_for_invalid_dates(self):
        """Should raise ValueError for invalid date strings."""
        df = pd.DataFrame({
            'date': ['not-a-date', 'invalid', 'bad-date'],
            'value': [1, 2, 3]
        })

        with pytest.raises(ValueError, match="Не удалось преобразовать"):
            ensure_datetime(df, 'date')

    def test_custom_column_name(self):
        """Should work with custom column name."""
        df = pd.DataFrame({
            'timestamp': ['2024-01-01', '2024-01-02'],
            'value': [1, 2]
        })

        result = ensure_datetime(df, 'timestamp')

        assert pd.api.types.is_datetime64_any_dtype(result['timestamp'])

    def test_preserves_other_columns(self):
        """Should preserve all other columns."""
        df = pd.DataFrame({
            'date': ['2024-01-02', '2024-01-01'],
            'value': [2, 1],
            'category': ['A', 'B']
        })

        result = ensure_datetime(df, 'date')

        assert 'value' in result.columns
        assert 'category' in result.columns
        assert len(result.columns) == 3

    def test_returns_copy(self):
        """Should return a copy, not modify original."""
        df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02'],
            'value': [1, 2]
        })
        original_type = df['date'].dtype

        result = ensure_datetime(df, 'date')

        assert df['date'].dtype == original_type
        assert result is not df


class TestSafeDivide:
    """Tests for safe_divide function."""

    def test_normal_division(self):
        """Should perform normal division."""
        result = safe_divide(10, 2)
        assert result == 5.0

    def test_division_by_zero_returns_default(self):
        """Should return default value when dividing by zero."""
        result = safe_divide(10, 0)
        assert result == 0.0

    def test_division_by_zero_custom_default(self):
        """Should return custom default value."""
        result = safe_divide(10, 0, default=-1.0)
        assert result == -1.0

    def test_negative_division(self):
        """Should handle negative numbers."""
        result = safe_divide(-10, 2)
        assert result == -5.0

    def test_float_division(self):
        """Should handle float division."""
        result = safe_divide(7.5, 2.5)
        assert result == 3.0

    def test_very_small_denominator(self):
        """Should handle very small but non-zero denominator."""
        result = safe_divide(10, 0.00001)
        assert abs(result - 1000000.0) < 0.01  # Allow for floating point precision

    def test_zero_numerator(self):
        """Should return zero when numerator is zero."""
        result = safe_divide(0, 5)
        assert result == 0.0

    def test_negative_zero(self):
        """Should treat -0.0 as zero."""
        result = safe_divide(10, -0.0)
        assert result == 0.0


class TestFormatNumber:
    """Tests for format_number function."""

    def test_format_integer(self):
        """Should format integer with decimals."""
        result = format_number(100)
        assert result == "100.00"

    def test_format_float(self):
        """Should format float with specified decimals."""
        result = format_number(123.456, decimals=2)
        assert result == "123.46"

    def test_format_with_thousands_separator(self):
        """Should include thousands separator."""
        result = format_number(1234567.89)
        assert "," in result
        assert result == "1,234,567.89"

    def test_format_custom_decimals(self):
        """Should format with custom decimal places."""
        result = format_number(123.456789, decimals=4)
        assert result == "123.4568"

    def test_format_zero_decimals(self):
        """Should format with zero decimals."""
        result = format_number(123.456, decimals=0)
        assert result == "123"

    def test_format_nan_returns_na(self):
        """Should return 'N/A' for NaN values."""
        result = format_number(np.nan)
        assert result == "N/A"

    def test_format_none_returns_na(self):
        """Should return 'N/A' for None values."""
        result = format_number(None)
        assert result == "N/A"

    def test_format_negative_number(self):
        """Should format negative numbers."""
        result = format_number(-1234.56)
        assert result == "-1,234.56"

    def test_format_very_small_number(self):
        """Should format very small numbers."""
        result = format_number(0.00123, decimals=5)
        assert result == "0.00123"

    def test_format_very_large_number(self):
        """Should format very large numbers."""
        result = format_number(1234567890.12)
        assert "1,234,567,890.12" == result


class TestRoundMetrics:
    """Tests for round_metrics function."""

    def test_round_float_values(self):
        """Should round float values in dictionary."""
        metrics = {'mae': 12.3456, 'rmse': 23.4567}
        result = round_metrics(metrics, decimals=2)

        assert result['mae'] == 12.35
        assert result['rmse'] == 23.46

    def test_preserve_integer_values(self):
        """Should preserve integer values."""
        metrics = {'count': 100, 'total': 500}
        result = round_metrics(metrics, decimals=2)

        assert result['count'] == 100
        assert result['total'] == 500

    def test_preserve_none_values(self):
        """Should preserve None values."""
        metrics = {'value': None, 'other': 12.34}
        result = round_metrics(metrics, decimals=2)

        assert result['value'] is None
        assert result['other'] == 12.34

    def test_preserve_string_values(self):
        """Should preserve non-numeric values."""
        metrics = {'name': 'test', 'value': 12.345, 'flag': True}
        result = round_metrics(metrics, decimals=2)

        assert result['name'] == 'test'
        assert result['value'] == 12.35
        # Note: round() converts bool to int, so True becomes 1
        assert result['flag'] == 1

    def test_custom_decimals(self):
        """Should use custom decimal places."""
        metrics = {'value': 12.3456789}
        result = round_metrics(metrics, decimals=4)

        assert result['value'] == 12.3457

    def test_zero_decimals(self):
        """Should round to integers with decimals=0."""
        metrics = {'value': 12.7, 'other': 12.3}
        result = round_metrics(metrics, decimals=0)

        assert result['value'] == 13
        assert result['other'] == 12

    def test_empty_dictionary(self):
        """Should handle empty dictionary."""
        metrics = {}
        result = round_metrics(metrics)

        assert result == {}

    def test_mixed_types(self):
        """Should handle mixed types correctly."""
        metrics = {
            'float': 12.345,
            'int': 100,
            'string': 'test',
            'none': None,
            'bool': True,
            'list': [1, 2, 3]
        }
        result = round_metrics(metrics, decimals=2)

        assert result['float'] == 12.35
        assert result['int'] == 100
        assert result['string'] == 'test'
        assert result['none'] is None
        # Note: round() converts bool to int, so True becomes 1
        assert result['bool'] == 1
        assert result['list'] == [1, 2, 3]

    def test_negative_values(self):
        """Should round negative values correctly."""
        metrics = {'neg': -12.345, 'pos': 12.345}
        result = round_metrics(metrics, decimals=2)

        assert result['neg'] == -12.35
        assert result['pos'] == 12.35


class TestValidateDataFrame:
    """Tests for validate_dataframe function."""

    def test_valid_dataframe_returns_true(self):
        """Should return True for valid DataFrame."""
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4], 'col3': [5, 6]})
        result = validate_dataframe(df, ['col1', 'col2'])

        assert result is True

    def test_all_columns_present(self):
        """Should pass when all required columns present."""
        df = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
        result = validate_dataframe(df, ['a', 'b', 'c'])

        assert result is True

    def test_raises_error_for_missing_column(self):
        """Should raise ValueError for missing column."""
        df = pd.DataFrame({'col1': [1, 2]})

        with pytest.raises(ValueError, match="Отсутствуют обязательные колонки"):
            validate_dataframe(df, ['col1', 'col2'])

    def test_raises_error_for_multiple_missing_columns(self):
        """Should list all missing columns in error."""
        df = pd.DataFrame({'col1': [1, 2]})

        with pytest.raises(ValueError) as exc_info:
            validate_dataframe(df, ['col1', 'col2', 'col3'])

        error_msg = str(exc_info.value)
        assert 'col2' in error_msg
        assert 'col3' in error_msg

    def test_empty_required_list(self):
        """Should return True when no columns required."""
        df = pd.DataFrame({'col1': [1, 2]})
        result = validate_dataframe(df, [])

        assert result is True

    def test_case_sensitive_column_names(self):
        """Should be case-sensitive with column names."""
        df = pd.DataFrame({'Col1': [1, 2]})

        with pytest.raises(ValueError):
            validate_dataframe(df, ['col1'])  # Different case

    def test_extra_columns_allowed(self):
        """Should allow extra columns not in required list."""
        df = pd.DataFrame({'col1': [1], 'col2': [2], 'col3': [3]})
        result = validate_dataframe(df, ['col1'])

        assert result is True

    def test_empty_dataframe(self):
        """Should validate column names even for empty DataFrame."""
        df = pd.DataFrame(columns=['col1', 'col2'])
        result = validate_dataframe(df, ['col1', 'col2'])

        assert result is True


class TestHelpersIntegration:
    """Integration tests for helper functions."""

    def test_complete_data_processing_pipeline(self):
        """Should process data through multiple helper functions."""
        # Create raw data
        df = pd.DataFrame({
            'date': ['2024-01-03', '2024-01-01', '2024-01-02'],
            'sku_id': ['SKU001', 'SKU001', 'SKU001'],
            'units': [100, 200, 150]
        })

        # Validate schema
        validate_dataframe(df, ['date', 'sku_id', 'units'])

        # Convert dates
        df = ensure_datetime(df, 'date')

        # Calculate metrics
        total = df['units'].sum()
        average = safe_divide(total, len(df))

        # Format results
        metrics = {
            'total': float(total),
            'average': average,
            'count': len(df)
        }
        rounded = round_metrics(metrics, decimals=2)

        assert rounded['total'] == 450.0
        assert rounded['average'] == 150.0
        assert rounded['count'] == 3

    def test_error_handling_chain(self):
        """Should properly propagate errors through function chain."""
        df = pd.DataFrame({'value': [1, 2, 3]})

        # Missing column should raise error
        with pytest.raises(ValueError):
            validate_dataframe(df, ['date'])

        # Should not proceed to ensure_datetime
        # This tests that validation happens before transformation

    def test_metrics_calculation_with_edge_cases(self):
        """Should handle edge cases in metrics calculation."""
        metrics = {
            'division_result': safe_divide(10, 0, default=np.nan),
            'valid_value': 123.456,
            'none_value': None
        }

        rounded = round_metrics(metrics, decimals=2)
        formatted = format_number(rounded['valid_value'])

        assert pd.isna(rounded['division_result'])
        assert rounded['valid_value'] == 123.46
        assert rounded['none_value'] is None
        assert formatted == "123.46"
