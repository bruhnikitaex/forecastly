"""
Тесты для ETL-конвейера Forecastly.

Проверяет загрузку, очистку, валидацию данных и построение признаков.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from src.etl.validate import (
    ValidationError,
    validate_required_columns,
    validate_date_column,
    validate_date_range,
    validate_numeric_column,
    validate_no_duplicates,
    validate_sku_count,
    validate_data_completeness,
    validate_csv_file,
    validate_model_data
)
from src.etl.clean_data import (
    ensure_columns,
    remove_outliers_iqr,
    fill_date_gaps,
    clean_sales
)
from src.etl.feature_builder import build_features
from src.utils.helpers import ensure_datetime


class TestValidateRequiredColumns:
    """Тесты для validate_required_columns."""

    def test_valid_columns(self):
        """Не должно быть ошибки при наличии всех колонок."""
        df = pd.DataFrame({'date': [1], 'sku_id': ['A'], 'units': [10]})
        # Не должно выбрасывать исключение
        validate_required_columns(df, ['date', 'sku_id'])

    def test_missing_columns_raises_error(self):
        """Должна быть ошибка при отсутствии колонки."""
        df = pd.DataFrame({'date': [1]})
        with pytest.raises(ValidationError, match="Отсутствуют обязательные колонки"):
            validate_required_columns(df, ['date', 'sku_id'])


class TestValidateDateColumn:
    """Тесты для validate_date_column."""

    def test_valid_dates(self):
        """Корректные даты должны парситься."""
        df = pd.DataFrame({'date': ['2024-01-01', '2024-01-02']})
        result = validate_date_column(df, 'date')
        assert pd.api.types.is_datetime64_any_dtype(result['date'])

    def test_invalid_dates_raises_error(self):
        """Некорректные даты должны вызывать ошибку."""
        df = pd.DataFrame({'date': ['not-a-date', 'also-not']})
        with pytest.raises(ValidationError):
            validate_date_column(df, 'date')


class TestValidateDateRange:
    """Тесты для validate_date_range."""

    def test_sufficient_date_range(self):
        """Достаточный диапазон дат не должен вызывать ошибку."""
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        df = pd.DataFrame({'date': dates})
        # Не должно выбрасывать исключение
        validate_date_range(df, 'date', min_days=30)

    def test_insufficient_date_range_raises_error(self):
        """Недостаточный диапазон дат должен вызывать ошибку."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        df = pd.DataFrame({'date': dates})
        with pytest.raises(ValidationError, match="Диапазон дат слишком мал"):
            validate_date_range(df, 'date', min_days=30)


class TestValidateNumericColumn:
    """Тесты для validate_numeric_column."""

    def test_valid_numeric_column(self):
        """Корректная числовая колонка."""
        df = pd.DataFrame({'units': [10, 20, 30]})
        result = validate_numeric_column(df, 'units')
        assert result['units'].dtype in [np.int64, np.float64]

    def test_negative_values_raises_error(self):
        """Отрицательные значения должны вызывать ошибку."""
        df = pd.DataFrame({'units': [10, -5, 30]})
        with pytest.raises(ValidationError, match="отрицательных значений"):
            validate_numeric_column(df, 'units', allow_negative=False)

    def test_nan_values_filled_with_zero(self):
        """NaN значения должны заполняться нулями."""
        df = pd.DataFrame({'units': [10, np.nan, 30]})
        result = validate_numeric_column(df, 'units')
        assert result['units'].isna().sum() == 0
        assert result['units'].iloc[1] == 0


class TestValidateNoDuplicates:
    """Тесты для validate_no_duplicates."""

    def test_no_duplicates(self):
        """Отсутствие дубликатов не должно вызывать проблем."""
        df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02'],
            'sku_id': ['A', 'A']
        })
        # Не должно выбрасывать исключение (только warning)
        validate_no_duplicates(df, ['date', 'sku_id'])

    def test_with_duplicates_logs_warning(self):
        """Дубликаты должны логироваться как warning."""
        df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-01'],
            'sku_id': ['A', 'A']
        })
        # Не должно выбрасывать исключение
        validate_no_duplicates(df, ['date', 'sku_id'])


class TestValidateSkuCount:
    """Тесты для validate_sku_count."""

    def test_sufficient_skus(self):
        """Достаточное количество SKU."""
        df = pd.DataFrame({'sku_id': ['A', 'B', 'C']})
        validate_sku_count(df, 'sku_id', min_skus=2)

    def test_insufficient_skus_raises_error(self):
        """Недостаточное количество SKU должно вызывать ошибку."""
        df = pd.DataFrame({'sku_id': ['A']})
        with pytest.raises(ValidationError, match="Недостаточно уникальных SKU"):
            validate_sku_count(df, 'sku_id', min_skus=2)


class TestValidateDataCompleteness:
    """Тесты для validate_data_completeness."""

    def test_complete_data(self):
        """Полные данные не должны вызывать ошибку."""
        df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02'],
            'sku_id': ['A', 'B'],
            'units': [10, 20]
        })
        validate_data_completeness(df, ['date', 'sku_id', 'units'])

    def test_too_many_missing_raises_error(self):
        """Слишком много пропусков должно вызывать ошибку."""
        df = pd.DataFrame({
            'date': ['2024-01-01'] * 10,
            'sku_id': ['A'] * 10,
            'units': [np.nan] * 10  # 100% пропусков
        })
        with pytest.raises(ValidationError, match="пропусков"):
            validate_data_completeness(df, ['units'], max_missing_percent=50.0)


class TestEnsureColumns:
    """Тесты для ensure_columns."""

    def test_creates_optional_columns(self):
        """Опциональные колонки должны создаваться."""
        df = pd.DataFrame({
            'date': ['2024-01-01'],
            'sku_id': ['A']
        })
        result = ensure_columns(df)
        assert 'store_id' in result.columns
        assert 'price' in result.columns
        assert 'promo_flag' in result.columns

    def test_missing_required_raises_error(self):
        """Отсутствие обязательных колонок должно вызывать ошибку."""
        df = pd.DataFrame({'some_col': [1]})
        with pytest.raises(ValueError, match="обязательные колонки"):
            ensure_columns(df)


class TestRemoveOutliersIQR:
    """Тесты для remove_outliers_iqr."""

    def test_clips_outliers(self):
        """Выбросы должны обрезаться."""
        df = pd.DataFrame({
            'sku_id': ['A'] * 10,
            'store_id': ['S1'] * 10,
            'units': [10, 10, 10, 10, 10, 10, 10, 10, 10, 1000]  # 1000 - выброс
        })
        result = remove_outliers_iqr(df, 'units', ['sku_id', 'store_id'])
        assert result['units'].max() < 1000

    def test_preserves_normal_values(self):
        """Нормальные значения должны сохраняться."""
        df = pd.DataFrame({
            'sku_id': ['A'] * 5,
            'store_id': ['S1'] * 5,
            'units': [10, 11, 12, 11, 10]
        })
        result = remove_outliers_iqr(df, 'units', ['sku_id', 'store_id'])
        assert list(result['units']) == [10, 11, 12, 11, 10]


class TestFillDateGaps:
    """Тесты для fill_date_gaps."""

    def test_fills_missing_dates(self):
        """Пропущенные даты должны заполняться."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-01-03']),  # пропущен 2024-01-02
            'sku_id': ['A', 'A'],
            'store_id': ['S1', 'S1'],
            'units': [10, 12]
        })
        result = fill_date_gaps(df)
        assert len(result) == 3  # Должно быть 3 даты

    def test_preserves_original_values(self):
        """Исходные значения должны сохраняться."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'sku_id': ['A', 'A'],
            'store_id': ['S1', 'S1'],
            'units': [10, 20]
        })
        result = fill_date_gaps(df)
        assert 10 in result['units'].values
        assert 20 in result['units'].values


class TestCleanSales:
    """Тесты для clean_sales."""

    def test_full_cleaning_pipeline(self):
        """Полный конвейер очистки."""
        df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'sku_id': ['A', 'A', 'A'],
            'units': [10, 20, 30]
        })
        result = clean_sales(df)

        # Проверяем что создались опциональные колонки
        assert 'store_id' in result.columns
        assert 'price' in result.columns

        # Проверяем что даты datetime
        assert pd.api.types.is_datetime64_any_dtype(result['date'])


class TestBuildFeatures:
    """Тесты для build_features."""

    def test_creates_time_features(self):
        """Временные признаки должны создаваться."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'sku_id': ['A'] * 30,
            'store_id': ['S1'] * 30,
            'units': list(range(30))
        })
        result = build_features(df)

        assert 'dow' in result.columns  # day of week
        assert 'week' in result.columns
        assert 'month' in result.columns

    def test_creates_lag_features(self):
        """Лаговые признаки должны создаваться."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'sku_id': ['A'] * 30,
            'store_id': ['S1'] * 30,
            'units': list(range(30))
        })
        result = build_features(df)

        assert 'units_lag_1' in result.columns
        assert 'units_lag_7' in result.columns


class TestEnsureDatetime:
    """Тесты для ensure_datetime."""

    def test_converts_string_to_datetime(self):
        """Строки должны конвертироваться в datetime."""
        df = pd.DataFrame({'date': ['2024-01-01', '2024-01-02']})
        result = ensure_datetime(df, 'date')
        assert pd.api.types.is_datetime64_any_dtype(result['date'])

    def test_sorts_by_date(self):
        """Данные должны сортироваться по дате."""
        df = pd.DataFrame({'date': ['2024-01-02', '2024-01-01']})
        result = ensure_datetime(df, 'date')
        assert result['date'].iloc[0] < result['date'].iloc[1]


class TestValidateModelData:
    """Тесты для validate_model_data."""

    def test_valid_model_data(self):
        """Валидные данные для модели."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=50, freq='D'),
            'sku_id': ['A'] * 50,
            'units': list(range(50))
        })
        # Не должно выбрасывать исключение
        validate_model_data(df, min_samples=30)

    def test_insufficient_samples_raises_error(self):
        """Недостаточное количество строк должно вызывать ошибку."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10, freq='D'),
            'sku_id': ['A'] * 10,
            'units': list(range(10))
        })
        with pytest.raises(ValidationError, match="Недостаточно данных"):
            validate_model_data(df, min_samples=30)

    def test_missing_required_columns_raises_error(self):
        """Отсутствие обязательных колонок должно вызывать ошибку."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=50, freq='D'),
            'sku_id': ['A'] * 50
            # units отсутствует
        })
        with pytest.raises(ValidationError, match="Отсутствуют обязательные колонки"):
            validate_model_data(df, min_samples=30)


class TestValidateCsvFile:
    """Интеграционные тесты для validate_csv_file."""

    def test_valid_csv_file(self, tmp_path):
        """Валидный CSV файл."""
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=60, freq='D'),
            'sku_id': ['SKU001'] * 60,
            'units': [10] * 60
        })
        df.to_csv(csv_path, index=False)

        result = validate_csv_file(csv_path)
        assert len(result) == 60

    def test_nonexistent_file_raises_error(self, tmp_path):
        """Несуществующий файл должен вызывать ошибку."""
        with pytest.raises(FileNotFoundError):
            validate_csv_file(tmp_path / "nonexistent.csv")

    def test_empty_csv_raises_error(self, tmp_path):
        """Пустой CSV должен вызывать ошибку."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("date,sku_id,units\n")

        with pytest.raises(ValidationError, match="пуст"):
            validate_csv_file(csv_path)


class TestValidationEdgeCases:
    """Additional edge case tests for validation functions."""

    def test_validate_with_extra_columns(self):
        """Should pass with extra columns not in required list."""
        df = pd.DataFrame({
            'date': ['2024-01-01'],
            'sku_id': ['A'],
            'units': [10],
            'extra_col': ['extra']
        })
        # Should not raise error
        validate_required_columns(df, ['date', 'sku_id', 'units'])

    def test_validate_date_with_different_formats(self):
        """Should handle different date formats."""
        df1 = pd.DataFrame({'date': ['2024-01-01', '2024-01-02']})
        df2 = pd.DataFrame({'date': ['01/01/2024', '01/02/2024']})
        df3 = pd.DataFrame({'date': ['2024/01/01', '2024/01/02']})

        result1 = validate_date_column(df1, 'date')
        result2 = validate_date_column(df2, 'date')
        result3 = validate_date_column(df3, 'date')

        assert pd.api.types.is_datetime64_any_dtype(result1['date'])
        assert pd.api.types.is_datetime64_any_dtype(result2['date'])
        assert pd.api.types.is_datetime64_any_dtype(result3['date'])

    def test_validate_numeric_with_float_strings(self):
        """Should handle numeric strings."""
        df = pd.DataFrame({'units': ['10.5', '20.3', '30.1']})
        result = validate_numeric_column(df, 'units')
        assert result['units'].dtype in [np.float64]

    def test_validate_date_range_edge_dates(self):
        """Should handle edge cases in date ranges."""
        # Exactly min_days
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        df = pd.DataFrame({'date': dates})
        validate_date_range(df, 'date', min_days=30)

        # One day less should fail
        dates = pd.date_range('2024-01-01', periods=29, freq='D')
        df = pd.DataFrame({'date': dates})
        with pytest.raises(ValidationError):
            validate_date_range(df, 'date', min_days=30)


class TestCleaningEdgeCases:
    """Edge case tests for cleaning functions."""

    def test_remove_outliers_all_same_values(self):
        """Should handle data with all same values."""
        df = pd.DataFrame({
            'sku_id': ['A'] * 10,
            'store_id': ['S1'] * 10,
            'units': [10] * 10
        })
        result = remove_outliers_iqr(df, 'units', ['sku_id', 'store_id'])
        assert list(result['units']) == [10] * 10

    def test_remove_outliers_small_group(self):
        """Should handle small groups gracefully."""
        df = pd.DataFrame({
            'sku_id': ['A', 'A', 'A'],
            'store_id': ['S1', 'S1', 'S1'],
            'units': [10, 11, 100]
        })
        result = remove_outliers_iqr(df, 'units', ['sku_id', 'store_id'])
        # With only 3 values, IQR method should still work
        assert len(result) == 3

    def test_fill_date_gaps_no_gaps(self):
        """Should handle data with no gaps."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=7, freq='D'),
            'sku_id': ['A'] * 7,
            'store_id': ['S1'] * 7,
            'units': list(range(7))
        })
        result = fill_date_gaps(df)
        assert len(result) == 7

    def test_fill_date_gaps_multiple_skus(self):
        """Should fill gaps for each SKU separately."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-01-03', '2024-01-01', '2024-01-03']),
            'sku_id': ['A', 'A', 'B', 'B'],
            'store_id': ['S1', 'S1', 'S1', 'S1'],
            'units': [10, 12, 20, 22]
        })
        result = fill_date_gaps(df)
        # Should have filled dates for both SKUs
        assert len(result) > 4


class TestFeatureBuilderEdgeCases:
    """Edge case tests for feature building."""

    def test_build_features_short_series(self):
        """Should handle short time series."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10, freq='D'),
            'sku_id': ['A'] * 10,
            'store_id': ['S1'] * 10,
            'units': list(range(10))
        })
        result = build_features(df)

        # Lag features will have NaN for first values
        assert result['units_lag_1'].isna().sum() > 0
        assert result['units_lag_7'].isna().sum() > 0

    def test_build_features_leap_year(self):
        """Should handle leap year dates."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-02-28', periods=3, freq='D'),
            'sku_id': ['A'] * 3,
            'store_id': ['S1'] * 3,
            'units': [10, 11, 12]
        })
        result = build_features(df)

        # 2024 is leap year, so Feb 29 should exist
        assert len(result) == 3
        assert pd.Timestamp('2024-02-29') in result['date'].values

    def test_build_features_year_boundary(self):
        """Should handle year boundary transitions."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-12-30', periods=5, freq='D'),
            'sku_id': ['A'] * 5,
            'store_id': ['S1'] * 5,
            'units': list(range(5))
        })
        result = build_features(df)

        # Should handle transition from 2023 to 2024
        assert 'week' in result.columns
        assert 'month' in result.columns


class TestDataPipelineIntegration:
    """Integration tests for complete data pipeline."""

    def test_full_pipeline_with_real_like_data(self):
        """Should process realistic sales data through full pipeline."""
        # Create realistic sales data
        dates = []
        sku_ids = []
        units = []

        for sku in ['SKU001', 'SKU002', 'SKU003']:
            for i in range(90):
                dates.append(pd.Timestamp('2024-01-01') + pd.Timedelta(days=i))
                sku_ids.append(sku)
                # Realistic sales pattern with weekly seasonality
                base = 50
                weekly = 10 * np.sin(i * 2 * np.pi / 7)
                noise = np.random.randn() * 5
                units.append(max(0, base + weekly + noise))

        df = pd.DataFrame({
            'date': dates,
            'sku_id': sku_ids,
            'units': units
        })

        # Run through pipeline
        cleaned = clean_sales(df)
        features = build_features(cleaned)

        # Validate output
        assert len(features) > 0
        assert 'date' in features.columns
        assert 'sku_id' in features.columns
        assert 'units' in features.columns
        assert 'dow' in features.columns
        assert pd.api.types.is_datetime64_any_dtype(features['date'])

    def test_pipeline_preserves_sku_counts(self):
        """Pipeline should preserve SKU counts."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30, freq='D').tolist() * 3,
            'sku_id': ['A'] * 30 + ['B'] * 30 + ['C'] * 30,
            'units': list(range(90))
        })

        original_skus = df['sku_id'].nunique()

        cleaned = clean_sales(df)
        features = build_features(cleaned)

        assert features['sku_id'].nunique() == original_skus

    def test_pipeline_handles_missing_values(self):
        """Pipeline should handle missing values appropriately."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'sku_id': ['A'] * 30,
            'units': [10] * 20 + [np.nan] * 10
        })

        cleaned = clean_sales(df)

        # Missing values should be filled with 0
        assert cleaned['units'].isna().sum() == 0


class TestValidationWithMultipleSKUs:
    """Tests for validation with multiple SKUs."""

    def test_validate_multiple_skus_sufficient(self):
        """Should pass with sufficient SKUs."""
        df = pd.DataFrame({
            'sku_id': ['A', 'B', 'C', 'D', 'E']
        })
        validate_sku_count(df, 'sku_id', min_skus=3)

    def test_validate_data_completeness_per_sku(self):
        """Data completeness check across multiple SKUs."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30, freq='D').tolist() * 2,
            'sku_id': ['A'] * 30 + ['B'] * 30,
            'units': [10] * 30 + [np.nan] * 30
        })

        # Overall 50% missing
        with pytest.raises(ValidationError):
            validate_data_completeness(df, ['units'], max_missing_percent=40.0)


class TestOutlierDetection:
    """Comprehensive outlier detection tests."""

    def test_outlier_detection_symmetric(self):
        """Should detect outliers on both ends."""
        df = pd.DataFrame({
            'sku_id': ['A'] * 12,
            'store_id': ['S1'] * 12,
            'units': [10, 10, 10, 10, 10, 1, 10, 10, 10, 10, 10, 100]
        })
        result = remove_outliers_iqr(df, 'units', ['sku_id', 'store_id'])

        # Both 1 and 100 should be clipped
        assert result['units'].min() > 1
        assert result['units'].max() < 100

    def test_outlier_preserves_median_range(self):
        """Should preserve values in median range."""
        df = pd.DataFrame({
            'sku_id': ['A'] * 20,
            'store_id': ['S1'] * 20,
            'units': list(range(10, 30))
        })
        result = remove_outliers_iqr(df, 'units', ['sku_id', 'store_id'])

        # Median values should be unchanged
        median_idx = len(df) // 2
        assert result['units'].iloc[median_idx] == df['units'].iloc[median_idx]
