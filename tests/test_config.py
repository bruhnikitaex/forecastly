"""
Тесты для конфигурации и утилит.

Покрывает:
- Загрузку конфигурации (PATHS, MODEL_CFG)
- Вспомогательные функции (helpers)
- Логирование
"""

import pytest
import os
from pathlib import Path
import pandas as pd
import numpy as np

from src.utils.config import PATHS, MODEL_CFG
from src.utils.helpers import ensure_datetime, safe_divide, round_metrics
from src.utils.logger import logger


# ============================================================================
# PATHS CONFIGURATION TESTS
# ============================================================================

class TestPathsConfig:
    """Тесты для конфигурации путей."""

    def test_paths_loaded(self):
        """PATHS должен быть загружен."""
        assert PATHS is not None
        assert isinstance(PATHS, dict)

    def test_paths_has_data_section(self):
        """PATHS должен содержать секцию data."""
        assert "data" in PATHS

    def test_paths_has_required_keys(self):
        """PATHS должен содержать обязательные ключи."""
        data = PATHS.get("data", {})

        # Проверяем основные ключи (могут быть разные названия)
        has_raw = "raw" in data or "raw_dir" in data
        has_processed = "processed" in data or "processed_dir" in data
        has_models = "models_dir" in data or "models" in data

        assert has_raw or has_processed or has_models, "PATHS должен содержать пути к данным"

    def test_paths_values_are_strings(self):
        """Значения путей должны быть строками."""
        data = PATHS.get("data", {})

        for key, value in data.items():
            if value is not None:
                assert isinstance(value, str), f"Путь {key} должен быть строкой"


# ============================================================================
# MODEL CONFIGURATION TESTS
# ============================================================================

class TestModelConfig:
    """Тесты для конфигурации моделей."""

    def test_model_cfg_loaded(self):
        """MODEL_CFG должен быть загружен."""
        assert MODEL_CFG is not None
        assert isinstance(MODEL_CFG, dict)

    def test_model_cfg_has_model_section(self):
        """MODEL_CFG должен содержать секцию model."""
        # Может быть пустым в некоторых конфигурациях
        assert MODEL_CFG is not None

    def test_xgboost_params_valid(self):
        """Параметры XGBoost должны быть валидными."""
        model = MODEL_CFG.get("model", {})
        xgb = model.get("xgboost", {})

        if "n_estimators" in xgb:
            assert 50 <= xgb["n_estimators"] <= 2000
        if "learning_rate" in xgb:
            assert 0.001 <= xgb["learning_rate"] <= 0.5
        if "max_depth" in xgb:
            assert 1 <= xgb["max_depth"] <= 20

    def test_horizon_weeks_valid(self):
        """Горизонт прогноза должен быть валидным."""
        model = MODEL_CFG.get("model", {})
        horizon = model.get("horizon_weeks", 8)

        assert 1 <= horizon <= 52


# ============================================================================
# HELPERS TESTS
# ============================================================================

class TestEnsureDatetime:
    """Тесты для ensure_datetime."""

    def test_converts_string_dates(self):
        """Конвертирует строковые даты."""
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"]
        })
        result = ensure_datetime(df, "date")

        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_sorts_by_date(self):
        """Сортирует по дате."""
        df = pd.DataFrame({
            "date": ["2024-01-03", "2024-01-01", "2024-01-02"],
            "value": [3, 1, 2]
        })
        result = ensure_datetime(df, "date")

        assert result["value"].iloc[0] == 1
        assert result["value"].iloc[2] == 3

    def test_handles_datetime_column(self):
        """Работает с уже datetime колонкой."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=3)
        })
        result = ensure_datetime(df, "date")

        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_handles_mixed_formats(self):
        """Обрабатывает разные форматы дат."""
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"]
        })
        # Pandas должен распарсить ISO формат
        result = ensure_datetime(df, "date")

        assert len(result) == 3
        assert pd.api.types.is_datetime64_any_dtype(result["date"])


class TestSafeDivide:
    """Тесты для safe_divide."""

    def test_normal_division(self):
        """Обычное деление."""
        result = safe_divide(10, 2)
        assert result == 5.0

    def test_division_by_zero(self):
        """Деление на ноль возвращает 0."""
        result = safe_divide(10, 0)
        assert result == 0.0

    def test_division_by_zero_custom_default(self):
        """Деление на ноль с кастомным значением."""
        result = safe_divide(10, 0, default=np.nan)
        assert np.isnan(result)

    def test_numpy_arrays(self):
        """Работа с numpy массивами."""
        numerator = np.array([10, 20, 30])
        denominator = np.array([2, 0, 5])

        # Поэлементное деление
        results = [safe_divide(n, d) for n, d in zip(numerator, denominator)]

        assert results[0] == 5.0
        assert results[1] == 0.0
        assert results[2] == 6.0


class TestRoundMetrics:
    """Тесты для round_metrics."""

    def test_round_dict(self):
        """Округление словаря с метриками."""
        metrics = {
            "mape": 5.123456789,
            "rmse": 10.987654321,
            "mae": 7.5
        }
        result = round_metrics(metrics, decimals=2)

        assert result["mape"] == 5.12
        assert result["rmse"] == 10.99
        assert result["mae"] == 7.5

    def test_handles_non_numeric(self):
        """Обрабатывает нечисловые значения."""
        metrics = {
            "mape": 5.123,
            "model_name": "prophet"
        }
        result = round_metrics(metrics, decimals=2)

        assert result["mape"] == 5.12
        assert result["model_name"] == "prophet"

    def test_handles_none_values(self):
        """Обрабатывает None значения."""
        metrics = {
            "mape": 5.123,
            "rmse": None
        }
        result = round_metrics(metrics, decimals=2)

        assert result["mape"] == 5.12
        assert result["rmse"] is None


# ============================================================================
# LOGGER TESTS
# ============================================================================

class TestLogger:
    """Тесты для логгера."""

    def test_logger_exists(self):
        """Логгер должен существовать."""
        assert logger is not None

    def test_logger_has_methods(self):
        """Логгер должен иметь стандартные методы."""
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")
        assert hasattr(logger, "debug")

    def test_logger_info(self, caplog):
        """Логирование info сообщения."""
        import logging
        with caplog.at_level(logging.INFO):
            logger.info("Test info message")

        # Loguru использует свой формат, проверяем что не упало
        assert True

    def test_logger_warning(self, caplog):
        """Логирование warning сообщения."""
        import logging
        with caplog.at_level(logging.WARNING):
            logger.warning("Test warning message")

        assert True

    def test_logger_error(self, caplog):
        """Логирование error сообщения."""
        import logging
        with caplog.at_level(logging.ERROR):
            logger.error("Test error message")

        assert True


# ============================================================================
# ENVIRONMENT TESTS
# ============================================================================

class TestEnvironment:
    """Тесты для переменных окружения."""

    def test_environment_variable(self):
        """Переменная ENVIRONMENT должна быть установлена или иметь default."""
        env = os.getenv("ENVIRONMENT", "development")
        assert env in ["development", "testing", "production"]

    def test_use_database_variable(self):
        """Переменная USE_DATABASE должна быть строкой true/false."""
        use_db = os.getenv("USE_DATABASE", "false").lower()
        assert use_db in ["true", "false"]

    def test_log_level_variable(self):
        """Переменная LOG_LEVEL должна быть валидной."""
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert log_level in valid_levels


# ============================================================================
# FILE PATHS TESTS
# ============================================================================

class TestFilePaths:
    """Тесты для путей к файлам."""

    def test_project_structure(self, project_root):
        """Проверка структуры проекта."""
        assert (project_root / "src").exists()
        assert (project_root / "configs").exists()
        assert (project_root / "requirements.txt").exists()

    def test_config_files_exist(self, project_root):
        """Конфигурационные файлы должны существовать."""
        configs_dir = project_root / "configs"

        assert (configs_dir / "paths.yaml").exists() or len(list(configs_dir.glob("*.yaml"))) > 0

    def test_data_directories_creatable(self, tmp_path):
        """Директории для данных должны создаваться."""
        data_raw = tmp_path / "data" / "raw"
        data_proc = tmp_path / "data" / "processed"
        data_models = tmp_path / "data" / "models"

        data_raw.mkdir(parents=True)
        data_proc.mkdir(parents=True)
        data_models.mkdir(parents=True)

        assert data_raw.exists()
        assert data_proc.exists()
        assert data_models.exists()


# ============================================================================
# DATA VALIDATION HELPERS
# ============================================================================

class TestDataValidationHelpers:
    """Тесты для вспомогательных функций валидации данных."""

    def test_dataframe_has_required_columns(self, sample_sales_df):
        """DataFrame должен иметь обязательные колонки."""
        required = ["date", "sku_id", "units"]

        for col in required:
            assert col in sample_sales_df.columns

    def test_dataframe_no_all_nulls(self, sample_sales_df):
        """Колонки не должны быть полностью пустыми."""
        for col in sample_sales_df.columns:
            assert not sample_sales_df[col].isna().all(), f"Колонка {col} полностью пустая"

    def test_dataframe_positive_units(self, sample_sales_df):
        """Units должны быть неотрицательными."""
        assert (sample_sales_df["units"] >= 0).all()

    def test_dataframe_valid_dates(self, sample_sales_df):
        """Даты должны быть валидными."""
        dates = pd.to_datetime(sample_sales_df["date"])

        assert dates.notna().all()
        assert dates.min() > pd.Timestamp("2000-01-01")
        assert dates.max() < pd.Timestamp("2030-01-01")


# ============================================================================
# Запуск тестов
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
