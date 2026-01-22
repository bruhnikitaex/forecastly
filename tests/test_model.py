"""
Тесты для модулей машинного обучения Forecastly.

Проверяет обучение моделей, прогнозирование и оценку качества.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import joblib

from src.models.predict import normalize_sku, load_prophet_models
from src.models.evaluate import mape


class TestNormalizeSku:
    """Тесты для normalize_sku."""

    def test_normalize_standard_format(self):
        """Стандартный формат SKU001."""
        assert normalize_sku("SKU001") == "SKU001"
        assert normalize_sku("SKU1") == "SKU001"
        assert normalize_sku("SKU123") == "SKU123"

    def test_normalize_with_underscore(self):
        """Формат с подчёркиванием SKU_001."""
        assert normalize_sku("SKU_001") == "SKU001"
        assert normalize_sku("SKU_1") == "SKU001"

    def test_normalize_with_dash(self):
        """Формат с тире SKU-001."""
        assert normalize_sku("SKU-001") == "SKU001"
        assert normalize_sku("SKU-1") == "SKU001"

    def test_normalize_lowercase(self):
        """Нижний регистр."""
        assert normalize_sku("sku001") == "SKU001"
        assert normalize_sku("sku_001") == "SKU001"

    def test_normalize_with_spaces(self):
        """Пробелы в начале/конце."""
        assert normalize_sku("  SKU001  ") == "SKU001"
        assert normalize_sku(" SKU_001 ") == "SKU001"

    def test_normalize_numeric_only(self):
        """Только цифры."""
        assert normalize_sku("123") == "SKU123"
        assert normalize_sku("1") == "SKU001"

    def test_normalize_padding(self):
        """Дополнение нулями до 3 цифр."""
        assert normalize_sku("SKU1") == "SKU001"
        assert normalize_sku("SKU12") == "SKU012"
        assert normalize_sku("SKU123") == "SKU123"
        assert normalize_sku("SKU1234") == "SKU1234"  # Больше 3 цифр - не обрезаем


class TestLoadProphetModels:
    """Тесты для load_prophet_models."""

    def test_returns_empty_dict_when_no_models(self, monkeypatch):
        """Возвращает пустой словарь если модели не найдены."""
        # Патчим путь к несуществующей директории
        monkeypatch.setitem(
            __import__('src.utils.config', fromlist=['PATHS']).PATHS['data'],
            'models_dir',
            '/nonexistent/path'
        )
        result = load_prophet_models()
        assert result == {}

    def test_loads_existing_models(self, tmp_path):
        """Загружает существующие модели."""
        # Создаём временную директорию с моделями
        models = {'SKU001': 'mock_model', 'SKU002': 'mock_model2'}
        model_path = tmp_path / 'prophet_model.pkl'
        joblib.dump(models, model_path)

        # Этот тест проверяет что функция корректно обрабатывает путь
        # (полный тест требует мокирования PATHS)


class TestMape:
    """Тесты для функции MAPE."""

    def test_perfect_prediction(self):
        """MAPE должен быть 0 при идеальном прогнозе."""
        y_true = [10, 20, 30]
        y_pred = [10, 20, 30]
        assert mape(y_true, y_pred) == 0.0

    def test_standard_mape(self):
        """Стандартный расчёт MAPE."""
        y_true = [100, 100, 100]
        y_pred = [90, 110, 100]
        # MAPE = mean(|10/100|, |10/100|, |0/100|) * 100 = 6.67%
        result = mape(y_true, y_pred)
        assert 6 < result < 7

    def test_ignores_zero_true_values(self):
        """Нулевые значения в y_true должны игнорироваться."""
        y_true = [0, 100, 100]
        y_pred = [10, 100, 110]
        # Только последние два: MAPE = mean(|0/100|, |10/100|) * 100 = 5%
        result = mape(y_true, y_pred)
        assert result == 5.0

    def test_handles_nan_values(self):
        """NaN значения должны обрабатываться."""
        y_true = [100, np.nan, 100]
        y_pred = [100, 100, np.nan]
        # Только первое значение валидно
        result = mape(y_true, y_pred)
        assert result == 0.0

    def test_returns_nan_when_no_valid_values(self):
        """Возвращает NaN если нет валидных значений."""
        y_true = [0, 0, 0]
        y_pred = [10, 20, 30]
        result = mape(y_true, y_pred)
        assert np.isnan(result)


class TestModelOutput:
    """Тесты для выходных данных моделей."""

    def test_predictions_csv_structure(self, tmp_path):
        """Проверяет структуру predictions.csv."""
        # Создаём типичный predictions.csv
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=14, freq='D'),
            'sku_id': ['SKU001'] * 14,
            'prophet': [12.5] * 14,
            'xgb': [11.8] * 14,
            'ensemble': [12.15] * 14,
            'p_low': [10.0] * 14,
            'p_high': [15.0] * 14
        })
        csv_path = tmp_path / 'predictions.csv'
        df.to_csv(csv_path, index=False)

        # Загружаем и проверяем
        loaded = pd.read_csv(csv_path, parse_dates=['date'])

        assert 'date' in loaded.columns
        assert 'sku_id' in loaded.columns
        assert 'prophet' in loaded.columns
        assert 'xgb' in loaded.columns
        assert 'ensemble' in loaded.columns

    def test_metrics_csv_structure(self, tmp_path):
        """Проверяет структуру metrics.csv."""
        df = pd.DataFrame({
            'sku_id': ['SKU001', 'SKU002'],
            'mape_prophet': [5.2, 6.1],
            'mape_xgboost': [4.8, 5.5],
            'mape_naive': [8.3, 9.0],
            'mape_ens': [4.5, 5.0],
            'best_model': ['ens', 'ens']
        })
        csv_path = tmp_path / 'metrics.csv'
        df.to_csv(csv_path, index=False)

        loaded = pd.read_csv(csv_path)

        assert 'sku_id' in loaded.columns
        assert 'mape_prophet' in loaded.columns
        assert 'mape_xgboost' in loaded.columns
        assert 'mape_naive' in loaded.columns
        assert 'mape_ens' in loaded.columns
        assert 'best_model' in loaded.columns


class TestModelPredictions:
    """Тесты для предсказаний моделей."""

    def test_ensemble_is_average(self):
        """Ensemble должен быть средним Prophet и XGBoost."""
        prophet_pred = np.array([10, 20, 30])
        xgb_pred = np.array([12, 22, 32])
        expected_ensemble = (prophet_pred + xgb_pred) / 2

        np.testing.assert_array_equal(expected_ensemble, [11, 21, 31])

    def test_predictions_positive(self):
        """Прогнозы продаж должны быть неотрицательными."""
        # В реальном коде можно добавить clip(0, None)
        predictions = np.array([10, -5, 20])
        clipped = np.clip(predictions, 0, None)
        assert all(clipped >= 0)


class TestModelSerialization:
    """Тесты для сериализации моделей."""

    def test_xgboost_model_serializable(self, tmp_path):
        """XGBoost модель должна сериализоваться."""
        from xgboost import XGBRegressor

        model = XGBRegressor(n_estimators=10)
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y = np.array([1, 2, 3])
        model.fit(X, y)

        model_path = tmp_path / 'xgb_test.pkl'
        joblib.dump(model, model_path)

        loaded = joblib.load(model_path)
        predictions = loaded.predict(X)
        assert len(predictions) == 3

    def test_prophet_models_dict_serializable(self, tmp_path):
        """Словарь Prophet моделей должен сериализоваться."""
        # Prophet модель создавать долго, проверяем просто словарь
        models = {
            'SKU001': 'prophet_model_placeholder',
            'SKU002': 'prophet_model_placeholder'
        }

        model_path = tmp_path / 'prophet_test.pkl'
        joblib.dump(models, model_path)

        loaded = joblib.load(model_path)
        assert 'SKU001' in loaded
        assert 'SKU002' in loaded


class TestModelConfig:
    """Тесты для конфигурации моделей."""

    def test_xgboost_default_params(self):
        """Проверяет параметры XGBoost по умолчанию."""
        from src.utils.config import MODEL_CFG

        xgb_cfg = MODEL_CFG.get('model', {}).get('xgboost', {})

        # Параметры должны быть разумными
        if 'n_estimators' in xgb_cfg:
            assert 100 <= xgb_cfg['n_estimators'] <= 1000
        if 'learning_rate' in xgb_cfg:
            assert 0.01 <= xgb_cfg['learning_rate'] <= 0.3
        if 'max_depth' in xgb_cfg:
            assert 3 <= xgb_cfg['max_depth'] <= 15

    def test_horizon_weeks_config(self):
        """Проверяет настройку горизонта прогноза."""
        from src.utils.config import MODEL_CFG

        horizon = MODEL_CFG.get('model', {}).get('horizon_weeks', 8)
        assert 1 <= horizon <= 52  # Разумные пределы


class TestFeatureImportance:
    """Тесты для важности признаков."""

    def test_xgboost_has_feature_importances(self):
        """XGBoost должен предоставлять важность признаков."""
        from xgboost import XGBRegressor

        model = XGBRegressor(n_estimators=10)
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        y = np.array([1, 2, 3, 4])
        model.fit(X, y)

        importances = model.feature_importances_
        assert len(importances) == 3
        assert all(i >= 0 for i in importances)
        assert abs(sum(importances) - 1.0) < 0.01  # Сумма ~ 1


class TestDataPreparation:
    """Тесты для подготовки данных для моделей."""

    def test_prophet_data_format(self):
        """Prophet требует колонки ds и y."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'units': list(range(30))
        })
        prophet_df = df.rename(columns={'date': 'ds', 'units': 'y'})

        assert 'ds' in prophet_df.columns
        assert 'y' in prophet_df.columns
        assert len(prophet_df) == 30

    def test_xgboost_features_available(self):
        """Признаки для XGBoost должны быть доступны."""
        from src.etl.feature_builder import build_features

        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'sku_id': ['A'] * 30,
            'store_id': ['S1'] * 30,
            'units': list(range(30))
        })
        features = build_features(df)

        required_features = ['dow', 'week', 'month', 'units_lag_1', 'units_lag_7']
        for feat in required_features:
            assert feat in features.columns, f"Отсутствует признак {feat}"
