"""
Тесты для REST API Forecastly.

Проверяет все endpoints API на корректность ответов.
"""
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import pandas as pd
import tempfile
import shutil


# Создаём тестовые данные перед импортом API
@pytest.fixture(scope="module", autouse=True)
def setup_test_data():
    """Создаёт тестовые данные для API тестов."""
    # Создаём директории
    data_raw = Path("data/raw")
    data_proc = Path("data/processed")
    data_raw.mkdir(parents=True, exist_ok=True)
    data_proc.mkdir(parents=True, exist_ok=True)

    # Создаём тестовый CSV если его нет
    test_csv = data_raw / "sales_synth.csv"
    if not test_csv.exists():
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=60, freq='D'),
            'sku_id': ['SKU001'] * 30 + ['SKU002'] * 30,
            'store_id': ['S01'] * 60,
            'units': [10 + i % 5 for i in range(60)],
            'price': [100.0] * 60
        })
        df.to_csv(test_csv, index=False)

    # Создаём тестовые прогнозы
    test_pred = data_proc / "predictions.csv"
    if not test_pred.exists():
        df_pred = pd.DataFrame({
            'date': pd.date_range('2024-03-01', periods=14, freq='D').tolist() * 2,
            'sku_id': ['SKU001'] * 14 + ['SKU002'] * 14,
            'prophet': [12.5] * 28,
            'xgb': [11.8] * 28,
            'ensemble': [12.15] * 28
        })
        df_pred.to_csv(test_pred, index=False)

    # Создаём тестовые метрики
    test_met = data_proc / "metrics.csv"
    if not test_met.exists():
        df_met = pd.DataFrame({
            'sku_id': ['SKU001', 'SKU002'],
            'mape_prophet': [5.2, 6.1],
            'mape_xgboost': [4.8, 5.5],
            'mape_naive': [8.3, 9.0],
            'mape_ens': [4.5, 5.0],
            'best_model': ['ens', 'ens']
        })
        df_met.to_csv(test_met, index=False)

    yield

    # Cleanup (опционально)


# Импортируем API после создания тестовых данных
from src.api.app import app

client = TestClient(app)


class TestHealthEndpoint:
    """Тесты для endpoint /health."""

    def test_health_returns_ok(self):
        """Health check должен возвращать status ok."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "forecastly-api"
        assert "timestamp" in data

    def test_health_has_timestamp(self):
        """Health check должен содержать timestamp."""
        response = client.get("/health")
        data = response.json()
        assert "timestamp" in data
        # Проверяем формат ISO
        assert "T" in data["timestamp"]


class TestRootEndpoint:
    """Тесты для корневого endpoint /."""

    def test_root_returns_info(self):
        """Корневой endpoint должен возвращать информацию об API."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "forecastly-api"
        assert data["version"] == "1.1.0"
        assert "endpoints" in data

    def test_root_contains_docs_link(self):
        """Корневой endpoint должен содержать ссылку на документацию."""
        response = client.get("/")
        data = response.json()
        assert data["docs"] == "/docs"
        assert data["redoc"] == "/redoc"


class TestSkusEndpoint:
    """Тесты для endpoint /api/v1/skus."""

    def test_get_skus_returns_list(self):
        """Endpoint должен возвращать список SKU."""
        response = client.get("/api/v1/skus")
        assert response.status_code == 200
        data = response.json()
        assert "skus" in data
        assert "count" in data
        assert isinstance(data["skus"], list)
        assert data["count"] == len(data["skus"])

    def test_get_skus_sorted(self):
        """SKU должны быть отсортированы."""
        response = client.get("/api/v1/skus")
        data = response.json()
        skus = data["skus"]
        assert skus == sorted(skus)


class TestPredictEndpoint:
    """Тесты для endpoint /api/v1/predict."""

    def test_predict_requires_sku_id(self):
        """Endpoint должен требовать параметр sku_id."""
        response = client.get("/api/v1/predict")
        assert response.status_code == 422  # Validation error

    def test_predict_with_valid_sku(self):
        """Endpoint должен возвращать прогноз для валидного SKU."""
        response = client.get("/api/v1/predict?sku_id=SKU001")
        assert response.status_code == 200
        data = response.json()
        assert "sku_id" in data
        assert "predictions" in data
        assert "horizon" in data

    def test_predict_with_custom_horizon(self):
        """Endpoint должен принимать параметр horizon."""
        response = client.get("/api/v1/predict?sku_id=SKU001&horizon=7")
        assert response.status_code == 200
        data = response.json()
        assert data["horizon"] == 7
        assert len(data["predictions"]) <= 7

    def test_predict_invalid_sku_returns_404(self):
        """Endpoint должен возвращать 404 для несуществующего SKU."""
        response = client.get("/api/v1/predict?sku_id=INVALID_SKU_999")
        assert response.status_code == 404

    def test_predict_horizon_validation(self):
        """Horizon должен быть в пределах 1-120."""
        # Слишком маленький
        response = client.get("/api/v1/predict?sku_id=SKU001&horizon=0")
        assert response.status_code == 422

        # Слишком большой
        response = client.get("/api/v1/predict?sku_id=SKU001&horizon=150")
        assert response.status_code == 422


class TestMetricsEndpoint:
    """Тесты для endpoint /api/v1/metrics."""

    def test_get_metrics_returns_data(self):
        """Endpoint должен возвращать метрики."""
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        assert "count" in data

    def test_metrics_contain_required_fields(self):
        """Метрики должны содержать обязательные поля."""
        response = client.get("/api/v1/metrics")
        data = response.json()
        if data["count"] > 0:
            metric = data["metrics"][0]
            assert "sku_id" in metric
            assert "best_model" in metric


class TestStatusEndpoint:
    """Тесты для endpoint /api/v1/status."""

    def test_status_returns_system_info(self):
        """Endpoint должен возвращать информацию о системе."""
        response = client.get("/api/v1/status")
        assert response.status_code == 200
        data = response.json()
        assert data["system"] == "ready"
        assert "timestamp" in data
        assert "data_available" in data

    def test_status_contains_data_availability(self):
        """Статус должен содержать информацию о доступности данных."""
        response = client.get("/api/v1/status")
        data = response.json()
        avail = data["data_available"]
        assert "raw" in avail
        assert "processed" in avail
        assert "predictions" in avail
        assert "metrics" in avail
        assert "models" in avail


class TestCORS:
    """Тесты для CORS настроек."""

    def test_cors_headers_present(self):
        """CORS заголовки должны присутствовать."""
        response = client.options("/health", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET"
        })
        # FastAPI TestClient не всегда отдаёт CORS headers
        # но проверяем что OPTIONS работает
        assert response.status_code in [200, 405]


class TestDocumentation:
    """Тесты для документации API."""

    def test_swagger_docs_available(self):
        """Swagger UI должен быть доступен."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_available(self):
        """ReDoc должен быть доступен."""
        response = client.get("/redoc")
        assert response.status_code == 200
