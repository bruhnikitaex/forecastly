"""
Общие фикстуры для тестов Forecastly.

Предоставляет:
- Тестовую базу данных (SQLite in-memory)
- Тестовые данные (DataFrame, CSV)
- Мок-пользователей и токены
- Тестовый клиент API
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Generator
import tempfile
import shutil

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

# Устанавливаем тестовое окружение
os.environ.setdefault('ENVIRONMENT', 'testing')
os.environ.setdefault('USE_DATABASE', 'false')
os.environ.setdefault('LOG_LEVEL', 'WARNING')


# ============================================================================
# DATABASE FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def test_db_url():
    """URL для тестовой SQLite базы данных."""
    return "sqlite:///./test_forecastly.db"


@pytest.fixture(scope="function")
def db_session():
    """
    Фикстура для тестовой сессии БД.

    Создаёт in-memory SQLite базу для каждого теста.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from src.db.models import Base

    # In-memory SQLite
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False}
    )

    # Создаём таблицы
    Base.metadata.create_all(bind=engine)

    # Создаём сессию
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()

    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def db_with_user(db_session):
    """
    Фикстура с БД и тестовым пользователем.
    """
    from src.db.models import User
    from src.auth.security import get_password_hash

    user = User(
        email="test@example.com",
        username="testuser",
        hashed_password=get_password_hash("TestPassword123"),
        is_active=True,
        role="analyst"
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    yield db_session, user


# ============================================================================
# TEST DATA FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def sample_sales_df():
    """Создаёт тестовый DataFrame с продажами."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=90, freq='D')

    data = []
    for sku in ['SKU001', 'SKU002', 'SKU003']:
        for date in dates:
            data.append({
                'date': date,
                'sku_id': sku,
                'store_id': 'S01',
                'units': np.random.randint(5, 50),
                'price': np.random.uniform(50, 150),
                'promo_flag': np.random.choice([0, 1], p=[0.8, 0.2])
            })

    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def sample_predictions_df():
    """Создаёт тестовый DataFrame с прогнозами."""
    dates = pd.date_range('2024-04-01', periods=14, freq='D')

    data = []
    for sku in ['SKU001', 'SKU002']:
        for date in dates:
            prophet = np.random.uniform(10, 30)
            xgb = np.random.uniform(10, 30)
            ensemble = (prophet + xgb) / 2
            data.append({
                'date': date,
                'sku_id': sku,
                'prophet': prophet,
                'xgb': xgb,
                'ensemble': ensemble,
                'p_low': ensemble * 0.8,
                'p_high': ensemble * 1.2
            })

    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def sample_metrics_df():
    """Создаёт тестовый DataFrame с метриками."""
    return pd.DataFrame({
        'sku_id': ['SKU001', 'SKU002', 'SKU003'],
        'mape_prophet': [5.2, 6.1, 7.3],
        'mape_xgboost': [4.8, 5.5, 6.2],
        'mape_naive': [8.3, 9.0, 10.5],
        'mape_ens': [4.5, 5.0, 5.8],
        'best_model': ['ens', 'ens', 'ens']
    })


@pytest.fixture(scope="function")
def temp_data_dir(sample_sales_df, sample_predictions_df, sample_metrics_df):
    """
    Создаёт временную директорию с тестовыми данными.

    Возвращает путь к директории.
    """
    temp_dir = tempfile.mkdtemp()

    # Создаём структуру директорий
    raw_dir = Path(temp_dir) / "data" / "raw"
    proc_dir = Path(temp_dir) / "data" / "processed"
    models_dir = Path(temp_dir) / "data" / "models"

    raw_dir.mkdir(parents=True)
    proc_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)

    # Сохраняем тестовые данные
    sample_sales_df.to_csv(raw_dir / "sales_synth.csv", index=False)
    sample_predictions_df.to_csv(proc_dir / "predictions.csv", index=False)
    sample_metrics_df.to_csv(proc_dir / "metrics.csv", index=False)

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


# ============================================================================
# API CLIENT FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def test_client():
    """
    Создаёт тестовый клиент FastAPI.
    """
    from fastapi.testclient import TestClient
    from src.api.app import app

    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="function")
def authenticated_client(test_client, db_with_user):
    """
    Создаёт аутентифицированный тестовый клиент.
    """
    from src.auth.security import create_access_token

    db_session, user = db_with_user
    token = create_access_token(data={"sub": str(user.id)})

    test_client.headers["Authorization"] = f"Bearer {token}"

    yield test_client, user

    # Cleanup
    test_client.headers.pop("Authorization", None)


# ============================================================================
# MOCK USER FIXTURES
# ============================================================================

@pytest.fixture
def mock_user():
    """Создаёт мок-объект пользователя."""
    class MockUser:
        def __init__(self):
            self.id = 1
            self.email = "test@example.com"
            self.username = "testuser"
            self.hashed_password = "hashed_password"
            self.is_active = True
            self.is_superuser = False
            self.role = "analyst"
            self.failed_login_attempts = 0
            self.locked_until = None
            self.last_failed_login = None
            self.last_login = None

    return MockUser()


@pytest.fixture
def mock_locked_user(mock_user):
    """Создаёт заблокированного пользователя."""
    mock_user.failed_login_attempts = 5
    mock_user.locked_until = datetime.now(timezone.utc) + timedelta(minutes=15)
    mock_user.last_failed_login = datetime.now(timezone.utc)
    return mock_user


# ============================================================================
# TOKEN FIXTURES
# ============================================================================

@pytest.fixture
def valid_access_token():
    """Создаёт валидный access token."""
    from src.auth.security import create_access_token
    return create_access_token(data={"sub": "1"})


@pytest.fixture
def valid_refresh_token():
    """Создаёт валидный refresh token."""
    from src.auth.security import create_refresh_token
    return create_refresh_token(data={"sub": "1"})


@pytest.fixture
def expired_token():
    """Создаёт истёкший token."""
    from src.auth.security import create_access_token
    return create_access_token(
        data={"sub": "1"},
        expires_delta=timedelta(seconds=-1)
    )


# ============================================================================
# UTILITY FIXTURES
# ============================================================================

@pytest.fixture
def strong_password():
    """Возвращает надёжный пароль."""
    return "SecurePassword123!"


@pytest.fixture
def weak_passwords():
    """Возвращает список слабых паролей."""
    return [
        "short",           # Слишком короткий
        "nouppercase123",  # Нет заглавных
        "NOLOWERCASE123",  # Нет строчных
        "NoDigitsHere",    # Нет цифр
        "password123",     # Распространённый
        "aaaa1111Bb",      # Повторяющиеся символы
    ]


@pytest.fixture(scope="session")
def project_root():
    """Возвращает путь к корню проекта."""
    return Path(__file__).parent.parent


# ============================================================================
# SKIP CONDITIONS
# ============================================================================

def pytest_configure(config):
    """Добавляет кастомные маркеры."""
    config.addinivalue_line(
        "markers", "db: тесты, требующие базу данных"
    )
    config.addinivalue_line(
        "markers", "slow: медленные тесты"
    )
    config.addinivalue_line(
        "markers", "integration: интеграционные тесты"
    )


@pytest.fixture
def skip_if_no_db():
    """Пропускает тест если БД не настроена."""
    if os.getenv('USE_DATABASE', 'false').lower() != 'true':
        pytest.skip("БД не настроена (USE_DATABASE != true)")


@pytest.fixture
def skip_if_no_prophet():
    """Пропускает тест если Prophet не установлен."""
    try:
        from prophet import Prophet
    except ImportError:
        pytest.skip("Prophet не установлен")
