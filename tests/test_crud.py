"""
Тесты для CRUD операций с базой данных.

Покрывает:
- SKU операции
- Prediction операции
- ForecastRun операции
- Metric операции
- User операции
"""

import pytest
import pandas as pd
from datetime import datetime, date, timezone, timedelta

from src.db.models import SKU, Prediction, ForecastRun, Metric, User, APIKey
from src.db import crud


def bcrypt_available():
    """Проверяет доступность bcrypt."""
    try:
        from passlib.hash import bcrypt
        bcrypt.hash("test")
        return True
    except Exception:
        return False


class TestSKUCrud:
    """Тесты для CRUD операций с SKU."""

    def test_create_sku(self, db_session):
        """Создание нового SKU."""
        sku = crud.get_or_create_sku(db_session, "SKU001")

        assert sku is not None
        assert sku.sku_id == "SKU001"
        assert sku.is_active is True

    def test_get_sku(self, db_session):
        """Получение SKU по идентификатору."""
        # Создаём SKU
        crud.get_or_create_sku(db_session, "SKU002")

        # Получаем
        sku = crud.get_sku(db_session, "SKU002")

        assert sku is not None
        assert sku.sku_id == "SKU002"

    def test_get_sku_case_insensitive(self, db_session):
        """Получение SKU регистронезависимо."""
        crud.get_or_create_sku(db_session, "SKU003")

        # Пробуем разный регистр
        sku = crud.get_sku(db_session, "sku003")
        assert sku is not None

    def test_get_nonexistent_sku(self, db_session):
        """Получение несуществующего SKU."""
        sku = crud.get_sku(db_session, "NONEXISTENT")
        assert sku is None

    def test_get_all_skus(self, db_session):
        """Получение списка всех SKU."""
        # Создаём несколько SKU
        crud.get_or_create_sku(db_session, "SKU010")
        crud.get_or_create_sku(db_session, "SKU011")
        crud.get_or_create_sku(db_session, "SKU012")

        skus = crud.get_all_skus(db_session)

        assert len(skus) >= 3
        sku_ids = [s.sku_id for s in skus]
        assert "SKU010" in sku_ids
        assert "SKU011" in sku_ids

    def test_get_all_skus_with_limit(self, db_session):
        """Получение SKU с лимитом."""
        for i in range(10):
            crud.get_or_create_sku(db_session, f"SKU1{i:02d}")

        skus = crud.get_all_skus(db_session, limit=5)
        assert len(skus) == 5

    def test_get_or_create_returns_existing(self, db_session):
        """get_or_create возвращает существующий SKU."""
        sku1 = crud.get_or_create_sku(db_session, "SKU020")
        sku2 = crud.get_or_create_sku(db_session, "SKU020")

        assert sku1.id == sku2.id


class TestForecastRunCrud:
    """Тесты для CRUD операций с ForecastRun."""

    def test_create_forecast_run(self, db_session):
        """Создание записи о запуске прогноза."""
        run = crud.create_forecast_run(db_session, horizon=14)

        assert run is not None
        assert run.run_id is not None
        assert run.horizon == 14
        assert run.status == "running"

    def test_complete_forecast_run(self, db_session):
        """Завершение запуска прогноза."""
        run = crud.create_forecast_run(db_session, horizon=7)
        crud.complete_forecast_run(db_session, run.run_id, records_count=100)

        # Получаем обновлённый run
        db_session.refresh(run)

        assert run.status == "completed"
        assert run.records_count == 100
        assert run.completed_at is not None

    def test_fail_forecast_run(self, db_session):
        """Провал запуска прогноза."""
        run = crud.create_forecast_run(db_session, horizon=14)
        crud.fail_forecast_run(db_session, run.run_id, "Test error message")

        db_session.refresh(run)

        assert run.status == "failed"
        assert run.error_message == "Test error message"

    def test_get_forecast_runs(self, db_session):
        """Получение списка запусков."""
        for i in range(5):
            crud.create_forecast_run(db_session, horizon=14)

        runs = crud.get_forecast_runs(db_session)

        assert len(runs) >= 5

    def test_get_forecast_runs_with_pagination(self, db_session):
        """Пагинация списка запусков."""
        for i in range(10):
            crud.create_forecast_run(db_session, horizon=14)

        runs = crud.get_forecast_runs(db_session, skip=0, limit=5)
        assert len(runs) == 5


class TestPredictionCrud:
    """Тесты для CRUD операций с Prediction."""

    def test_bulk_create_predictions(self, db_session, sample_predictions_df):
        """Массовое создание прогнозов."""
        run = crud.create_forecast_run(db_session, horizon=14)

        count = crud.bulk_create_predictions(
            db_session,
            sample_predictions_df,
            run.id
        )

        # Завершаем run чтобы он был доступен
        crud.complete_forecast_run(db_session, run.run_id, count)

        assert count > 0

    def test_get_predictions_by_sku(self, db_session, sample_predictions_df):
        """Получение прогнозов по SKU."""
        run = crud.create_forecast_run(db_session, horizon=14)
        count = crud.bulk_create_predictions(db_session, sample_predictions_df, run.id)

        # Завершаем run чтобы он был виден как 'completed'
        crud.complete_forecast_run(db_session, run.run_id, count)

        predictions = crud.get_predictions_by_sku(db_session, "SKU001", horizon=10)

        assert len(predictions) > 0
        assert all(p.sku.sku_id == "SKU001" for p in predictions)

    def test_get_predictions_by_sku_with_horizon(self, db_session, sample_predictions_df):
        """Получение прогнозов с ограничением горизонта."""
        run = crud.create_forecast_run(db_session, horizon=14)
        count = crud.bulk_create_predictions(db_session, sample_predictions_df, run.id)

        # Завершаем run
        crud.complete_forecast_run(db_session, run.run_id, count)

        predictions = crud.get_predictions_by_sku(db_session, "SKU001", horizon=7)

        assert len(predictions) <= 7


class TestMetricCrud:
    """Тесты для CRUD операций с Metric."""

    def test_bulk_create_metrics(self, db_session, sample_metrics_df):
        """Массовое создание метрик."""
        count = crud.bulk_create_metrics(db_session, sample_metrics_df)
        assert count == len(sample_metrics_df)

    def test_get_metrics(self, db_session, sample_metrics_df):
        """Получение метрик."""
        crud.bulk_create_metrics(db_session, sample_metrics_df)

        metrics = crud.get_metrics(db_session)

        assert len(metrics) > 0
        assert "sku_id" in metrics[0]


class TestDatabaseStats:
    """Тесты для статистики БД."""

    def test_get_database_stats(self, db_session):
        """Получение статистики базы данных."""
        # Создаём тестовые данные
        crud.get_or_create_sku(db_session, "SKU100")
        crud.get_or_create_sku(db_session, "SKU101")
        crud.create_forecast_run(db_session, horizon=14)

        stats = crud.get_database_stats(db_session)

        assert "skus" in stats
        assert "predictions" in stats
        assert "forecast_runs" in stats
        assert stats["skus"] >= 2


@pytest.mark.skipif(not bcrypt_available(), reason="bcrypt backend не установлен")
class TestUserCrud:
    """Тесты для CRUD операций с User (если есть в crud.py)."""

    def test_create_user(self, db_session):
        """Создание пользователя напрямую через модель."""
        from src.auth.security import get_password_hash

        user = User(
            email="newuser@example.com",
            username="newuser",
            hashed_password=get_password_hash("Password123"),
            is_active=True,
            role="viewer"
        )
        db_session.add(user)
        db_session.commit()

        assert user.id is not None
        assert user.email == "newuser@example.com"

    def test_user_unique_email(self, db_session):
        """Email должен быть уникальным."""
        from src.auth.security import get_password_hash
        from sqlalchemy.exc import IntegrityError

        user1 = User(
            email="unique@example.com",
            hashed_password=get_password_hash("Password123")
        )
        db_session.add(user1)
        db_session.commit()

        user2 = User(
            email="unique@example.com",
            hashed_password=get_password_hash("Password456")
        )
        db_session.add(user2)

        with pytest.raises(IntegrityError):
            db_session.commit()

        db_session.rollback()

    def test_user_is_locked_property(self, db_session):
        """Проверка свойства is_locked."""
        from src.auth.security import get_password_hash

        user = User(
            email="locktest@example.com",
            hashed_password=get_password_hash("Password123")
        )
        db_session.add(user)
        db_session.commit()

        # Не заблокирован
        assert user.is_locked is False

        # Блокируем
        user.locked_until = datetime.now(timezone.utc) + timedelta(minutes=15)
        db_session.commit()

        assert user.is_locked is True

        # Блокировка истекла
        user.locked_until = datetime.now(timezone.utc) - timedelta(minutes=1)
        db_session.commit()

        assert user.is_locked is False


@pytest.mark.skipif(not bcrypt_available(), reason="bcrypt backend не установлен")
class TestAPIKeyCrud:
    """Тесты для CRUD операций с API ключами."""

    def test_create_api_key(self, db_session):
        """Создание API ключа."""
        from src.auth.security import get_password_hash, generate_api_key, hash_api_key

        # Создаём пользователя
        user = User(
            email="apiuser@example.com",
            hashed_password=get_password_hash("Password123")
        )
        db_session.add(user)
        db_session.commit()

        # Создаём API ключ
        raw_key, key_hash = generate_api_key()

        api_key = APIKey(
            key_hash=key_hash,
            key_prefix=raw_key[6:14],
            name="Test Key",
            user_id=user.id,
            is_active=True
        )
        db_session.add(api_key)
        db_session.commit()

        assert api_key.id is not None
        assert api_key.key_hash == key_hash

    def test_api_key_verification(self, db_session):
        """Проверка API ключа."""
        from src.auth.security import get_password_hash, generate_api_key, hash_api_key

        user = User(
            email="verifyuser@example.com",
            hashed_password=get_password_hash("Password123")
        )
        db_session.add(user)
        db_session.commit()

        raw_key, key_hash = generate_api_key()

        api_key = APIKey(
            key_hash=key_hash,
            key_prefix=raw_key[6:14],
            name="Verify Key",
            user_id=user.id
        )
        db_session.add(api_key)
        db_session.commit()

        # Проверяем что хэш совпадает
        assert hash_api_key(raw_key) == key_hash


class TestRelationships:
    """Тесты для связей между моделями."""

    def test_sku_predictions_relationship(self, db_session, sample_predictions_df):
        """Связь SKU -> Predictions."""
        run = crud.create_forecast_run(db_session, horizon=14)
        crud.bulk_create_predictions(db_session, sample_predictions_df, run.id)

        sku = crud.get_sku(db_session, "SKU001")
        assert sku is not None
        assert len(sku.predictions) > 0

    def test_forecast_run_predictions_relationship(self, db_session, sample_predictions_df):
        """Связь ForecastRun -> Predictions."""
        run = crud.create_forecast_run(db_session, horizon=14)
        crud.bulk_create_predictions(db_session, sample_predictions_df, run.id)

        db_session.refresh(run)
        assert len(run.predictions) > 0

    @pytest.mark.skipif(not bcrypt_available(), reason="bcrypt backend не установлен")
    def test_user_api_keys_relationship(self, db_session):
        """Связь User -> APIKeys."""
        from src.auth.security import get_password_hash, generate_api_key

        user = User(
            email="reluser@example.com",
            hashed_password=get_password_hash("Password123")
        )
        db_session.add(user)
        db_session.commit()

        for i in range(3):
            raw_key, key_hash = generate_api_key()
            api_key = APIKey(
                key_hash=key_hash,
                key_prefix=raw_key[6:14],
                name=f"Key {i}",
                user_id=user.id
            )
            db_session.add(api_key)

        db_session.commit()
        db_session.refresh(user)

        assert len(user.api_keys) == 3

    def test_cascade_delete(self, db_session):
        """Каскадное удаление при удалении SKU."""
        # Создаём SKU с прогнозами
        sku = crud.get_or_create_sku(db_session, "SKU_DELETE")
        run = crud.create_forecast_run(db_session, horizon=7)

        pred = Prediction(
            sku_id=sku.id,
            forecast_run_id=run.id,
            date=date(2024, 1, 1),
            prophet=10.0,
            xgb=11.0,
            ensemble=10.5
        )
        db_session.add(pred)
        db_session.commit()

        # Удаляем SKU
        db_session.delete(sku)
        db_session.commit()

        # Прогноз тоже должен быть удалён
        remaining = db_session.query(Prediction).filter(
            Prediction.sku_id == sku.id
        ).count()

        assert remaining == 0


# ============================================================================
# Запуск тестов
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
