#!/usr/bin/env python
"""
Скрипт инициализации базы данных Forecastly.

Использование:
    python -m src.db.init_db              # Создать таблицы
    python -m src.db.init_db --sync       # Синхронизировать данные из CSV
    python -m src.db.init_db --reset      # Пересоздать таблицы (УДАЛИТ ВСЕ ДАННЫЕ)

Переменные окружения:
    DATABASE_URL      - Полный URL подключения
    POSTGRES_HOST     - Хост PostgreSQL (по умолчанию localhost)
    POSTGRES_PORT     - Порт (по умолчанию 5432)
    POSTGRES_DB       - Имя базы (по умолчанию forecastly)
    POSTGRES_USER     - Пользователь (по умолчанию forecastly)
    POSTGRES_PASSWORD - Пароль
    SQLITE_PATH       - Путь к SQLite файлу (если не указан PostgreSQL)
"""

import argparse
import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import logger
from src.db.database import init_db, drop_all_tables, check_connection, DATABASE_URL
from src.db.models import Base


def main():
    parser = argparse.ArgumentParser(description='Инициализация базы данных Forecastly')
    parser.add_argument('--reset', action='store_true',
                        help='Пересоздать все таблицы (УДАЛИТ ВСЕ ДАННЫЕ!)')
    parser.add_argument('--sync', action='store_true',
                        help='Синхронизировать данные из CSV файлов')
    parser.add_argument('--check', action='store_true',
                        help='Только проверить подключение')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Инициализация базы данных Forecastly")
    logger.info("=" * 60)

    # Выводим информацию о подключении (скрываем пароль)
    if '@' in DATABASE_URL:
        safe_url = DATABASE_URL.split('@')[-1]
        logger.info(f"База данных: {safe_url}")
    else:
        logger.info(f"База данных: {DATABASE_URL}")

    # Проверяем подключение
    if not check_connection():
        logger.error("Не удалось подключиться к базе данных!")
        sys.exit(1)

    logger.info("✓ Подключение успешно")

    if args.check:
        logger.info("Режим проверки - завершение")
        return

    # Сброс базы данных
    if args.reset:
        confirm = input("ВНИМАНИЕ: Все данные будут удалены! Продолжить? (yes/no): ")
        if confirm.lower() != 'yes':
            logger.info("Операция отменена")
            return

        logger.warning("Удаление всех таблиц...")
        drop_all_tables()

    # Создание таблиц
    logger.info("Создание таблиц...")
    init_db()
    logger.info("✓ Таблицы созданы")

    # Синхронизация данных
    if args.sync:
        sync_data()

    logger.info("=" * 60)
    logger.info("Инициализация завершена успешно!")
    logger.info("=" * 60)


def sync_data():
    """Синхронизирует данные из CSV файлов в базу данных."""
    import pandas as pd
    from src.db.database import get_db_session
    from src.db import crud
    from src.utils.config import PATHS

    logger.info("Синхронизация данных из CSV...")

    DATA_RAW = Path(PATHS["data"].get("raw", "data/raw"))
    DATA_PROC = Path(PATHS["data"].get("processed", "data/processed"))

    # Нормализуем путь
    if DATA_PROC.suffix:
        DATA_PROC = DATA_PROC.parent

    with get_db_session() as db:
        # 1. Загружаем SKU из сырых данных
        raw_path = DATA_RAW / "sales_synth.csv"
        if raw_path.exists():
            logger.info(f"Загрузка SKU из {raw_path}...")
            df = pd.read_csv(raw_path)
            if "sku_id" in df.columns:
                sku_ids = df["sku_id"].unique().tolist()
                skus = crud.bulk_create_skus(db, sku_ids)
                logger.info(f"  ✓ Загружено {len(skus)} SKU")

        # 2. Загружаем прогнозы
        pred_path = DATA_PROC / "predictions.csv"
        if pred_path.exists():
            logger.info(f"Загрузка прогнозов из {pred_path}...")
            df = pd.read_csv(pred_path, parse_dates=["date"])

            run = crud.create_forecast_run(db, horizon=14)
            count = crud.bulk_create_predictions(db, df, run.id)
            crud.complete_forecast_run(db, run.run_id, count)
            logger.info(f"  ✓ Загружено {count} прогнозов")

        # 3. Загружаем метрики
        met_path = DATA_PROC / "metrics.csv"
        if met_path.exists():
            logger.info(f"Загрузка метрик из {met_path}...")
            df = pd.read_csv(met_path)
            count = crud.bulk_create_metrics(db, df)
            logger.info(f"  ✓ Загружено {count} метрик")

        # 4. Загружаем историю продаж (если нужно)
        if raw_path.exists():
            logger.info(f"Загрузка истории продаж...")
            df = pd.read_csv(raw_path, parse_dates=["date"])
            count = crud.bulk_create_sales_history(db, df)
            logger.info(f"  ✓ Загружено {count} записей продаж")

    logger.info("✓ Синхронизация завершена")


def init_database():
    """
    Простая функция инициализации БД (алиас для использования в CLI).

    Создаёт все таблицы и проверяет подключение.
    """
    if not check_connection():
        raise RuntimeError("Не удалось подключиться к базе данных!")
    init_db()
    logger.info("✓ База данных инициализирована")


if __name__ == "__main__":
    main()
