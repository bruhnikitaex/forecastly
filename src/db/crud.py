"""
CRUD операции для работы с базой данных Forecastly.

Предоставляет функции для создания, чтения, обновления и удаления данных.
"""

from datetime import datetime, date, timezone
from typing import Optional
from uuid import uuid4
import pandas as pd

from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from src.db.models import SKU, Prediction, ForecastRun, Metric, SalesHistory
from src.utils.logger import logger


# ============================================================================
# SKU CRUD
# ============================================================================

def get_sku(db: Session, sku_id: str) -> Optional[SKU]:
    """Получить SKU по идентификатору."""
    return db.query(SKU).filter(SKU.sku_id == sku_id.upper()).first()


def get_sku_by_id(db: Session, id: int) -> Optional[SKU]:
    """Получить SKU по первичному ключу."""
    return db.query(SKU).filter(SKU.id == id).first()


def get_all_skus(db: Session, skip: int = 0, limit: int = 100) -> list[SKU]:
    """Получить список всех активных SKU."""
    return db.query(SKU).filter(SKU.is_active.is_(True)).offset(skip).limit(limit).all()


def get_sku_count(db: Session) -> int:
    """Получить количество SKU."""
    return db.query(func.count(SKU.id)).filter(SKU.is_active.is_(True)).scalar()


def create_sku(db: Session, sku_id: str, name: str = None,
               category: str = None, store_id: str = 'default') -> SKU:
    """Создать новый SKU."""
    sku = SKU(
        sku_id=sku_id.upper(),
        name=name,
        category=category,
        store_id=store_id
    )
    db.add(sku)
    db.commit()
    db.refresh(sku)
    logger.info(f"Создан SKU: {sku_id}")
    return sku


def get_or_create_sku(db: Session, sku_id: str, **kwargs) -> SKU:
    """Получить существующий или создать новый SKU."""
    sku = get_sku(db, sku_id)
    if not sku:
        sku = create_sku(db, sku_id, **kwargs)
    return sku


def bulk_create_skus(db: Session, sku_ids: list[str]) -> list[SKU]:
    """Массовое создание SKU."""
    created = []
    for sku_id in sku_ids:
        sku = get_or_create_sku(db, sku_id)
        created.append(sku)
    return created


# ============================================================================
# ForecastRun CRUD
# ============================================================================

def create_forecast_run(db: Session, horizon: int = 14,
                        model_type: str = 'ensemble') -> ForecastRun:
    """Создать новый запуск прогнозирования."""
    run = ForecastRun(
        run_id=str(uuid4()),
        horizon=horizon,
        model_type=model_type,
        status='running',
        started_at=datetime.now(timezone.utc)
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    logger.info(f"Создан ForecastRun: {run.run_id}")
    return run


def complete_forecast_run(db: Session, run_id: str,
                          records_count: int = 0) -> Optional[ForecastRun]:
    """Отметить запуск как завершённый."""
    run = db.query(ForecastRun).filter(ForecastRun.run_id == run_id).first()
    if run:
        run.status = 'completed'
        run.completed_at = datetime.now(timezone.utc)
        run.records_count = records_count
        db.commit()
        logger.info(f"ForecastRun завершён: {run_id}, записей: {records_count}")
    return run


def fail_forecast_run(db: Session, run_id: str,
                      error_message: str) -> Optional[ForecastRun]:
    """Отметить запуск как неудачный."""
    run = db.query(ForecastRun).filter(ForecastRun.run_id == run_id).first()
    if run:
        run.status = 'failed'
        run.completed_at = datetime.now(timezone.utc)
        run.error_message = error_message
        db.commit()
        logger.error(f"ForecastRun провален: {run_id}, ошибка: {error_message}")
    return run


def get_latest_forecast_run(db: Session) -> Optional[ForecastRun]:
    """Получить последний успешный запуск прогнозирования."""
    return db.query(ForecastRun)\
        .filter(ForecastRun.status == 'completed')\
        .order_by(desc(ForecastRun.completed_at))\
        .first()


def get_forecast_runs(db: Session, skip: int = 0, limit: int = 20) -> list[ForecastRun]:
    """Получить историю запусков прогнозирования."""
    return db.query(ForecastRun)\
        .order_by(desc(ForecastRun.started_at))\
        .offset(skip).limit(limit).all()


# ============================================================================
# Prediction CRUD
# ============================================================================

def create_prediction(db: Session, sku_id: int, forecast_run_id: int,
                      prediction_date: date, prophet: float = None,
                      xgb: float = None, ensemble: float = None,
                      p_low: float = None, p_high: float = None) -> Prediction:
    """Создать запись прогноза."""
    pred = Prediction(
        sku_id=sku_id,
        forecast_run_id=forecast_run_id,
        date=prediction_date,
        prophet=prophet,
        xgb=xgb,
        ensemble=ensemble,
        p_low=p_low,
        p_high=p_high
    )
    db.add(pred)
    return pred


def bulk_create_predictions(db: Session, predictions_df: pd.DataFrame,
                            forecast_run_id: int) -> int:
    """
    Массовое создание прогнозов из DataFrame.

    Args:
        db: Сессия базы данных
        predictions_df: DataFrame с колонками: sku_id, date, prophet, xgb, ensemble, p_low, p_high
        forecast_run_id: ID запуска прогнозирования

    Returns:
        Количество созданных записей
    """
    count = 0
    for _, row in predictions_df.iterrows():
        # Получаем или создаём SKU
        sku = get_or_create_sku(db, row['sku_id'])

        # Создаём прогноз
        pred = Prediction(
            sku_id=sku.id,
            forecast_run_id=forecast_run_id,
            date=row['date'] if isinstance(row['date'], date) else pd.to_datetime(row['date']).date(),
            prophet=row.get('prophet'),
            xgb=row.get('xgb'),
            ensemble=row.get('ensemble'),
            p_low=row.get('p_low'),
            p_high=row.get('p_high')
        )
        db.add(pred)
        count += 1

    db.commit()
    logger.info(f"Создано {count} прогнозов для run_id={forecast_run_id}")
    return count


def get_predictions_by_sku(db: Session, sku_id: str,
                           horizon: int = 14,
                           run_id: int = None) -> list[Prediction]:
    """
    Получить прогнозы по SKU.

    Args:
        db: Сессия базы данных
        sku_id: Идентификатор SKU
        horizon: Количество дней
        run_id: ID конкретного запуска (опционально, по умолчанию - последний)

    Returns:
        Список прогнозов
    """
    sku = get_sku(db, sku_id)
    if not sku:
        return []

    query = db.query(Prediction).filter(Prediction.sku_id == sku.id)

    if run_id:
        query = query.filter(Prediction.forecast_run_id == run_id)
    else:
        # Берём последний успешный запуск
        latest_run = get_latest_forecast_run(db)
        if latest_run:
            query = query.filter(Prediction.forecast_run_id == latest_run.id)

    return query.order_by(Prediction.date).limit(horizon).all()


def get_predictions_dataframe(db: Session, sku_id: str = None,
                              run_id: int = None) -> pd.DataFrame:
    """
    Получить прогнозы как DataFrame.

    Args:
        db: Сессия базы данных
        sku_id: Идентификатор SKU (опционально)
        run_id: ID запуска (опционально)

    Returns:
        DataFrame с прогнозами
    """
    query = db.query(
        SKU.sku_id,
        Prediction.date,
        Prediction.prophet,
        Prediction.xgb,
        Prediction.ensemble,
        Prediction.p_low,
        Prediction.p_high
    ).join(SKU)

    if sku_id:
        query = query.filter(SKU.sku_id == sku_id.upper())

    if run_id:
        query = query.filter(Prediction.forecast_run_id == run_id)
    else:
        latest_run = get_latest_forecast_run(db)
        if latest_run:
            query = query.filter(Prediction.forecast_run_id == latest_run.id)

    results = query.order_by(SKU.sku_id, Prediction.date).all()

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results, columns=['sku_id', 'date', 'prophet', 'xgb', 'ensemble', 'p_low', 'p_high'])


# ============================================================================
# Metric CRUD
# ============================================================================

def create_metric(db: Session, sku_id: int, forecast_run_id: int = None,
                  mape_prophet: float = None, mape_xgboost: float = None,
                  mape_ensemble: float = None, mape_naive: float = None,
                  best_model: str = None) -> Metric:
    """Создать запись метрики."""
    metric = Metric(
        sku_id=sku_id,
        forecast_run_id=forecast_run_id,
        mape_prophet=mape_prophet,
        mape_xgboost=mape_xgboost,
        mape_ensemble=mape_ensemble,
        mape_naive=mape_naive,
        best_model=best_model
    )
    db.add(metric)
    db.commit()
    db.refresh(metric)
    return metric


def bulk_create_metrics(db: Session, metrics_df: pd.DataFrame,
                        forecast_run_id: int = None) -> int:
    """
    Массовое создание метрик из DataFrame.

    Args:
        db: Сессия базы данных
        metrics_df: DataFrame с колонками: sku_id, mape_prophet, mape_xgboost, mape_ens, mape_naive, best_model
        forecast_run_id: ID запуска прогнозирования

    Returns:
        Количество созданных записей
    """
    count = 0
    for _, row in metrics_df.iterrows():
        sku = get_or_create_sku(db, row['sku_id'])

        metric = Metric(
            sku_id=sku.id,
            forecast_run_id=forecast_run_id,
            mape_prophet=row.get('mape_prophet'),
            mape_xgboost=row.get('mape_xgboost'),
            mape_ensemble=row.get('mape_ens'),
            mape_naive=row.get('mape_naive'),
            best_model=row.get('best_model')
        )
        db.add(metric)
        count += 1

    db.commit()
    logger.info(f"Создано {count} метрик")
    return count


def get_metrics(db: Session, run_id: int = None) -> list[dict]:
    """
    Получить метрики качества.

    Args:
        db: Сессия базы данных
        run_id: ID запуска (опционально)

    Returns:
        Список словарей с метриками
    """
    query = db.query(
        SKU.sku_id,
        Metric.mape_prophet,
        Metric.mape_xgboost,
        Metric.mape_ensemble,
        Metric.mape_naive,
        Metric.best_model,
        Metric.created_at
    ).join(SKU)

    if run_id:
        query = query.filter(Metric.forecast_run_id == run_id)

    results = query.order_by(SKU.sku_id).all()

    return [
        {
            'sku_id': r.sku_id,
            'mape_prophet': r.mape_prophet,
            'mape_xgboost': r.mape_xgboost,
            'mape_ens': r.mape_ensemble,
            'mape_naive': r.mape_naive,
            'best_model': r.best_model,
            'created_at': r.created_at.isoformat() if r.created_at else None
        }
        for r in results
    ]


# ============================================================================
# SalesHistory CRUD
# ============================================================================

def bulk_create_sales_history(db: Session, sales_df: pd.DataFrame) -> int:
    """
    Массовая загрузка истории продаж.

    Args:
        db: Сессия базы данных
        sales_df: DataFrame с колонками: sku_id, date, units, revenue, price, promo_flag

    Returns:
        Количество загруженных записей
    """
    count = 0
    for _, row in sales_df.iterrows():
        sku = get_or_create_sku(db, row['sku_id'])

        sale = SalesHistory(
            sku_id=sku.id,
            date=row['date'] if isinstance(row['date'], date) else pd.to_datetime(row['date']).date(),
            units=row.get('units', 0),
            revenue=row.get('revenue'),
            price=row.get('price'),
            promo_flag=row.get('promo_flag', False)
        )
        db.add(sale)
        count += 1

        # Commit каждые 1000 записей для экономии памяти
        if count % 1000 == 0:
            db.commit()

    db.commit()
    logger.info(f"Загружено {count} записей истории продаж")
    return count


def get_sales_history(db: Session, sku_id: str = None,
                      start_date: date = None,
                      end_date: date = None) -> pd.DataFrame:
    """
    Получить историю продаж.

    Args:
        db: Сессия базы данных
        sku_id: Идентификатор SKU (опционально)
        start_date: Начальная дата (опционально)
        end_date: Конечная дата (опционально)

    Returns:
        DataFrame с историей продаж
    """
    query = db.query(
        SKU.sku_id,
        SalesHistory.date,
        SalesHistory.units,
        SalesHistory.revenue,
        SalesHistory.price,
        SalesHistory.promo_flag
    ).join(SKU)

    if sku_id:
        query = query.filter(SKU.sku_id == sku_id.upper())
    if start_date:
        query = query.filter(SalesHistory.date >= start_date)
    if end_date:
        query = query.filter(SalesHistory.date <= end_date)

    results = query.order_by(SKU.sku_id, SalesHistory.date).all()

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results, columns=['sku_id', 'date', 'units', 'revenue', 'price', 'promo_flag'])


# ============================================================================
# Utility functions
# ============================================================================

def get_database_stats(db: Session) -> dict:
    """Получить статистику базы данных."""
    return {
        'skus': db.query(func.count(SKU.id)).scalar(),
        'predictions': db.query(func.count(Prediction.id)).scalar(),
        'forecast_runs': db.query(func.count(ForecastRun.id)).scalar(),
        'metrics': db.query(func.count(Metric.id)).scalar(),
        'sales_records': db.query(func.count(SalesHistory.id)).scalar()
    }
