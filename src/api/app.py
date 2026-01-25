"""
REST API для системы прогнозирования продаж Forecastly.

Предоставляет endpoints для:
- Получения списка SKU
- Получения прогнозов по SKU
- Пересчёта прогнозов
- Получения метрик качества
- Управления базой данных

Версионирование: /api/v1/...
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.utils.config import PATHS
from src.utils.logger import logger
from src.api.exceptions import (
    register_exception_handlers,
    SKUNotFoundException,
    PredictionNotFoundException,
    ForecastGenerationException,
    DatabaseException,
)
from src.monitoring import get_metrics_collector
from src.api.middleware import MetricsMiddleware, RequestLoggingMiddleware

# Инициализация rate limiter
# Лимиты настраиваются через RATE_LIMIT env variable (формат: "100/minute")
_rate_limit = os.getenv('RATE_LIMIT', '100/minute')
limiter = Limiter(key_func=get_remote_address, default_limits=[_rate_limit])

# Проверяем, включена ли база данных
USE_DATABASE = os.getenv('USE_DATABASE', 'false').lower() == 'true'

if USE_DATABASE:
    from src.db.database import get_db, init_db, check_connection, DATABASE_URL
    from src.db import crud
    # Импортируем auth router только если БД включена
    from src.auth.router import router as auth_router, auth_limiter

# конфиг
try:
    DATA_RAW = Path(PATHS["data"].get("raw", "data/raw"))
    DATA_PROC = Path(PATHS["data"].get("processed", "data/processed"))
except Exception:
    # fallback
    DATA_RAW = Path("data/raw")
    DATA_PROC = Path("data/processed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager для инициализации и завершения приложения."""
    # Startup
    logger.info("Запуск Forecastly API...")
    if USE_DATABASE:
        logger.info(f"Режим базы данных включён: {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else 'SQLite'}")
        try:
            init_db()
            logger.info("✓ База данных инициализирована")
        except Exception as e:
            logger.error(f"Ошибка инициализации БД: {e}")
    else:
        logger.info("Режим файловой системы (CSV/Parquet)")

    yield

    # Shutdown (если нужна очистка ресурсов)
    logger.info("Остановка Forecastly API...")


app = FastAPI(
    title="Forecastly API",
    description="API для системы анализа и прогнозирования продаж",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Регистрация exception handlers
register_exception_handlers(app)

# Middleware (порядок важен - выполняются в обратном порядке)
app.add_middleware(MetricsMiddleware)
app.add_middleware(RequestLoggingMiddleware)

# Подключаем rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Подключаем auth router (только если БД включена)
if USE_DATABASE:
    app.include_router(auth_router, prefix="/api/v1")
    # Добавляем auth limiter state
    app.state.auth_limiter = auth_limiter

# CORS middleware - настраивается через переменные окружения
# В production задайте CORS_ORIGINS="https://yourdomain.com,https://app.yourdomain.com"
_cors_origins_env = os.getenv('CORS_ORIGINS', '')
_environment = os.getenv('ENVIRONMENT', 'development').lower()

if _cors_origins_env:
    # Используем заданные origins
    _cors_origins = [origin.strip() for origin in _cors_origins_env.split(',') if origin.strip()]
    _cors_credentials = True
elif _environment == 'production':
    # В production без явного указания - только localhost (безопасно)
    _cors_origins = ["http://localhost:8501", "http://localhost:3000"]
    _cors_credentials = True
    logger.warning("CORS_ORIGINS не задан в production. Используются только localhost origins.")
else:
    # В development разрешаем всё
    # Примечание: wildcard "*" несовместим с credentials=True по спецификации CORS
    _cors_origins = ["*"]
    _cors_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)


def _normalize_processed_path(p: Path) -> Path:
    """
    Нормализует путь к директории обработанных данных.
    """
    if p.suffix:
        return p.parent
    return p


def _clean_json(df: pd.DataFrame) -> list[dict]:
    """
    Преобразует DataFrame в JSON-safe список словарей.
    """
    recs = df.to_dict(orient="records")
    cleaned = []
    for r in recs:
        for k, v in list(r.items()):
            if isinstance(v, float):
                if np.isnan(v) or np.isinf(v):
                    r[k] = None
            # Преобразуем даты в строки
            elif hasattr(v, 'isoformat'):
                r[k] = v.isoformat() if hasattr(v, 'isoformat') else str(v)
        cleaned.append(r)
    return cleaned


def get_optional_db():
    """Dependency для опциональной работы с БД."""
    if USE_DATABASE:
        from src.db.database import SessionLocal
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
    else:
        yield None


# ============================================================================
# HEALTH & STATUS
# ============================================================================

@app.get("/health")
def health():
    """Проверка статуса API (health check)."""
    logger.info('Health check запрос')

    status = {
        "status": "ok",
        "service": "forecastly-api",
        "version": "1.1.0",
        "timestamp": datetime.now().isoformat(),
        "database_mode": USE_DATABASE
    }

    if USE_DATABASE:
        status["database_connected"] = check_connection()

    return status


@app.get("/", tags=["Root"])
def root():
    """Корневой endpoint с информацией об API."""
    return {
        "service": "forecastly-api",
        "version": "1.1.0",
        "database_mode": USE_DATABASE,
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "health": "/health",
            "skus": "/api/v1/skus",
            "predict": "/api/v1/predict?sku_id=SKU001&horizon=14",
            "rebuild": "/api/v1/predict/rebuild?horizon=14",
            "metrics": "/api/v1/metrics",
            "status": "/api/v1/status",
            "db_stats": "/api/v1/db/stats" if USE_DATABASE else None,
            "forecast_runs": "/api/v1/forecast-runs" if USE_DATABASE else None
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/v1/status", tags=["System"])
def system_status(db: Session = Depends(get_optional_db)):
    """Возвращает статус системы и информацию о доступных данных."""
    try:
        DATA_PROC_ = _normalize_processed_path(DATA_PROC)

        status = {
            "system": "ready",
            "timestamp": datetime.now().isoformat(),
            "database_mode": USE_DATABASE,
            "data_available": {
                "raw": (DATA_RAW / "sales_synth.csv").exists(),
                "processed": (DATA_PROC_ / "processed.parquet").exists(),
                "predictions": (DATA_PROC_ / "predictions.csv").exists(),
                "metrics": (DATA_PROC_ / "metrics.csv").exists(),
                "models": {
                    "prophet": (Path(PATHS['data']['models_dir']) / 'prophet_model.pkl').exists(),
                    "xgboost": (Path(PATHS['data']['models_dir']) / 'xgboost_model.pkl').exists()
                }
            }
        }

        # Добавляем статистику БД если включена
        if USE_DATABASE and db:
            try:
                status["database"] = crud.get_database_stats(db)
            except Exception as e:
                status["database"] = {"error": str(e)}

        logger.info('✓ Запрос статуса системы выполнен')
        return status

    except Exception as e:
        logger.error(f"Ошибка при получении статуса: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка: {str(e)}")


# ============================================================================
# SKU
# ============================================================================

@app.get("/api/v1/skus", tags=["SKU"])
def get_skus_v1(db: Session = Depends(get_optional_db)):
    """Возвращает список уникальных SKU."""
    try:
        skus = set()

        # Если БД включена, пробуем получить из неё
        if USE_DATABASE and db:
            try:
                db_skus = crud.get_all_skus(db, limit=1000)
                if db_skus:
                    skus.update([s.sku_id for s in db_skus])
                    logger.info(f"Загружено {len(skus)} SKU из БД")
            except Exception as e:
                logger.warning(f"Ошибка чтения SKU из БД: {e}")

        # Fallback на CSV файлы
        if not skus:
            DATA_PROC_ = _normalize_processed_path(DATA_PROC)
            pred_path = DATA_PROC_ / "predictions.csv"
            raw_path = DATA_RAW / "sales_synth.csv"

            if raw_path.exists():
                try:
                    df_raw = pd.read_csv(raw_path)
                    if "sku_id" in df_raw.columns:
                        skus.update(df_raw["sku_id"].astype(str).tolist())
                except Exception as e:
                    logger.warning(f"Ошибка чтения сырых данных: {str(e)}")

            if pred_path.exists():
                try:
                    df_pred = pd.read_csv(pred_path)
                    if "sku_id" in df_pred.columns:
                        skus.update(df_pred["sku_id"].astype(str).tolist())
                except Exception as e:
                    logger.warning(f"Ошибка чтения прогнозов: {str(e)}")

        if not skus:
            logger.warning('Нет доступных SKU')
            raise HTTPException(
                status_code=404,
                detail="Нет доступных SKU. Запустите ETL процесс."
            )

        logger.info(f"✓ Запрос SKU выполнен, найдено {len(skus)} SKU")
        return {"skus": sorted(list(skus)), "count": len(skus)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при получении SKU: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")


# ============================================================================
# PREDICTIONS
# ============================================================================

@app.get("/api/v1/predict", tags=["Predictions"])
def get_predict_v1(
    sku_id: str = Query(..., description="SKU товара (например SKU001)"),
    horizon: int = Query(14, ge=1, le=120, description="Горизонт прогноза в днях"),
    db: Session = Depends(get_optional_db)
):
    """Возвращает прогноз по конкретному SKU."""
    try:
        sku_id_norm = sku_id.strip().upper().replace("SKU_", "SKU").replace("SKU-", "SKU")

        # Пробуем получить из БД
        if USE_DATABASE and db:
            try:
                predictions = crud.get_predictions_by_sku(db, sku_id_norm, horizon)
                if predictions:
                    result = []
                    for p in predictions:
                        result.append({
                            "date": p.date.isoformat(),
                            "prophet": p.prophet,
                            "xgb": p.xgb,
                            "ensemble": p.ensemble,
                            "p_low": p.p_low,
                            "p_high": p.p_high
                        })
                    logger.info(f"✓ Прогноз из БД для {sku_id_norm}, {len(result)} записей")
                    return {
                        "sku_id": sku_id_norm,
                        "horizon": horizon,
                        "count": len(result),
                        "source": "database",
                        "predictions": result
                    }
            except Exception as e:
                logger.warning(f"Ошибка чтения прогнозов из БД: {e}")

        # Fallback на CSV
        DATA_PROC_ = _normalize_processed_path(DATA_PROC)
        pred_path = DATA_PROC_ / "predictions.csv"

        if not pred_path.exists():
            logger.error(f'Файл predictions.csv не найден по пути {pred_path}')
            raise PredictionNotFoundException("Прогнозы не найдены. Запустите процесс прогнозирования.")

        df = pd.read_csv(pred_path, parse_dates=["date"])
        df_sku = df[df["sku_id"].astype(str).str.upper() == sku_id_norm].copy()

        if df_sku.empty:
            logger.warning(f"SKU '{sku_id_norm}' не найден в прогнозах")
            available_skus = df["sku_id"].unique().tolist()
            raise SKUNotFoundException(sku_id_norm, available_skus)

        df_sku = df_sku.sort_values("date").head(horizon)

        logger.info(f"✓ Прогноз из CSV для {sku_id_norm}, {len(df_sku)} записей")
        return {
            "sku_id": sku_id_norm,
            "horizon": horizon,
            "count": len(df_sku),
            "source": "csv",
            "predictions": _clean_json(df_sku)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при получении прогноза: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")


@app.post("/api/v1/predict/rebuild", tags=["Predictions"])
def rebuild_predict_v1(
    horizon: int = Query(14, ge=1, le=120),
    save_to_db: bool = Query(False, description="Сохранить результаты в БД"),
    db: Session = Depends(get_optional_db)
):
    """Пересчитывает прогнозы."""
    logger.info(f'Запрос на пересчёт прогнозов с горизонтом {horizon} дней')

    try:
        cmd = ["python", "-m", "src.models.predict", "--horizon", str(horizon)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            logger.error(f'Ошибка при пересчёте прогнозов: {result.stderr}')
            raise ForecastGenerationException(
                message="Ошибка при пересчёте прогнозов",
                stderr=result.stderr
            )

        response = {
            "status": "ok",
            "message": "Прогнозы пересчитаны успешно",
            "horizon": horizon,
            "timestamp": datetime.now().isoformat()
        }

        # Сохраняем в БД если запрошено
        if save_to_db and USE_DATABASE and db:
            try:
                DATA_PROC_ = _normalize_processed_path(DATA_PROC)
                pred_path = DATA_PROC_ / "predictions.csv"

                if pred_path.exists():
                    df = pd.read_csv(pred_path, parse_dates=["date"])

                    # Создаём запись о запуске
                    run = crud.create_forecast_run(db, horizon=horizon)

                    # Сохраняем прогнозы
                    count = crud.bulk_create_predictions(db, df, run.id)
                    crud.complete_forecast_run(db, run.run_id, count)

                    response["saved_to_db"] = True
                    response["records_saved"] = count
                    response["run_id"] = run.run_id

            except Exception as e:
                logger.error(f"Ошибка сохранения в БД: {e}")
                response["saved_to_db"] = False
                response["db_error"] = str(e)

        logger.info("Прогнозы пересчитаны успешно")
        return response

    except subprocess.TimeoutExpired:
        logger.error('Время ожидания прогноза истекло (timeout > 5 мин)')
        raise HTTPException(
            status_code=504,
            detail="Превышено время ожидания (максимум 5 минут)"
        )
    except FileNotFoundError:
        logger.error('Модуль src.models.predict не найден')
        raise HTTPException(
            status_code=500,
            detail="Модуль predict.py не найден"
        )
    except Exception as e:
        logger.error(f"Неожиданная ошибка при пересчёте: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка: {str(e)}")


# ============================================================================
# METRICS
# ============================================================================

@app.get("/api/v1/metrics", tags=["Metrics"])
def get_metrics_v1(db: Session = Depends(get_optional_db)):
    """Возвращает метрики качества прогнозирования."""
    try:
        # Пробуем получить из БД
        if USE_DATABASE and db:
            try:
                metrics = crud.get_metrics(db)
                if metrics:
                    logger.info(f"✓ Метрики из БД для {len(metrics)} SKU")
                    return {
                        "count": len(metrics),
                        "source": "database",
                        "metrics": metrics
                    }
            except Exception as e:
                logger.warning(f"Ошибка чтения метрик из БД: {e}")

        # Fallback на CSV
        DATA_PROC_ = _normalize_processed_path(DATA_PROC)
        met_path = DATA_PROC_ / "metrics.csv"

        if not met_path.exists():
            logger.warning('Файл metrics.csv не найден')
            raise HTTPException(
                status_code=404,
                detail="Файл metrics.csv не найден. Запустите процесс оценки качества."
            )

        df = pd.read_csv(met_path)
        logger.info(f"✓ Метрики из CSV для {len(df)} SKU")
        return {
            "count": len(df),
            "source": "csv",
            "metrics": _clean_json(df)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при получении метрик: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")


# ============================================================================
# DATABASE ENDPOINTS (только если USE_DATABASE=true)
# ============================================================================

@app.get("/api/v1/db/stats", tags=["Database"])
def get_db_stats(db: Session = Depends(get_optional_db)):
    """Возвращает статистику базы данных."""
    if not USE_DATABASE:
        raise HTTPException(
            status_code=400,
            detail="Режим базы данных не включён. Установите USE_DATABASE=true"
        )

    try:
        stats = crud.get_database_stats(db)
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Ошибка получения статистики БД: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/forecast-runs", tags=["Database"])
def get_forecast_runs(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_optional_db)
):
    """Возвращает историю запусков прогнозирования."""
    if not USE_DATABASE:
        raise HTTPException(
            status_code=400,
            detail="Режим базы данных не включён"
        )

    try:
        runs = crud.get_forecast_runs(db, skip=skip, limit=limit)
        return {
            "count": len(runs),
            "runs": [
                {
                    "run_id": r.run_id,
                    "horizon": r.horizon,
                    "model_type": r.model_type,
                    "status": r.status,
                    "started_at": r.started_at.isoformat() if r.started_at else None,
                    "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                    "records_count": r.records_count,
                    "error_message": r.error_message
                }
                for r in runs
            ]
        }
    except Exception as e:
        logger.error(f"Ошибка получения истории запусков: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/db/sync", tags=["Database"])
def sync_csv_to_db(db: Session = Depends(get_optional_db)):
    """Синхронизирует данные из CSV файлов в базу данных."""
    if not USE_DATABASE:
        raise HTTPException(
            status_code=400,
            detail="Режим базы данных не включён"
        )

    try:
        DATA_PROC_ = _normalize_processed_path(DATA_PROC)
        result = {"synced": {}}

        # Синхронизируем прогнозы
        pred_path = DATA_PROC_ / "predictions.csv"
        if pred_path.exists():
            df = pd.read_csv(pred_path, parse_dates=["date"])
            run = crud.create_forecast_run(db, horizon=14)
            count = crud.bulk_create_predictions(db, df, run.id)
            crud.complete_forecast_run(db, run.run_id, count)
            result["synced"]["predictions"] = count

        # Синхронизируем метрики
        met_path = DATA_PROC_ / "metrics.csv"
        if met_path.exists():
            df = pd.read_csv(met_path)
            count = crud.bulk_create_metrics(db, df)
            result["synced"]["metrics"] = count

        result["status"] = "ok"
        result["timestamp"] = datetime.now().isoformat()

        logger.info(f"Синхронизация завершена: {result}")
        return result

    except Exception as e:
        logger.error(f"Ошибка синхронизации: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# MONITORING ENDPOINTS
# ==============================================================================

@app.get("/metrics", tags=["Monitoring"])
def get_prometheus_metrics():
    """
    Get application metrics in Prometheus format.

    Returns metrics for:
    - API request latency
    - Model prediction time
    - System resources
    """
    collector = get_metrics_collector()
    return JSONResponse(
        content=collector.get_metrics_text(),
        media_type="text/plain"
    )


@app.get("/api/v1/monitoring/metrics", tags=["Monitoring"])
def get_metrics_json():
    """Get application metrics in JSON format."""
    collector = get_metrics_collector()
    return collector.get_metrics_json()


@app.get("/api/v1/monitoring/health-detailed", tags=["Monitoring"])
def get_detailed_health():
    """
    Get detailed health information including metrics.
    """
    collector = get_metrics_collector()
    collector.collect_system_metrics()

    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "system": collector.system_metrics,
        "database_mode": USE_DATABASE,
        "components": {
            "api": "healthy",
            "models": {
                "prophet": (Path(PATHS['data']['models_dir']) / 'prophet_model.pkl').exists(),
                "xgboost": (Path(PATHS['data']['models_dir']) / 'xgboost_model.pkl').exists(),
            },
            "data": {
                "raw": (DATA_RAW / "sales_synth.csv").exists(),
                "processed": (_normalize_processed_path(DATA_PROC) / "processed.parquet").exists(),
            }
        }
    }
