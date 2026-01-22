"""
REST API для системы прогнозирования продаж Forecastly.

Предоставляет endpoints для:
- Получения списка SKU
- Получения прогнозов по SKU
- Пересчёта прогнозов
- Получения метрик качества

Версионирование: /api/v1/...
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime
from src.utils.config import PATHS
from src.utils.logger import logger

# конфиг
try:
    DATA_RAW = Path(PATHS["data"].get("raw", "data/raw"))
    DATA_PROC = Path(PATHS["data"].get("processed", "data/processed"))
except Exception:
    # fallback
    DATA_RAW = Path("data/raw")
    DATA_PROC = Path("data/processed")


from src.utils.logger import logger

app = FastAPI(
    title="Forecastly API",
    description="API для системы анализа и прогнозирования продаж",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _normalize_processed_path(p: Path) -> Path:
    """
    Нормализует путь к директории обработанных данных.
    
    Если в конфиге указан файл (с расширением), возвращает родительскую директорию.
    
    Args:
        p: Путь для нормализации.
    
    Returns:
        Путь к директории.
    """
    if p.suffix:  # .parquet, .csv и т.п.
        return p.parent
    return p


def _clean_json(df: pd.DataFrame) -> list[dict]:
    """
    Преобразует DataFrame в JSON-safe список словарей.
    
    Заменяет NaN и Inf значения на None для корректной сериализации.
    
    Args:
        df: DataFrame для преобразования.
    
    Returns:
        Список словарей с чистыми значениями.
    """
    recs = df.to_dict(orient="records")
    cleaned = []
    for r in recs:
        for k, v in list(r.items()):
            if isinstance(v, float):
                if np.isnan(v) or np.isinf(v):
                    r[k] = None
        cleaned.append(r)
    return cleaned


@app.get("/health")
def health():
    """
    Проверка статуса API (health check).
    
    Returns:
        JSON с информацией о статусе сервиса.
    """
    logger.info('Health check запрос')
    return {
        "status": "ok",
        "service": "forecastly-api",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/v1/skus", tags=["SKU"])
def get_skus_v1():
    """
    Возвращает список уникальных SKU из доступных данных.
    
    Проверяет как сырые данные, так и прогнозы.
    
    Returns:
        JSON с массивом уникальных SKU.
        
    Raises:
        HTTPException 404: Если нет доступных данных.
    """
    try:
        DATA_PROC_ = _normalize_processed_path(DATA_PROC)
        pred_path = DATA_PROC_ / "predictions.csv"
        raw_path = DATA_RAW / "sales_synth.csv"

        skus = set()

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


@app.get("/api/v1/predict", tags=["Predictions"])
def get_predict_v1(
    sku_id: str = Query(..., description="SKU товара (например SKU001)"),
    horizon: int = Query(14, ge=1, le=120, description="Горизонт прогноза в днях")
):
    """
    Возвращает прогноз по конкретному SKU.
    
    Извлекает данные из уже посчитанного файла predictions.csv.
    
    Args:
        sku_id: Идентификатор товара.
        horizon: Количество дней для прогноза (1-120).
    
    Returns:
        JSON с массивом прогнозов по датам и моделям.
        
    Raises:
        HTTPException 404: Если файл прогнозов не найден или SKU не существует.
        HTTPException 422: Если параметры некорректны.
    """
    try:
        DATA_PROC_ = _normalize_processed_path(DATA_PROC)
        pred_path = DATA_PROC_ / "predictions.csv"

        if not pred_path.exists():
            logger.error(f'Файл predictions.csv не найден по пути {pred_path}')
            raise HTTPException(
                status_code=404,
                detail="Файл predictions.csv не найден. Запустите процесс прогнозирования."
            )

        df = pd.read_csv(pred_path, parse_dates=["date"])
        
        # нормализуем SKU
        sku_id_norm = sku_id.strip().upper().replace("SKU_", "SKU").replace("SKU-", "SKU")
        df_sku = df[df["sku_id"].astype(str).str.upper() == sku_id_norm].copy()

        if df_sku.empty:
            logger.warning(f"SKU '{sku_id_norm}' не найден в прогнозах")
            available_skus = df["sku_id"].unique()[:5].tolist()
            raise HTTPException(
                status_code=404,
                detail=f"Прогноз для SKU '{sku_id_norm}' не найден. "
                        f"Доступные SKU: {available_skus}..."
            )

        # ограничим по горизонту
        df_sku = df_sku.sort_values("date").head(horizon)

        logger.info(f"✓ Прогноз получен для {sku_id_norm}, {len(df_sku)} записей")
        return {
            "sku_id": sku_id_norm,
            "horizon": horizon,
            "count": len(df_sku),
            "predictions": _clean_json(df_sku)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при получении прогноза: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")


@app.post("/api/v1/predict/rebuild", tags=["Predictions"])
def rebuild_predict_v1(horizon: int = Query(14, ge=1, le=120)):
    """
    Пересчитывает прогнозы через команду predict.py.
    
    Это дорогостоящая операция, выполняется синхронно.
    
    Args:
        horizon: Горизонт прогноза в днях.
    
    Returns:
        JSON с информацией о завершении операции.
        
    Raises:
        HTTPException 500: Если процесс обучения/прогноза не удался.
    """
    logger.info(f'Запрос на пересчёт прогнозов с горизонтом {horizon} дней')
    
    try:
        cmd = ["python", "-m", "src.models.predict", "--horizon", str(horizon)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            logger.error(f'Ошибка при пересчёте прогнозов: {result.stderr}')
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка при пересчёте прогнозов: {result.stderr[:500]}"
            )
        
        logger.info("Прогнозы пересчитаны успешно")
        return {
            "status": "ok",
            "message": "Прогнозы пересчитаны успешно",
            "horizon": horizon,
            "timestamp": datetime.now().isoformat()
        }
    
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


@app.get("/api/v1/metrics", tags=["Metrics"])
def get_metrics_v1():
    """
    Возвращает метрики качества прогнозирования.
    
    Содержит MAPE (Mean Absolute Percentage Error) по разным моделям
    и лучшую модель для каждого SKU.
    
    Returns:
        JSON с массивом метрик по SKU.
        
    Raises:
        HTTPException 404: Если файл метрик не найден.
    """
    try:
        DATA_PROC_ = _normalize_processed_path(DATA_PROC)
        met_path = DATA_PROC_ / "metrics.csv"

        if not met_path.exists():
            logger.warning('Файл metrics.csv не найден')
            raise HTTPException(
                status_code=404,
                detail="Файл metrics.csv не найден. Запустите процесс оценки качества."
            )

        df = pd.read_csv(met_path)
        logger.info(f"✓ Метрики получены для {len(df)} SKU")
        return {
            "count": len(df),
            "metrics": _clean_json(df)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при получении метрик: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")


@app.get("/", tags=["Root"])
def root():
    """
    Корневой endpoint с информацией об API.
    
    Returns:
        JSON с ссылками на документацию и список endpoints.
    """
    return {
        "service": "forecastly-api",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "health": "/health",
            "skus": "/api/v1/skus",
            "predict": "/api/v1/predict?sku_id=SKU001&horizon=14",
            "rebuild": "/api/v1/predict/rebuild?horizon=14",
            "metrics": "/api/v1/metrics"
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/v1/status", tags=["System"])
def system_status():
    """
    Возвращает статус системы и информацию о доступных данных.
    
    Returns:
        JSON с информацией о доступных файлах и готовности системы.
    """
    try:
        DATA_PROC_ = _normalize_processed_path(DATA_PROC)
        
        status = {
            "system": "ready",
            "timestamp": datetime.now().isoformat(),
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
        
        logger.info('✓ Запрос статуса системы выполнен')
        return status
    
    except Exception as e:
        logger.error(f"Ошибка при получении статуса: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка: {str(e)}")
