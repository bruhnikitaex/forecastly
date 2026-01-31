"""
API роутер для управления данными.

Endpoints:
- POST /api/v1/data/upload - Загрузка CSV/XLSX файлов
- GET /api/v1/data/datasets - Список загрузок и статусы
- GET /api/v1/forecast/batch - Пакетный прогноз по списку SKU
"""

import os
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from io import BytesIO

import pandas as pd
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from sqlalchemy.orm import Session

from src.utils.config import PATHS
from src.utils.logger import logger

router = APIRouter(tags=["Data"])

# Paths
DATA_RAW = Path(PATHS.get("data", {}).get("raw", "data/raw"))
DATA_PROC = Path(PATHS.get("data", {}).get("processed", "data/processed"))
if DATA_PROC.suffix:
    DATA_PROC = DATA_PROC.parent

USE_DATABASE = os.getenv("USE_DATABASE", "false").lower() == "true"


def _get_optional_db():
    if USE_DATABASE:
        from src.db.database import SessionLocal
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
    else:
        yield None


def _clean_json(df: pd.DataFrame) -> list:
    recs = df.to_dict(orient="records")
    for r in recs:
        for k, v in list(r.items()):
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                r[k] = None
            elif hasattr(v, "isoformat"):
                r[k] = v.isoformat()
    return recs


# ============================================================================
# DATA UPLOAD
# ============================================================================

@router.post("/data/upload")
async def upload_data(
    file: UploadFile = File(...),
    db: Session = Depends(_get_optional_db),
):
    """
    Загрузка данных из CSV или XLSX файла.

    Поддерживаемые форматы: .csv, .xlsx
    Обязательные поля: date, sku_id, qty (или units)
    """
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()

    if ext not in (".csv", ".xlsx"):
        raise HTTPException(400, "Поддерживаемые форматы: CSV, XLSX")

    content = await file.read()
    file_hash = hashlib.md5(content).hexdigest()[:12]

    try:
        if ext == ".csv":
            df = pd.read_csv(BytesIO(content))
        else:
            df = pd.read_excel(BytesIO(content), engine="openpyxl")
    except Exception as e:
        raise HTTPException(400, f"Ошибка чтения файла: {e}")

    # Нормализация колонок
    col_map = {"qty": "units", "quantity": "units", "sales": "units"}
    df.columns = [c.strip().lower() for c in df.columns]
    df.rename(columns=col_map, inplace=True)

    # Валидация
    required = {"date", "sku_id"}
    missing = required - set(df.columns)
    if missing:
        raise HTTPException(400, f"Отсутствуют обязательные поля: {missing}")

    if "units" not in df.columns:
        raise HTTPException(400, "Не найдено поле количества (qty/units/quantity/sales)")

    # Сохраняем
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"upload_{ts}_{file_hash}.csv"
    save_path = DATA_RAW / save_name
    df.to_csv(save_path, index=False)

    # DQ отчёт
    dq = {
        "rows": len(df),
        "columns": list(df.columns),
        "sku_count": int(df["sku_id"].nunique()),
        "date_range": {
            "min": str(df["date"].min()),
            "max": str(df["date"].max()),
        },
        "missing_pct": {
            col: round(float(df[col].isna().mean()) * 100, 1)
            for col in df.columns
        },
        "duplicates": int(df.duplicated().sum()),
    }

    # Простая проверка выбросов по IQR
    if "units" in df.columns:
        q1 = df["units"].quantile(0.25)
        q3 = df["units"].quantile(0.75)
        iqr = q3 - q1
        outliers = int(((df["units"] < q1 - 1.5 * iqr) | (df["units"] > q3 + 1.5 * iqr)).sum())
        dq["outliers_units"] = outliers

    logger.info(f"Загружен файл {filename}: {len(df)} строк, {dq['sku_count']} SKU")

    return {
        "status": "ok",
        "filename": filename,
        "saved_as": save_name,
        "timestamp": datetime.now().isoformat(),
        "dq_report": dq,
    }


# ============================================================================
# DATASETS LIST
# ============================================================================

@router.get("/data/datasets")
def list_datasets():
    """Список загруженных датасетов и их статусы."""
    datasets = []

    if DATA_RAW.exists():
        for f in sorted(DATA_RAW.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                size = f.stat().st_size
                modified = datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                # Quick row count
                with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                    row_count = sum(1 for _ in fh) - 1
                datasets.append({
                    "name": f.name,
                    "size_bytes": size,
                    "rows": row_count,
                    "modified": modified,
                    "path": str(f),
                })
            except Exception:
                datasets.append({"name": f.name, "error": "Не удалось прочитать"})

    return {"count": len(datasets), "datasets": datasets}


# ============================================================================
# POSTGRESQL IMPORT
# ============================================================================

@router.post("/data/import/postgres")
def import_from_postgres(
    table: str = Query("sales", description="Имя таблицы в PostgreSQL"),
    host: str = Query("localhost", description="Хост PostgreSQL"),
    port: int = Query(5432, ge=1, le=65535),
    database: str = Query(..., description="Имя базы данных"),
    user: str = Query(..., description="Пользователь"),
    password: str = Query(..., description="Пароль"),
    db: Session = Depends(_get_optional_db),
):
    """
    Импорт данных о продажах из внешней PostgreSQL базы данных.

    Читает таблицу продаж и сохраняет в формате CSV для дальнейшей обработки.
    Обязательные поля в таблице: date, sku_id, qty/units.
    """
    try:
        from sqlalchemy import create_engine

        conn_str = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        engine = create_engine(conn_str)

        df = pd.read_sql_table(table, engine)
        logger.info(f"Загружено {len(df)} строк из {host}:{port}/{database}.{table}")

        # Нормализация колонок
        df.columns = [c.strip().lower() for c in df.columns]
        col_map = {"qty": "units", "quantity": "units", "sales": "units"}
        df.rename(columns=col_map, inplace=True)

        required = {"date", "sku_id"}
        missing = required - set(df.columns)
        if missing:
            raise HTTPException(400, f"Отсутствуют обязательные поля: {missing}")
        if "units" not in df.columns:
            raise HTTPException(400, "Не найдено поле количества (qty/units/quantity/sales)")

        # Сохраняем
        DATA_RAW.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"pg_import_{ts}.csv"
        save_path = DATA_RAW / save_name
        df.to_csv(save_path, index=False)

        return {
            "status": "ok",
            "source": f"{host}:{port}/{database}.{table}",
            "saved_as": save_name,
            "rows": len(df),
            "sku_count": int(df["sku_id"].nunique()),
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка импорта из PostgreSQL: {e}")
        raise HTTPException(500, f"Ошибка подключения к PostgreSQL: {e}")


# ============================================================================
# BATCH FORECAST
# ============================================================================

@router.get("/forecast/batch")
def batch_forecast(
    sku_ids: str = Query(..., description="Список SKU через запятую (SKU001,SKU002,...)"),
    horizon: int = Query(14, ge=1, le=30),
):
    """
    Пакетный прогноз по списку SKU.

    Args:
        sku_ids: SKU через запятую
        horizon: Горизонт прогноза (1-30 дней)

    Returns:
        Прогнозы для каждого SKU
    """
    sku_list = [s.strip().upper() for s in sku_ids.split(",") if s.strip()]
    if not sku_list:
        raise HTTPException(400, "Передайте хотя бы один SKU")
    if len(sku_list) > 100:
        raise HTTPException(400, "Максимум 100 SKU в одном запросе")

    pred_path = DATA_PROC / "predictions.csv"
    if not pred_path.exists():
        raise HTTPException(404, "Прогнозы не найдены. Запустите процесс прогнозирования.")

    df = pd.read_csv(pred_path, parse_dates=["date"])
    df["sku_id_upper"] = df["sku_id"].astype(str).str.upper()

    results = {}
    not_found = []

    for sku in sku_list:
        sku_norm = sku.replace("SKU_", "SKU").replace("SKU-", "SKU")
        df_sku = df[df["sku_id_upper"] == sku_norm].sort_values("date").head(horizon)

        if df_sku.empty:
            not_found.append(sku)
            continue

        results[sku_norm] = _clean_json(df_sku.drop(columns=["sku_id_upper"], errors="ignore"))

    return {
        "horizon": horizon,
        "requested": len(sku_list),
        "found": len(results),
        "not_found": not_found,
        "predictions": results,
    }
