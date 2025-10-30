from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import subprocess
import pandas as pd
import numpy as np

# если у тебя есть свой конфиг
try:
    from src.utils.config import PATHS
    DATA_RAW = Path(PATHS["data"]["raw_dir"])
    DATA_PROC = Path(PATHS["data"]["processed"])
except Exception:
    # fallback, если нет конфига
    DATA_RAW = Path("data/raw")
    DATA_PROC = Path("data/processed")

app = FastAPI(
    title="Forecastly API",
    description="API для системы анализа и прогнозирования продаж",
    version="1.0.0",
)

# CORS, чтобы можно было дёргать из дашборда/браузера
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _normalize_processed_path(p: Path) -> Path:
    """
    Если в конфиге указали data/processed/processed.parquet — возьмём родителя.
    """
    if p.suffix:  # .parquet, .csv и т.п.
        return p.parent
    return p


def _clean_json(df: pd.DataFrame) -> list[dict]:
    """
    Преобразуем DataFrame в список dict и чистим NaN/inf → None,
    чтобы FastAPI смог отдать JSON без ошибок.
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
    return {"status": "ok", "service": "forecastly-api"}


@app.get("/skus")
def get_skus():
    """
    Возвращает список SKU из сырых данных или из прогноза.
    """
    DATA_PROC_ = _normalize_processed_path(DATA_PROC)
    pred_path = DATA_PROC_ / "predictions.csv"
    raw_path = DATA_RAW / "sales_synth.csv"

    skus = set()

    if raw_path.exists():
        df_raw = pd.read_csv(raw_path)
        if "sku_id" in df_raw.columns:
            skus.update(df_raw["sku_id"].astype(str).tolist())

    if pred_path.exists():
        df_pred = pd.read_csv(pred_path)
        if "sku_id" in df_pred.columns:
            skus.update(df_pred["sku_id"].astype(str).tolist())

    return {"skus": sorted(list(skus))}


@app.get("/predict")
def get_predict(
    sku_id: str = Query(..., description="Идентификатор товара, например SKU001"),
    horizon: int = Query(14, ge=1, le=120, description="Горизонт прогноза в днях")
):
    """
    Возвращает прогноз по конкретному SKU из уже посчитанного файла data/processed/predictions.csv.
    Если файла нет — 404.
    Если по SKU нет строк — 404.
    """
    DATA_PROC_ = _normalize_processed_path(DATA_PROC)
    pred_path = DATA_PROC_ / "predictions.csv"

    if not pred_path.exists():
        raise HTTPException(status_code=404, detail="predictions.csv not found. Run forecasting first.")

    df = pd.read_csv(pred_path, parse_dates=["date"])
    # нормализуем SKU, потому что в проекте они крутятся вокруг SKU001
    sku_id_norm = sku_id.strip().upper().replace("SKU_", "SKU").replace("SKU-", "SKU")

    df_sku = df[df["sku_id"].astype(str).str.upper() == sku_id_norm].copy()

    if df_sku.empty:
        raise HTTPException(status_code=404, detail=f"no predictions for SKU '{sku_id_norm}'")

    # ограничим по горизонту, если хотим
    df_sku = df_sku.sort_values("date").head(horizon)

    return {
        "sku_id": sku_id_norm,
        "horizon": horizon,
        "predictions": _clean_json(df_sku)
    }


@app.post("/predict/rebuild")
def rebuild_predict(horizon: int = 14):
    """
    Пересчитать прогноз из API.
    Просто вызывает: python -m src.models.predict --horizon X
    """
    cmd = ["python", "-m", "src.models.predict", "--horizon", str(horizon)]
    try:
        # stdout можно не сохранять, но на будущее оставим capture_output
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"prediction script failed: {result.stderr}"
            )
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="python not found or module src.models.predict not found")

    return {
        "status": "ok",
        "message": "predictions recalculated",
        "horizon": horizon
    }


@app.get("/metrics")
def get_metrics():
    """
    Возвращает метрики из data/processed/metrics.csv
    """
    DATA_PROC_ = _normalize_processed_path(DATA_PROC)
    met_path = DATA_PROC_ / "metrics.csv"

    if not met_path.exists():
        raise HTTPException(status_code=404, detail="metrics.csv not found. Run evaluation first.")

    df = pd.read_csv(met_path)
    return {"metrics": _clean_json(df)}


# опционально: корень можно отдать как "живой" API
@app.get("/")
def root():
    return {
        "service": "forecastly-api",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": ["/health", "/skus", "/predict", "/metrics", "/predict/rebuild"]
    }
