from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pathlib import Path
import pandas as pd
import numpy as np
import re

app = FastAPI(title="Forecastly API", version="1.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

PRED_PATH = Path("data/processed/processed.parquet/predictions.csv")
RAW_PATH  = Path("data/raw/sales_synth.csv")


def _load_csv(path: Path):
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    df = pd.read_csv(path)
    return df

def _sanitize(df: pd.DataFrame) -> list[dict]:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.where(pd.notna(df), None)
    return jsonable_encoder(df.to_dict(orient="records"))

@app.get("/")
def root():
    return {"name": "Forecastly API", "endpoints": ["/health", "/skus", "/predict"]}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/skus")
def get_skus():
    df = _load_csv(RAW_PATH)
    return {"skus": sorted(df["sku_id"].astype(str).unique().tolist())}

@app.get("/predict")
def get_predict(sku_id: str = Query(..., description="SKU ID, напр. SKU_001"), horizon: int = 14):
    dfp = _load_csv(PRED_PATH)
    matches = [c for c in dfp["sku_id"].astype(str).unique() if c.lower() == sku_id.lower()]
    if not matches:
        raise HTTPException(status_code=404, detail=f"SKU {sku_id} not found")
    sku = matches[0]
    data = dfp[dfp["sku_id"] == sku].sort_values("date").tail(horizon)
    if data.empty:
        raise HTTPException(status_code=404, detail="No prediction data found.")
    return JSONResponse(content=_sanitize(data))
