# src/models/predict.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load
from src.utils.config import PATHS
from src.utils.logger import logger

def predict_all(horizon: int = 14):
    logger.info(f"Prediction start: horizon={horizon}")

    processed_file = Path(PATHS['data']['processed'])      # .../processed.parquet (файл)
    processed_dir  = processed_file.parent                 # .../data/processed (папка)
    models_dir     = Path(PATHS['data']['models_dir'])
    out_csv        = processed_dir / 'predictions.csv'

    df = pd.read_parquet(processed_file)
    df['date'] = pd.to_datetime(df['date'])
    skus = df['sku_id'].unique().tolist()

    # загрузка моделей
    prophet_models = None
    lgbm_model = None
    try:
        prophet_models = load(models_dir / 'prophet_model.pkl')  # dict: sku -> model
    except Exception as e:
        logger.warning(f"Prophet models not loaded: {e}")
    try:
        lgbm_model = load(models_dir / 'lgbm_model.pkl')
    except Exception as e:
        logger.warning(f"LGBM not loaded: {e}")

    results = []
    for sku in skus:
        hist = df[df['sku_id'] == sku].sort_values('date').copy()
        if hist.empty:
            continue
        last_date = hist['date'].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq='D')

        # Prophet per-SKU
        prophet_forecast = [np.nan] * horizon
        if isinstance(prophet_models, dict) and sku in prophet_models:
            try:
                fut = pd.DataFrame({'ds': future_dates})
                p = prophet_models[sku].predict(fut)['yhat'].to_numpy()
                prophet_forecast = p
            except Exception as e:
                logger.warning(f"Prophet predict failed for {sku}: {e}")

        # LGBM простая авто-рекурсия на 7-дневных роллингах
        lgbm_forecast = [np.nan] * horizon
        if lgbm_model is not None:
            try:
                series = hist['units'].astype(float).tolist()
                preds = []
                for _ in range(horizon):
                    feat = pd.Series(series).rolling(7, min_periods=1).mean().iloc[-1]
                    pred = float(lgbm_model.predict(np.array([[feat]]))[0])
                    pred = max(0.0, pred)
                    preds.append(pred)
                    series.append(pred)
                lgbm_forecast = preds
            except Exception as e:
                logger.warning(f"LGBM predict failed for {sku}: {e}")

        df_out = pd.DataFrame({
            'date': future_dates,
            'sku_id': sku,
            'prophet': prophet_forecast,
            'lgbm': lgbm_forecast
        })
        # ансамбль, если обе колонки есть
        df_out['ensemble'] = df_out[['prophet','lgbm']].mean(axis=1, skipna=True)
        results.append(df_out)

    pred_all = pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=['date','sku_id','prophet','lgbm','ensemble'])
    processed_dir.mkdir(parents=True, exist_ok=True)
    pred_all.to_csv(out_csv, index=False)
    logger.info(f"Predictions saved -> {out_csv}")
    print(f"✅ Прогноз сохранён: {out_csv} (строк: {len(pred_all)})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=14)
    args = ap.parse_args()
    predict_all(horizon=args.horizon)
