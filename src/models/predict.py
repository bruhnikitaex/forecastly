# src/models/predict.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load
from src.utils.config import PATHS
from src.utils.logger import logger


def _weekday(ts: pd.Timestamp) -> int:
    # 0..6 (пн..вс), как в train
    return ts.weekday()


def _weeknum(ts: pd.Timestamp) -> int:
    # номер недели года (ISO)
    try:
        return int(ts.isocalendar().week)
    except Exception:
        # совместимость со старыми pandas
        return int(ts.isocalendar()[1])


def predict_all(horizon: int = 14):
    """
    Делает прогноз по КАЖДОМУ SKU на horizon дней вперёд.
      - Prophet: отдельная модель на каждый SKU (dict{sku: Prophet})
      - LGBM: использует тот же набор фич, что и при обучении:
              dow, week, month, units_lag_1, units_lag_7 (авто-рекурсия)
      - Ensemble = среднее Prophet и LGBM (по доступным значениям)
    Сохраняет результат в data/processed/predictions.csv
    """
    logger.info(f"Prediction start: horizon={horizon}")

    processed_file = Path(PATHS['data']['processed'])   # .../processed.parquet (ФАЙЛ)
    processed_dir = processed_file.parent               # .../data/processed (ПАПКА)
    models_dir = Path(PATHS['data']['models_dir'])
    out_csv = processed_dir / 'predictions.csv'

    # --- данные ---
    df = pd.read_parquet(processed_file)
    df['date'] = pd.to_datetime(df['date'])
    skus = df['sku_id'].unique().tolist()
    logger.info(f"SKUs found: {len(skus)}")

    # --- загрузка моделей ---
    prophet_models = None
    lgbm_model = None
    try:
        prophet_models = load(models_dir / 'prophet_model.pkl')  # dict: {sku: Prophet()}
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
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1),
                                     periods=horizon, freq='D')

        # --- Prophet per-SKU ---
        prophet_forecast = [np.nan] * horizon
        if isinstance(prophet_models, dict) and (sku in prophet_models):
            try:
                fut = pd.DataFrame({'ds': future_dates})
                pred = prophet_models[sku].predict(fut)
                prophet_forecast = pred['yhat'].to_numpy()
                p_low  = pred['yhat_lower'].to_numpy()
                p_high = pred['yhat_upper'].to_numpy()

            except Exception as e:
                logger.warning(f"Prophet predict failed for {sku}: {e}")

        # --- LGBM: генерируем те же фичи, что в train_lgbm ---
        lgbm_forecast = [np.nan] * horizon
        if lgbm_model is not None:
            try:
                # авто-рекурсия: используем историю + собственные прогнозы для лагов
                series = hist['units'].astype(float).tolist()
                preds = []
                for i in range(1, horizon + 1):
                    d = last_date + pd.Timedelta(days=i)
                    dow = _weekday(d)
                    week = _weeknum(d)
                    month = d.month
                    lag1 = series[-1] if len(series) >= 1 else 0.0
                    lag7 = series[-7] if len(series) >= 7 else lag1

                    X_row = pd.DataFrame({
                        'dow': [dow],
                        'week': [week],
                        'month': [month],
                        'units_lag_1': [lag1],
                        'units_lag_7': [lag7]
                    })

                    yhat = float(lgbm_model.predict(X_row)[0])
                    yhat = max(0.0, yhat)  # без отрицательных
                    preds.append(yhat)
                    series.append(yhat)  # чтобы следующие шаги имели корректные лаги

                lgbm_forecast = preds
            except Exception as e:
                logger.warning(f"LGBM predict failed for {sku}: {e}")

        # --- сборка ---
        df_out = pd.DataFrame({
            'date': future_dates,
            'sku_id': sku,
            'prophet': prophet_forecast,
            'p_low': p_low if 'p_low' in locals() else [np.nan]*horizon,
            'p_high': p_high if 'p_high' in locals() else [np.nan]*horizon,
            'lgbm': lgbm_forecast
        })

# привести к числам (NaN, если не получилось)
        for c in ['prophet','p_low','p_high','lgbm','ensemble']:
            if c in pred_all.columns:
                pred_all[c] = pd.to_numeric(pred_all[c], errors='coerce')


        # ансамбль по доступным
        df_out['ensemble'] = df_out[['prophet', 'lgbm']].mean(axis=1, skipna=True)

        results.append(df_out)

    # --- объединение и сохранение ---
    if results:
        pred_all = pd.concat(results, ignore_index=True)
    else:
        pred_all = pd.DataFrame(columns=['date', 'sku_id', 'prophet', 'lgbm', 'ensemble'])

    for c in ['prophet', 'lgbm', 'ensemble']:
        if c in pred_all.columns:
            pred_all[c] = pd.to_numeric(pred_all[c], errors='coerce')

    processed_dir.mkdir(parents=True, exist_ok=True)
    pred_all.to_csv(out_csv, index=False)
    logger.info(f"Predictions saved -> {out_csv}")
    print(f"✅ Прогноз сохранён: {out_csv} (строк: {len(pred_all)})")

    return out_csv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=14)
    args = ap.parse_args()
    predict_all(horizon=args.horizon)
