import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from prophet import Prophet
from lightgbm import LGBMRegressor

from src.utils.logger import logger
from src.utils.config import PATHS
from src.etl.load_data import load_sales
from src.etl.clean_data import clean_sales
from src.etl.feature_builder import build_features


def predict(horizon: int = 14):
    """Создание прогноза для всех SKU."""
    logger.info(f"Start predicting, horizon={horizon}")
    df = load_sales()
    df = clean_sales(df)
    df = build_features(df)
    df = df.sort_values(['sku_id', 'store_id', 'date'])

    results = []

    for sku, g in df.groupby('sku_id'):
        g = g.sort_values('date')
        last_date = g['date'].max()

        # Prophet
        try:
            m = Prophet()
            df_p = g[['date','units']].rename(columns={'date':'ds','units':'y'})
            m.fit(df_p)
            fut = pd.DataFrame({'ds':[last_date + pd.Timedelta(days=i) for i in range(1, horizon+1)]})
            fc = m.predict(fut)
            prophet_pred = fc[['ds','yhat','yhat_lower','yhat_upper']]
            prophet_pred.columns = ['date','prophet','p_low','p_high']
        except Exception as e:
            logger.warning(f"Prophet failed on {sku}: {e}")
            prophet_pred = pd.DataFrame(columns=['date','prophet','p_low','p_high'])

        # LightGBM
        try:
            model_path = Path(PATHS['data']['models_dir']) / 'lgbm_model.pkl'
            model = joblib.load(model_path)
            g_feat = build_features(g.copy())
            last_rows = g_feat.tail(60).copy()
            X_last = last_rows[['dow','week','month','units_lag_1','units_lag_7']].fillna(0)
            preds = []
            cur = X_last.iloc[-1:].copy()
            for i in range(horizon):
                p = model.predict(cur)[0]
                preds.append(p)
                # обновим лаги
                cur['units_lag_1'] = p
                cur['units_lag_7'] = preds[i-6] if i >= 6 else p
            lgbm_pred = pd.DataFrame({
                'date': [last_date + pd.Timedelta(days=i+1) for i in range(horizon)],
                'lgbm': preds
            })
        except Exception as e:
            logger.warning(f"LGBM failed on {sku}: {e}")
            lgbm_pred = pd.DataFrame(columns=['date','lgbm'])

        # Ensemble
        df_merge = pd.merge(prophet_pred, lgbm_pred, on='date', how='outer')
        df_merge['ensemble'] = df_merge[['prophet','lgbm']].mean(axis=1)
        df_merge['sku_id'] = sku
        results.append(df_merge)

    all_pred = pd.concat(results, ignore_index=True)

    # === Исправление: гарантируем, что processed — это директория ===
    base = Path(PATHS['data']['processed'])
    if base.exists() and not base.is_dir():
        base.unlink()  # если файл — удаляем
    base.mkdir(parents=True, exist_ok=True)

    out_path = base / 'predictions.csv'

    # сохраняем прогноз
    all_pred.to_csv(out_path, index=False)
    logger.info(f"✅ Predictions saved to {out_path}")
    print(f"✅ Predictions saved to {out_path}")
    return out_path




if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=14)
    args = ap.parse_args()
    predict(horizon=args.horizon)
