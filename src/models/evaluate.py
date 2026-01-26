# src/models/evaluate.py
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.logger import logger
from src.utils.config import PATHS
from src.etl.load_data import load_sales
from src.etl.clean_data import clean_sales
from src.etl.feature_builder import build_features


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = (y_true > 0) & np.isfinite(y_true) & np.isfinite(y_pred)
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan


def evaluate(horizon: int = 14):
    logger.info(f"Evaluation start: horizon={horizon}")

    df = load_sales()
    df = clean_sales(df)
    df = build_features(df)
    df = df.sort_values(['sku_id', 'store_id', 'date'])

    from prophet import Prophet
    from xgboost import XGBRegressor

    results = []

    for sku, g in df.groupby('sku_id'):
        sku = str(sku)
        g = g.sort_values('date')

        if len(g) < 90:
            continue

        train = g.iloc[:-horizon].copy()
        test = g.iloc[-horizon:].copy()

        # Prophet
        try:
            m = Prophet()
            df_p = train[['date', 'units']].rename(columns={'date': 'ds', 'units': 'y'})
            m.fit(df_p)
            fc = m.predict(test[['date']].rename(columns={'date': 'ds'}))
            y_p = fc['yhat'].values
        except Exception:
            y_p = np.full(len(test), np.nan)

        # XGBoost
        try:
            X_cols = ['dow', 'week', 'month', 'units_lag_1', 'units_lag_7']
            train_feat = build_features(train.copy())
            test_feat = build_features(test.copy())
            model = XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                random_state=42
            )
            model.fit(train_feat[X_cols].fillna(0), train_feat['units'])
            y_x = model.predict(test_feat[X_cols].fillna(0))
        except Exception:
            y_x = np.full(len(test), np.nan)

        # Naive (последнее значение)
        y_n = np.full(len(test), train['units'].iloc[-1])

        y_e = np.nanmean(np.vstack([
            np.asarray(y_p),
            np.asarray(y_x)
        ]), axis=0)
        m_prophet = mape(test['units'], y_p)
        m_xgb = mape(test['units'], y_x)
        m_naive = mape(test['units'], y_n)
        m_ens = mape(test['units'], y_e)

        best = min(
            [('prophet', m_prophet), ('xgboost', m_xgb), ('naive', m_naive), ('ens', m_ens)],
            key=lambda x: (x[1] if not np.isnan(x[1]) else 999)
        )[0]

        results.append({
            'sku_id': sku,
            'mape_prophet': round(m_prophet, 2),
            'mape_xgboost': round(m_xgb, 2),
            'mape_naive': round(m_naive, 2),
            'mape_ens': round(m_ens, 2),
            'best_model': best
        })

    met = pd.DataFrame(results)
    out_path = Path(PATHS['data']['processed']) / 'metrics.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    met.to_csv(out_path, index=False)

    logger.info(f"Evaluation complete. Results saved to {out_path}")
    print(f"[OK] Evaluation complete. Results saved to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=14)
    args = ap.parse_args()
    evaluate(horizon=args.horizon)
