# src/models/predict.py
"""
Модуль прогнозирования продаж.

Использует предобученные модели Prophet и XGBoost для генерации прогнозов.
Если модели не найдены - обучает их на лету.
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from prophet import Prophet

from src.utils.logger import logger
from src.utils.config import PATHS
from src.etl.load_data import load_sales
from src.etl.clean_data import clean_sales
from src.etl.feature_builder import build_features


def normalize_sku(s) -> str:
    """
    Приводим ВСЁ к виду SKUxxx (без подчёркивания),
    потому что в дашборде используется именно такой формат.
    """
    s = str(s).strip().upper()
    # убираем возможные варианты
    s = s.replace("SKU_", "SKU")
    s = s.replace("SKU-", "SKU")
    if s.startswith("SKU"):
        num = s[3:]
        num = "".join(ch for ch in num if ch.isdigit())
        return f"SKU{num.zfill(3)}"
    # если вдруг совсем другой формат
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits:
        return f"SKU{digits.zfill(3)}"
    return s


def load_prophet_models() -> dict:
    """
    Загружает предобученные модели Prophet.

    Returns:
        Словарь {sku_id: Prophet_model} или пустой словарь.
    """
    model_path = Path(PATHS['data']['models_dir']) / 'prophet_model.pkl'
    if model_path.exists():
        try:
            models = joblib.load(model_path)
            logger.info(f"Загружено {len(models)} предобученных Prophet моделей")
            return models
        except Exception as e:
            logger.warning(f"Не удалось загрузить Prophet модели: {e}")
    return {}


def predict(horizon: int = 14):
    logger.info(f"Start predicting, horizon={horizon}")

    # 1. загрузили и подготовили данные
    df = load_sales()
    df = clean_sales(df)
    df = build_features(df)
    df = df.sort_values(['sku_id', 'store_id', 'date'])

    # Загружаем предобученные модели
    prophet_models = load_prophet_models()

    results = []

    # 2. проходимся по каждому SKU
    for sku, g in df.groupby('sku_id'):
        sku_norm = normalize_sku(sku)
        g = g.sort_values('date')
        last_date = g['date'].max()

        # -------- Prophet --------
        try:
            # Пытаемся использовать предобученную модель
            if sku in prophet_models:
                m = prophet_models[sku]
                logger.debug(f"Используем предобученную модель Prophet для {sku_norm}")
            elif sku_norm in prophet_models:
                m = prophet_models[sku_norm]
                logger.debug(f"Используем предобученную модель Prophet для {sku_norm}")
            else:
                # Обучаем новую модель если нет предобученной
                logger.info(f"Обучение новой Prophet модели для {sku_norm}")
                m = Prophet()
                df_p = g[['date', 'units']].rename(columns={'date': 'ds', 'units': 'y'})
                m.fit(df_p)

            fut = pd.DataFrame({'ds': [last_date + pd.Timedelta(days=i) for i in range(1, horizon + 1)]})
            fc = m.predict(fut)
            prophet_pred = fc[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            prophet_pred.columns = ['date', 'prophet', 'p_low', 'p_high']
        except Exception as e:
            logger.warning(f"Prophet failed on {sku_norm}: {e}")
            prophet_pred = pd.DataFrame(columns=['date', 'prophet', 'p_low', 'p_high'])

        # -------- XGBoost --------
        try:
            model_path = Path(PATHS['data']['models_dir']) / 'xgboost_model.pkl'
            model = joblib.load(model_path)

            g_feat = build_features(g.copy())
            last_rows = g_feat.tail(60).copy()
            X_last = last_rows[['dow', 'week', 'month', 'units_lag_1', 'units_lag_7']].fillna(0)

            preds = []
            cur = X_last.iloc[-1:].copy()
            for i in range(horizon):
                p = model.predict(cur)[0]
                preds.append(float(p))
                # обновим лаги
                cur['units_lag_1'] = p
                cur['units_lag_7'] = preds[i - 6] if i >= 6 else p

            xgb_pred = pd.DataFrame({
                'date': [last_date + pd.Timedelta(days=i + 1) for i in range(horizon)],
                'xgb': preds
            })
        except Exception as e:
            logger.warning(f"XGBoost failed on {sku_norm}: {e}")
            xgb_pred = pd.DataFrame(columns=['date', 'xgb'])

        # -------- Ensemble --------
        df_merge = pd.merge(prophet_pred, xgb_pred, on='date', how='outer')
        df_merge['ensemble'] = df_merge[['prophet', 'xgb']].mean(axis=1)
        df_merge['sku_id'] = sku_norm
        results.append(df_merge)

    # 3. склеили всё
    all_pred = pd.concat(results, ignore_index=True)

    # 4. нормализуем путь (если в конфиге был кривой)
    base = Path(PATHS['data']['processed'])
    # если в конфиге указано .../processed.parquet — берём родителя
    if base.suffix:
        base = base.parent
    if base.exists() and not base.is_dir():
        base.unlink()
    base.mkdir(parents=True, exist_ok=True)

    # основной файл, который читает дашборд
    out_path = base / 'predictions.csv'
    all_pred.to_csv(out_path, index=False)

    # 5. сохраняем версию в историю
    hist_dir = base / "history"
    hist_dir.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%d_%H%M%S")
    hist_path = hist_dir / f"predictions_{ts}.csv"
    all_pred.to_csv(hist_path, index=False)

    logger.info(f"✅ Predictions saved to {out_path}")
    logger.info(f"🗂  History saved to {hist_path}")
    print(f"✅ Predictions saved to {out_path}")
    print(f"🗂  History saved to {hist_path}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=14)
    args = ap.parse_args()
    predict(horizon=args.horizon)
