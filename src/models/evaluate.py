# src/models/evaluate.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from lightgbm import LGBMRegressor
from prophet import Prophet

from src.utils.config import PATHS
from src.utils.logger import logger
from src.etl.load_data import load_sales
from src.etl.clean_data import clean_sales
from src.etl.feature_builder import build_features


# ---------- метрики ----------
def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0

def smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100.0

def rmse(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


# ---------- нормализация колонок ----------
def _first_match(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводим датафрейм к стандартным именам:
    date, sku_id, store_id, units, price, promo_flag, category
    Понимаем популярные синонимы и создаём store_id при отсутствии.
    """
    df = df.copy()
    cols = {c: c for c in df.columns}

    # кандидаты имён
    date_col  = _first_match(cols, ["date", "ds", "day"])
    sku_col   = _first_match(cols, ["sku_id", "sku", "item_id", "product_id", "product"])
    store_col = _first_match(cols, ["store_id", "store", "shop_id"])
    units_col = _first_match(cols, ["units", "qty", "quantity", "sales", "y", "target"])
    price_col = _first_match(cols, ["price", "prc"])
    promo_col = _first_match(cols, ["promo_flag", "promo", "is_promo"])
    cat_col   = _first_match(cols, ["category", "cat"])

    # проверим критические
    missing = []
    if date_col is None:  missing.append("date")
    if sku_col  is None:  missing.append("sku_id")
    if units_col is None: missing.append("units")
    if missing:
        raise KeyError(f"Не найдены критические колонки: {missing}. Найденные колонки: {list(df.columns)}")

    # переименуем
    ren = {date_col: "date", sku_col: "sku_id", units_col: "units"}
    if store_col: ren[store_col] = "store_id"
    if price_col: ren[price_col] = "price"
    if promo_col: ren[promo_col] = "promo_flag"
    if cat_col:   ren[cat_col]   = "category"

    df = df.rename(columns=ren)

    # если store_id нет — создадим фиктивный
    if "store_id" not in df.columns:
        df["store_id"] = "STORE_001"

    # приведение типов
    df["date"] = pd.to_datetime(df["date"])
    df["sku_id"] = df["sku_id"].astype(str)
    df["store_id"] = df["store_id"].astype(str)
    df["units"] = pd.to_numeric(df["units"], errors="coerce")

    # price/promo_flag — по возможности
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    else:
        df["price"] = 100.0
    if "promo_flag" in df.columns:
        df["promo_flag"] = pd.to_numeric(df["promo_flag"], errors="coerce").fillna(0).astype(int)
    else:
        df["promo_flag"] = 0

    return df


# ---------- основная логика ----------
def evaluate(horizon: int = 14) -> Path:
    """
    На каждом SKU: делаем «ретро-прогноз» следующих H дней,
    сравниваем с фактом и сохраняем метрики в CSV.
    """
    logger.info(f"Evaluation start: horizon={horizon}")

    # 1) Источник данных → клининг → нормализация имён
    df = load_sales()              # data/raw/sales_synth.csv
    df = clean_sales(df)           # клининг (проверено)
    df = normalize_columns(df)     # гарантируем date/sku_id/store_id/units/...

    # порядок
    df = df.sort_values(['sku_id', 'store_id', 'date'])

    # 2) Куда сохраняем
    out_dir = Path(PATHS['data']['processed']).parent  # .../data/processed
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / 'metrics.csv'

    results = []

    for sku, gsku in df.groupby('sku_id', sort=False):
        gsku = gsku.sort_values('date').copy()

        last_date = gsku['date'].max()
        cutoff = last_date - pd.Timedelta(days=horizon)

        train = gsku[gsku['date'] <= cutoff].copy()
        test  = gsku[gsku['date'] >  cutoff].copy()

        if len(test) == 0 or len(train) < 30:
            continue

        # --- Prophet ---
        yhat_prophet = [np.nan] * len(test)
        try:
            df_p = train[['date', 'units']].rename(columns={'date': 'ds', 'units': 'y'})
            m = Prophet(seasonality_mode='additive')
            m.fit(df_p)
            fut = pd.DataFrame({'ds': test['date']})
            fc = m.predict(fut)
            yhat_prophet = fc['yhat'].clip(lower=0).to_numpy()
        except Exception as e:
            logger.warning(f"Prophet failed on {sku}: {e}")

        # --- LGBM ---
        yhat_lgbm = [np.nan] * len(test)
        try:
            # строим фичи
            train_f = build_features(train.copy())
            ctx = train.tail(60)  # немного истории перед тестом, чтобы лаги посчитались
            test_f  = build_features(pd.concat([ctx, test], ignore_index=True))
            test_f = test_f.iloc[-len(test):].copy()

            feature_cols = [
                'dow','week','month','promo_flag','price',
                'rolling_mean_7','rolling_std_7',
                'units_lag_1','units_lag_2','units_lag_3',
                'units_lag_7','units_lag_14','units_lag_28'
            ]
            feature_cols = [c for c in feature_cols if c in train_f.columns]

            Xtr = train_f[feature_cols].fillna(0.0)
            ytr = train_f['units'].astype(float).values

            model = LGBMRegressor(
                n_estimators=400, learning_rate=0.05, num_leaves=31,
                subsample=0.9, colsample_bytree=0.9, random_state=42
            )
            model.fit(Xtr, ytr)

            Xte = test_f[feature_cols].fillna(0.0)
            yhat_lgbm = model.predict(Xte)
            yhat_lgbm = np.clip(yhat_lgbm, 0, None)
        except Exception as e:
            logger.warning(f"LGBM failed on {sku}: {e}")

        # --- Naive ---
        try:
            last_val = float(train['units'].iloc[-1])
        except Exception:
            last_val = np.nan
        naive_level = np.array([last_val] * len(test), dtype=float)

        try:
            past = gsku[gsku['date'] <= cutoff].tail(len(test) + 7)['units'].to_numpy()
            if len(past) >= len(test) + 7:
                naive_week = past[-len(test)-7:-7]
            else:
                naive_week = naive_level
        except Exception:
            naive_week = naive_level

        naive = np.nanmean(np.vstack([naive_level, naive_week]), axis=0)
        naive = np.clip(naive, 0, None)

        # --- Ensemble ---
        ens = np.nanmean(np.vstack([yhat_prophet, yhat_lgbm]), axis=0)
        ens = np.clip(ens, 0, None)

        y_true = test['units'].astype(float).values

        row = {
            'sku_id': sku,
            'horizon': horizon,
            'mape_prophet': mape(y_true, yhat_prophet),
            'mape_lgbm':    mape(y_true, yhat_lgbm),
            'mape_naive':   mape(y_true, naive),
            'mape_ens':     mape(y_true, ens),
            'smape_prophet': smape(y_true, yhat_prophet),
            'smape_lgbm':    smape(y_true, yhat_lgbm),
            'smape_naive':   smape(y_true, naive),
            'smape_ens':     smape(y_true, ens),
            'rmse_prophet': rmse(y_true, yhat_prophet),
            'rmse_lgbm':    rmse(y_true, yhat_lgbm),
            'rmse_naive':   rmse(y_true, naive),
            'rmse_ens':     rmse(y_true, ens),
        }

        model_map = {
            'prophet': row['mape_prophet'],
            'lgbm':    row['mape_lgbm'],
            'naive':   row['mape_naive'],
            'ensemble':row['mape_ens'],
        }
        # аккуратно выбираем лучшую (учитывая NaN)
        best = min(
            (k for k in model_map.keys() if not pd.isna(model_map[k])),
            key=lambda k: model_map[k],
            default='ensemble'
        )
        row['best_model'] = best

        results.append(row)

    met = pd.DataFrame(results).sort_values(['mape_ens','mape_prophet','mape_lgbm'], na_position='last')
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    met.to_csv(out_csv, index=False)
    logger.info(f"Metrics saved -> {out_csv} (rows={len(met)})")
    print(f"✅ Metrics saved: {out_csv}  rows={len(met)}")
    return out_csv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=14)
    args = ap.parse_args()
    evaluate(horizon=args.horizon)
