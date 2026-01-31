# src/models/evaluate.py
"""
Модуль оценки качества моделей прогнозирования.

Включает:
- Метрики: MAE, RMSE, MAPE, sMAPE
- Rolling/expanding window backtesting (2-3 фолда)
- Сравнение Prophet, XGBoost, LightGBM, Naive, Ensemble
- Выбор лучшей модели per SKU
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.logger import logger
from src.utils.config import PATHS
from src.etl.load_data import load_sales
from src.etl.clean_data import clean_sales
from src.etl.feature_builder import build_features


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return np.nan
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error (с защитой от деления на 0)."""
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    mask = (y_true > 0) & np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return np.nan
    denom = (np.abs(y_true[mask]) + np.abs(y_pred[mask])) / 2
    denom = np.where(denom == 0, 1, denom)
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom) * 100)


def _train_predict_prophet(train_data, test_dates):
    """Обучает Prophet и возвращает прогноз."""
    from prophet import Prophet
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    df_p = train_data[['date', 'units']].rename(columns={'date': 'ds', 'units': 'y'})
    m.fit(df_p)
    fc = m.predict(pd.DataFrame({'ds': test_dates}))
    return fc['yhat'].values


def _train_predict_xgb(train_data, test_data, feature_cols):
    """Обучает XGBoost и возвращает прогноз."""
    from xgboost import XGBRegressor
    train_feat = build_features(train_data.copy())
    test_feat = build_features(test_data.copy())
    model = XGBRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        random_state=42, verbosity=0
    )
    model.fit(train_feat[feature_cols].fillna(0), train_feat['units'])
    return model.predict(test_feat[feature_cols].fillna(0))


def _train_predict_lgbm(train_data, test_data, feature_cols):
    """Обучает LightGBM и возвращает прогноз."""
    from lightgbm import LGBMRegressor
    train_feat = build_features(train_data.copy())
    test_feat = build_features(test_data.copy())
    model = LGBMRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        num_leaves=31, random_state=42, verbosity=-1
    )
    model.fit(train_feat[feature_cols].fillna(0), train_feat['units'])
    return model.predict(test_feat[feature_cols].fillna(0))


def evaluate(horizon: int = 14, n_folds: int = 3):
    """
    Оценивает модели с помощью rolling window backtesting.

    Args:
        horizon: Горизонт прогноза (дни).
        n_folds: Количество фолдов для backtesting (2-3).
    """
    logger.info(f"Evaluation start: horizon={horizon}, folds={n_folds}")

    df = load_sales()
    df = clean_sales(df)
    df = build_features(df)
    df = df.sort_values(['sku_id', 'store_id', 'date'])

    xgb_features = ['dow', 'week', 'month', 'units_lag_1', 'units_lag_7']
    lgbm_features = ['dow', 'week', 'month', 'units_lag_1', 'units_lag_7',
                      'units_lag_14', 'units_ma_7', 'units_ma_28']
    lgbm_features = [c for c in lgbm_features if c in df.columns]

    results = []

    for sku, g in df.groupby('sku_id'):
        sku = str(sku)
        g = g.sort_values('date').reset_index(drop=True)

        min_data = horizon * (n_folds + 1)
        if len(g) < min_data:
            if len(g) >= 90:
                n_folds_sku = max(2, (len(g) // horizon) - 1)
                n_folds_sku = min(n_folds_sku, n_folds)
            else:
                continue
        else:
            n_folds_sku = n_folds

        # Собираем метрики по фолдам
        fold_metrics = {m: {metric: [] for metric in ['mae', 'rmse', 'mape', 'smape']}
                        for m in ['prophet', 'xgboost', 'lightgbm', 'naive', 'ensemble']}

        for fold in range(n_folds_sku):
            test_end = len(g) - fold * horizon
            test_start = test_end - horizon
            if test_start < 30:
                break

            train = g.iloc[:test_start].copy()
            test = g.iloc[test_start:test_end].copy()
            y_true = test['units'].values

            # Prophet
            try:
                y_prophet = _train_predict_prophet(train, test['date'].values)
            except Exception:
                y_prophet = np.full(len(test), np.nan)

            # XGBoost
            try:
                y_xgb = _train_predict_xgb(train, test, xgb_features)
            except Exception:
                y_xgb = np.full(len(test), np.nan)

            # LightGBM
            try:
                y_lgbm = _train_predict_lgbm(train, test, lgbm_features)
            except Exception:
                y_lgbm = np.full(len(test), np.nan)

            # Naive (последнее значение)
            y_naive = np.full(len(test), train['units'].iloc[-1])

            # Ensemble (среднее доступных моделей)
            preds_stack = []
            for arr in [y_prophet, y_xgb, y_lgbm]:
                if not np.all(np.isnan(arr)):
                    preds_stack.append(arr)
            y_ensemble = np.nanmean(preds_stack, axis=0) if preds_stack else np.full(len(test), np.nan)

            # Считаем метрики
            predictions = {
                'prophet': y_prophet, 'xgboost': y_xgb,
                'lightgbm': y_lgbm, 'naive': y_naive, 'ensemble': y_ensemble
            }
            for model_name, y_pred in predictions.items():
                fold_metrics[model_name]['mae'].append(mae(y_true, y_pred))
                fold_metrics[model_name]['rmse'].append(rmse(y_true, y_pred))
                fold_metrics[model_name]['mape'].append(mape(y_true, y_pred))
                fold_metrics[model_name]['smape'].append(smape(y_true, y_pred))

        # Среднее по фолдам
        row = {'sku_id': sku, 'n_folds': n_folds_sku}
        for model_name in ['prophet', 'xgboost', 'lightgbm', 'naive', 'ensemble']:
            prefix = model_name if model_name != 'ensemble' else 'ens'
            for metric_name in ['mae', 'rmse', 'mape', 'smape']:
                vals = [v for v in fold_metrics[model_name][metric_name] if not np.isnan(v)]
                avg = np.mean(vals) if vals else np.nan
                col_name = f"{metric_name}_{prefix}"
                row[col_name] = round(avg, 2) if not np.isnan(avg) else np.nan

        # Backward compatibility: mape_prophet, mape_xgboost, mape_naive, mape_ens
        row['mape_prophet'] = row.get('mape_prophet', np.nan)
        row['mape_xgboost'] = row.get('mape_xgboost', np.nan)
        row['mape_naive'] = row.get('mape_naive', np.nan)
        row['mape_ens'] = row.get('mape_ens', np.nan)
        row['mape_lightgbm'] = row.get('mape_lightgbm', np.nan)

        # Выбор лучшей модели
        candidates = [
            ('prophet', row.get('mape_prophet', np.nan)),
            ('xgboost', row.get('mape_xgboost', np.nan)),
            ('lightgbm', row.get('mape_lightgbm', np.nan)),
            ('naive', row.get('mape_naive', np.nan)),
            ('ens', row.get('mape_ens', np.nan)),
        ]
        best = min(candidates, key=lambda x: x[1] if not np.isnan(x[1]) else 999)
        row['best_model'] = best[0]

        results.append(row)

    met = pd.DataFrame(results)

    # Сохраняем
    base = Path(PATHS['data']['processed'])
    if base.suffix:
        base = base.parent
    base.mkdir(parents=True, exist_ok=True)
    out_path = base / 'metrics.csv'
    met.to_csv(out_path, index=False)

    logger.info(f"Evaluation complete. {len(met)} SKUs evaluated with {n_folds} folds.")
    logger.info(f"Results saved to {out_path}")
    print(f"[OK] Evaluation complete. Results saved to {out_path}")

    # Вывод сводки
    if not met.empty:
        print(f"\nСводка метрик (средние по всем SKU):")
        for model_name in ['prophet', 'xgboost', 'lightgbm', 'naive', 'ens']:
            mape_col = f'mape_{model_name}'
            mae_col = f'mae_{model_name}'
            rmse_col = f'rmse_{model_name}'
            if mape_col in met.columns:
                print(f"  {model_name:>10}: MAPE={met[mape_col].mean():.1f}%  "
                      f"MAE={met.get(mae_col, pd.Series([np.nan])).mean():.1f}  "
                      f"RMSE={met.get(rmse_col, pd.Series([np.nan])).mean():.1f}")
        print(f"\nЛучшая модель (по частоте): {met['best_model'].value_counts().index[0]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=14)
    ap.add_argument("--folds", type=int, default=3)
    args = ap.parse_args()
    evaluate(horizon=args.horizon, n_folds=args.folds)
