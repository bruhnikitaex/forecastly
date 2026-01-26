"""
Модуль построения признаков для моделей прогнозирования.

Включает календарные признаки, лаги и скользящие средние.
"""

import pandas as pd
from src.utils.logger import logger


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет календарные признаки к данным.

    Args:
        df: DataFrame с колонкой date.

    Returns:
        DataFrame с добавленными колонками: dow, week, month.
    """
    logger.info('Adding calendar features...')
    df['dow'] = df['date'].dt.dayofweek
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    df['month'] = df['date'].dt.month
    return df

def add_lag_rolling(df: pd.DataFrame, lags=(1, 7, 14), windows=(7, 28)) -> pd.DataFrame:
    """
    Добавляет лаговые признаки и скользящие средние.

    Args:
        df: DataFrame с колонками sku_id, store_id, date, units.
        lags: Кортеж лагов для создания (по умолчанию 1, 7, 14 дней).
        windows: Кортеж окон для скользящих средних (по умолчанию 7 и 28 дней).

    Returns:
        DataFrame с добавленными признаками units_lag_N и units_ma_N.
    """
    logger.info(f'Adding lags {lags} and rolling means {windows} per sku-store')
    df = df.sort_values(['sku_id', 'store_id', 'date'])
    for lag in lags:
        df[f'units_lag_{lag}'] = df.groupby(['sku_id', 'store_id'])['units'].shift(lag)
    for window in windows:
        df[f'units_ma_{window}'] = df.groupby(['sku_id', 'store_id'])['units'].rolling(window=window, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Главная функция построения признаков для прогнозирования.

    Выполняет последовательно:
    1. Добавление календарных признаков (dow, week, month)
    2. Добавление лагов и скользящих средних
    3. Заполнение пропущенных значений нулями

    Args:
        df: DataFrame с колонками date, sku_id, store_id, units.

    Returns:
        DataFrame с добавленными признаками.
    """
    df = add_calendar_features(df)
    df = add_lag_rolling(df)
    df = df.fillna(0)
    return df
