"""
Модуль построения признаков для моделей прогнозирования.

Включает календарные признаки, лаги, скользящие средние и праздники.
"""

import pandas as pd
import numpy as np
from src.utils.logger import logger


# Российские праздники (фиксированные даты)
RU_HOLIDAYS_FIXED = [
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),  # Новогодние
    (2, 23),  # День защитника Отечества
    (3, 8),   # Международный женский день
    (5, 1),   # Праздник Весны и Труда
    (5, 9),   # День Победы
    (6, 12),  # День России
    (11, 4),  # День народного единства
]


def is_ru_holiday(date: pd.Timestamp) -> bool:
    """Проверяет, является ли дата российским праздником."""
    return (date.month, date.day) in RU_HOLIDAYS_FIXED


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет календарные признаки к данным.

    Args:
        df: DataFrame с колонкой date.

    Returns:
        DataFrame с колонками: dow, week, month, is_weekend, is_holiday,
        day_of_year, quarter.
    """
    logger.info('Adding calendar features...')
    df['dow'] = df['date'].dt.dayofweek
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    df['month'] = df['date'].dt.month
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    df['is_holiday'] = df['date'].apply(is_ru_holiday).astype(int)
    df['day_of_year'] = df['date'].dt.dayofyear
    df['quarter'] = df['date'].dt.quarter
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
