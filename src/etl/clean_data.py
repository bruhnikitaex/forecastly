"""
Модуль очистки и предварительной обработки данных о продажах.

Включает функции для проверки колонок, удаления выбросов, заполнения пропусков дат.
"""

import pandas as pd
import numpy as np
from src.utils.helpers import ensure_datetime
from src.utils.logger import logger

REQUIRED = ["date", "sku_id"]  # минимальный набор
OPTIONAL = ["store_id", "price", "promo_flag"]


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Проверяет наличие обязательных колонок и создаёт опциональные если их нет.
    
    Args:
        df: DataFrame для проверки.
    
    Returns:
        DataFrame с гарантированными колонками.
    
    Raises:
        ValueError: Если отсутствуют обязательные колонки.
    """
    # минимальные проверки
    missing_required = [c for c in REQUIRED if c not in df.columns]
    if missing_required:
        logger.error(f"Отсутствуют обязательные колонки: {missing_required}")
        raise ValueError(f"В исходных данных отсутствуют обязательные колонки: {missing_required}")
    # создаём опциональные при отсутствии
    if "store_id" not in df.columns:
        df["store_id"] = "S01"
    if "price" not in df.columns:
        df["price"] = 0.0
    if "promo_flag" not in df.columns:
        df["promo_flag"] = 0
    logger.info(f"✓ Все обязательные колонки присутствуют")
    return df

def remove_outliers_iqr(df: pd.DataFrame, col='units', group_cols=['sku_id','store_id']) -> pd.DataFrame:
    """
    Удаляет выбросы используя межквартильный размах (IQR).
    
    Применяет ограничение: lower = Q1 - 1.5*IQR, upper = Q3 + 1.5*IQR
    
    Args:
        df: DataFrame с данными.
        col: Название колонки для удаления выбросов (по умолчанию 'units').
        group_cols: Список колонок для группировки перед обработкой.
    
    Returns:
        DataFrame с обрезанными выбросами.
    """
    def _clip(g):
        q1 = g[col].quantile(0.25)
        q3 = g[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
        original_sum = g[col].sum()
        g[col] = g[col].clip(lower=max(lo,0), upper=hi)
        new_sum = g[col].sum()
        if original_sum != new_sum:
            logger.debug(f"Обнаружены выбросы в {g.name}, обрезаны значения")
        return g
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return df.groupby(group_cols, group_keys=False).apply(_clip)

def fill_date_gaps(df: pd.DataFrame, start=None, end=None):
    """
    Заполняет пропуски дат для каждой пары (sku_id, store_id).
    
    Для каждого товара в каждом магазине создаётся полный календарь дат,
    пропуски заполняются методом forward fill и backward fill.
    
    Args:
        df: DataFrame с колонкой 'date'.
        start: Начальная дата диапазона (по умолчанию берётся из данных).
        end: Конечная дата диапазона (по умолчанию берётся из данных).
    
    Returns:
        DataFrame с заполненными пропусками дат.
    
    Raises:
        ValueError: Если отсутствует колонка 'date'.
    """
    logger.info('Заполнение пропусков дат для каждой sku-store пары...')
    
    try:
        df = ensure_datetime(df, 'date')
    except Exception as e:
        logger.error(f"Ошибка при преобразовании дат: {str(e)}")
        raise

    # на всякий случай, если нет store_id — создаём S01
    if 'store_id' not in df.columns:
        df['store_id'] = 'S01'

    if start is None:
        start = df['date'].min()
    if end is None:
        end = df['date'].max()

    try:
        # полный календарь по всем sku/store
        idx = pd.MultiIndex.from_product(
            [pd.date_range(start, end, freq='D'),
             df['sku_id'].unique().tolist(),
             df['store_id'].unique().tolist()],
            names=['date', 'sku_id', 'store_id']
        )

        # reindex -> получаем все недостающие даты строками с NaN
        df = (df.set_index(['date', 'sku_id', 'store_id'])
                .reindex(idx)
                .reset_index())

        # какие колонки заполняем? (все, кроме ключей)
        key_cols = ['date', 'sku_id', 'store_id']
        fill_cols = [c for c in df.columns if c not in key_cols]

        # сначала прямое заполнение NaN для ожидаемых числовых
        if 'units' in df.columns:
            df['units'] = df['units'].astype('float')

        # ffill/bfill внутри каждой группы sku-store
        for c in fill_cols:
            df[c] = df.groupby(['sku_id', 'store_id'])[c].ffill().bfill()

        # финальные приведения типов/значений
        if 'units' in df.columns:
            df['units'] = df['units'].fillna(0).astype(int)
        if 'promo_flag' in df.columns:
            df['promo_flag'] = df['promo_flag'].fillna(0).astype(int)
        if 'price' in df.columns:
            df['price'] = df['price'].fillna(0.0).astype(float)
        
        logger.info(f"✓ Пропуски дат успешно заполнены")
        return df
    
    except Exception as e:
        logger.error(f"Ошибка при заполнении пропусков дат: {str(e)}")
        raise


def clean_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Главная функция очистки данных о продажах.
    
    Выполняет последовательно:
    1. Проверку и создание обязательных колонок
    2. Преобразование дат в datetime
    3. Заполнение пропусков в числовых данных
    4. Удаление дубликатов
    5. Заполнение пропусков дат
    6. Удаление выбросов по IQR
    
    Args:
        df: DataFrame с исходными данными о продажах.
    
    Returns:
        Очищенный DataFrame.
    
    Raises:
        ValueError: Если отсутствуют обязательные колонки.
    """
    logger.info('Начало очистки данных...')
    
    try:
        df = ensure_columns(df)
        df = ensure_datetime(df, 'date')
        
        # базовые заполнения
        if 'units' not in df.columns:
            df['units'] = 0
        for c in ['units','price','promo_flag']:
            if c in df.columns:
                df[c] = df[c].fillna(0)
        
        # убрать дубляжи
        dup_count = df.duplicated(subset=['date','sku_id','store_id']).sum()
        if dup_count > 0:
            logger.warning(f"⚠ Найдено {dup_count} дубликатов, удаляются")
        df = df.drop_duplicates(subset=['date','sku_id','store_id'])
        
        # заполнить пропуски дат в разрезе SKU/Store
        df = fill_date_gaps(df)
        
        # удалить выбросы по IQR, если есть units
        if 'units' in df.columns:
            df = remove_outliers_iqr(df, 'units', ['sku_id','store_id'])
        
        logger.info(f"✅ Очистка завершена успешно!")
        return df
    
    except Exception as e:
        logger.error(f"Ошибка при очистке данных: {str(e)}")
        raise
