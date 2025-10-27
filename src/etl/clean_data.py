import pandas as pd
import numpy as np
from src.utils.helpers import ensure_datetime
from src.utils.logger import logger

REQUIRED = ["date", "sku_id"]  # минимальный набор
OPTIONAL = ["store_id", "price", "promo_flag"]

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    # минимальные проверки
    missing_required = [c for c in REQUIRED if c not in df.columns]
    if missing_required:
        raise ValueError(f"В исходных данных отсутствуют обязательные колонки: {missing_required}")
    # создаём опциональные при отсутствии
    if "store_id" not in df.columns:
        df["store_id"] = "S01"
    if "price" not in df.columns:
        df["price"] = 0.0
    if "promo_flag" not in df.columns:
        df["promo_flag"] = 0
    return df

def remove_outliers_iqr(df: pd.DataFrame, col='units', group_cols=['sku_id','store_id']):
    def _clip(g):
        q1 = g[col].quantile(0.25)
        q3 = g[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
        g[col] = g[col].clip(lower=max(lo,0), upper=hi)
        return g
    return df.groupby(group_cols, group_keys=False).apply(_clip)

def fill_date_gaps(df: pd.DataFrame, start=None, end=None):
    """
    Заполняем пропуски дат для каждой пары (sku_id, store_id).
    Никаких merge — только reindex + ffill/bfill по группам.
    """
    logger.info('Filling date gaps per sku-store...')
    df = ensure_datetime(df, 'date')

    # на всякий случай, если нет store_id — создаём S01
    if 'store_id' not in df.columns:
        df['store_id'] = 'S01'

    if start is None:
        start = df['date'].min()
    if end is None:
        end = df['date'].max()

    # полный календарь по всем sku/store
    idx = pd.MultiIndex.from_product(
        [pd.date_range(start, end, freq='D'),
         df['sku_id'].unique(),
         df['store_id'].unique()],
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

    return df


def clean_sales(df: pd.DataFrame) -> pd.DataFrame:
    logger.info('Cleaning data...')
    df = ensure_columns(df)
    df = ensure_datetime(df, 'date')
    # базовые заполнения
    if 'units' not in df.columns:
        df['units'] = 0
    for c in ['units','price','promo_flag']:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    # убрать дубляжи
    df = df.drop_duplicates(subset=['date','sku_id','store_id'])
    # заполнить пропуски дат в разрезе SKU/Store
    df = fill_date_gaps(df)
    # удалить выбросы по IQR, если есть units
    if 'units' in df.columns:
        df = remove_outliers_iqr(df, 'units', ['sku_id','store_id'])
    return df
