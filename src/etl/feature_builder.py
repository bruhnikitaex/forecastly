import pandas as pd
from src.utils.logger import logger

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info('Adding calendar features...')
    df['dow'] = df['date'].dt.dayofweek
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    df['month'] = df['date'].dt.month
    return df

def add_lag_rolling(df: pd.DataFrame, lags=(1,7,14), windows=(7,28)):
    logger.info(f'Adding lags {lags} and rolling means {windows} per sku-store')
    df = df.sort_values(['sku_id','store_id','date'])
    for l in lags:
        df[f'units_lag_{l}'] = df.groupby(['sku_id','store_id'])['units'].shift(l)
    for w in windows:
        df[f'units_ma_{w}'] = df.groupby(['sku_id','store_id'])['units'].rolling(window=w, min_periods=1).mean().reset_index(level=[0,1], drop=True)
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_calendar_features(df)
    df = add_lag_rolling(df)
    df = df.fillna(0)
    return df
