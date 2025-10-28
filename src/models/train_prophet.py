# src/models/train_prophet.py
from pathlib import Path
import pandas as pd
from prophet import Prophet
from joblib import dump
from src.utils.config import PATHS
from src.utils.logger import logger

def train():
    processed_file = Path(PATHS['data']['processed'])      # .../processed.parquet (файл)
    models_dir     = Path(PATHS['data']['models_dir'])
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(processed_file)
    df = df[['date', 'sku_id', 'units']].copy()
    df['date'] = pd.to_datetime(df['date'])
    df.rename(columns={'date':'ds', 'units':'y'}, inplace=True)

    models = {}
    for sku in df['sku_id'].unique():
        df_sku = df[df['sku_id'] == sku][['ds','y']].sort_values('ds')
        if len(df_sku) < 30 or df_sku['y'].sum() == 0:
            logger.warning(f"Skip {sku}: not enough data")
            continue
        m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
        m.fit(df_sku)
        models[sku] = m
        logger.info(f"Prophet model trained for {sku} ({len(df_sku)} rows)")

    dump(models, models_dir / 'prophet_model.pkl')
    print(f"✅ Обучено моделей Prophet: {len(models)} → {models_dir / 'prophet_model.pkl'}")

if __name__ == "__main__":
    train()
