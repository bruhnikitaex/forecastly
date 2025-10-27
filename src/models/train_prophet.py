import pandas as pd
import joblib
from prophet import Prophet
from pathlib import Path
from src.utils.config import PATHS, MODEL_CFG
from src.utils.logger import logger
from src.etl.load_data import load_sales
from src.etl.clean_data import clean_sales
from src.etl.feature_builder import build_features

OUT = Path(PATHS['data']['models_dir']) / 'prophet_model.pkl'

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby('date', as_index=False)['units'].sum()
    agg = agg.rename(columns={'date':'ds','units':'y'})
    return agg

def train():
    df = load_sales()
    df = clean_sales(df)
    df = build_features(df)
    train_df = prepare(df)
    logger.info('Training Prophet...')
    m = Prophet(
        weekly_seasonality=MODEL_CFG['model']['prophet']['weekly_seasonality'],
        yearly_seasonality=MODEL_CFG['model']['prophet']['yearly_seasonality'],
        seasonality_mode=MODEL_CFG['model']['prophet']['seasonality_mode'],
    )
    m.fit(train_df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(m, OUT)
    logger.info(f'Model saved to {OUT}')

if __name__ == '__main__':
    train()
