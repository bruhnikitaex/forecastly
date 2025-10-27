import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from lightgbm import LGBMRegressor
from src.utils.config import PATHS, MODEL_CFG
from src.utils.logger import logger
from src.etl.load_data import load_sales
from src.etl.clean_data import clean_sales
from src.etl.feature_builder import build_features

OUT = Path(PATHS['data']['models_dir']) / 'lgbm_model.pkl'

def train():
    df = load_sales()
    df = clean_sales(df)
    df = build_features(df)
    X = df[['dow','week','month','units_lag_1','units_lag_7']].copy()
    y = df['units'].values
    mask = ~np.isnan(y)
    X, y = X[mask], y[mask]

    logger.info('Training LightGBM...')
    model = LGBMRegressor(
        n_estimators=MODEL_CFG['model']['lgbm']['n_estimators'],
        learning_rate=MODEL_CFG['model']['lgbm']['learning_rate'],
        num_leaves=MODEL_CFG['model']['lgbm']['num_leaves'],
        random_state=42
    )
    model.fit(X, y)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT)
    logger.info(f'Model saved to {OUT}')

if __name__ == '__main__':
    train()
