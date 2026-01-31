"""
Модуль обучения LightGBM модели для прогнозирования продаж.

LightGBM - быстрая альтернатива XGBoost для табличных данных.
"""

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
from src.etl.validate import validate_model_data

OUT = Path(PATHS['data']['models_dir']) / 'lightgbm_model.pkl'


def train():
    """
    Обучает LightGBM модель на исторических данных продаж.

    Процесс:
    1. Загрузка данных
    2. Очистка данных
    3. Валидация данных
    4. Построение признаков
    5. Обучение модели
    6. Сохранение модели
    """

    logger.info('=' * 60)
    logger.info('LightGBM: Запуск обучения модели')
    logger.info('=' * 60)

    try:
        logger.info('Этап 1: Загрузка данных о продажах...')
        df = load_sales()
        logger.info(f'  Загружено {len(df)} записей')

        logger.info('Этап 2: Очистка и нормализация данных...')
        df = clean_sales(df)
        logger.info(f'  После очистки: {len(df)} записей')

        logger.info('Этап 3: Валидация данных...')
        validate_model_data(df)
        logger.info('  Данные прошли валидацию')

        logger.info('Этап 4: Построение признаков...')
        df = build_features(df)
        logger.info(f'  Построено признаков: {df.shape[1] - 4}')

        feature_cols = ['dow', 'week', 'month', 'units_lag_1', 'units_lag_7',
                        'units_lag_14', 'units_ma_7', 'units_ma_28']
        # Используем только те признаки, которые есть
        feature_cols = [c for c in feature_cols if c in df.columns]

        X = df[feature_cols].copy()
        y = df['units'].values.astype(float)
        mask = np.isfinite(y)
        X, y = X[mask], y[mask]

        if len(y) == 0:
            raise ValueError('Недостаточно валидных данных для обучения')

        logger.info(f'  Размер обучающей выборки: {len(X)} строк')
        logger.info(f'  Используемые признаки: {feature_cols}')

        logger.info('Этап 5: Обучение LightGBM модели...')
        params = MODEL_CFG.get('model', {}).get('lightgbm', {})

        model = LGBMRegressor(
            n_estimators=params.get('n_estimators', 500),
            learning_rate=params.get('learning_rate', 0.05),
            max_depth=params.get('max_depth', 6),
            num_leaves=params.get('num_leaves', 31),
            random_state=42,
            verbosity=-1
        )

        model.fit(X, y)
        logger.info(f'  Модель успешно обучена!')
        logger.info(f'  Качество (R2): {model.score(X, y):.4f}')

        logger.info('Этап 6: Сохранение модели...')
        OUT.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, OUT)
        logger.info(f'  Модель сохранена в {OUT}')

        logger.info('=' * 60)
        logger.info('LightGBM: Обучение завершено успешно!')
        logger.info('=' * 60)
        print(f"[OK] LightGBM model saved to {OUT}")

        return model

    except Exception as e:
        logger.error(f'Ошибка при обучении LightGBM: {str(e)}', exc_info=True)
        raise


if __name__ == '__main__':
    train()
