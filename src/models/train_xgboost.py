"""
Модуль обучения XGBoost модели для прогнозирования продаж.

Выполняет загрузку, очистку данных, построение признаков и обучение модели.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from xgboost import XGBRegressor
from src.utils.config import PATHS, MODEL_CFG
from src.utils.logger import logger
from src.etl.load_data import load_sales
from src.etl.clean_data import clean_sales
from src.etl.feature_builder import build_features
from src.etl.validate import validate_model_data

OUT = Path(PATHS['data']['models_dir']) / 'xgboost_model.pkl'


def train():
    """
    Обучает XGBoost модель на исторических данных продаж.
    
    Процесс:
    1. Загрузка данных
    2. Очистка данных
    3. Валидация данных
    4. Построение признаков
    5. Обучение модели
    6. Сохранение модели
    
    Raises:
        FileNotFoundError: Если файл с данными не найден.
        ValueError: Если данные некорректны или недостаточны.
    """
    
    logger.info('=' * 60)
    logger.info('🤖 Запуск обучения XGBoost модели')
    logger.info('=' * 60)
    
    try:
        # Этап 1: Загрузка данных
        logger.info('Этап 1: Загрузка данных о продажах...')
        df = load_sales()
        logger.info(f'  Загружено {len(df)} записей')
        
        # Этап 2: Очистка данных
        logger.info('Этап 2: Очистка и нормализация данных...')
        df = clean_sales(df)
        logger.info(f'  После очистки: {len(df)} записей')
        
        # Этап 3: Валидация данных
        logger.info('Этап 3: Валидация данных...')
        validate_model_data(df)
        logger.info('  ✓ Данные прошли валидацию')
        
        # Этап 4: Построение признаков
        logger.info('Этап 4: Построение признаков...')
        df = build_features(df)
        logger.info(f'  Построено признаков: {df.shape[1] - 4}')
        
        # Этап 5: Подготовка данных для обучения
        feature_cols = ['dow','week','month','units_lag_1','units_lag_7']
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            logger.error(f'Отсутствуют признаки: {missing_cols}')
            raise ValueError(f'Отсутствуют необходимые признаки: {missing_cols}')
        
        X = df[feature_cols].copy()
        y = df['units'].values.astype(float)
        mask = np.isfinite(y)
        X, y = X[mask], y[mask]
        
        if len(y) == 0:
            logger.error('После фильтрации валидных данных не осталось')
            raise ValueError('Недостаточно валидных данных для обучения')
        
        logger.info(f'  Размер обучающей выборки: {len(X)} строк')
        logger.info(f'  Используемые признаки: {feature_cols}')

        # Этап 6: Обучение модели
        logger.info('Этап 5: Обучение XGBoost модели...')
        params = MODEL_CFG.get('model', {}).get('xgboost', {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 6
        })
        
        model = XGBRegressor(
            n_estimators=params.get('n_estimators', 500),
            learning_rate=params.get('learning_rate', 0.05),
            max_depth=params.get('max_depth', 6),
            random_state=42,
            verbosity=1
        )
        
        model.fit(X, y)
        logger.info(f'  ✓ Модель успешно обучена!')
        logger.info(f'  Качество (R²): {model.score(X, y):.4f}')
        
        # Этап 7: Сохранение модели
        logger.info('Этап 6: Сохранение модели...')
        OUT.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, OUT)
        logger.info(f'  ✓ Модель сохранена в {OUT}')
        
        logger.info('=' * 60)
        logger.info('✅ Обучение завершено успешно!')
        logger.info('=' * 60)
        
        return model
        
    except Exception as e:
        logger.error(f'❌ Ошибка при обучении модели: {str(e)}', exc_info=True)
        raise


if __name__ == '__main__':
    train()
