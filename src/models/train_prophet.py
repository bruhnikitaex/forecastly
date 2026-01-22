"""
Модуль обучения Prophet модели для прогнозирования продаж.

Prophet использует временные ряды с поддержкой сезонности и тренда.
"""

from pathlib import Path
import pandas as pd
from prophet import Prophet
from joblib import dump
from src.utils.config import PATHS
from src.utils.logger import logger
from src.etl.validate import validate_model_data


def train() -> None:
    """
    Обучает отдельные Prophet модели для каждого SKU.
    
    Этапы:
    1. Загрузка обработанных данных
    2. Подготовка данных в формате Prophet (ds, y)
    3. Для каждого SKU:
       - Проверка достаточности данных (минимум 30 дней)
       - Обучение модели с поддержкой сезонности
       - Сохранение обученной модели
    4. Сохранение всех моделей в словарь pickle
    
    Raises:
        FileNotFoundError: Если файл с обработанными данными не найден.
        Exception: Если произойдёт ошибка при обучении.
    
    Logs:
        INFO: Информация о ходе обучения.
        WARNING: Пропусты SKU с недостаточными данными.
        ERROR: Критические ошибки при обучении.
    """
    logger.info('=' * 60)
    logger.info('Начало обучения Prophet моделей...')
    
    try:
        # Загрузка обработанных данных
        processed_file = Path(PATHS['data']['processed'])
        models_dir = Path(PATHS['data']['models_dir'])
        models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f'Загрузка данных из {processed_file}')
        
        # Всегда загружаем raw данные и обрабатываем их для обучения Prophet
        # (predictions.csv - это прогнозы, недостаточно для переобучения)
        from src.etl.load_data import load_sales
        from src.etl.clean_data import clean_sales
        
        df = load_sales()
        df = clean_sales(df)
        
        logger.info(f'Загружено {len(df)} строк')
        
        # Подготовка данных - выбираем правильные колонки в зависимости от источника
        if 'units' in df.columns:
            # Если загружены raw или обработанные данные
            df = df[['date', 'sku_id', 'units']].copy()
            df['date'] = pd.to_datetime(df['date'])
            df.rename(columns={'date':'ds', 'units':'y'}, inplace=True)
        elif 'prophet' in df.columns:
            # Если загружены predictions (из predict.py)
            df = df[['date', 'sku_id', 'prophet']].copy()
            df.rename(columns={'date':'ds', 'prophet':'y'}, inplace=True)
            df['ds'] = pd.to_datetime(df['ds'])
        else:
            logger.error(f"❌ Не найдены ожидаемые колонки. Доступные: {df.columns.tolist()}")
            raise ValueError(f"Неизвестный формат данных. Ожидаются 'units' или 'prophet'")
        
        models = {}
        skus = df['sku_id'].unique()
        logger.info(f'Будет обучено моделей для {len(skus)} SKU')
        
        # Обучение модели для каждого SKU
        for idx, sku in enumerate(skus, 1):
            try:
                df_sku = df[df['sku_id'] == sku][['ds','y']].sort_values('ds')
                
                # Валидация данных SKU
                if len(df_sku) < 30:
                    logger.warning(f"Пропуск {sku}: недостаточно данных ({len(df_sku)} дней < 30)")
                    continue
                
                if df_sku['y'].sum() == 0:
                    logger.warning(f"Пропуск {sku}: нулевые продажи")
                    continue
                
                logger.info(f"  [{idx}] Обучение Prophet для {sku} ({len(df_sku)} дней)...")
                
                # Обучение модели с сезонностью
                m = Prophet(
                    interval_width=0.95
                )
                m.fit(df_sku)
                models[sku] = m
                logger.info(f"    OK Модель успешно обучена для {sku}")
            
            except Exception as e:
                logger.error(f"  Ошибка при обучении {sku}: {str(e)}")
                continue
        
        if not models:
            logger.error('Не удалось обучить ни одну модель Prophet')
            raise ValueError('Обучение Prophet прервано: нет валидных SKU')
        
        # Сохранение моделей
        model_path = models_dir / 'prophet_model.pkl'
        dump(models, model_path)
        logger.info(f'Модели сохранены в {model_path}')
        
        logger.info('=' * 60)
        logger.info(f'Обучение Prophet завершено успешно!')
        print(f"[OK] Обучено моделей Prophet: {len(models)} -> {model_path}")
        
    except Exception as e:
        logger.error(f'Критическая ошибка при обучении Prophet: {str(e)}', exc_info=True)
        print(f"[ERROR] Ошибка при обучении Prophet: {str(e)}")
        raise


if __name__ == "__main__":
    train()
