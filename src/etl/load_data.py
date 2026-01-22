"""
Модуль загрузки сырых данных о продажах.

Предоставляет функции для чтения данных из CSV файлов.
"""

import pandas as pd
from pathlib import Path
from src.utils.config import PATHS
from src.utils.logger import logger


def load_sales(path: str | None = None) -> pd.DataFrame:
    """
    Загружает данные о продажах из CSV файла.
    
    Если путь не указан, использует путь из конфига (PATHS['data']['raw']).
    
    Args:
        path: Опциональный путь к CSV файлу. Если None, используется путь по умолчанию.
    
    Returns:
        pandas.DataFrame с загруженными данными о продажах.
    
    Raises:
        FileNotFoundError: Если файл не найден.
        pd.errors.ParserError: Если файл не является корректным CSV.
    
    Example:
        >>> df = load_sales('data/raw/sales_synth.csv')
        >>> print(df.shape)
        (1000, 5)
    """
    csv_path = Path(path) if path else Path(PATHS['data']['raw'])
    
    if not csv_path.exists():
        logger.error(f'Файл не найден: {csv_path}')
        raise FileNotFoundError(f'CSV файл не найден по пути: {csv_path}')
    
    try:
        logger.info(f'Загрузка данных из {csv_path}')
        df = pd.read_csv(csv_path)
        logger.info(f'Успешно загружено {len(df)} строк, {len(df.columns)} колонок')
        return df
    except pd.errors.ParserError as e:
        logger.error(f'Ошибка при парсинге CSV: {str(e)}')
        raise
    except Exception as e:
        logger.error(f'Неожиданная ошибка при загрузке данных: {str(e)}')
        raise


if __name__ == '__main__':
    df = load_sales()
    print(df.head())
