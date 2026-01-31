"""
Модуль загрузки сырых данных о продажах.

Поддерживаемые форматы: CSV, XLSX.
"""

import pandas as pd
from pathlib import Path
from src.utils.config import PATHS
from src.utils.logger import logger


def load_sales(path: str | None = None) -> pd.DataFrame:
    """
    Загружает данные о продажах из CSV или XLSX файла.

    Если путь не указан, использует путь из конфига (PATHS['data']['raw']).

    Args:
        path: Опциональный путь к файлу. Если None, используется путь по умолчанию.

    Returns:
        pandas.DataFrame с загруженными данными о продажах.

    Raises:
        FileNotFoundError: Если файл не найден.
        ValueError: Если формат файла не поддерживается.
    """
    file_path = Path(path) if path else Path(PATHS['data']['raw'])

    if not file_path.exists():
        logger.error(f'Файл не найден: {file_path}')
        raise FileNotFoundError(f'Файл не найден по пути: {file_path}')

    ext = file_path.suffix.lower()

    try:
        logger.info(f'Загрузка данных из {file_path}')

        if ext == '.xlsx':
            df = pd.read_excel(file_path, engine='openpyxl')
        elif ext in ('.csv', ''):
            df = pd.read_csv(file_path)
        else:
            raise ValueError(
                f'Неподдерживаемый формат файла: {ext}. Используйте CSV или XLSX.'
            )

        # Нормализация колонок
        df.columns = [c.strip().lower() for c in df.columns]
        col_map = {"qty": "units", "quantity": "units", "sales": "units"}
        df.rename(columns=col_map, inplace=True)

        logger.info(f'Успешно загружено {len(df)} строк, {len(df.columns)} колонок')
        return df
    except (pd.errors.ParserError, ValueError) as e:
        logger.error(f'Ошибка при чтении файла: {str(e)}')
        raise
    except Exception as e:
        logger.error(f'Неожиданная ошибка при загрузке данных: {str(e)}')
        raise


if __name__ == '__main__':
    df = load_sales()
    print(df.head())
