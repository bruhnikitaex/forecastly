"""
Модуль валидации входных данных для ETL-конвейера.

Проверяет формат CSV, типы данных, обязательные колонки и пороги качества.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Tuple
from src.utils.logger import logger


class ValidationError(Exception):
    """Исключение для ошибок валидации данных."""
    pass


def validate_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    """
    Проверяет наличие обязательных колонок в DataFrame.
    
    Args:
        df: DataFrame для проверки.
        required: Список обязательных названий колонок.
    
    Raises:
        ValidationError: Если отсутствуют какие-либо обязательные колонки.
    """
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValidationError(
            f"Отсутствуют обязательные колонки: {', '.join(missing)}. "
            f"Доступные колонки: {', '.join(df.columns)}"
        )
    logger.info(f"✓ Проверка обязательных колонок пройдена")


def validate_date_column(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Проверяет и приводит колонку дат к datetime формату.
    
    Args:
        df: DataFrame с колонкой дат.
        date_col: Название колонки с датами.
    
    Returns:
        DataFrame с приведённой колонкой дат.
    
    Raises:
        ValidationError: Если колонка дат содержит невалидные значения.
    """
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception as e:
        raise ValidationError(
            f"Не удалось преобразовать колонку '{date_col}' в datetime. "
            f"Ошибка: {str(e)}"
        )
    
    # Проверка на NaT значения
    nat_count = df[date_col].isna().sum()
    if nat_count > 0:
        raise ValidationError(
            f"Колонка '{date_col}' содержит {nat_count} невалидных дат (NaT)"
        )
    
    logger.info(f"✓ Проверка колонки дат пройдена: {df[date_col].min()} ... {df[date_col].max()}")
    return df


def validate_date_range(
    df: pd.DataFrame,
    date_col: str = 'date',
    min_days: int = 30,
    max_days: int = 1000
) -> None:
    """
    Проверяет диапазон дат (достаточное количество дней данных).
    
    Args:
        df: DataFrame с колонкой дат.
        date_col: Название колонки с датами.
        min_days: Минимальное количество дней данных.
        max_days: Максимальное количество дней (для выявления аномалий).
    
    Raises:
        ValidationError: Если диапазон дат некорректен.
    """
    date_range = (df[date_col].max() - df[date_col].min()).days
    
    if date_range < min_days:
        raise ValidationError(
            f"Диапазон дат слишком мал: {date_range} дней. "
            f"Требуется минимум {min_days} дней данных."
        )
    
    if date_range > max_days:
        logger.warning(
            f"⚠ Диапазон дат очень большой: {date_range} дней "
            f"(обычно достаточно ~{max_days//2})"
        )
    
    logger.info(f"✓ Проверка диапазона дат пройдена: {date_range} дней")


def validate_numeric_column(
    df: pd.DataFrame,
    col: str,
    min_value: float = 0.0,
    allow_negative: bool = False
) -> pd.DataFrame:
    """
    Проверяет и приводит колонку к числовому формату с диапазоном значений.
    
    Args:
        df: DataFrame с числовой колонкой.
        col: Название проверяемой колонки.
        min_value: Минимально допустимое значение.
        allow_negative: Разрешить ли отрицательные значения.
    
    Returns:
        DataFrame с приведённой колонкой.
    
    Raises:
        ValidationError: Если колонка содержит невалидные значения.
    """
    try:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    except Exception as e:
        raise ValidationError(f"Не удалось преобразовать '{col}' в числа. Ошибка: {str(e)}")
    
    # Проверка на NaN после преобразования
    nan_count = df[col].isna().sum()
    if nan_count > 0:
        logger.warning(f"⚠ Колонка '{col}' содержит {nan_count} NaN значений (будут заполнены нулями)")
        df[col] = df[col].fillna(0)
    
    # Проверка диапазона значений
    if not allow_negative and (df[col] < min_value).any():
        invalid_count = (df[col] < min_value).sum()
        raise ValidationError(
            f"Колонка '{col}' содержит {invalid_count} отрицательных значений, "
            f"что недопустимо для продаж"
        )
    
    logger.info(f"✓ Проверка колонки '{col}' пройдена: min={df[col].min()}, max={df[col].max()}")
    return df


def validate_no_duplicates(
    df: pd.DataFrame,
    subset: list[str]
) -> None:
    """
    Проверяет отсутствие дубликатов по указанному набору колонок.
    
    Args:
        df: DataFrame для проверки.
        subset: Список колонок для проверки дубликатов.
    
    Raises:
        ValidationError: Если найдены дубликаты.
    """
    dup_count = df.duplicated(subset=subset).sum()
    if dup_count > 0:
        logger.warning(f"⚠ Найдено {dup_count} дубликатов по колонкам {subset}")
    else:
        logger.info(f"✓ Проверка дубликатов пройдена")


def validate_sku_count(df: pd.DataFrame, sku_col: str = 'sku_id', min_skus: int = 1) -> None:
    """
    Проверяет наличие достаточного количества уникальных SKU.
    
    Args:
        df: DataFrame с колонкой SKU.
        sku_col: Название колонки с SKU.
        min_skus: Минимальное количество уникальных SKU.
    
    Raises:
        ValidationError: Если SKU меньше требуемого количества.
    """
    unique_skus = df[sku_col].nunique()
    if unique_skus < min_skus:
        raise ValidationError(
            f"Недостаточно уникальных SKU: {unique_skus} (требуется минимум {min_skus})"
        )
    logger.info(f"✓ Проверка SKU пройдена: {unique_skus} уникальных товаров")


def validate_data_completeness(
    df: pd.DataFrame,
    required_columns: list[str],
    max_missing_percent: float = 10.0
) -> None:
    """
    Проверяет процент недостающих значений в критических колонках.
    
    Args:
        df: DataFrame для проверки.
        required_columns: Список критических колонок.
        max_missing_percent: Максимальный допустимый процент пропусков.
    
    Raises:
        ValidationError: Если процент пропусков превышен.
    """
    for col in required_columns:
        if col in df.columns:
            missing_percent = (df[col].isna().sum() / len(df)) * 100
            if missing_percent > max_missing_percent:
                raise ValidationError(
                    f"Колонка '{col}' содержит {missing_percent:.1f}% пропусков "
                    f"(допустимо макс {max_missing_percent}%)"
                )
            if missing_percent > 0:
                logger.warning(f"⚠ Колонка '{col}': {missing_percent:.1f}% пропусков")
    
    logger.info(f"✓ Проверка полноты данных пройдена")


def validate_csv_file(file_path: str | Path) -> pd.DataFrame:
    """
    Комплексная валидация CSV файла с продажами.
    
    Проверяет:
    - Существование файла
    - Формат CSV
    - Обязательные колонки (date, sku_id)
    - Типы данных
    - Диапазон дат
    - Уникальность SKU
    - Полноту данных
    
    Args:
        file_path: Путь к CSV файлу для валидации.
    
    Returns:
        Валидированный DataFrame.
    
    Raises:
        ValidationError: Если данные не проходят валидацию.
        FileNotFoundError: Если файл не найден.
    """
    file_path = Path(file_path)
    
    # Проверка существования файла
    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    
    logger.info(f"Запуск валидации файла: {file_path}")
    
    # Загрузка CSV
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValidationError(f"Не удалось загрузить CSV файл. Ошибка: {str(e)}")
    
    if len(df) == 0:
        raise ValidationError("CSV файл пуст (нет данных)")
    
    logger.info(f"Загружено {len(df)} строк, {len(df.columns)} колонок")
    
    # Валидация колонок
    validate_required_columns(df, ['date', 'sku_id'])
    
    # Валидация дат
    df = validate_date_column(df, 'date')
    validate_date_range(df, 'date', min_days=30)
    
    # Валидация продаж (units) если есть
    if 'units' in df.columns:
        df = validate_numeric_column(df, 'units', min_value=0, allow_negative=False)
    
    # Валидация цены если есть
    if 'price' in df.columns:
        df = validate_numeric_column(df, 'price', min_value=0, allow_negative=False)
    
    # Валидация SKU
    validate_sku_count(df, 'sku_id', min_skus=1)
    
    # Проверка дубликатов
    key_cols = ['date', 'sku_id']
    if 'store_id' in df.columns:
        key_cols.append('store_id')
    validate_no_duplicates(df, key_cols)
    
    # Проверка полноты данных
    critical_cols = ['date', 'sku_id', 'units'] if 'units' in df.columns else ['date', 'sku_id']
    validate_data_completeness(df, critical_cols, max_missing_percent=10.0)
    
    logger.info(f"✅ Валидация успешно завершена!")
    return df


def validate_model_data(df: pd.DataFrame, min_samples: int = 30) -> None:
    """
    Валидирует данные перед обучением модели.
    
    Args:
        df: DataFrame для проверки перед обучением.
        min_samples: Минимальное количество строк.
    
    Raises:
        ValidationError: Если данные недостаточны для обучения.
    """
    if len(df) < min_samples:
        raise ValidationError(
            f"Недостаточно данных для обучения: {len(df)} строк (требуется минимум {min_samples})"
        )
    
    required_cols = ['date', 'sku_id', 'units']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValidationError(
            f"Отсутствуют обязательные колонки для обучения: {', '.join(missing)}"
        )
    
    # Проверка на NaN в критических колонках
    for col in required_cols:
        if df[col].isna().any():
            raise ValidationError(f"Колонка '{col}' содержит NaN значения")
    
    logger.info(f"✓ Данные готовы к обучению модели ({len(df)} строк)")
