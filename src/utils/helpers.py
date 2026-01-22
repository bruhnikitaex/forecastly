"""
Вспомогательные функции для обработки данных.
"""
import pandas as pd
from src.utils.logger import logger


def ensure_datetime(df: pd.DataFrame, col: str = 'date') -> pd.DataFrame:
    """
    Преобразует колонку в datetime и сортирует DataFrame.

    Args:
        df: DataFrame с колонкой дат.
        col: Название колонки с датами.

    Returns:
        DataFrame с преобразованной колонкой, отсортированный по дате.

    Raises:
        ValueError: Если колонка не существует или не может быть преобразована.
    """
    if col not in df.columns:
        raise ValueError(f"Колонка '{col}' не найдена в DataFrame. Доступные: {df.columns.tolist()}")

    try:
        df = df.copy()
        df[col] = pd.to_datetime(df[col])
        return df.sort_values(col)
    except Exception as e:
        logger.error(f"Ошибка при преобразовании колонки '{col}' в datetime: {e}")
        raise ValueError(f"Не удалось преобразовать колонку '{col}' в datetime: {e}")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Безопасное деление с обработкой деления на ноль.

    Args:
        numerator: Числитель.
        denominator: Знаменатель.
        default: Значение по умолчанию при делении на ноль.

    Returns:
        Результат деления или default.
    """
    if denominator == 0:
        return default
    return numerator / denominator


def format_number(value: float, decimals: int = 2) -> str:
    """
    Форматирует число для отображения.

    Args:
        value: Число для форматирования.
        decimals: Количество знаков после запятой.

    Returns:
        Отформатированная строка.
    """
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def round_metrics(metrics: dict, decimals: int = 2) -> dict:
    """
    Округляет числовые значения в словаре метрик.

    Args:
        metrics: Словарь с метриками.
        decimals: Количество знаков после запятой.

    Returns:
        Словарь с округлёнными значениями.
    """
    result = {}
    for key, value in metrics.items():
        if value is None:
            result[key] = None
        elif isinstance(value, (int, float)):
            result[key] = round(value, decimals)
        else:
            result[key] = value
    return result


def validate_dataframe(df: pd.DataFrame, required_cols: list[str]) -> bool:
    """
    Проверяет наличие обязательных колонок в DataFrame.

    Args:
        df: DataFrame для проверки.
        required_cols: Список обязательных колонок.

    Returns:
        True если все колонки присутствуют.

    Raises:
        ValueError: Если отсутствуют обязательные колонки.
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют обязательные колонки: {missing}")
    return True
