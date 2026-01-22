"""
Модуль управления конфигурацией приложения.

Загружает параметры из YAML файлов и environment переменных.
Поддерживает переопределение значений через .env файл.
"""

import yaml
import os
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv


def load_config(p: str) -> Dict[str, Any]:
    """
    Загружает YAML конфигурационный файл.
    
    Args:
        p: Путь к YAML файлу.
    
    Returns:
        Словарь с параметрами конфигурации.
    
    Raises:
        FileNotFoundError: Если файл не найден.
        yaml.YAMLError: Если YAML невалидный.
    """
    config_path = Path(p)
    if not config_path.exists():
        raise FileNotFoundError(f"Конфиг файл не найден: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Ошибка парсинга YAML {config_path}: {str(e)}")


def get_env_var(key: str, default: Any = None, var_type: type = str) -> Any:
    """
    Получает переменную окружения с опциональным типизацией.
    
    Args:
        key: Название переменной окружения.
        default: Значение по умолчанию если переменная не установлена.
        var_type: Тип переменной (str, int, bool, float).
    
    Returns:
        Значение переменной окружения или default.
    """
    value = os.getenv(key, default)
    if value is None:
        return None
    
    try:
        if var_type == bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        elif var_type == int:
            return int(value)
        elif var_type == float:
            return float(value)
        else:
            return str(value)
    except (ValueError, AttributeError):
        return default


# Загружаем .env файл
ROOT = Path(__file__).resolve().parents[2]
env_file = ROOT / '.env'
if env_file.exists():
    load_dotenv(env_file)


# Загружаем основные конфиги
ROOT = Path(__file__).resolve().parents[2]
PATHS = load_config(str(ROOT / 'configs' / 'paths.yaml'))
MODEL_CFG = load_config(str(ROOT / 'configs' / 'model.yaml'))

# Переопределяем пути из environment переменных если установлены
PATHS['data']['raw'] = get_env_var('DATA_RAW_DIR', PATHS['data'].get('raw', 'data/raw/sales_synth.csv'))
PATHS['data']['processed'] = get_env_var('DATA_PROCESSED_DIR', PATHS['data'].get('processed', 'data/processed'))
PATHS['data']['models_dir'] = get_env_var('MODELS_DIR', PATHS['data'].get('models_dir', 'data/models'))

# Переопределяем параметры моделей из env если установлены
if 'model' in MODEL_CFG and 'lgbm' in MODEL_CFG['model']:
    MODEL_CFG['model']['lgbm']['n_estimators'] = get_env_var(
        'LGBM_N_ESTIMATORS',
        MODEL_CFG['model']['lgbm'].get('n_estimators', 500),
        int
    )
    MODEL_CFG['model']['lgbm']['learning_rate'] = get_env_var(
        'LGBM_LEARNING_RATE',
        MODEL_CFG['model']['lgbm'].get('learning_rate', 0.05),
        float
    )
    MODEL_CFG['model']['lgbm']['num_leaves'] = get_env_var(
        'LGBM_NUM_LEAVES',
        MODEL_CFG['model']['lgbm'].get('num_leaves', 31),
        int
    )

# Environment (development/production)
ENVIRONMENT = get_env_var('ENVIRONMENT', 'development')

__all__ = ['PATHS', 'MODEL_CFG', 'ENVIRONMENT', 'get_env_var', 'load_config']
