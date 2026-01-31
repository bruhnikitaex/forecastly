"""
Модуль логирования Forecastly.

Поддерживает:
- Настраиваемый уровень логирования через LOG_LEVEL env variable
- Структурированные JSON логи (LOG_FORMAT=json)
- Ротация файлов (1 MB, хранение 10 файлов)
"""

import os
import sys
from pathlib import Path

from loguru import logger

LOG_DIR = Path(__file__).resolve().parents[2] / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / 'forecastly.log'

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
LOG_FORMAT = os.getenv('LOG_FORMAT', 'text').lower()

logger.remove()

if LOG_FORMAT == 'json':
    # Структурированные JSON логи для агрегации (Loki/ELK)
    json_fmt = (
        '{{"timestamp":"{time:YYYY-MM-DDTHH:mm:ss.SSSZ}",'
        '"level":"{level}",'
        '"module":"{module}",'
        '"function":"{function}",'
        '"line":{line},'
        '"message":"{message}"}}'
    )
    logger.add(sys.stderr, level=LOG_LEVEL, format=json_fmt)
    logger.add(LOG_PATH, level=LOG_LEVEL, format=json_fmt, rotation='1 MB', retention=10)
else:
    logger.add(sys.stderr, level=LOG_LEVEL)
    logger.add(LOG_PATH, level=LOG_LEVEL, rotation='1 MB', retention=10)

__all__ = ['logger']
