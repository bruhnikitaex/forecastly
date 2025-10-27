from loguru import logger
import sys
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[2] / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / 'app.log'

logger.remove()
logger.add(sys.stderr, level='INFO')
logger.add(LOG_PATH, level='INFO', rotation='1 MB', retention=10)

__all__ = ['logger']
