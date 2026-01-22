"""
Утилиты безопасности: хэширование паролей, JWT токены, API ключи.
"""

import os
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Any

from jose import jwt, JWTError
from passlib.context import CryptContext

from src.utils.logger import logger


# ============================================================================
# CONFIGURATION
# ============================================================================

# Секретный ключ для JWT (ОБЯЗАТЕЛЬНО заменить в production!)
SECRET_KEY = os.getenv('SECRET_KEY', 'forecastly-dev-secret-key-change-in-production-2024')
ALGORITHM = "HS256"

# Время жизни токенов
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', '30'))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv('REFRESH_TOKEN_EXPIRE_DAYS', '7'))

# Контекст для хэширования паролей (bcrypt)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ============================================================================
# PASSWORD HASHING
# ============================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Проверяет соответствие пароля его хэшу.

    Args:
        plain_password: Пароль в открытом виде
        hashed_password: Хэш пароля из БД

    Returns:
        True если пароль верный
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Создаёт хэш пароля.

    Args:
        password: Пароль в открытом виде

    Returns:
        Хэш пароля (bcrypt)
    """
    return pwd_context.hash(password)


# ============================================================================
# JWT TOKENS
# ============================================================================

def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Создаёт JWT access token.

    Args:
        data: Данные для включения в токен (обычно {"sub": user_id})
        expires_delta: Время жизни токена

    Returns:
        JWT токен
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(
    data: dict,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Создаёт JWT refresh token.

    Args:
        data: Данные для включения в токен
        expires_delta: Время жизни токена

    Returns:
        JWT refresh токен
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    })

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> Optional[dict]:
    """
    Декодирует JWT токен.

    Args:
        token: JWT токен

    Returns:
        Данные из токена или None если токен невалидный
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        logger.warning(f"JWT decode error: {e}")
        return None


def verify_token(token: str, token_type: str = "access") -> Optional[dict]:
    """
    Проверяет JWT токен и его тип.

    Args:
        token: JWT токен
        token_type: Ожидаемый тип токена (access или refresh)

    Returns:
        Данные из токена или None
    """
    payload = decode_token(token)

    if not payload:
        return None

    if payload.get("type") != token_type:
        logger.warning(f"Token type mismatch: expected {token_type}, got {payload.get('type')}")
        return None

    return payload


# ============================================================================
# API KEYS
# ============================================================================

def generate_api_key() -> tuple[str, str]:
    """
    Генерирует новый API ключ.

    Returns:
        Кортеж (raw_key, key_hash):
        - raw_key: Ключ для показа пользователю (только один раз!)
        - key_hash: Хэш для хранения в БД
    """
    # Генерируем 32-байтовый случайный ключ
    raw_key = f"fcast_{secrets.token_urlsafe(32)}"

    # Хэшируем для хранения (SHA-256)
    key_hash = hash_api_key(raw_key)

    return raw_key, key_hash


def hash_api_key(api_key: str) -> str:
    """
    Хэширует API ключ для хранения.

    Args:
        api_key: API ключ в открытом виде

    Returns:
        SHA-256 хэш ключа
    """
    return hashlib.sha256(api_key.encode()).hexdigest()


def get_api_key_prefix(api_key: str) -> str:
    """
    Получает префикс API ключа для идентификации.

    Args:
        api_key: API ключ

    Returns:
        Первые 8 символов после 'fcast_'
    """
    if api_key.startswith('fcast_'):
        return api_key[6:14]  # Первые 8 символов после префикса
    return api_key[:8]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_random_password(length: int = 16) -> str:
    """
    Генерирует случайный пароль.

    Args:
        length: Длина пароля

    Returns:
        Случайный пароль
    """
    return secrets.token_urlsafe(length)


def is_password_strong(password: str) -> tuple[bool, str]:
    """
    Проверяет надёжность пароля.

    Args:
        password: Пароль для проверки

    Returns:
        Кортеж (is_valid, message)
    """
    if len(password) < 8:
        return False, "Пароль должен быть не менее 8 символов"

    if not any(c.isupper() for c in password):
        return False, "Пароль должен содержать заглавную букву"

    if not any(c.islower() for c in password):
        return False, "Пароль должен содержать строчную букву"

    if not any(c.isdigit() for c in password):
        return False, "Пароль должен содержать цифру"

    return True, "OK"
