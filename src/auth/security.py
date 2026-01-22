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

# Секретный ключ для JWT
_DEFAULT_SECRET_KEY = 'forecastly-dev-secret-key-change-in-production-2024'
SECRET_KEY = os.getenv('SECRET_KEY', _DEFAULT_SECRET_KEY)
ALGORITHM = "HS256"

# Проверка безопасности: запрет использования default ключа в production
_ENVIRONMENT = os.getenv('ENVIRONMENT', 'development').lower()
if _ENVIRONMENT == 'production' and SECRET_KEY == _DEFAULT_SECRET_KEY:
    raise RuntimeError(
        "КРИТИЧЕСКАЯ ОШИБКА БЕЗОПАСНОСТИ: SECRET_KEY не задан в production!\n"
        "Установите переменную окружения SECRET_KEY с надёжным случайным значением.\n"
        "Пример: export SECRET_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
    )

# Время жизни токенов
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', '30'))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv('REFRESH_TOKEN_EXPIRE_DAYS', '7'))

# Account Lockout настройки
MAX_FAILED_LOGIN_ATTEMPTS = int(os.getenv('MAX_FAILED_LOGIN_ATTEMPTS', '5'))
LOCKOUT_DURATION_MINUTES = int(os.getenv('LOCKOUT_DURATION_MINUTES', '15'))
FAILED_LOGIN_RESET_MINUTES = int(os.getenv('FAILED_LOGIN_RESET_MINUTES', '60'))

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


# Password Policy настройки
MIN_PASSWORD_LENGTH = int(os.getenv('MIN_PASSWORD_LENGTH', '8'))
REQUIRE_UPPERCASE = os.getenv('REQUIRE_PASSWORD_UPPERCASE', 'true').lower() == 'true'
REQUIRE_LOWERCASE = os.getenv('REQUIRE_PASSWORD_LOWERCASE', 'true').lower() == 'true'
REQUIRE_DIGIT = os.getenv('REQUIRE_PASSWORD_DIGIT', 'true').lower() == 'true'
REQUIRE_SPECIAL = os.getenv('REQUIRE_PASSWORD_SPECIAL', 'false').lower() == 'true'

# Список распространённых паролей для проверки
COMMON_PASSWORDS = {
    'password', 'password123', '123456', '12345678', 'qwerty', 'abc123',
    'monkey', 'master', 'dragon', 'letmein', 'login', 'admin', 'welcome',
    'password1', 'p@ssw0rd', 'passw0rd', 'qwerty123', 'iloveyou', 'princess',
    'sunshine', 'football', 'baseball', 'superman', 'trustno1', 'access',
}


def is_password_strong(password: str) -> tuple[bool, list[str]]:
    """
    Проверяет надёжность пароля по настроенной политике.

    Args:
        password: Пароль для проверки

    Returns:
        Кортеж (is_valid, errors):
        - is_valid: True если пароль соответствует политике
        - errors: Список ошибок (пустой если пароль валиден)
    """
    errors = []

    # Проверка длины
    if len(password) < MIN_PASSWORD_LENGTH:
        errors.append(f"Пароль должен быть не менее {MIN_PASSWORD_LENGTH} символов")

    # Проверка заглавных букв
    if REQUIRE_UPPERCASE and not any(c.isupper() for c in password):
        errors.append("Пароль должен содержать заглавную букву")

    # Проверка строчных букв
    if REQUIRE_LOWERCASE and not any(c.islower() for c in password):
        errors.append("Пароль должен содержать строчную букву")

    # Проверка цифр
    if REQUIRE_DIGIT and not any(c.isdigit() for c in password):
        errors.append("Пароль должен содержать цифру")

    # Проверка специальных символов
    if REQUIRE_SPECIAL:
        special_chars = set('!@#$%^&*()_+-=[]{}|;:,.<>?')
        if not any(c in special_chars for c in password):
            errors.append("Пароль должен содержать специальный символ (!@#$%^&*...)")

    # Проверка на распространённые пароли
    if password.lower() in COMMON_PASSWORDS:
        errors.append("Этот пароль слишком распространённый, выберите другой")

    # Проверка на повторяющиеся символы (например, 'aaaa')
    if len(password) >= 4:
        for i in range(len(password) - 3):
            if password[i] == password[i+1] == password[i+2] == password[i+3]:
                errors.append("Пароль не должен содержать 4+ одинаковых символа подряд")
                break

    return len(errors) == 0, errors


def get_password_policy() -> dict:
    """
    Возвращает текущую политику паролей.

    Returns:
        Словарь с настройками политики
    """
    return {
        "min_length": MIN_PASSWORD_LENGTH,
        "require_uppercase": REQUIRE_UPPERCASE,
        "require_lowercase": REQUIRE_LOWERCASE,
        "require_digit": REQUIRE_DIGIT,
        "require_special": REQUIRE_SPECIAL,
        "common_passwords_blocked": True
    }


# ============================================================================
# ACCOUNT LOCKOUT
# ============================================================================

def check_account_lockout(user) -> tuple[bool, Optional[int]]:
    """
    Проверяет, заблокирован ли аккаунт.

    Args:
        user: Объект пользователя из БД

    Returns:
        Кортеж (is_locked, remaining_seconds):
        - is_locked: True если аккаунт заблокирован
        - remaining_seconds: Оставшееся время блокировки в секундах (или None)
    """
    if user.locked_until is None:
        return False, None

    now = datetime.utcnow()
    if now < user.locked_until:
        remaining = (user.locked_until - now).total_seconds()
        return True, int(remaining)

    return False, None


def should_reset_failed_attempts(user) -> bool:
    """
    Проверяет, нужно ли сбросить счётчик неудачных попыток.

    Счётчик сбрасывается если прошло достаточно времени с последней
    неудачной попытки (FAILED_LOGIN_RESET_MINUTES).

    Args:
        user: Объект пользователя из БД

    Returns:
        True если нужно сбросить счётчик
    """
    if user.last_failed_login is None:
        return True

    reset_threshold = datetime.utcnow() - timedelta(minutes=FAILED_LOGIN_RESET_MINUTES)
    return user.last_failed_login < reset_threshold


def record_failed_login(user, db_session) -> tuple[bool, Optional[datetime]]:
    """
    Записывает неудачную попытку входа и проверяет необходимость блокировки.

    Args:
        user: Объект пользователя из БД
        db_session: Сессия SQLAlchemy

    Returns:
        Кортеж (is_now_locked, locked_until):
        - is_now_locked: True если аккаунт был заблокирован этой попыткой
        - locked_until: Время до которого заблокирован (или None)
    """
    now = datetime.utcnow()

    # Сбрасываем счётчик если прошло достаточно времени
    if should_reset_failed_attempts(user):
        user.failed_login_attempts = 0

    # Увеличиваем счётчик
    user.failed_login_attempts += 1
    user.last_failed_login = now

    # Проверяем, достигнут ли лимит
    if user.failed_login_attempts >= MAX_FAILED_LOGIN_ATTEMPTS:
        user.locked_until = now + timedelta(minutes=LOCKOUT_DURATION_MINUTES)
        db_session.commit()
        logger.warning(
            f"[SECURITY] Аккаунт заблокирован: {user.email} | "
            f"Попыток: {user.failed_login_attempts} | "
            f"Заблокирован до: {user.locked_until.isoformat()}"
        )
        return True, user.locked_until

    db_session.commit()
    return False, None


def record_successful_login(user, db_session) -> None:
    """
    Записывает успешный вход и сбрасывает счётчик неудачных попыток.

    Args:
        user: Объект пользователя из БД
        db_session: Сессия SQLAlchemy
    """
    user.failed_login_attempts = 0
    user.locked_until = None
    user.last_failed_login = None
    user.last_login = datetime.utcnow()
    db_session.commit()


def unlock_account(user, db_session) -> None:
    """
    Разблокирует аккаунт (для админов).

    Args:
        user: Объект пользователя из БД
        db_session: Сессия SQLAlchemy
    """
    user.failed_login_attempts = 0
    user.locked_until = None
    user.last_failed_login = None
    db_session.commit()
    logger.info(f"[SECURITY] Аккаунт разблокирован администратором: {user.email}")
