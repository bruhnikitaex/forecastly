"""
Тесты для модуля безопасности.

Покрывает:
- JWT токены (access, refresh)
- API ключи (генерация, хэширование)
- Верификация токенов
- Account lockout
"""

import pytest
from datetime import datetime, timedelta, timezone
from jose import jwt

from src.auth.security import (
    # Password
    get_password_hash,
    verify_password,
    is_password_strong,
    get_password_policy,
    generate_random_password,

    # JWT
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_token,
    SECRET_KEY,
    ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    REFRESH_TOKEN_EXPIRE_DAYS,

    # API Keys
    generate_api_key,
    hash_api_key,
    get_api_key_prefix,

    # Account Lockout
    check_account_lockout,
    should_reset_failed_attempts,
    record_failed_login,
    record_successful_login,
    unlock_account,
    MAX_FAILED_LOGIN_ATTEMPTS,
    LOCKOUT_DURATION_MINUTES,
)


# ============================================================================
# JWT TOKEN TESTS
# ============================================================================

class TestAccessToken:
    """Тесты для access token."""

    def test_create_access_token(self):
        """Создание access token."""
        token = create_access_token(data={"sub": "user123"})

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 50  # JWT достаточно длинный

    def test_access_token_contains_user_id(self):
        """Access token содержит user ID."""
        token = create_access_token(data={"sub": "user456"})
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        assert payload["sub"] == "user456"

    def test_access_token_has_expiration(self):
        """Access token имеет время истечения."""
        token = create_access_token(data={"sub": "user789"})
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        assert "exp" in payload
        assert payload["exp"] > datetime.now(timezone.utc).timestamp()

    def test_access_token_type(self):
        """Access token имеет тип 'access'."""
        token = create_access_token(data={"sub": "user"})
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        assert payload["type"] == "access"

    def test_access_token_custom_expiration(self):
        """Access token с кастомным временем истечения."""
        token = create_access_token(
            data={"sub": "user"},
            expires_delta=timedelta(hours=2)
        )
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        # Проверяем что exp примерно через 2 часа
        exp_time = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        expected = datetime.now(timezone.utc) + timedelta(hours=2)

        assert abs((exp_time - expected).total_seconds()) < 10

    def test_access_token_has_iat(self):
        """Access token имеет время создания (iat)."""
        token = create_access_token(data={"sub": "user"})
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        assert "iat" in payload


class TestRefreshToken:
    """Тесты для refresh token."""

    def test_create_refresh_token(self):
        """Создание refresh token."""
        token = create_refresh_token(data={"sub": "user123"})

        assert token is not None
        assert isinstance(token, str)

    def test_refresh_token_type(self):
        """Refresh token имеет тип 'refresh'."""
        token = create_refresh_token(data={"sub": "user"})
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        assert payload["type"] == "refresh"

    def test_refresh_token_longer_expiration(self):
        """Refresh token живёт дольше access token."""
        access = create_access_token(data={"sub": "user"})
        refresh = create_refresh_token(data={"sub": "user"})

        access_payload = jwt.decode(access, SECRET_KEY, algorithms=[ALGORITHM])
        refresh_payload = jwt.decode(refresh, SECRET_KEY, algorithms=[ALGORITHM])

        # Refresh должен истекать позже
        assert refresh_payload["exp"] > access_payload["exp"]


class TestDecodeToken:
    """Тесты для декодирования токенов."""

    def test_decode_valid_token(self):
        """Декодирование валидного токена."""
        token = create_access_token(data={"sub": "user123", "role": "admin"})
        payload = decode_token(token)

        assert payload is not None
        assert payload["sub"] == "user123"
        assert payload["role"] == "admin"

    def test_decode_invalid_token(self):
        """Декодирование невалидного токена."""
        payload = decode_token("invalid.token.here")
        assert payload is None

    def test_decode_expired_token(self):
        """Декодирование истёкшего токена."""
        token = create_access_token(
            data={"sub": "user"},
            expires_delta=timedelta(seconds=-10)  # Уже истёк
        )
        payload = decode_token(token)
        assert payload is None

    def test_decode_tampered_token(self):
        """Декодирование изменённого токена."""
        token = create_access_token(data={"sub": "user"})
        # Изменяем токен
        tampered = token[:-5] + "xxxxx"
        payload = decode_token(tampered)
        assert payload is None


class TestVerifyToken:
    """Тесты для верификации токенов."""

    def test_verify_access_token(self):
        """Верификация access token."""
        token = create_access_token(data={"sub": "user"})
        payload = verify_token(token, token_type="access")

        assert payload is not None
        assert payload["type"] == "access"

    def test_verify_refresh_token(self):
        """Верификация refresh token."""
        token = create_refresh_token(data={"sub": "user"})
        payload = verify_token(token, token_type="refresh")

        assert payload is not None
        assert payload["type"] == "refresh"

    def test_verify_wrong_token_type(self):
        """Верификация токена с неправильным типом."""
        access_token = create_access_token(data={"sub": "user"})
        # Пробуем верифицировать как refresh
        payload = verify_token(access_token, token_type="refresh")

        assert payload is None

    def test_verify_invalid_token(self):
        """Верификация невалидного токена."""
        payload = verify_token("invalid", token_type="access")
        assert payload is None


# ============================================================================
# API KEY TESTS
# ============================================================================

class TestAPIKeyGeneration:
    """Тесты для генерации API ключей."""

    def test_generate_api_key(self):
        """Генерация API ключа."""
        raw_key, key_hash = generate_api_key()

        assert raw_key is not None
        assert key_hash is not None
        assert raw_key.startswith("fcast_")
        assert len(raw_key) > 40

    def test_api_key_uniqueness(self):
        """API ключи должны быть уникальными."""
        keys = [generate_api_key()[0] for _ in range(10)]
        assert len(keys) == len(set(keys))

    def test_api_key_hash_consistency(self):
        """Хэш API ключа должен быть консистентным."""
        raw_key, key_hash = generate_api_key()

        # Хэшируем тот же ключ ещё раз
        second_hash = hash_api_key(raw_key)

        assert key_hash == second_hash

    def test_hash_api_key(self):
        """Хэширование API ключа."""
        key = "fcast_test_key_12345"
        hashed = hash_api_key(key)

        assert hashed is not None
        assert len(hashed) == 64  # SHA-256 hex
        assert hashed != key

    def test_get_api_key_prefix(self):
        """Получение префикса API ключа."""
        raw_key, _ = generate_api_key()
        prefix = get_api_key_prefix(raw_key)

        assert len(prefix) == 8
        assert raw_key[6:14] == prefix


# ============================================================================
# PASSWORD TESTS
# ============================================================================

def bcrypt_available():
    """Проверяет доступность bcrypt."""
    try:
        from passlib.hash import bcrypt
        bcrypt.hash("test")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not bcrypt_available(), reason="bcrypt backend не установлен")
class TestPasswordHashing:
    """Тесты для хэширования паролей."""

    def test_hash_password(self):
        """Хэширование пароля."""
        password = "SecurePassword123"
        hashed = get_password_hash(password)

        assert hashed is not None
        assert hashed != password
        assert hashed.startswith("$2b$")  # bcrypt

    def test_verify_correct_password(self):
        """Проверка правильного пароля."""
        password = "SecurePassword123"
        hashed = get_password_hash(password)

        assert verify_password(password, hashed) is True

    def test_verify_wrong_password(self):
        """Проверка неправильного пароля."""
        hashed = get_password_hash("SecurePassword123")

        assert verify_password("WrongPassword", hashed) is False

    def test_hash_salt_uniqueness(self):
        """Каждый хэш должен иметь уникальную соль."""
        password = "SamePassword123"
        hash1 = get_password_hash(password)
        hash2 = get_password_hash(password)

        assert hash1 != hash2  # Разные соли
        # Но оба должны верифицироваться
        assert verify_password(password, hash1)
        assert verify_password(password, hash2)


class TestPasswordPolicy:
    """Тесты для политики паролей."""

    def test_strong_password(self):
        """Надёжный пароль проходит валидацию."""
        is_valid, errors = is_password_strong("SecurePass123!")
        assert is_valid is True
        assert len(errors) == 0

    def test_password_too_short(self):
        """Короткий пароль не проходит."""
        is_valid, errors = is_password_strong("Abc1")
        assert is_valid is False
        assert any("8 символов" in e for e in errors)

    def test_password_no_uppercase(self):
        """Пароль без заглавных букв."""
        is_valid, errors = is_password_strong("securepass123")
        assert is_valid is False
        assert any("заглавную" in e for e in errors)

    def test_password_no_lowercase(self):
        """Пароль без строчных букв."""
        is_valid, errors = is_password_strong("SECUREPASS123")
        assert is_valid is False
        assert any("строчную" in e for e in errors)

    def test_password_no_digit(self):
        """Пароль без цифр."""
        is_valid, errors = is_password_strong("SecurePassWord")
        assert is_valid is False
        assert any("цифру" in e for e in errors)

    def test_common_password(self):
        """Распространённый пароль."""
        is_valid, errors = is_password_strong("Password123")
        assert is_valid is False
        assert any("распространённый" in e for e in errors)

    def test_repeated_characters(self):
        """Пароль с повторяющимися символами."""
        is_valid, errors = is_password_strong("Aaaa1111Bb")
        assert is_valid is False
        assert any("одинаковых символа" in e for e in errors)

    def test_get_password_policy(self):
        """Получение политики паролей."""
        policy = get_password_policy()

        assert "min_length" in policy
        assert "require_uppercase" in policy
        assert "require_lowercase" in policy
        assert "require_digit" in policy
        assert policy["min_length"] >= 8

    def test_generate_random_password(self):
        """Генерация случайного пароля."""
        password = generate_random_password(16)

        assert len(password) >= 16
        # Должен быть уникальным
        password2 = generate_random_password(16)
        assert password != password2


# ============================================================================
# ACCOUNT LOCKOUT TESTS
# ============================================================================

class TestAccountLockout:
    """Тесты для механизма блокировки аккаунта."""

    def test_account_not_locked(self, mock_user):
        """Аккаунт без блокировки."""
        is_locked, remaining = check_account_lockout(mock_user)

        assert is_locked is False
        assert remaining is None

    def test_account_locked(self, mock_locked_user):
        """Заблокированный аккаунт."""
        is_locked, remaining = check_account_lockout(mock_locked_user)

        assert is_locked is True
        assert remaining is not None
        assert remaining > 0

    def test_lock_expired(self, mock_user):
        """Истёкшая блокировка."""
        mock_user.locked_until = datetime.now(timezone.utc) - timedelta(minutes=1)

        is_locked, remaining = check_account_lockout(mock_user)

        assert is_locked is False
        assert remaining is None

    def test_should_reset_no_previous_attempts(self, mock_user):
        """Сброс счётчика: нет предыдущих попыток."""
        assert should_reset_failed_attempts(mock_user) is True

    def test_should_reset_old_attempt(self, mock_user):
        """Сброс счётчика: старая попытка."""
        mock_user.last_failed_login = datetime.now(timezone.utc) - timedelta(hours=2)

        assert should_reset_failed_attempts(mock_user) is True

    def test_should_not_reset_recent_attempt(self, mock_user):
        """Не сбрасывать: недавняя попытка."""
        mock_user.last_failed_login = datetime.now(timezone.utc) - timedelta(minutes=5)

        assert should_reset_failed_attempts(mock_user) is False

    def test_lockout_constants(self):
        """Проверка констант блокировки."""
        assert MAX_FAILED_LOGIN_ATTEMPTS >= 3
        assert MAX_FAILED_LOGIN_ATTEMPTS <= 10
        assert LOCKOUT_DURATION_MINUTES >= 5
        assert LOCKOUT_DURATION_MINUTES <= 60


@pytest.mark.skipif(not bcrypt_available(), reason="bcrypt backend не установлен")
class TestAccountLockoutWithDB:
    """Тесты блокировки с реальной БД."""

    def test_record_failed_login(self, db_with_user):
        """Запись неудачной попытки входа."""
        db_session, user = db_with_user

        initial_attempts = user.failed_login_attempts

        is_locked, locked_until = record_failed_login(user, db_session)

        assert user.failed_login_attempts == initial_attempts + 1
        assert user.last_failed_login is not None

    def test_lockout_after_max_attempts(self, db_with_user):
        """Блокировка после максимума попыток."""
        db_session, user = db_with_user

        # Симулируем MAX_FAILED_LOGIN_ATTEMPTS - 1 попыток
        user.failed_login_attempts = MAX_FAILED_LOGIN_ATTEMPTS - 1
        db_session.commit()

        # Последняя попытка должна заблокировать
        is_locked, locked_until = record_failed_login(user, db_session)

        assert is_locked is True
        assert locked_until is not None
        assert user.locked_until is not None

    def test_record_successful_login(self, db_with_user):
        """Успешный вход сбрасывает счётчик."""
        db_session, user = db_with_user

        # Устанавливаем неудачные попытки
        user.failed_login_attempts = 3
        user.locked_until = datetime.now(timezone.utc) + timedelta(minutes=5)
        db_session.commit()

        record_successful_login(user, db_session)

        assert user.failed_login_attempts == 0
        assert user.locked_until is None
        assert user.last_login is not None

    def test_unlock_account(self, db_with_user):
        """Разблокировка аккаунта администратором."""
        db_session, user = db_with_user

        # Блокируем
        user.failed_login_attempts = MAX_FAILED_LOGIN_ATTEMPTS
        user.locked_until = datetime.now(timezone.utc) + timedelta(minutes=15)
        db_session.commit()

        # Разблокируем
        unlock_account(user, db_session)

        assert user.failed_login_attempts == 0
        assert user.locked_until is None


class TestEdgeCases:
    """Тесты граничных случаев."""

    def test_empty_token_data(self):
        """Токен с пустыми данными."""
        token = create_access_token(data={})
        payload = decode_token(token)

        assert payload is not None
        assert "exp" in payload
        assert "type" in payload

    def test_token_with_special_characters(self):
        """Токен с спецсимволами в данных."""
        token = create_access_token(data={"sub": "user@example.com/test"})
        payload = decode_token(token)

        assert payload["sub"] == "user@example.com/test"

    @pytest.mark.skipif(not bcrypt_available(), reason="bcrypt backend не установлен")
    def test_very_long_password(self):
        """Очень длинный пароль."""
        long_password = "Aa1" + "x" * 1000
        hashed = get_password_hash(long_password)

        assert verify_password(long_password, hashed)

    @pytest.mark.skipif(not bcrypt_available(), reason="bcrypt backend не установлен")
    def test_unicode_password(self):
        """Пароль с юникодом."""
        unicode_password = "СекретныйПароль123"
        hashed = get_password_hash(unicode_password)

        assert verify_password(unicode_password, hashed)


# ============================================================================
# Запуск тестов
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
