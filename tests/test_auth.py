"""
Тесты для модуля аутентификации.

Покрывает:
- Password policy validation
- Account lockout механизм
- Audit logging
"""

import pytest
from datetime import datetime, timedelta, timezone

# Импорты для тестирования
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.auth.security import (
    is_password_strong,
    get_password_policy,
    verify_password,
    get_password_hash,
    check_account_lockout,
    should_reset_failed_attempts,
    MAX_FAILED_LOGIN_ATTEMPTS,
    LOCKOUT_DURATION_MINUTES,
    FAILED_LOGIN_RESET_MINUTES
)


# ============================================================================
# PASSWORD POLICY TESTS
# ============================================================================

class TestPasswordPolicy:
    """Тесты для валидации паролей."""

    def test_password_too_short(self):
        """Пароль короче минимальной длины."""
        is_valid, errors = is_password_strong("Abc123")
        assert not is_valid
        assert any("8 символов" in e for e in errors)

    def test_password_no_uppercase(self):
        """Пароль без заглавных букв."""
        is_valid, errors = is_password_strong("abcdefgh123")
        assert not is_valid
        assert any("заглавную" in e for e in errors)

    def test_password_no_lowercase(self):
        """Пароль без строчных букв."""
        is_valid, errors = is_password_strong("ABCDEFGH123")
        assert not is_valid
        assert any("строчную" in e for e in errors)

    def test_password_no_digit(self):
        """Пароль без цифр."""
        is_valid, errors = is_password_strong("AbcdefgHijk")
        assert not is_valid
        assert any("цифру" in e for e in errors)

    def test_password_common(self):
        """Распространённый пароль."""
        is_valid, errors = is_password_strong("Password123")
        assert not is_valid
        assert any("распространённый" in e for e in errors)

    def test_password_repeated_chars(self):
        """Пароль с повторяющимися символами."""
        is_valid, errors = is_password_strong("Abcd1111xyz")
        assert not is_valid
        assert any("одинаковых символа" in e for e in errors)

    def test_password_valid(self):
        """Валидный пароль."""
        is_valid, errors = is_password_strong("SecurePass123!")
        assert is_valid
        assert len(errors) == 0

    def test_password_valid_minimal(self):
        """Минимально валидный пароль."""
        is_valid, errors = is_password_strong("Abcdefg1")
        assert is_valid
        assert len(errors) == 0

    def test_password_multiple_errors(self):
        """Пароль с несколькими ошибками."""
        is_valid, errors = is_password_strong("abc")
        assert not is_valid
        assert len(errors) >= 2  # Минимум: короткий + нет заглавной + нет цифры

    def test_get_password_policy(self):
        """Проверка структуры политики паролей."""
        policy = get_password_policy()
        assert "min_length" in policy
        assert "require_uppercase" in policy
        assert "require_lowercase" in policy
        assert "require_digit" in policy
        assert "require_special" in policy
        assert policy["min_length"] >= 8


# ============================================================================
# PASSWORD HASHING TESTS
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
    """Тесты для хеширования паролей."""

    def test_hash_password(self):
        """Хеширование пароля."""
        password = "TestPassword123"
        hashed = get_password_hash(password)
        assert hashed != password
        assert hashed.startswith("$2b$")  # bcrypt prefix

    def test_verify_correct_password(self):
        """Проверка правильного пароля."""
        password = "TestPassword123"
        hashed = get_password_hash(password)
        assert verify_password(password, hashed)

    def test_verify_wrong_password(self):
        """Проверка неправильного пароля."""
        password = "TestPassword123"
        hashed = get_password_hash(password)
        assert not verify_password("WrongPassword", hashed)

    def test_hash_uniqueness(self):
        """Разные хеши для одного пароля (salt)."""
        password = "TestPassword123"
        hash1 = get_password_hash(password)
        hash2 = get_password_hash(password)
        assert hash1 != hash2  # Разные соли


# ============================================================================
# ACCOUNT LOCKOUT TESTS
# ============================================================================

class MockUser:
    """Mock объект пользователя для тестов lockout."""

    def __init__(self):
        self.failed_login_attempts = 0
        self.locked_until = None
        self.last_failed_login = None


class TestAccountLockout:
    """Тесты для механизма блокировки аккаунта."""

    def test_account_not_locked(self):
        """Аккаунт без блокировки."""
        user = MockUser()
        is_locked, remaining = check_account_lockout(user)
        assert not is_locked
        assert remaining is None

    def test_account_locked(self):
        """Заблокированный аккаунт."""
        user = MockUser()
        user.locked_until = datetime.now(timezone.utc) + timedelta(minutes=10)

        is_locked, remaining = check_account_lockout(user)
        assert is_locked
        assert remaining is not None
        assert remaining > 0
        assert remaining <= 600  # 10 минут в секундах

    def test_account_lock_expired(self):
        """Истёкшая блокировка."""
        user = MockUser()
        user.locked_until = datetime.now(timezone.utc) - timedelta(minutes=1)

        is_locked, remaining = check_account_lockout(user)
        assert not is_locked
        assert remaining is None

    def test_should_reset_no_failed_login(self):
        """Сброс счётчика: нет предыдущих попыток."""
        user = MockUser()
        assert should_reset_failed_attempts(user)

    def test_should_reset_old_failed_login(self):
        """Сброс счётчика: старая неудачная попытка."""
        user = MockUser()
        user.last_failed_login = datetime.now(timezone.utc) - timedelta(minutes=FAILED_LOGIN_RESET_MINUTES + 1)
        assert should_reset_failed_attempts(user)

    def test_should_not_reset_recent_failed_login(self):
        """Не сбрасывать: недавняя неудачная попытка."""
        user = MockUser()
        user.last_failed_login = datetime.now(timezone.utc) - timedelta(minutes=5)
        assert not should_reset_failed_attempts(user)

    def test_lockout_constants(self):
        """Проверка констант блокировки."""
        assert MAX_FAILED_LOGIN_ATTEMPTS >= 3
        assert MAX_FAILED_LOGIN_ATTEMPTS <= 10
        assert LOCKOUT_DURATION_MINUTES >= 5
        assert LOCKOUT_DURATION_MINUTES <= 60
        assert FAILED_LOGIN_RESET_MINUTES >= 30


# ============================================================================
# INTEGRATION TESTS (требуют БД)
# ============================================================================

class TestAuthIntegration:
    """
    Интеграционные тесты для auth модуля.

    Примечание: Эти тесты пропускаются если БД недоступна.
    Для запуска установите USE_DATABASE=true и настройте подключение.
    """

    @pytest.fixture
    def db_session(self):
        """Фикстура для сессии БД."""
        import os
        if os.getenv('USE_DATABASE', 'false').lower() != 'true':
            pytest.skip("БД не настроена (USE_DATABASE != true)")

        from src.db.database import SessionLocal
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def test_audit_log_structure(self, db_session):
        """Проверка структуры таблицы audit log."""
        from src.db.models import SecurityAuditLog

        # Проверяем что можем создать запись (без сохранения)
        log = SecurityAuditLog(
            event_type='test_event',
            user_email='test@example.com',
            ip_address='127.0.0.1',
            severity='info'
        )
        assert log.event_type == 'test_event'
        assert log.severity == 'info'


# ============================================================================
# Запуск тестов
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
