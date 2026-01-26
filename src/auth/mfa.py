"""
Multi-Factor Authentication (MFA) с поддержкой TOTP.

Использует алгоритм TOTP (Time-based One-Time Password) для генерации
одноразовых кодов, совместимых с Google Authenticator, Authy и др.
"""

import pyotp
import qrcode
import io
import base64
import secrets
from typing import Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.utils.logger import logger


@dataclass
class MFASetup:
    """Данные для настройки MFA."""
    secret: str
    provisioning_uri: str
    qr_code_base64: str
    backup_codes: list[str]


@dataclass
class MFAVerification:
    """Результат верификации MFA."""
    success: bool
    method: str  # 'totp' or 'backup_code'
    remaining_backup_codes: Optional[int] = None


class MFAManager:
    """Менеджер Multi-Factor Authentication."""

    BACKUP_CODES_COUNT = 10
    BACKUP_CODE_LENGTH = 8
    ISSUER_NAME = "Forecastly"

    def __init__(self):
        self._used_backup_codes: dict[str, set[str]] = {}

    def generate_secret(self) -> str:
        """Генерирует новый секретный ключ для TOTP."""
        return pyotp.random_base32()

    def setup_mfa(self, user_email: str, secret: Optional[str] = None) -> MFASetup:
        """
        Настраивает MFA для пользователя.

        Args:
            user_email: Email пользователя
            secret: Опциональный секретный ключ (если уже есть)

        Returns:
            MFASetup с секретом, QR-кодом и резервными кодами
        """
        if secret is None:
            secret = self.generate_secret()

        # Создаём TOTP объект
        totp = pyotp.TOTP(secret)

        # Генерируем provisioning URI для QR-кода
        provisioning_uri = totp.provisioning_uri(
            name=user_email,
            issuer_name=self.ISSUER_NAME
        )

        # Генерируем QR-код
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")

        # Конвертируем в base64
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        qr_code_base64 = base64.b64encode(buffer.getvalue()).decode()

        # Генерируем резервные коды
        backup_codes = self._generate_backup_codes()

        logger.info(f"MFA setup generated for user: {user_email}")

        return MFASetup(
            secret=secret,
            provisioning_uri=provisioning_uri,
            qr_code_base64=qr_code_base64,
            backup_codes=backup_codes
        )

    def _generate_backup_codes(self) -> list[str]:
        """Генерирует резервные коды."""
        codes = []
        for _ in range(self.BACKUP_CODES_COUNT):
            # Генерируем код вида: XXXX-XXXX
            code = secrets.token_hex(self.BACKUP_CODE_LENGTH // 2).upper()
            formatted = f"{code[:4]}-{code[4:]}"
            codes.append(formatted)
        return codes

    def verify_totp(self, secret: str, code: str, valid_window: int = 1) -> bool:
        """
        Проверяет TOTP код.

        Args:
            secret: Секретный ключ пользователя
            code: Введённый код
            valid_window: Окно валидности (по умолчанию ±1 период = ±30 сек)

        Returns:
            True если код верный
        """
        try:
            totp = pyotp.TOTP(secret)
            return totp.verify(code, valid_window=valid_window)
        except Exception as e:
            logger.warning(f"TOTP verification error: {e}")
            return False

    def verify_backup_code(
        self,
        user_id: str,
        code: str,
        stored_codes: list[str]
    ) -> Tuple[bool, list[str]]:
        """
        Проверяет резервный код.

        Args:
            user_id: ID пользователя
            code: Введённый код
            stored_codes: Список сохранённых резервных кодов

        Returns:
            Tuple (успех, оставшиеся коды)
        """
        # Нормализуем код (убираем дефисы, приводим к верхнему регистру)
        normalized = code.replace("-", "").upper()

        for stored in stored_codes:
            stored_normalized = stored.replace("-", "").upper()
            if secrets.compare_digest(normalized, stored_normalized):
                # Удаляем использованный код
                remaining = [c for c in stored_codes if c != stored]
                logger.info(f"Backup code used for user {user_id}, {len(remaining)} remaining")
                return True, remaining

        return False, stored_codes

    def verify(
        self,
        user_id: str,
        secret: str,
        code: str,
        backup_codes: list[str]
    ) -> Tuple[MFAVerification, list[str]]:
        """
        Проверяет MFA код (TOTP или резервный).

        Args:
            user_id: ID пользователя
            secret: TOTP секрет
            code: Введённый код
            backup_codes: Список резервных кодов

        Returns:
            Tuple (результат верификации, обновлённые резервные коды)
        """
        # Сначала пробуем TOTP (6 цифр)
        if len(code.replace("-", "")) == 6 and code.replace("-", "").isdigit():
            if self.verify_totp(secret, code):
                return MFAVerification(
                    success=True,
                    method="totp"
                ), backup_codes

        # Пробуем резервный код
        success, remaining_codes = self.verify_backup_code(user_id, code, backup_codes)
        if success:
            return MFAVerification(
                success=True,
                method="backup_code",
                remaining_backup_codes=len(remaining_codes)
            ), remaining_codes

        return MFAVerification(success=False, method="none"), backup_codes

    def get_current_code(self, secret: str) -> str:
        """
        Получает текущий TOTP код (для тестирования).

        Args:
            secret: Секретный ключ

        Returns:
            Текущий 6-значный код
        """
        totp = pyotp.TOTP(secret)
        return totp.now()

    def regenerate_backup_codes(self) -> list[str]:
        """Генерирует новый набор резервных кодов."""
        return self._generate_backup_codes()


# Singleton instance
_mfa_manager: Optional[MFAManager] = None


def get_mfa_manager() -> MFAManager:
    """Получает экземпляр MFA менеджера."""
    global _mfa_manager
    if _mfa_manager is None:
        _mfa_manager = MFAManager()
    return _mfa_manager
