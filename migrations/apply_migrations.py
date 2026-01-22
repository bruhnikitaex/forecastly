"""
Скрипт для применения миграций базы данных.

Использование:
    python -m migrations.apply_migrations

Или для SQLite:
    python -m migrations.apply_migrations --sqlite
"""

import os
import sys
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from src.db.database import engine, DATABASE_URL
from src.utils.logger import logger


def apply_migration_001():
    """Применяет миграцию 001: Account Lockout Fields."""

    logger.info("Применение миграции 001: Account Lockout Fields...")

    # SQL для PostgreSQL и SQLite совместимый
    statements = []

    if 'sqlite' in DATABASE_URL.lower():
        # SQLite не поддерживает IF NOT EXISTS для ALTER TABLE
        # Проверяем существование колонок через pragma
        statements = [
            # Для SQLite нужно проверять и добавлять колонки по одной
            """
            -- SQLite version: добавляем колонки если их нет
            -- Примечание: SQLite не поддерживает ADD COLUMN IF NOT EXISTS
            -- Эти команды вызовут ошибку если колонки уже существуют
            """,
        ]
        # Для SQLite используем try/except в Python
        columns_to_add = [
            ("failed_login_attempts", "INTEGER DEFAULT 0"),
            ("locked_until", "TIMESTAMP NULL"),
            ("last_failed_login", "TIMESTAMP NULL"),
        ]

        with engine.connect() as conn:
            # Получаем текущие колонки
            result = conn.execute(text("PRAGMA table_info(users)"))
            existing_columns = {row[1] for row in result.fetchall()}

            for col_name, col_type in columns_to_add:
                if col_name not in existing_columns:
                    try:
                        conn.execute(text(f"ALTER TABLE users ADD COLUMN {col_name} {col_type}"))
                        conn.commit()
                        logger.info(f"  Добавлена колонка: {col_name}")
                    except Exception as e:
                        logger.warning(f"  Колонка {col_name} уже существует или ошибка: {e}")
                else:
                    logger.info(f"  Колонка {col_name} уже существует")

            logger.info("✓ Миграция 001 применена успешно (SQLite)")
            return

    else:
        # PostgreSQL
        statements = [
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS failed_login_attempts INTEGER DEFAULT 0",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS locked_until TIMESTAMP NULL",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS last_failed_login TIMESTAMP NULL",
            "CREATE INDEX IF NOT EXISTS ix_users_locked_until ON users(locked_until) WHERE locked_until IS NOT NULL",
        ]

    with engine.connect() as conn:
        for stmt in statements:
            if stmt.strip() and not stmt.strip().startswith('--'):
                try:
                    conn.execute(text(stmt))
                    logger.info(f"  Выполнено: {stmt[:50]}...")
                except Exception as e:
                    logger.warning(f"  Предупреждение: {e}")
        conn.commit()

    logger.info("✓ Миграция 001 применена успешно")


def apply_migration_002():
    """Применяет миграцию 002: Security Audit Log Table."""

    logger.info("Применение миграции 002: Security Audit Log Table...")

    if 'sqlite' in DATABASE_URL.lower():
        # SQLite версия
        statements = [
            """
            CREATE TABLE IF NOT EXISTS security_audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type VARCHAR(50) NOT NULL,
                user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                user_email VARCHAR(255),
                ip_address VARCHAR(45),
                user_agent VARCHAR(500),
                details TEXT,
                severity VARCHAR(20) DEFAULT 'info',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            "CREATE INDEX IF NOT EXISTS ix_audit_event_type ON security_audit_logs(event_type)",
            "CREATE INDEX IF NOT EXISTS ix_audit_user_id ON security_audit_logs(user_id)",
            "CREATE INDEX IF NOT EXISTS ix_audit_user_email ON security_audit_logs(user_email)",
            "CREATE INDEX IF NOT EXISTS ix_audit_created_at ON security_audit_logs(created_at)",
            "CREATE INDEX IF NOT EXISTS ix_audit_severity ON security_audit_logs(severity)",
        ]
    else:
        # PostgreSQL версия
        statements = [
            """
            CREATE TABLE IF NOT EXISTS security_audit_logs (
                id SERIAL PRIMARY KEY,
                event_type VARCHAR(50) NOT NULL,
                user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                user_email VARCHAR(255),
                ip_address VARCHAR(45),
                user_agent VARCHAR(500),
                details TEXT,
                severity VARCHAR(20) DEFAULT 'info',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            "CREATE INDEX IF NOT EXISTS ix_audit_event_type ON security_audit_logs(event_type)",
            "CREATE INDEX IF NOT EXISTS ix_audit_user_id ON security_audit_logs(user_id)",
            "CREATE INDEX IF NOT EXISTS ix_audit_user_email ON security_audit_logs(user_email)",
            "CREATE INDEX IF NOT EXISTS ix_audit_created_at ON security_audit_logs(created_at)",
            "CREATE INDEX IF NOT EXISTS ix_audit_severity ON security_audit_logs(severity)",
            "CREATE INDEX IF NOT EXISTS ix_audit_event_created ON security_audit_logs(event_type, created_at)",
            "CREATE INDEX IF NOT EXISTS ix_audit_user_created ON security_audit_logs(user_id, created_at)",
            "CREATE INDEX IF NOT EXISTS ix_audit_severity_created ON security_audit_logs(severity, created_at)",
        ]

    with engine.connect() as conn:
        for stmt in statements:
            if stmt.strip():
                try:
                    conn.execute(text(stmt))
                    logger.info(f"  Выполнено: {stmt[:60].strip()}...")
                except Exception as e:
                    logger.warning(f"  Предупреждение: {e}")
        conn.commit()

    logger.info("✓ Миграция 002 применена успешно")


def main():
    """Применяет все миграции."""
    logger.info("=" * 60)
    logger.info("Начало применения миграций")
    logger.info(f"База данных: {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else DATABASE_URL}")
    logger.info("=" * 60)

    try:
        apply_migration_001()
        apply_migration_002()
        logger.info("=" * 60)
        logger.info("Все миграции применены успешно!")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Ошибка применения миграций: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
