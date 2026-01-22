-- ============================================================================
-- Migration: Add Account Lockout Fields
-- Version: 001
-- Date: 2026-01-22
-- Description: Добавляет поля для защиты от brute force атак
-- ============================================================================

-- Добавляем поля для отслеживания неудачных попыток входа
ALTER TABLE users ADD COLUMN IF NOT EXISTS failed_login_attempts INTEGER DEFAULT 0;
ALTER TABLE users ADD COLUMN IF NOT EXISTS locked_until TIMESTAMP NULL;
ALTER TABLE users ADD COLUMN IF NOT EXISTS last_failed_login TIMESTAMP NULL;

-- Создаём индекс для быстрого поиска заблокированных аккаунтов
CREATE INDEX IF NOT EXISTS ix_users_locked_until ON users(locked_until) WHERE locked_until IS NOT NULL;

-- ============================================================================
-- Rollback (если нужно откатить миграцию):
-- ============================================================================
-- ALTER TABLE users DROP COLUMN IF EXISTS failed_login_attempts;
-- ALTER TABLE users DROP COLUMN IF EXISTS locked_until;
-- ALTER TABLE users DROP COLUMN IF EXISTS last_failed_login;
-- DROP INDEX IF EXISTS ix_users_locked_until;
