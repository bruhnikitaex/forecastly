-- ============================================================================
-- Migration: Add Security Audit Log Table
-- Version: 002
-- Date: 2026-01-22
-- Description: Создаёт таблицу для хранения событий безопасности
-- ============================================================================

-- Создаём таблицу audit log
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
);

-- Индексы для быстрого поиска
CREATE INDEX IF NOT EXISTS ix_audit_event_type ON security_audit_logs(event_type);
CREATE INDEX IF NOT EXISTS ix_audit_user_id ON security_audit_logs(user_id);
CREATE INDEX IF NOT EXISTS ix_audit_user_email ON security_audit_logs(user_email);
CREATE INDEX IF NOT EXISTS ix_audit_created_at ON security_audit_logs(created_at);
CREATE INDEX IF NOT EXISTS ix_audit_severity ON security_audit_logs(severity);

-- Составные индексы
CREATE INDEX IF NOT EXISTS ix_audit_event_created ON security_audit_logs(event_type, created_at);
CREATE INDEX IF NOT EXISTS ix_audit_user_created ON security_audit_logs(user_id, created_at);
CREATE INDEX IF NOT EXISTS ix_audit_severity_created ON security_audit_logs(severity, created_at);

-- ============================================================================
-- Rollback (если нужно откатить миграцию):
-- ============================================================================
-- DROP TABLE IF EXISTS security_audit_logs;
