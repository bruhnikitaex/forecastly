"""
Сервис для записи событий безопасности в журнал аудита.

Используется для логирования всех важных событий аутентификации
и авторизации для последующего анализа и расследования инцидентов.
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Optional, Any
from sqlalchemy.orm import Session
from fastapi import Request

from src.db.models import SecurityAuditLog, AuditEventType
from src.utils.logger import logger


def get_client_info(request: Optional[Request]) -> tuple[str, str]:
    """
    Извлекает информацию о клиенте из запроса.

    Args:
        request: FastAPI Request объект

    Returns:
        Кортеж (ip_address, user_agent)
    """
    if request is None:
        return None, None

    # Получаем IP (учитываем прокси)
    ip_address = request.headers.get('X-Forwarded-For')
    if ip_address:
        ip_address = ip_address.split(',')[0].strip()
    else:
        ip_address = request.client.host if request.client else None

    # Получаем User-Agent
    user_agent = request.headers.get('User-Agent', '')[:500]  # Ограничиваем длину

    return ip_address, user_agent


def log_security_event(
    db: Session,
    event_type: str,
    user_id: Optional[int] = None,
    user_email: Optional[str] = None,
    request: Optional[Request] = None,
    details: Optional[dict] = None,
    severity: str = 'info'
) -> SecurityAuditLog:
    """
    Записывает событие безопасности в журнал аудита.

    Args:
        db: Сессия SQLAlchemy
        event_type: Тип события (из AuditEventType)
        user_id: ID пользователя (если применимо)
        user_email: Email пользователя
        request: FastAPI Request для извлечения IP и User-Agent
        details: Дополнительные детали события (словарь)
        severity: Уровень важности (info, warning, critical)

    Returns:
        Созданная запись SecurityAuditLog
    """
    ip_address, user_agent = get_client_info(request)

    audit_log = SecurityAuditLog(
        event_type=event_type,
        user_id=user_id,
        user_email=user_email,
        ip_address=ip_address,
        user_agent=user_agent,
        details=json.dumps(details, ensure_ascii=False) if details else None,
        severity=severity
    )

    db.add(audit_log)
    db.commit()

    # Также логируем в файл для быстрого доступа
    log_message = f"[AUDIT] {event_type} | User: {user_email or 'N/A'} | IP: {ip_address or 'N/A'}"
    if details:
        log_message += f" | Details: {details}"

    if severity == 'critical':
        logger.error(log_message)
    elif severity == 'warning':
        logger.warning(log_message)
    else:
        logger.info(log_message)

    return audit_log


# ============================================================================
# Удобные функции для конкретных событий
# ============================================================================

def log_login_success(db: Session, user_id: int, user_email: str, request: Request):
    """Логирует успешный вход."""
    return log_security_event(
        db=db,
        event_type=AuditEventType.LOGIN_SUCCESS,
        user_id=user_id,
        user_email=user_email,
        request=request,
        severity='info'
    )


def log_login_failed(
    db: Session,
    user_email: str,
    request: Request,
    reason: str = 'invalid_credentials',
    user_id: Optional[int] = None,
    attempts_left: Optional[int] = None
):
    """Логирует неудачную попытку входа."""
    details = {'reason': reason}
    if attempts_left is not None:
        details['attempts_left'] = attempts_left

    return log_security_event(
        db=db,
        event_type=AuditEventType.LOGIN_FAILED,
        user_id=user_id,
        user_email=user_email,
        request=request,
        details=details,
        severity='warning'
    )


def log_account_locked(
    db: Session,
    user_id: int,
    user_email: str,
    request: Request,
    locked_until: datetime,
    failed_attempts: int
):
    """Логирует блокировку аккаунта."""
    return log_security_event(
        db=db,
        event_type=AuditEventType.ACCOUNT_LOCKED,
        user_id=user_id,
        user_email=user_email,
        request=request,
        details={
            'locked_until': locked_until.isoformat(),
            'failed_attempts': failed_attempts
        },
        severity='critical'
    )


def log_account_unlocked(
    db: Session,
    user_id: int,
    user_email: str,
    admin_email: str,
    request: Optional[Request] = None
):
    """Логирует разблокировку аккаунта администратором."""
    return log_security_event(
        db=db,
        event_type=AuditEventType.ACCOUNT_UNLOCKED,
        user_id=user_id,
        user_email=user_email,
        request=request,
        details={'unlocked_by': admin_email},
        severity='info'
    )


def log_user_registered(db: Session, user_id: int, user_email: str, request: Request):
    """Логирует регистрацию нового пользователя."""
    return log_security_event(
        db=db,
        event_type=AuditEventType.USER_REGISTERED,
        user_id=user_id,
        user_email=user_email,
        request=request,
        severity='info'
    )


def log_password_changed(db: Session, user_id: int, user_email: str, request: Optional[Request] = None):
    """Логирует успешную смену пароля."""
    return log_security_event(
        db=db,
        event_type=AuditEventType.PASSWORD_CHANGED,
        user_id=user_id,
        user_email=user_email,
        request=request,
        severity='info'
    )


def log_password_change_failed(
    db: Session,
    user_id: int,
    user_email: str,
    reason: str,
    request: Optional[Request] = None
):
    """Логирует неудачную попытку смены пароля."""
    return log_security_event(
        db=db,
        event_type=AuditEventType.PASSWORD_CHANGE_FAILED,
        user_id=user_id,
        user_email=user_email,
        request=request,
        details={'reason': reason},
        severity='warning'
    )


def log_api_key_created(
    db: Session,
    user_id: int,
    user_email: str,
    key_name: str,
    key_prefix: str,
    request: Optional[Request] = None
):
    """Логирует создание API ключа."""
    return log_security_event(
        db=db,
        event_type=AuditEventType.API_KEY_CREATED,
        user_id=user_id,
        user_email=user_email,
        request=request,
        details={'key_name': key_name, 'key_prefix': key_prefix},
        severity='info'
    )


def log_api_key_deleted(
    db: Session,
    user_id: int,
    user_email: str,
    key_name: str,
    request: Optional[Request] = None
):
    """Логирует удаление API ключа."""
    return log_security_event(
        db=db,
        event_type=AuditEventType.API_KEY_DELETED,
        user_id=user_id,
        user_email=user_email,
        request=request,
        details={'key_name': key_name},
        severity='info'
    )


def log_user_role_changed(
    db: Session,
    user_id: int,
    user_email: str,
    old_role: str,
    new_role: str,
    admin_email: str,
    request: Optional[Request] = None
):
    """Логирует изменение роли пользователя."""
    return log_security_event(
        db=db,
        event_type=AuditEventType.USER_ROLE_CHANGED,
        user_id=user_id,
        user_email=user_email,
        request=request,
        details={
            'old_role': old_role,
            'new_role': new_role,
            'changed_by': admin_email
        },
        severity='warning'
    )


def log_user_deactivated(
    db: Session,
    user_id: int,
    user_email: str,
    admin_email: str,
    request: Optional[Request] = None
):
    """Логирует деактивацию пользователя."""
    return log_security_event(
        db=db,
        event_type=AuditEventType.USER_DEACTIVATED,
        user_id=user_id,
        user_email=user_email,
        request=request,
        details={'deactivated_by': admin_email},
        severity='warning'
    )


# ============================================================================
# Функции для чтения и анализа логов
# ============================================================================

def get_audit_logs(
    db: Session,
    event_type: Optional[str] = None,
    user_id: Optional[int] = None,
    user_email: Optional[str] = None,
    severity: Optional[str] = None,
    ip_address: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    skip: int = 0,
    limit: int = 100
) -> list[SecurityAuditLog]:
    """
    Получает журнал аудита с фильтрацией.

    Args:
        db: Сессия SQLAlchemy
        event_type: Фильтр по типу события
        user_id: Фильтр по ID пользователя
        user_email: Фильтр по email (частичное совпадение)
        severity: Фильтр по уровню важности
        ip_address: Фильтр по IP адресу
        start_date: Начало периода
        end_date: Конец периода
        skip: Пропустить записей
        limit: Максимум записей

    Returns:
        Список записей аудита
    """
    query = db.query(SecurityAuditLog)

    if event_type:
        query = query.filter(SecurityAuditLog.event_type == event_type)
    if user_id:
        query = query.filter(SecurityAuditLog.user_id == user_id)
    if user_email:
        query = query.filter(SecurityAuditLog.user_email.ilike(f'%{user_email}%'))
    if severity:
        query = query.filter(SecurityAuditLog.severity == severity)
    if ip_address:
        query = query.filter(SecurityAuditLog.ip_address == ip_address)
    if start_date:
        query = query.filter(SecurityAuditLog.created_at >= start_date)
    if end_date:
        query = query.filter(SecurityAuditLog.created_at <= end_date)

    return query.order_by(SecurityAuditLog.created_at.desc()).offset(skip).limit(limit).all()


def get_failed_login_attempts(
    db: Session,
    hours: int = 24,
    min_attempts: int = 3
) -> list[dict]:
    """
    Получает IP адреса с множественными неудачными попытками входа.

    Полезно для обнаружения brute force атак.

    Args:
        db: Сессия SQLAlchemy
        hours: За последние N часов
        min_attempts: Минимальное количество попыток

    Returns:
        Список словарей с IP и количеством попыток
    """
    from sqlalchemy import func

    since = datetime.now(timezone.utc) - timedelta(hours=hours)

    results = db.query(
        SecurityAuditLog.ip_address,
        func.count(SecurityAuditLog.id).label('attempts')
    ).filter(
        SecurityAuditLog.event_type == AuditEventType.LOGIN_FAILED,
        SecurityAuditLog.created_at >= since,
        SecurityAuditLog.ip_address.isnot(None)
    ).group_by(
        SecurityAuditLog.ip_address
    ).having(
        func.count(SecurityAuditLog.id) >= min_attempts
    ).all()

    return [{'ip_address': r.ip_address, 'attempts': r.attempts} for r in results]


def get_security_stats(db: Session, hours: int = 24) -> dict:
    """
    Получает статистику безопасности за период.

    Args:
        db: Сессия SQLAlchemy
        hours: За последние N часов

    Returns:
        Словарь со статистикой
    """
    from sqlalchemy import func

    since = datetime.now(timezone.utc) - timedelta(hours=hours)

    # Подсчёт событий по типам
    event_counts = db.query(
        SecurityAuditLog.event_type,
        func.count(SecurityAuditLog.id).label('count')
    ).filter(
        SecurityAuditLog.created_at >= since
    ).group_by(
        SecurityAuditLog.event_type
    ).all()

    # Подсчёт по severity
    severity_counts = db.query(
        SecurityAuditLog.severity,
        func.count(SecurityAuditLog.id).label('count')
    ).filter(
        SecurityAuditLog.created_at >= since
    ).group_by(
        SecurityAuditLog.severity
    ).all()

    return {
        'period_hours': hours,
        'events_by_type': {r.event_type: r.count for r in event_counts},
        'events_by_severity': {r.severity: r.count for r in severity_counts},
        'total_events': sum(r.count for r in event_counts),
        'suspicious_ips': get_failed_login_attempts(db, hours, 5)
    }
