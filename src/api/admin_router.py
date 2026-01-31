"""
Административный API роутер.

Endpoints:
- GET/POST/PATCH /api/v1/admin/users - Управление пользователями
- GET/POST/PATCH /api/v1/admin/roles - Управление ролями
- GET /api/v1/admin/audit - Журнал аудита
"""

import json
from typing import Optional, List
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from src.db.database import get_db
from src.db.models import User, SecurityAuditLog
from src.auth.dependencies import require_role
from src.auth.security import get_password_hash, is_password_strong
from src.auth.schemas import UserResponse, Message
from src.auth.audit import get_audit_logs, get_security_stats
from src.utils.logger import logger


router = APIRouter(prefix="/admin", tags=["Admin"])

ADMIN_ROLE = require_role(["admin"])


# ============================================================================
# USERS
# ============================================================================

@router.get("/users", response_model=List[UserResponse])
def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    role: Optional[str] = None,
    is_active: Optional[bool] = None,
    current_user: User = Depends(ADMIN_ROLE),
    db: Session = Depends(get_db),
):
    """Список пользователей с фильтрацией и пагинацией."""
    q = db.query(User)
    if role:
        q = q.filter(User.role == role)
    if is_active is not None:
        q = q.filter(User.is_active == is_active)
    users = q.offset(skip).limit(limit).all()
    return users


@router.post("/users", response_model=UserResponse, status_code=201)
def create_user(
    email: str,
    password: str,
    role: str = "viewer",
    username: Optional[str] = None,
    current_user: User = Depends(ADMIN_ROLE),
    db: Session = Depends(get_db),
):
    """Создание пользователя администратором."""
    if role not in ("admin", "analyst", "viewer"):
        raise HTTPException(400, "Допустимые роли: admin, analyst, viewer")

    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(400, "Email уже зарегистрирован")

    is_valid, errors = is_password_strong(password)
    if not is_valid:
        raise HTTPException(400, detail={"message": "Слабый пароль", "errors": errors})

    user = User(
        email=email,
        username=username or email.split("@")[0],
        hashed_password=get_password_hash(password),
        role=role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info(f"Админ {current_user.email} создал пользователя {email} (роль: {role})")
    return user


@router.patch("/users/{user_id}", response_model=UserResponse)
def update_user(
    user_id: int,
    role: Optional[str] = None,
    is_active: Optional[bool] = None,
    username: Optional[str] = None,
    current_user: User = Depends(ADMIN_ROLE),
    db: Session = Depends(get_db),
):
    """Обновление пользователя (роль, статус, имя)."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "Пользователь не найден")

    if role:
        if role not in ("admin", "analyst", "viewer"):
            raise HTTPException(400, "Допустимые роли: admin, analyst, viewer")
        user.role = role

    if is_active is not None:
        if user_id == current_user.id and not is_active:
            raise HTTPException(400, "Нельзя деактивировать свой аккаунт")
        user.is_active = is_active

    if username is not None:
        user.username = username

    db.commit()
    db.refresh(user)
    logger.info(f"Админ {current_user.email} обновил пользователя {user.email}")
    return user


# ============================================================================
# ROLES
# ============================================================================

@router.get("/roles")
def list_roles(current_user: User = Depends(ADMIN_ROLE)):
    """Список доступных ролей и их описания."""
    return {
        "roles": [
            {
                "name": "admin",
                "description": "Управление пользователями, настройками, полный доступ",
                "permissions": ["read", "write", "admin", "manage_users", "view_audit"]
            },
            {
                "name": "analyst",
                "description": "Загрузка данных, обучение моделей, просмотр метрик",
                "permissions": ["read", "write", "train", "export"]
            },
            {
                "name": "viewer",
                "description": "Просмотр дашборда, прогнозов и отчётов",
                "permissions": ["read"]
            },
        ]
    }


# ============================================================================
# AUDIT
# ============================================================================

@router.get("/audit")
def get_audit(
    event_type: Optional[str] = None,
    user_email: Optional[str] = None,
    severity: Optional[str] = None,
    hours: int = Query(24, ge=1, le=720),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    current_user: User = Depends(ADMIN_ROLE),
    db: Session = Depends(get_db),
):
    """Журнал аудита с фильтрацией."""
    start_date = datetime.now(timezone.utc) - timedelta(hours=hours)

    logs = get_audit_logs(
        db=db,
        event_type=event_type,
        user_email=user_email,
        severity=severity,
        start_date=start_date,
        skip=skip,
        limit=limit,
    )

    return {
        "count": len(logs),
        "filters": {
            "event_type": event_type,
            "user_email": user_email,
            "severity": severity,
            "hours": hours,
        },
        "logs": [
            {
                "id": log.id,
                "event_type": log.event_type,
                "user_id": log.user_id,
                "user_email": log.user_email,
                "ip_address": log.ip_address,
                "severity": log.severity,
                "details": json.loads(log.details) if log.details else None,
                "created_at": log.created_at.isoformat(),
            }
            for log in logs
        ],
    }


@router.get("/audit/stats")
def get_audit_stats(
    hours: int = Query(24, ge=1, le=720),
    current_user: User = Depends(ADMIN_ROLE),
    db: Session = Depends(get_db),
):
    """Статистика безопасности."""
    return get_security_stats(db, hours)
