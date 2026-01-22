"""
FastAPI dependencies для аутентификации и авторизации.
"""

import json
from datetime import datetime
from typing import Optional, List
from functools import wraps

from fastapi import Depends, HTTPException, status, Security
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from sqlalchemy.orm import Session

from src.db.database import get_db
from src.db.models import User, APIKey
from src.auth.security import decode_token, hash_api_key
from src.utils.logger import logger


# OAuth2 схема для JWT токенов
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/login",
    auto_error=False  # Не выбрасывать ошибку автоматически
)

# Схема для API ключей в заголовке
api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False
)


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    api_key: Optional[str] = Depends(api_key_header),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """
    Получает текущего пользователя из JWT токена или API ключа.

    Поддерживает два способа аутентификации:
    1. JWT токен в заголовке Authorization: Bearer <token>
    2. API ключ в заголовке X-API-Key: <key>

    Args:
        token: JWT токен
        api_key: API ключ
        db: Сессия БД

    Returns:
        User или None

    Raises:
        HTTPException 401: Если токен/ключ невалидный
    """
    # Пробуем JWT токен
    if token:
        payload = decode_token(token)
        if payload and payload.get("type") == "access":
            user_id = payload.get("sub")
            if user_id:
                user = db.query(User).filter(User.id == int(user_id)).first()
                if user and user.is_active:
                    return user

    # Пробуем API ключ
    if api_key:
        key_hash = hash_api_key(api_key)
        db_key = db.query(APIKey).filter(
            APIKey.key_hash == key_hash,
            APIKey.is_active == True
        ).first()

        if db_key:
            # Проверяем срок действия
            if db_key.expires_at and db_key.expires_at < datetime.utcnow():
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API ключ истёк"
                )

            # Обновляем время последнего использования
            db_key.last_used_at = datetime.utcnow()
            db.commit()

            # Получаем пользователя
            user = db.query(User).filter(User.id == db_key.user_id).first()
            if user and user.is_active:
                # Сохраняем разрешения ключа в user для проверки
                user._api_key_permissions = json.loads(db_key.permissions)
                return user

    return None


async def get_current_active_user(
    current_user: Optional[User] = Depends(get_current_user)
) -> User:
    """
    Требует аутентифицированного активного пользователя.

    Args:
        current_user: Текущий пользователь

    Returns:
        User

    Raises:
        HTTPException 401: Если пользователь не аутентифицирован
        HTTPException 403: Если пользователь деактивирован
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Требуется аутентификация",
            headers={"WWW-Authenticate": "Bearer"}
        )

    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Аккаунт деактивирован"
        )

    return current_user


async def get_current_superuser(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """
    Требует суперпользователя.

    Args:
        current_user: Текущий пользователь

    Returns:
        User (суперпользователь)

    Raises:
        HTTPException 403: Если пользователь не суперпользователь
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Требуются права администратора"
        )

    return current_user


def require_role(allowed_roles: List[str]):
    """
    Декоратор/dependency для проверки роли пользователя.

    Usage:
        @app.get("/admin")
        def admin_endpoint(user: User = Depends(require_role(["admin"]))):
            ...

    Args:
        allowed_roles: Список допустимых ролей

    Returns:
        Dependency function
    """
    async def role_checker(
        current_user: User = Depends(get_current_active_user)
    ) -> User:
        if current_user.role not in allowed_roles and not current_user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Требуется одна из ролей: {', '.join(allowed_roles)}"
            )
        return current_user

    return role_checker


def require_permissions(required_permissions: List[str]):
    """
    Dependency для проверки разрешений API ключа.

    Usage:
        @app.post("/data")
        def write_data(user: User = Depends(require_permissions(["write"]))):
            ...

    Args:
        required_permissions: Список требуемых разрешений

    Returns:
        Dependency function
    """
    async def permission_checker(
        current_user: User = Depends(get_current_active_user)
    ) -> User:
        # Суперпользователи имеют все разрешения
        if current_user.is_superuser:
            return current_user

        # Если аутентификация через JWT - разрешаем всё
        if not hasattr(current_user, '_api_key_permissions'):
            return current_user

        # Проверяем разрешения API ключа
        user_permissions = current_user._api_key_permissions

        # admin имеет все разрешения
        if "admin" in user_permissions:
            return current_user

        for perm in required_permissions:
            if perm not in user_permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Требуется разрешение: {perm}"
                )

        return current_user

    return permission_checker


async def get_api_key_user(
    api_key: str = Depends(api_key_header),
    db: Session = Depends(get_db)
) -> User:
    """
    Получает пользователя только по API ключу (без JWT).

    Args:
        api_key: API ключ
        db: Сессия БД

    Returns:
        User

    Raises:
        HTTPException 401: Если ключ невалидный
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Требуется API ключ в заголовке X-API-Key"
        )

    key_hash = hash_api_key(api_key)
    db_key = db.query(APIKey).filter(
        APIKey.key_hash == key_hash,
        APIKey.is_active == True
    ).first()

    if not db_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Недействительный API ключ"
        )

    if db_key.expires_at and db_key.expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API ключ истёк"
        )

    # Обновляем время последнего использования
    db_key.last_used_at = datetime.utcnow()
    db.commit()

    user = db.query(User).filter(User.id == db_key.user_id).first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Пользователь не найден или деактивирован"
        )

    user._api_key_permissions = json.loads(db_key.permissions)
    return user


# ============================================================================
# OPTIONAL AUTH (для endpoints, доступных и без авторизации)
# ============================================================================

async def get_optional_user(
    token: Optional[str] = Depends(oauth2_scheme),
    api_key: Optional[str] = Depends(api_key_header),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """
    Опционально получает текущего пользователя.

    Не выбрасывает ошибку, если пользователь не аутентифицирован.
    Полезно для endpoints, доступных всем, но с дополнительными
    возможностями для авторизованных пользователей.

    Returns:
        User или None
    """
    try:
        return await get_current_user(token, api_key, db)
    except HTTPException:
        return None
