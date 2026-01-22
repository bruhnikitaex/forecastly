"""
FastAPI router для аутентификации.

Endpoints:
- POST /auth/register - Регистрация
- POST /auth/login - Вход (получение токенов)
- POST /auth/refresh - Обновление access token
- POST /auth/logout - Выход (отзыв refresh token)
- GET /auth/me - Данные текущего пользователя
- PUT /auth/me - Обновление данных пользователя
- POST /auth/change-password - Смена пароля
- POST /auth/api-keys - Создание API ключа
- GET /auth/api-keys - Список API ключей
- DELETE /auth/api-keys/{key_id} - Удаление API ключа
"""

import os
import json
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

# Rate limiting
from slowapi import Limiter
from slowapi.util import get_remote_address

# Инициализация rate limiter для auth endpoints
# Более строгие лимиты для auth endpoints
_auth_rate_limit = os.getenv('AUTH_RATE_LIMIT', '5/minute')
auth_limiter = Limiter(key_func=get_remote_address)

from src.db.database import get_db
from src.db.models import User, APIKey, RefreshToken, SecurityAuditLog, AuditEventType
from src.auth.audit import (
    log_login_success,
    log_login_failed,
    log_account_locked,
    log_account_unlocked,
    log_user_registered,
    log_password_changed,
    log_password_change_failed,
    log_api_key_created,
    log_api_key_deleted,
    log_user_role_changed,
    log_user_deactivated,
    get_audit_logs,
    get_security_stats
)
from src.auth.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    decode_token,
    generate_api_key,
    hash_api_key,
    get_api_key_prefix,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    # Account lockout
    check_account_lockout,
    record_failed_login,
    record_successful_login,
    unlock_account,
    MAX_FAILED_LOGIN_ATTEMPTS,
    LOCKOUT_DURATION_MINUTES,
    # Password policy
    is_password_strong,
    get_password_policy
)
from src.auth.schemas import (
    UserCreate,
    UserUpdate,
    UserResponse,
    Token,
    LoginRequest,
    RefreshRequest,
    PasswordChangeRequest,
    APIKeyCreate,
    APIKeyResponse,
    APIKeyCreatedResponse,
    APIKeyList,
    Message
)
from src.auth.dependencies import (
    get_current_active_user,
    get_current_superuser
)
from src.utils.logger import logger


router = APIRouter(prefix="/auth", tags=["Authentication"])


# ============================================================================
# REGISTRATION & LOGIN
# ============================================================================

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
@auth_limiter.limit("3/minute")  # Строгий лимит на регистрацию
def register(
    request: Request,
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """
    Регистрация нового пользователя.

    Args:
        user_data: Данные пользователя (email, password, username, company)

    Returns:
        Созданный пользователь

    Raises:
        400: Email уже зарегистрирован или пароль не соответствует политике
    """
    client_ip = get_remote_address(request)

    # Проверяем надёжность пароля
    is_valid, password_errors = is_password_strong(user_data.password)
    if not is_valid:
        logger.warning(f"[SECURITY] Отклонена регистрация из-за слабого пароля: {user_data.email} | IP: {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Пароль не соответствует требованиям безопасности",
                "errors": password_errors,
                "policy": get_password_policy()
            }
        )

    # Проверяем, не занят ли email
    existing = db.query(User).filter(User.email == user_data.email).first()
    if existing:
        logger.warning(f"[SECURITY] Попытка регистрации существующего email: {user_data.email} | IP: {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email уже зарегистрирован"
        )

    # Создаём пользователя
    user = User(
        email=user_data.email,
        username=user_data.username,
        company=user_data.company,
        hashed_password=get_password_hash(user_data.password),
        role="viewer"  # По умолчанию - viewer
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    # Записываем в audit log
    log_user_registered(db, user.id, user.email, request)

    logger.info(f"[SECURITY] Зарегистрирован новый пользователь: {user.email} | IP: {client_ip}")
    return user


@router.get("/password-policy")
def get_password_requirements():
    """
    Возвращает текущую политику паролей.

    Полезно для отображения требований на форме регистрации.
    """
    return get_password_policy()


@router.post("/login", response_model=Token)
@auth_limiter.limit("5/minute")  # Строгий лимит на логин для защиты от brute force
def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    Вход в систему (OAuth2 compatible).

    Использует стандартную форму OAuth2 (username, password).
    В качестве username передаётся email.

    Returns:
        access_token, refresh_token

    Raises:
        401: Неверные учётные данные
        403: Аккаунт деактивирован или заблокирован
    """
    client_ip = get_remote_address(request)

    # Ищем пользователя по email
    user = db.query(User).filter(User.email == form_data.username).first()

    # Проверяем блокировку аккаунта (если пользователь существует)
    if user:
        is_locked, remaining_seconds = check_account_lockout(user)
        if is_locked:
            remaining_minutes = (remaining_seconds // 60) + 1
            logger.warning(
                f"[SECURITY] Попытка входа в заблокированный аккаунт: {user.email} | "
                f"IP: {client_ip} | Осталось: {remaining_minutes} мин"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Аккаунт временно заблокирован. Попробуйте через {remaining_minutes} мин."
            )

    # Проверяем учётные данные
    if not user or not verify_password(form_data.password, user.hashed_password):
        # Записываем неудачную попытку (если пользователь существует)
        if user:
            is_now_locked, locked_until = record_failed_login(user, db)
            attempts_left = MAX_FAILED_LOGIN_ATTEMPTS - user.failed_login_attempts

            # Audit log для неудачной попытки
            log_login_failed(db, user.email, request, 'invalid_password', user.id, attempts_left)

            if is_now_locked:
                # Audit log для блокировки
                log_account_locked(db, user.id, user.email, request, locked_until, user.failed_login_attempts)

                logger.warning(
                    f"[SECURITY] Аккаунт заблокирован после {MAX_FAILED_LOGIN_ATTEMPTS} неудачных попыток: "
                    f"{user.email} | IP: {client_ip}"
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Аккаунт заблокирован на {LOCKOUT_DURATION_MINUTES} мин после {MAX_FAILED_LOGIN_ATTEMPTS} неудачных попыток"
                )

            logger.warning(
                f"[SECURITY] Неудачная попытка входа: {form_data.username} | "
                f"IP: {client_ip} | Осталось попыток: {attempts_left}"
            )
        else:
            # Audit log для несуществующего email
            log_login_failed(db, form_data.username, request, 'email_not_found')
            logger.warning(f"[SECURITY] Неудачная попытка входа (email не найден): {form_data.username} | IP: {client_ip}")

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный email или пароль",
            headers={"WWW-Authenticate": "Bearer"}
        )

    if not user.is_active:
        logger.warning(f"[SECURITY] Попытка входа деактивированного пользователя: {user.email} | IP: {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Аккаунт деактивирован"
        )

    # Успешный вход - сбрасываем счётчик неудачных попыток
    record_successful_login(user, db)

    # Создаём токены
    access_token = create_access_token(
        data={"sub": str(user.id), "email": user.email, "role": user.role}
    )
    refresh_token = create_refresh_token(
        data={"sub": str(user.id)}
    )

    # Сохраняем refresh token в БД
    token_hash = hash_api_key(refresh_token)  # Используем тот же алгоритм
    db_token = RefreshToken(
        token_hash=token_hash,
        user_id=user.id,
        expires_at=datetime.utcnow() + timedelta(days=7)
    )
    db.add(db_token)
    db.commit()

    # Audit log для успешного входа
    log_login_success(db, user.id, user.email, request)

    logger.info(f"[SECURITY] Успешный вход: {user.email} | IP: {client_ip}")

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/token", response_model=Token)
@auth_limiter.limit("5/minute")  # Строгий лимит на логин для защиты от brute force
def login_json(
    request: Request,
    login_data: LoginRequest,
    db: Session = Depends(get_db)
):
    """
    Вход в систему (JSON format).

    Альтернатива /login для клиентов, предпочитающих JSON.

    Returns:
        access_token, refresh_token

    Raises:
        401: Неверные учётные данные
        403: Аккаунт деактивирован или заблокирован
    """
    client_ip = get_remote_address(request)

    user = db.query(User).filter(User.email == login_data.email).first()

    # Проверяем блокировку аккаунта (если пользователь существует)
    if user:
        is_locked, remaining_seconds = check_account_lockout(user)
        if is_locked:
            remaining_minutes = (remaining_seconds // 60) + 1
            logger.warning(
                f"[SECURITY] Попытка входа в заблокированный аккаунт: {user.email} | "
                f"IP: {client_ip} | Осталось: {remaining_minutes} мин"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Аккаунт временно заблокирован. Попробуйте через {remaining_minutes} мин."
            )

    # Проверяем учётные данные
    if not user or not verify_password(login_data.password, user.hashed_password):
        # Записываем неудачную попытку (если пользователь существует)
        if user:
            is_now_locked, locked_until = record_failed_login(user, db)
            attempts_left = MAX_FAILED_LOGIN_ATTEMPTS - user.failed_login_attempts

            if is_now_locked:
                logger.warning(
                    f"[SECURITY] Аккаунт заблокирован после {MAX_FAILED_LOGIN_ATTEMPTS} неудачных попыток: "
                    f"{user.email} | IP: {client_ip}"
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Аккаунт заблокирован на {LOCKOUT_DURATION_MINUTES} мин после {MAX_FAILED_LOGIN_ATTEMPTS} неудачных попыток"
                )

            logger.warning(
                f"[SECURITY] Неудачная попытка входа: {login_data.email} | "
                f"IP: {client_ip} | Осталось попыток: {attempts_left}"
            )
        else:
            logger.warning(f"[SECURITY] Неудачная попытка входа (email не найден): {login_data.email} | IP: {client_ip}")

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный email или пароль"
        )

    if not user.is_active:
        logger.warning(f"[SECURITY] Попытка входа деактивированного пользователя: {user.email} | IP: {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Аккаунт деактивирован"
        )

    # Успешный вход - сбрасываем счётчик неудачных попыток
    record_successful_login(user, db)

    access_token = create_access_token(
        data={"sub": str(user.id), "email": user.email, "role": user.role}
    )
    refresh_token = create_refresh_token(
        data={"sub": str(user.id)}
    )

    token_hash = hash_api_key(refresh_token)
    db_token = RefreshToken(
        token_hash=token_hash,
        user_id=user.id,
        expires_at=datetime.utcnow() + timedelta(days=7)
    )
    db.add(db_token)
    db.commit()

    logger.info(f"[SECURITY] Успешный вход: {user.email} | IP: {client_ip}")

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/refresh", response_model=Token)
def refresh_token(
    refresh_data: RefreshRequest,
    db: Session = Depends(get_db)
):
    """
    Обновление access token с помощью refresh token.

    Returns:
        Новые access_token и refresh_token
    """
    # Декодируем токен
    payload = decode_token(refresh_data.refresh_token)

    if not payload or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Недействительный refresh token"
        )

    # Проверяем токен в БД
    token_hash = hash_api_key(refresh_data.refresh_token)
    db_token = db.query(RefreshToken).filter(
        RefreshToken.token_hash == token_hash,
        RefreshToken.is_revoked == False
    ).first()

    if not db_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token отозван или не найден"
        )

    if db_token.expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token истёк"
        )

    # Получаем пользователя
    user_id = payload.get("sub")
    user = db.query(User).filter(User.id == int(user_id)).first()

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Пользователь не найден или деактивирован"
        )

    # Отзываем старый токен
    db_token.is_revoked = True

    # Создаём новые токены
    access_token = create_access_token(
        data={"sub": str(user.id), "email": user.email, "role": user.role}
    )
    new_refresh_token = create_refresh_token(
        data={"sub": str(user.id)}
    )

    # Сохраняем новый refresh token
    new_token_hash = hash_api_key(new_refresh_token)
    new_db_token = RefreshToken(
        token_hash=new_token_hash,
        user_id=user.id,
        expires_at=datetime.utcnow() + timedelta(days=7)
    )
    db.add(new_db_token)
    db.commit()

    return Token(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/logout", response_model=Message)
def logout(
    refresh_data: RefreshRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Выход из системы (отзыв refresh token).
    """
    token_hash = hash_api_key(refresh_data.refresh_token)
    db_token = db.query(RefreshToken).filter(
        RefreshToken.token_hash == token_hash,
        RefreshToken.user_id == current_user.id
    ).first()

    if db_token:
        db_token.is_revoked = True
        db.commit()

    logger.info(f"Пользователь вышел: {current_user.email}")
    return Message(message="Выход выполнен успешно")


# ============================================================================
# USER PROFILE
# ============================================================================

@router.get("/me", response_model=UserResponse)
def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """Получение данных текущего пользователя."""
    return current_user


@router.put("/me", response_model=UserResponse)
def update_current_user(
    user_data: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Обновление данных текущего пользователя."""
    if user_data.username is not None:
        current_user.username = user_data.username
    if user_data.company is not None:
        current_user.company = user_data.company
    if user_data.password is not None:
        current_user.hashed_password = get_password_hash(user_data.password)

    db.commit()
    db.refresh(current_user)

    logger.info(f"Пользователь обновлён: {current_user.email}")
    return current_user


@router.post("/change-password", response_model=Message)
def change_password(
    password_data: PasswordChangeRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Смена пароля."""
    # Проверяем текущий пароль
    if not verify_password(password_data.current_password, current_user.hashed_password):
        logger.warning(f"[SECURITY] Неудачная попытка смены пароля (неверный текущий): {current_user.email}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Неверный текущий пароль"
        )

    # Проверяем надёжность нового пароля
    is_valid, password_errors = is_password_strong(password_data.new_password)
    if not is_valid:
        logger.warning(f"[SECURITY] Отклонена смена пароля из-за слабого пароля: {current_user.email}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Новый пароль не соответствует требованиям безопасности",
                "errors": password_errors,
                "policy": get_password_policy()
            }
        )

    # Проверяем, что новый пароль отличается от текущего
    if verify_password(password_data.new_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Новый пароль должен отличаться от текущего"
        )

    current_user.hashed_password = get_password_hash(password_data.new_password)
    db.commit()

    logger.info(f"[SECURITY] Пароль успешно изменён: {current_user.email}")
    return Message(message="Пароль успешно изменён")


# ============================================================================
# API KEYS
# ============================================================================

@router.post("/api-keys", response_model=APIKeyCreatedResponse, status_code=status.HTTP_201_CREATED)
def create_api_key(
    key_data: APIKeyCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Создание нового API ключа.

    ВАЖНО: Сам ключ показывается только один раз при создании!
    Сохраните его в надёжном месте.
    """
    # Генерируем ключ
    raw_key, key_hash = generate_api_key()
    key_prefix = get_api_key_prefix(raw_key)

    # Вычисляем дату истечения
    expires_at = None
    if key_data.expires_in_days:
        expires_at = datetime.utcnow() + timedelta(days=key_data.expires_in_days)

    # Создаём запись
    api_key = APIKey(
        key_hash=key_hash,
        key_prefix=key_prefix,
        name=key_data.name,
        user_id=current_user.id,
        expires_at=expires_at,
        permissions=json.dumps(key_data.permissions)
    )

    db.add(api_key)
    db.commit()
    db.refresh(api_key)

    logger.info(f"Создан API ключ '{key_data.name}' для {current_user.email}")

    return APIKeyCreatedResponse(
        id=api_key.id,
        name=api_key.name,
        key_prefix=api_key.key_prefix,
        is_active=api_key.is_active,
        expires_at=api_key.expires_at,
        last_used_at=api_key.last_used_at,
        created_at=api_key.created_at,
        permissions=json.loads(api_key.permissions),
        api_key=raw_key  # Показываем только один раз!
    )


@router.get("/api-keys", response_model=APIKeyList)
def list_api_keys(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Получение списка API ключей пользователя."""
    keys = db.query(APIKey).filter(APIKey.user_id == current_user.id).all()

    return APIKeyList(
        count=len(keys),
        keys=[
            APIKeyResponse(
                id=k.id,
                name=k.name,
                key_prefix=k.key_prefix,
                is_active=k.is_active,
                expires_at=k.expires_at,
                last_used_at=k.last_used_at,
                created_at=k.created_at,
                permissions=json.loads(k.permissions)
            )
            for k in keys
        ]
    )


@router.delete("/api-keys/{key_id}", response_model=Message)
def delete_api_key(
    key_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Удаление (деактивация) API ключа."""
    api_key = db.query(APIKey).filter(
        APIKey.id == key_id,
        APIKey.user_id == current_user.id
    ).first()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API ключ не найден"
        )

    api_key.is_active = False
    db.commit()

    logger.info(f"Деактивирован API ключ '{api_key.name}' для {current_user.email}")
    return Message(message="API ключ деактивирован")


# ============================================================================
# ADMIN ENDPOINTS
# ============================================================================

@router.get("/users", response_model=List[UserResponse])
def list_users(
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_current_superuser),
    db: Session = Depends(get_db)
):
    """Получение списка всех пользователей (только для админов)."""
    users = db.query(User).offset(skip).limit(limit).all()
    return users


@router.put("/users/{user_id}/role", response_model=UserResponse)
def update_user_role(
    user_id: int,
    role: str,
    current_user: User = Depends(get_current_superuser),
    db: Session = Depends(get_db)
):
    """Изменение роли пользователя (только для админов)."""
    if role not in ["admin", "analyst", "viewer"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Недопустимая роль. Допустимые: admin, analyst, viewer"
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Пользователь не найден"
        )

    user.role = role
    db.commit()
    db.refresh(user)

    logger.info(f"Роль пользователя {user.email} изменена на {role}")
    return user


@router.delete("/users/{user_id}", response_model=Message)
def deactivate_user(
    user_id: int,
    current_user: User = Depends(get_current_superuser),
    db: Session = Depends(get_db)
):
    """Деактивация пользователя (только для админов)."""
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Нельзя деактивировать свой аккаунт"
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Пользователь не найден"
        )

    user.is_active = False
    db.commit()

    logger.info(f"Пользователь {user.email} деактивирован администратором {current_user.email}")
    return Message(message="Пользователь деактивирован")


# ============================================================================
# ACCOUNT LOCKOUT MANAGEMENT (ADMIN)
# ============================================================================

@router.get("/users/{user_id}/lockout-status")
def get_lockout_status(
    user_id: int,
    current_user: User = Depends(get_current_superuser),
    db: Session = Depends(get_db)
):
    """
    Получение статуса блокировки пользователя (только для админов).

    Returns:
        Информация о блокировке аккаунта
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Пользователь не найден"
        )

    is_locked, remaining_seconds = check_account_lockout(user)

    return {
        "user_id": user.id,
        "email": user.email,
        "is_locked": is_locked,
        "failed_login_attempts": user.failed_login_attempts,
        "locked_until": user.locked_until.isoformat() if user.locked_until else None,
        "remaining_seconds": remaining_seconds,
        "last_failed_login": user.last_failed_login.isoformat() if user.last_failed_login else None,
        "max_attempts": MAX_FAILED_LOGIN_ATTEMPTS,
        "lockout_duration_minutes": LOCKOUT_DURATION_MINUTES
    }


@router.post("/users/{user_id}/unlock", response_model=Message)
def admin_unlock_account(
    user_id: int,
    current_user: User = Depends(get_current_superuser),
    db: Session = Depends(get_db)
):
    """
    Разблокировка аккаунта пользователя (только для админов).

    Сбрасывает счётчик неудачных попыток и снимает блокировку.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Пользователь не найден"
        )

    was_locked = user.is_locked if hasattr(user, 'is_locked') else (user.locked_until is not None)

    unlock_account(user, db)

    if was_locked:
        logger.info(f"[SECURITY] Админ {current_user.email} разблокировал аккаунт {user.email}")
        return Message(message=f"Аккаунт {user.email} разблокирован")
    else:
        logger.info(f"[SECURITY] Админ {current_user.email} сбросил счётчик попыток для {user.email}")
        return Message(message=f"Счётчик неудачных попыток для {user.email} сброшен")


@router.get("/locked-accounts")
def list_locked_accounts(
    current_user: User = Depends(get_current_superuser),
    db: Session = Depends(get_db)
):
    """
    Получение списка заблокированных аккаунтов (только для админов).

    Returns:
        Список пользователей с активной блокировкой
    """
    now = datetime.utcnow()

    # Находим пользователей с активной блокировкой
    locked_users = db.query(User).filter(
        User.locked_until != None,
        User.locked_until > now
    ).all()

    return {
        "count": len(locked_users),
        "locked_accounts": [
            {
                "user_id": u.id,
                "email": u.email,
                "username": u.username,
                "failed_attempts": u.failed_login_attempts,
                "locked_until": u.locked_until.isoformat(),
                "remaining_minutes": int((u.locked_until - now).total_seconds() // 60) + 1
            }
            for u in locked_users
        ]
    }


# ============================================================================
# SECURITY AUDIT LOG (ADMIN)
# ============================================================================

@router.get("/audit-logs")
def get_audit_log_entries(
    event_type: Optional[str] = None,
    user_email: Optional[str] = None,
    severity: Optional[str] = None,
    hours: int = 24,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_superuser),
    db: Session = Depends(get_db)
):
    """
    Получение журнала аудита безопасности (только для админов).

    Args:
        event_type: Фильтр по типу события
        user_email: Фильтр по email (частичное совпадение)
        severity: Фильтр по уровню (info, warning, critical)
        hours: За последние N часов
        skip: Пропустить записей
        limit: Максимум записей (до 500)

    Returns:
        Список событий аудита
    """
    from datetime import timedelta

    if limit > 500:
        limit = 500

    start_date = datetime.utcnow() - timedelta(hours=hours)

    logs = get_audit_logs(
        db=db,
        event_type=event_type,
        user_email=user_email,
        severity=severity,
        start_date=start_date,
        skip=skip,
        limit=limit
    )

    return {
        "count": len(logs),
        "filters": {
            "event_type": event_type,
            "user_email": user_email,
            "severity": severity,
            "hours": hours
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
                "created_at": log.created_at.isoformat()
            }
            for log in logs
        ]
    }


@router.get("/audit-stats")
def get_audit_statistics(
    hours: int = 24,
    current_user: User = Depends(get_current_superuser),
    db: Session = Depends(get_db)
):
    """
    Получение статистики безопасности (только для админов).

    Args:
        hours: За последние N часов (по умолчанию 24)

    Returns:
        Статистика событий и подозрительные IP
    """
    stats = get_security_stats(db, hours)
    return stats


@router.get("/audit-events")
def list_audit_event_types(
    current_user: User = Depends(get_current_superuser)
):
    """
    Список всех типов событий аудита (для фильтрации).
    """
    return {
        "event_types": {
            "authentication": [
                AuditEventType.LOGIN_SUCCESS,
                AuditEventType.LOGIN_FAILED,
                AuditEventType.LOGOUT,
                AuditEventType.TOKEN_REFRESH
            ],
            "account_lockout": [
                AuditEventType.ACCOUNT_LOCKED,
                AuditEventType.ACCOUNT_UNLOCKED
            ],
            "user_management": [
                AuditEventType.USER_REGISTERED,
                AuditEventType.PASSWORD_CHANGED,
                AuditEventType.PASSWORD_CHANGE_FAILED,
                AuditEventType.PROFILE_UPDATED
            ],
            "api_keys": [
                AuditEventType.API_KEY_CREATED,
                AuditEventType.API_KEY_DELETED,
                AuditEventType.API_KEY_USED
            ],
            "admin_actions": [
                AuditEventType.USER_ROLE_CHANGED,
                AuditEventType.USER_DEACTIVATED,
                AuditEventType.USER_ACTIVATED
            ],
            "security": [
                AuditEventType.SUSPICIOUS_ACTIVITY,
                AuditEventType.RATE_LIMIT_EXCEEDED
            ]
        },
        "severities": ["info", "warning", "critical"]
    }
