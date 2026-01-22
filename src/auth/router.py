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

import json
from datetime import datetime, timedelta
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from src.db.database import get_db
from src.db.models import User, APIKey, RefreshToken
from src.auth.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    decode_token,
    generate_api_key,
    hash_api_key,
    get_api_key_prefix,
    ACCESS_TOKEN_EXPIRE_MINUTES
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
def register(
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
        400: Email уже зарегистрирован
    """
    # Проверяем, не занят ли email
    existing = db.query(User).filter(User.email == user_data.email).first()
    if existing:
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

    logger.info(f"Зарегистрирован новый пользователь: {user.email}")
    return user


@router.post("/login", response_model=Token)
def login(
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
    """
    # Ищем пользователя по email
    user = db.query(User).filter(User.email == form_data.username).first()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный email или пароль",
            headers={"WWW-Authenticate": "Bearer"}
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Аккаунт деактивирован"
        )

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

    # Обновляем время последнего входа
    user.last_login = datetime.utcnow()
    db.commit()

    logger.info(f"Пользователь вошёл: {user.email}")

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/token", response_model=Token)
def login_json(
    login_data: LoginRequest,
    db: Session = Depends(get_db)
):
    """
    Вход в систему (JSON format).

    Альтернатива /login для клиентов, предпочитающих JSON.

    Returns:
        access_token, refresh_token
    """
    user = db.query(User).filter(User.email == login_data.email).first()

    if not user or not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный email или пароль"
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Аккаунт деактивирован"
        )

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

    user.last_login = datetime.utcnow()
    db.commit()

    logger.info(f"Пользователь вошёл: {user.email}")

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
    if not verify_password(password_data.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Неверный текущий пароль"
        )

    current_user.hashed_password = get_password_hash(password_data.new_password)
    db.commit()

    logger.info(f"Пароль изменён: {current_user.email}")
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
