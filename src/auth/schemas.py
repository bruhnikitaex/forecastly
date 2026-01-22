"""
Pydantic схемы для аутентификации.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field, validator


# ============================================================================
# USER SCHEMAS
# ============================================================================

class UserBase(BaseModel):
    """Базовая схема пользователя."""
    email: EmailStr
    username: Optional[str] = None
    company: Optional[str] = None


class UserCreate(UserBase):
    """Схема для создания пользователя."""
    password: str = Field(..., min_length=8)

    @validator('password')
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Пароль должен быть не менее 8 символов')
        if not any(c.isupper() for c in v):
            raise ValueError('Пароль должен содержать заглавную букву')
        if not any(c.islower() for c in v):
            raise ValueError('Пароль должен содержать строчную букву')
        if not any(c.isdigit() for c in v):
            raise ValueError('Пароль должен содержать цифру')
        return v


class UserUpdate(BaseModel):
    """Схема для обновления пользователя."""
    username: Optional[str] = None
    company: Optional[str] = None
    password: Optional[str] = Field(None, min_length=8)


class UserResponse(UserBase):
    """Схема ответа с данными пользователя."""
    id: int
    is_active: bool
    is_superuser: bool
    role: str
    created_at: datetime
    last_login: Optional[datetime] = None

    class Config:
        from_attributes = True


class UserInDB(UserResponse):
    """Схема пользователя с хэшем пароля (для внутреннего использования)."""
    hashed_password: str


# ============================================================================
# AUTH SCHEMAS
# ============================================================================

class Token(BaseModel):
    """Схема ответа с токенами."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # Секунды до истечения access token


class TokenData(BaseModel):
    """Данные из токена."""
    user_id: Optional[int] = None
    email: Optional[str] = None
    role: Optional[str] = None


class LoginRequest(BaseModel):
    """Схема запроса на вход."""
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    """Схема запроса на обновление токена."""
    refresh_token: str


class PasswordChangeRequest(BaseModel):
    """Схема запроса на смену пароля."""
    current_password: str
    new_password: str = Field(..., min_length=8)

    @validator('new_password')
    def password_strength(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError('Пароль должен содержать заглавную букву')
        if not any(c.islower() for c in v):
            raise ValueError('Пароль должен содержать строчную букву')
        if not any(c.isdigit() for c in v):
            raise ValueError('Пароль должен содержать цифру')
        return v


# ============================================================================
# API KEY SCHEMAS
# ============================================================================

class APIKeyCreate(BaseModel):
    """Схема для создания API ключа."""
    name: str = Field(..., min_length=1, max_length=100)
    expires_in_days: Optional[int] = Field(None, ge=1, le=365)
    permissions: List[str] = Field(default=["read"])

    @validator('permissions')
    def validate_permissions(cls, v):
        allowed = {"read", "write", "admin"}
        for perm in v:
            if perm not in allowed:
                raise ValueError(f"Недопустимое разрешение: {perm}. Допустимые: {allowed}")
        return v


class APIKeyResponse(BaseModel):
    """Схема ответа с данными API ключа (без самого ключа)."""
    id: int
    name: str
    key_prefix: str
    is_active: bool
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    created_at: datetime
    permissions: List[str]

    class Config:
        from_attributes = True


class APIKeyCreatedResponse(APIKeyResponse):
    """Схема ответа при создании API ключа (включает сам ключ)."""
    api_key: str  # Показывается только один раз!


class APIKeyList(BaseModel):
    """Список API ключей."""
    count: int
    keys: List[APIKeyResponse]


# ============================================================================
# MESSAGE SCHEMAS
# ============================================================================

class Message(BaseModel):
    """Простое сообщение."""
    message: str


class ErrorResponse(BaseModel):
    """Схема ошибки."""
    detail: str
    code: Optional[str] = None
