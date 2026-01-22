"""
Модуль аутентификации и авторизации Forecastly.

Содержит:
- security.py: Хэширование паролей, JWT токены
- dependencies.py: FastAPI dependencies для защиты endpoints
- schemas.py: Pydantic схемы для auth
- router.py: Auth endpoints
"""

from src.auth.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    decode_token,
    generate_api_key
)
from src.auth.dependencies import (
    get_current_user,
    get_current_active_user,
    get_current_superuser,
    require_permissions,
    get_api_key_user
)

__all__ = [
    'verify_password',
    'get_password_hash',
    'create_access_token',
    'create_refresh_token',
    'decode_token',
    'generate_api_key',
    'get_current_user',
    'get_current_active_user',
    'get_current_superuser',
    'require_permissions',
    'get_api_key_user'
]
