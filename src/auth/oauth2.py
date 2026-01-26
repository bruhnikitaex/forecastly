"""
OAuth2/OIDC провайдеры для аутентификации.

Поддерживает:
- Google OAuth2
- Azure AD (Microsoft)
- GitHub
"""

import httpx
from typing import Optional, Dict, Any
from dataclasses import dataclass
from urllib.parse import urlencode
import secrets

from src.utils.logger import logger
from src.utils.config import get_env


@dataclass
class OAuth2Config:
    """Конфигурация OAuth2 провайдера."""
    client_id: str
    client_secret: str
    authorize_url: str
    token_url: str
    userinfo_url: str
    scopes: list[str]
    redirect_uri: str


class OAuth2Provider:
    """Базовый класс OAuth2 провайдера."""

    def __init__(self, config: OAuth2Config):
        self.config = config
        self._state_store: Dict[str, str] = {}

    def get_authorization_url(self, state: Optional[str] = None) -> tuple[str, str]:
        """
        Генерирует URL для авторизации.

        Returns:
            Tuple (authorization_url, state)
        """
        if state is None:
            state = secrets.token_urlsafe(32)

        params = {
            "client_id": self.config.client_id,
            "redirect_uri": self.config.redirect_uri,
            "response_type": "code",
            "scope": " ".join(self.config.scopes),
            "state": state,
        }

        url = f"{self.config.authorize_url}?{urlencode(params)}"
        self._state_store[state] = state
        return url, state

    async def exchange_code(self, code: str, state: str) -> Dict[str, Any]:
        """
        Обменивает authorization code на токены.

        Args:
            code: Authorization code от провайдера
            state: State параметр для CSRF защиты

        Returns:
            Dict с access_token, refresh_token, expires_in
        """
        if state not in self._state_store:
            raise ValueError("Invalid state parameter")

        del self._state_store[state]

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.config.token_url,
                data={
                    "client_id": self.config.client_id,
                    "client_secret": self.config.client_secret,
                    "code": code,
                    "redirect_uri": self.config.redirect_uri,
                    "grant_type": "authorization_code",
                },
                headers={"Accept": "application/json"},
            )

            if response.status_code != 200:
                logger.error(f"OAuth2 token exchange failed: {response.text}")
                raise ValueError(f"Token exchange failed: {response.status_code}")

            return response.json()

    async def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """
        Получает информацию о пользователе.

        Args:
            access_token: OAuth2 access token

        Returns:
            Dict с данными пользователя (email, name, etc.)
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(
                self.config.userinfo_url,
                headers={"Authorization": f"Bearer {access_token}"},
            )

            if response.status_code != 200:
                logger.error(f"Failed to get user info: {response.text}")
                raise ValueError(f"User info request failed: {response.status_code}")

            return response.json()


class GoogleOAuth2(OAuth2Provider):
    """Google OAuth2 провайдер."""

    def __init__(self, redirect_uri: str):
        config = OAuth2Config(
            client_id=get_env("GOOGLE_CLIENT_ID", ""),
            client_secret=get_env("GOOGLE_CLIENT_SECRET", ""),
            authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
            token_url="https://oauth2.googleapis.com/token",
            userinfo_url="https://www.googleapis.com/oauth2/v2/userinfo",
            scopes=["openid", "email", "profile"],
            redirect_uri=redirect_uri,
        )
        super().__init__(config)

    async def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Получает информацию о пользователе Google."""
        data = await super().get_user_info(access_token)
        return {
            "provider": "google",
            "provider_id": data.get("id"),
            "email": data.get("email"),
            "name": data.get("name"),
            "picture": data.get("picture"),
            "email_verified": data.get("verified_email", False),
        }


class AzureADOAuth2(OAuth2Provider):
    """Azure AD (Microsoft) OAuth2 провайдер."""

    def __init__(self, redirect_uri: str, tenant_id: str = "common"):
        config = OAuth2Config(
            client_id=get_env("AZURE_CLIENT_ID", ""),
            client_secret=get_env("AZURE_CLIENT_SECRET", ""),
            authorize_url=f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/authorize",
            token_url=f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
            userinfo_url="https://graph.microsoft.com/v1.0/me",
            scopes=["openid", "email", "profile", "User.Read"],
            redirect_uri=redirect_uri,
        )
        super().__init__(config)

    async def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Получает информацию о пользователе Azure AD."""
        data = await super().get_user_info(access_token)
        return {
            "provider": "azure",
            "provider_id": data.get("id"),
            "email": data.get("mail") or data.get("userPrincipalName"),
            "name": data.get("displayName"),
            "picture": None,  # Requires additional Graph API call
            "email_verified": True,  # Azure AD emails are verified
        }


class GitHubOAuth2(OAuth2Provider):
    """GitHub OAuth2 провайдер."""

    def __init__(self, redirect_uri: str):
        config = OAuth2Config(
            client_id=get_env("GITHUB_CLIENT_ID", ""),
            client_secret=get_env("GITHUB_CLIENT_SECRET", ""),
            authorize_url="https://github.com/login/oauth/authorize",
            token_url="https://github.com/login/oauth/access_token",
            userinfo_url="https://api.github.com/user",
            scopes=["read:user", "user:email"],
            redirect_uri=redirect_uri,
        )
        super().__init__(config)

    async def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Получает информацию о пользователе GitHub."""
        data = await super().get_user_info(access_token)

        # GitHub может не возвращать email в основном запросе
        email = data.get("email")
        if not email:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.github.com/user/emails",
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                if response.status_code == 200:
                    emails = response.json()
                    primary = next((e for e in emails if e.get("primary")), None)
                    if primary:
                        email = primary.get("email")

        return {
            "provider": "github",
            "provider_id": str(data.get("id")),
            "email": email,
            "name": data.get("name") or data.get("login"),
            "picture": data.get("avatar_url"),
            "email_verified": True,  # GitHub primary emails are verified
        }


# Фабрика провайдеров
_providers: Dict[str, OAuth2Provider] = {}


def get_oauth2_provider(provider_name: str, redirect_uri: str) -> OAuth2Provider:
    """
    Получает экземпляр OAuth2 провайдера.

    Args:
        provider_name: Имя провайдера (google, azure, github)
        redirect_uri: URI для редиректа после авторизации

    Returns:
        Экземпляр OAuth2Provider
    """
    cache_key = f"{provider_name}:{redirect_uri}"

    if cache_key not in _providers:
        if provider_name == "google":
            _providers[cache_key] = GoogleOAuth2(redirect_uri)
        elif provider_name == "azure":
            _providers[cache_key] = AzureADOAuth2(redirect_uri)
        elif provider_name == "github":
            _providers[cache_key] = GitHubOAuth2(redirect_uri)
        else:
            raise ValueError(f"Unknown OAuth2 provider: {provider_name}")

    return _providers[cache_key]


def get_available_providers() -> list[str]:
    """Возвращает список доступных OAuth2 провайдеров."""
    providers = []

    if get_env("GOOGLE_CLIENT_ID"):
        providers.append("google")
    if get_env("AZURE_CLIENT_ID"):
        providers.append("azure")
    if get_env("GITHUB_CLIENT_ID"):
        providers.append("github")

    return providers
