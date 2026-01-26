"""
FastAPI router для OAuth2/OIDC аутентификации.

Endpoints:
- GET /auth/oauth2/providers - Список доступных провайдеров
- GET /auth/oauth2/{provider}/authorize - Начало OAuth2 flow
- GET /auth/oauth2/{provider}/callback - Callback после авторизации
- POST /auth/oauth2/link - Привязка OAuth2 к существующему аккаунту
- DELETE /auth/oauth2/{provider}/unlink - Отвязка OAuth2 провайдера
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, status, Request, Depends, Query
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel

from src.db.database import get_db
from src.db.models import User, OAuth2Connection, RefreshToken
from src.auth.oauth2 import get_oauth2_provider, get_available_providers
from src.auth.security import (
    create_access_token,
    create_refresh_token,
    hash_api_key,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)
from src.auth.dependencies import get_current_active_user
from src.auth.audit import log_login_success
from src.utils.logger import logger
from src.utils.config import get_env


router = APIRouter(prefix="/auth/oauth2", tags=["OAuth2"])


# Pydantic models
class OAuth2ProviderInfo(BaseModel):
    """Information about an OAuth2 provider."""
    name: str
    display_name: str
    enabled: bool
    icon: Optional[str] = None


class OAuth2ProvidersResponse(BaseModel):
    """List of available OAuth2 providers."""
    providers: list[OAuth2ProviderInfo]


class OAuth2Token(BaseModel):
    """OAuth2 authentication token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict
    is_new_user: bool = False


class OAuth2LinkRequest(BaseModel):
    """Request to link OAuth2 provider to existing account."""
    provider: str
    code: str
    state: str


# Provider display configuration
PROVIDER_CONFIG = {
    "google": {
        "display_name": "Google",
        "icon": "google",
    },
    "azure": {
        "display_name": "Microsoft",
        "icon": "microsoft",
    },
    "github": {
        "display_name": "GitHub",
        "icon": "github",
    },
}


def _get_redirect_uri(request: Request, provider: str) -> str:
    """Generate the OAuth2 callback URL."""
    # Try to get from environment first
    base_url = get_env("OAUTH2_REDIRECT_BASE_URL", "")
    if not base_url:
        # Fallback to request URL
        scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
        host = request.headers.get("x-forwarded-host", request.url.netloc)
        base_url = f"{scheme}://{host}"

    return f"{base_url}/api/v1/auth/oauth2/{provider}/callback"


def _create_or_get_user(
    db: Session,
    provider: str,
    provider_id: str,
    email: str,
    name: Optional[str],
    picture: Optional[str],
    access_token: str,
    refresh_token: Optional[str],
) -> tuple[User, bool]:
    """
    Create or get user from OAuth2 data.

    Returns:
        Tuple of (user, is_new_user)
    """
    # Check if OAuth2 connection exists
    oauth_conn = db.query(OAuth2Connection).filter(
        OAuth2Connection.provider == provider,
        OAuth2Connection.provider_user_id == provider_id,
    ).first()

    if oauth_conn:
        # Existing connection - update tokens and return user
        oauth_conn.access_token = access_token
        oauth_conn.refresh_token = refresh_token
        oauth_conn.updated_at = datetime.now(timezone.utc)
        db.commit()

        user = db.query(User).filter(User.id == oauth_conn.user_id).first()
        return user, False

    # Check if user with this email exists
    user = db.query(User).filter(User.email == email).first()

    if user:
        # Link OAuth2 to existing user
        oauth_conn = OAuth2Connection(
            user_id=user.id,
            provider=provider,
            provider_user_id=provider_id,
            provider_email=email,
            access_token=access_token,
            refresh_token=refresh_token,
        )
        db.add(oauth_conn)
        db.commit()
        return user, False

    # Create new user
    user = User(
        email=email,
        username=name or email.split("@")[0],
        hashed_password="",  # No password for OAuth2 users
        is_active=True,
        role="viewer",
    )
    db.add(user)
    db.flush()

    # Create OAuth2 connection
    oauth_conn = OAuth2Connection(
        user_id=user.id,
        provider=provider,
        provider_user_id=provider_id,
        provider_email=email,
        access_token=access_token,
        refresh_token=refresh_token,
    )
    db.add(oauth_conn)
    db.commit()
    db.refresh(user)

    return user, True


@router.get("/providers", response_model=OAuth2ProvidersResponse)
async def list_providers():
    """
    Get list of available OAuth2 providers.

    Returns providers that are configured and enabled.
    """
    available = get_available_providers()

    providers = []
    for name, config in PROVIDER_CONFIG.items():
        providers.append(OAuth2ProviderInfo(
            name=name,
            display_name=config["display_name"],
            enabled=name in available,
            icon=config.get("icon"),
        ))

    return OAuth2ProvidersResponse(providers=providers)


@router.get("/{provider}/authorize")
async def authorize(
    provider: str,
    request: Request,
    redirect_url: Optional[str] = Query(None, description="URL to redirect after login"),
):
    """
    Start OAuth2 authorization flow.

    Redirects user to the provider's authorization page.
    """
    available = get_available_providers()

    if provider not in available:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Provider '{provider}' is not available or not configured"
        )

    redirect_uri = _get_redirect_uri(request, provider)

    try:
        oauth_provider = get_oauth2_provider(provider, redirect_uri)
        auth_url, state = oauth_provider.get_authorization_url()

        # Store redirect_url in state if provided
        # In production, you'd want to store this in a session or cache
        if redirect_url:
            # For simplicity, we'll append it to the auth URL as a query param
            # In production, use proper session storage
            auth_url = f"{auth_url}&redirect_url={redirect_url}"

        logger.info(f"OAuth2 authorization started for provider {provider}")

        return RedirectResponse(url=auth_url)

    except Exception as e:
        logger.error(f"OAuth2 authorization error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start OAuth2 flow: {str(e)}"
        )


@router.get("/{provider}/callback")
async def callback(
    provider: str,
    request: Request,
    code: str,
    state: str,
    error: Optional[str] = None,
    error_description: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    OAuth2 callback endpoint.

    Handles the callback from OAuth2 provider after user authorization.
    """
    # Check for errors from provider
    if error:
        logger.warning(f"OAuth2 callback error: {error} - {error_description}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_description or error
        )

    redirect_uri = _get_redirect_uri(request, provider)

    try:
        oauth_provider = get_oauth2_provider(provider, redirect_uri)

        # Exchange code for tokens
        tokens = await oauth_provider.exchange_code(code, state)
        access_token = tokens.get("access_token")
        refresh_token = tokens.get("refresh_token")

        if not access_token:
            raise ValueError("No access token received")

        # Get user info
        user_info = await oauth_provider.get_user_info(access_token)

        email = user_info.get("email")
        if not email:
            raise ValueError("Email not provided by OAuth2 provider")

        # Create or get user
        user, is_new_user = _create_or_get_user(
            db=db,
            provider=provider,
            provider_id=user_info.get("provider_id"),
            email=email,
            name=user_info.get("name"),
            picture=user_info.get("picture"),
            access_token=access_token,
            refresh_token=refresh_token,
        )

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is deactivated"
            )

        # Create application tokens
        app_access_token = create_access_token(
            data={"sub": str(user.id), "email": user.email, "role": user.role}
        )
        app_refresh_token = create_refresh_token(
            data={"sub": str(user.id)}
        )

        # Store refresh token
        token_hash = hash_api_key(app_refresh_token)
        db_token = RefreshToken(
            token_hash=token_hash,
            user_id=user.id,
            expires_at=datetime.now(timezone.utc) + timedelta(days=7)
        )
        db.add(db_token)
        db.commit()

        # Log successful login
        log_login_success(db, user.id, user.email, request)

        action = "registered and logged in" if is_new_user else "logged in"
        logger.info(f"[SECURITY] User {user.email} {action} via {provider}")

        # In production, redirect to frontend with tokens
        # For API, return tokens directly
        frontend_url = get_env("FRONTEND_URL", "")
        if frontend_url:
            # Redirect to frontend with tokens
            redirect_url = (
                f"{frontend_url}/auth/callback"
                f"?access_token={app_access_token}"
                f"&refresh_token={app_refresh_token}"
                f"&is_new_user={str(is_new_user).lower()}"
            )
            return RedirectResponse(url=redirect_url)

        # Return JSON response for API clients
        return OAuth2Token(
            access_token=app_access_token,
            refresh_token=app_refresh_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user={
                "id": user.id,
                "email": user.email,
                "username": user.username,
                "role": user.role,
            },
            is_new_user=is_new_user,
        )

    except ValueError as e:
        logger.error(f"OAuth2 callback validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"OAuth2 callback error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OAuth2 authentication failed: {str(e)}"
        )


@router.post("/link")
async def link_provider(
    request: Request,
    link_data: OAuth2LinkRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Link an OAuth2 provider to current user's account.

    Allows users to connect additional OAuth2 providers to their account.
    """
    provider = link_data.provider

    # Check if already linked
    existing = db.query(OAuth2Connection).filter(
        OAuth2Connection.user_id == current_user.id,
        OAuth2Connection.provider == provider,
    ).first()

    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Provider '{provider}' is already linked to your account"
        )

    redirect_uri = _get_redirect_uri(request, provider)

    try:
        oauth_provider = get_oauth2_provider(provider, redirect_uri)

        # Exchange code for tokens
        tokens = await oauth_provider.exchange_code(link_data.code, link_data.state)
        access_token = tokens.get("access_token")
        refresh_token = tokens.get("refresh_token")

        # Get user info
        user_info = await oauth_provider.get_user_info(access_token)

        # Check if this provider account is linked to another user
        existing_conn = db.query(OAuth2Connection).filter(
            OAuth2Connection.provider == provider,
            OAuth2Connection.provider_user_id == user_info.get("provider_id"),
        ).first()

        if existing_conn:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"This {provider} account is already linked to another user"
            )

        # Create connection
        oauth_conn = OAuth2Connection(
            user_id=current_user.id,
            provider=provider,
            provider_user_id=user_info.get("provider_id"),
            provider_email=user_info.get("email"),
            access_token=access_token,
            refresh_token=refresh_token,
        )
        db.add(oauth_conn)
        db.commit()

        logger.info(f"User {current_user.email} linked {provider} account")

        return {
            "status": "ok",
            "message": f"Successfully linked {provider} account",
            "provider": provider,
            "provider_email": user_info.get("email"),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OAuth2 link error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to link provider: {str(e)}"
        )


@router.delete("/{provider}/unlink")
async def unlink_provider(
    provider: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Unlink an OAuth2 provider from current user's account.
    """
    # Check if user has a password set (can't unlink if OAuth2 is only login method)
    if not current_user.hashed_password:
        # Count linked providers
        connections = db.query(OAuth2Connection).filter(
            OAuth2Connection.user_id == current_user.id
        ).count()

        if connections <= 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot unlink the only authentication method. Please set a password first."
            )

    # Find and delete connection
    connection = db.query(OAuth2Connection).filter(
        OAuth2Connection.user_id == current_user.id,
        OAuth2Connection.provider == provider,
    ).first()

    if not connection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider '{provider}' is not linked to your account"
        )

    db.delete(connection)
    db.commit()

    logger.info(f"User {current_user.email} unlinked {provider} account")

    return {
        "status": "ok",
        "message": f"Successfully unlinked {provider} account",
    }


@router.get("/connections")
async def list_connections(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    List OAuth2 providers connected to current user's account.
    """
    connections = db.query(OAuth2Connection).filter(
        OAuth2Connection.user_id == current_user.id
    ).all()

    return {
        "has_password": bool(current_user.hashed_password),
        "connections": [
            {
                "provider": c.provider,
                "provider_email": c.provider_email,
                "linked_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in connections
        ]
    }
