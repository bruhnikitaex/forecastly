"""
Middleware for FastAPI application.

Provides request tracking, logging, and monitoring.
"""

import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from src.utils.logger import logger
from src.monitoring import get_metrics_collector


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to track API request metrics."""

    async def dispatch(self, request: Request, call_next):
        """Process request and track metrics."""
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time

        # Track metrics
        collector = get_metrics_collector()
        collector.track_request(
            endpoint=request.url.path,
            method=request.method,
            duration=duration,
            status_code=response.status_code
        )

        # Add custom headers
        response.headers["X-Process-Time"] = str(duration)

        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log API requests."""

    async def dispatch(self, request: Request, call_next):
        """Process and log request."""
        start_time = time.time()

        # Log request
        logger.info(
            f"{request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else None,
            }
        )

        # Process request
        response = await call_next(request)

        # Log response
        duration = time.time() - start_time
        logger.info(
            f"Response {response.status_code} in {duration:.3f}s",
            extra={
                "status_code": response.status_code,
                "duration": duration,
                "path": request.url.path,
            }
        )

        return response
