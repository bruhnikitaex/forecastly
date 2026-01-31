"""
Middleware for FastAPI application.

Provides request tracking, logging, trace_id and monitoring.
"""

import time
import uuid

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

        response = await call_next(request)

        duration = time.time() - start_time

        collector = get_metrics_collector()
        collector.track_request(
            endpoint=request.url.path,
            method=request.method,
            duration=duration,
            status_code=response.status_code
        )

        response.headers["X-Process-Time"] = f"{duration:.4f}"

        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log API requests and add trace_id."""

    async def dispatch(self, request: Request, call_next):
        """Process and log request with trace_id."""
        start_time = time.time()

        # Generate trace_id for request tracing
        trace_id = request.headers.get("X-Trace-ID", uuid.uuid4().hex[:16])
        request.state.trace_id = trace_id

        logger.info(
            f"[{trace_id}] {request.method} {request.url.path}",
            extra={
                "trace_id": trace_id,
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else None,
            }
        )

        response = await call_next(request)

        duration = time.time() - start_time

        # Add trace_id to response headers
        response.headers["X-Trace-ID"] = trace_id

        logger.info(
            f"[{trace_id}] Response {response.status_code} in {duration:.3f}s",
            extra={
                "trace_id": trace_id,
                "status_code": response.status_code,
                "duration": duration,
                "path": request.url.path,
            }
        )

        return response
