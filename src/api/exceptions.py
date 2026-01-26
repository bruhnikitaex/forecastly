"""
Custom exceptions and error handlers for Forecastly API.

Provides centralized exception handling and structured error responses.
"""

from datetime import datetime
from typing import Any, Dict, Optional
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from src.utils.logger import logger


# ==============================================================================
# Custom Exception Classes
# ==============================================================================

class ForecastlyException(Exception):
    """Base exception for all Forecastly errors."""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)


class DataNotFoundException(ForecastlyException):
    """Exception raised when requested data is not found."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="DATA_NOT_FOUND",
            details=details,
        )


class SKUNotFoundException(DataNotFoundException):
    """Exception raised when SKU is not found."""

    def __init__(self, sku_id: str, available_skus: Optional[list] = None):
        details = {"sku_id": sku_id}
        if available_skus:
            details["available_skus"] = available_skus[:5]  # First 5 for hint
        super().__init__(
            message=f"SKU '{sku_id}' not found in forecasts",
            details=details,
        )


class PredictionNotFoundException(DataNotFoundException):
    """Exception raised when predictions are not found."""

    def __init__(self, message: str = "Predictions not found"):
        super().__init__(
            message=message,
            details={"hint": "Run forecast generation first"},
        )


class InvalidParameterException(ForecastlyException):
    """Exception raised when invalid parameters are provided."""

    def __init__(self, parameter: str, message: str, valid_range: Optional[str] = None):
        details = {"parameter": parameter}
        if valid_range:
            details["valid_range"] = valid_range
        super().__init__(
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="INVALID_PARAMETER",
            details=details,
        )


class ModelNotFoundException(ForecastlyException):
    """Exception raised when trained model is not found."""

    def __init__(self, model_name: str):
        super().__init__(
            message=f"Trained model '{model_name}' not found",
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="MODEL_NOT_FOUND",
            details={
                "model_name": model_name,
                "hint": "Train the model first using /models/train endpoint",
            },
        )


class DatabaseException(ForecastlyException):
    """Exception raised for database-related errors."""

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        details = {}
        if original_error:
            details["error_type"] = type(original_error).__name__
            details["error_message"] = str(original_error)
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="DATABASE_ERROR",
            details=details,
        )


class ForecastGenerationException(ForecastlyException):
    """Exception raised when forecast generation fails."""

    def __init__(self, message: str, stderr: Optional[str] = None):
        details = {}
        if stderr:
            details["stderr"] = stderr[:500]  # Limit error message length
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="FORECAST_GENERATION_FAILED",
            details=details,
        )


class RateLimitExceededException(ForecastlyException):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, retry_after: Optional[int] = None):
        details = {}
        if retry_after:
            details["retry_after_seconds"] = retry_after
        super().__init__(
            message="Rate limit exceeded. Please try again later.",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code="RATE_LIMIT_EXCEEDED",
            details=details,
        )


# ==============================================================================
# Exception Handlers
# ==============================================================================

async def forecastly_exception_handler(
    request: Request, exc: ForecastlyException
) -> JSONResponse:
    """
    Handler for all custom Forecastly exceptions.

    Returns structured error response with error code and details.
    """
    logger.error(
        f"ForecastlyException: {exc.error_code} - {exc.message}",
        extra={"details": exc.details, "path": request.url.path},
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "details": exc.details,
            },
            "path": str(request.url.path),
            "timestamp": datetime.now().isoformat(),
        },
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Handler for request validation errors.

    Returns user-friendly validation error messages.
    """
    errors = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"])
        errors.append(
            {
                "field": field,
                "message": error["msg"],
                "type": error["type"],
            }
        )

    logger.warning(
        f"Validation error on {request.url.path}",
        extra={"errors": errors},
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": {"validation_errors": errors},
            },
            "path": str(request.url.path),
        },
    )


async def sqlalchemy_exception_handler(
    request: Request, exc: SQLAlchemyError
) -> JSONResponse:
    """
    Handler for SQLAlchemy database errors.

    Provides user-friendly error messages while logging technical details.
    """
    logger.error(
        f"Database error: {type(exc).__name__}",
        exc_info=exc,
        extra={"path": request.url.path},
    )

    # Specific handling for integrity errors
    if isinstance(exc, IntegrityError):
        message = "Database integrity constraint violated"
        error_code = "INTEGRITY_ERROR"
    else:
        message = "Database operation failed"
        error_code = "DATABASE_ERROR"

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": error_code,
                "message": message,
                "details": {
                    "hint": "Check your request parameters and try again",
                },
            },
            "path": str(request.url.path),
        },
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handler for unexpected exceptions.

    Catches all unhandled exceptions to prevent leaking sensitive information.
    """
    logger.exception(
        f"Unhandled exception: {type(exc).__name__}",
        exc_info=exc,
        extra={"path": request.url.path},
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "details": {
                    "hint": "Please contact support if the problem persists",
                },
            },
            "path": str(request.url.path),
        },
    )


# ==============================================================================
# Helper Functions
# ==============================================================================

def register_exception_handlers(app):
    """
    Register all exception handlers with the FastAPI app.

    Args:
        app: FastAPI application instance
    """
    from fastapi.exceptions import RequestValidationError

    # Custom Forecastly exceptions
    app.add_exception_handler(ForecastlyException, forecastly_exception_handler)

    # Validation errors
    app.add_exception_handler(RequestValidationError, validation_exception_handler)

    # Database errors
    app.add_exception_handler(SQLAlchemyError, sqlalchemy_exception_handler)

    # Catch-all for unexpected errors
    app.add_exception_handler(Exception, generic_exception_handler)

    logger.info("✓ Exception handlers registered")
