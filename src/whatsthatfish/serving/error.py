from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError


class BaseAppException(Exception):
    """General exception for the FastAPI service"""

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class ResourceNotFoundException(BaseAppException):
    """Basic resource not found"""

    def __init__(self, message: str):
        super().__init__(message, status_code=404)


class ValidationException(BaseAppException):
    """Validation exception for invalid inputs"""

    def __init__(self, message: str):
        super().__init__(message, status_code=400)


class AuthenticationException(BaseAppException):
    """Missing, invalid, or expired credentials."""

    def __init__(self, message: str = "Not authenticated"):
        super().__init__(message, status_code=401)


class InvalidPredictionResponse(BaseAppException):
    """Invalid or empty response on prediction endpoint"""

    def __init__(self, message: str):
        super().__init__(message)


class InvalidPredictionRequest(RequestValidationError):
    """Incorrect inputs required for inference"""

    def __init__(self, message: str, body: Any, status_code: int = 422):
        self.message = message
        self.body = body
        self.status_code = status_code
        super().__init__(self.message, body=self.body)
