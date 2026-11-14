"""
FastAPI dependencies for authentication, authorization (roles) and rate limiting.
Every protected route declares one of these, so permissions are enforced by the backend itself
(hiding buttons in a UI is never relied upon).
"""
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import hmac
import jwt
import logging

from controllers import AuthController
from helpers.config import get_settings
from helpers.rate_limit import rate_limiter
from helpers.security import ROLE_ADMIN, ROLE_STUDENT
from models import ResponseSignal

logger = logging.getLogger('uvicorn.error')

bearer_scheme = HTTPBearer(auto_error=False)


def _error(code: int, signal: ResponseSignal, headers: dict = None):
    return HTTPException(status_code=code, detail={"signal": signal.value}, headers=headers)


def get_client_ip(request: Request):
    """Rate-limit key for the caller. Server-side frontends (Streamlit Cloud) make every request from their
    own IP, so they forward the end user's IP in X-Client-IP + the shared TRUSTED_FRONTEND_SECRET.
    The forwarded value is trusted ONLY with the right secret and is never used for authorization."""
    secret = get_settings().TRUSTED_FRONTEND_SECRET
    if secret:
        sent = request.headers.get("x-frontend-secret", "")
        forwarded = request.headers.get("x-client-ip", "").strip()[:64]
        if forwarded and hmac.compare_digest(sent.encode("utf-8"), secret.encode("utf-8")):
            return f"fe:{forwarded}"
    return request.client.host if request.client else "unknown"


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials is None or credentials.scheme.lower() != "bearer" or not credentials.credentials:
        raise _error(status.HTTP_401_UNAUTHORIZED, ResponseSignal.AUTH_REQUIRED, {"WWW-Authenticate": "Bearer"})
    try:
        return AuthController().decode_token(credentials.credentials)
    except jwt.ExpiredSignatureError:
        raise _error(status.HTTP_401_UNAUTHORIZED, ResponseSignal.AUTH_TOKEN_EXPIRED, {"WWW-Authenticate": "Bearer"})
    except (jwt.InvalidTokenError, PermissionError):
        raise _error(status.HTTP_401_UNAUTHORIZED, ResponseSignal.AUTH_INVALID_TOKEN, {"WWW-Authenticate": "Bearer"})


def require_admin(user: dict = Depends(get_current_user)):
    if user.get("role") != ROLE_ADMIN:
        raise _error(status.HTTP_403_FORBIDDEN, ResponseSignal.AUTH_FORBIDDEN)
    return user


def require_student_or_admin(user: dict = Depends(get_current_user)):
    if user.get("role") not in (ROLE_ADMIN, ROLE_STUDENT):
        raise _error(status.HTTP_403_FORBIDDEN, ResponseSignal.AUTH_FORBIDDEN)
    return user


def check_project_access(user: dict, project_id: str):
    """Students may only use the course project their token was issued for."""
    if user.get("role") == ROLE_STUDENT and user.get("project_id") != project_id:
        raise _error(status.HTTP_403_FORBIDDEN, ResponseSignal.AUTH_FORBIDDEN)


def _enforce(key: str, limit: int, window_seconds: int):
    allowed, retry_after = rate_limiter.hit(key, limit, window_seconds)
    if not allowed:
        logger.warning(f"Rate limit exceeded for {key.split(':')[0]}")
        raise _error(status.HTTP_429_TOO_MANY_REQUESTS, ResponseSignal.RATE_LIMIT_EXCEEDED,
                     {"Retry-After": str(retry_after)})


def login_rate_limit(request: Request):
    """Brute-force protection: all login attempts (admin + student) per client IP."""
    s = get_settings()
    _enforce(f"login:{get_client_ip(request)}", s.LOGIN_RATE_LIMIT, s.LOGIN_RATE_LIMIT_WINDOW_SECONDS)


def answer_rate_limit(user: dict = Depends(require_student_or_admin)):
    """Question limits per student session (admins are not limited)."""
    if user.get("role") == ROLE_STUDENT:
        s = get_settings()
        _enforce(f"answer-min:{user['sub']}", s.STUDENT_ANSWER_RATE_LIMIT_PER_MINUTE, 60)
        _enforce(f"answer-hour:{user['sub']}", s.STUDENT_ANSWER_RATE_LIMIT_PER_HOUR, 3600)
    return user
