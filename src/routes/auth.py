from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
import logging

from controllers import AuthController
from models import ResponseSignal
from .dependencies import get_current_user, login_rate_limit
from .schemes.auth import AdminLoginRequest, StudentLoginRequest

logger = logging.getLogger('uvicorn.error')

auth_router = APIRouter(
    prefix="/api/v1/auth",
    tags=["api_v1", "auth"],
)


@auth_router.post("/admin/login", dependencies=[Depends(login_rate_limit)])
async def admin_login(login_request: AdminLoginRequest):
    auth_controller = AuthController()
    if not auth_controller.is_admin_configured():
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            content={"signal": ResponseSignal.AUTH_NOT_CONFIGURED.value})

    token = auth_controller.authenticate_admin(login_request.username, login_request.password)
    if not token:
        logger.warning("Failed admin login attempt")
        return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED,
                            content={"signal": ResponseSignal.AUTH_LOGIN_FAILED.value})

    return JSONResponse(content={"signal": ResponseSignal.AUTH_LOGIN_SUCCESS.value, **token})


@auth_router.post("/student/login", dependencies=[Depends(login_rate_limit)])
async def student_login(login_request: StudentLoginRequest):
    auth_controller = AuthController()
    if not auth_controller.is_student_configured():
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            content={"signal": ResponseSignal.AUTH_NOT_CONFIGURED.value})

    token = auth_controller.authenticate_student(login_request.access_code, login_request.display_name)
    if not token:
        logger.warning("Failed student login attempt")
        return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED,
                            content={"signal": ResponseSignal.AUTH_LOGIN_FAILED.value})

    return JSONResponse(content={"signal": ResponseSignal.AUTH_LOGIN_SUCCESS.value,
                                 "project_id": auth_controller.app_settings.STUDENT_PROJECT_ID, **token})


@auth_router.get("/me")
async def who_am_i(user: dict = Depends(get_current_user)):
    """Lets a frontend validate its token with the backend (never trust frontend-only auth)."""
    return JSONResponse(content={
        "role": user["role"],
        "subject": user["sub"],
        "project_id": user.get("project_id"),
        "name": user.get("name"),
        "expires_at": user["exp"],
    })
