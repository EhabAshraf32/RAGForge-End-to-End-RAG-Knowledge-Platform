from .BaseController import BaseController
from helpers.security import (verify_password, create_access_token, decode_access_token,
                              ROLE_ADMIN, ROLE_STUDENT, MIN_SECRET_LENGTH)
import hmac
import uuid

# verified against when the username is wrong, so timing does not reveal valid usernames
_DUMMY_HASH = "scrypt:16384:8:1:AAAAAAAAAAAAAAAAAAAAAA:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"


class AuthController(BaseController):

    def __init__(self):
        super().__init__()

    def is_token_signing_configured(self):
        secret = self.app_settings.JWT_SECRET_KEY
        return bool(secret) and len(secret) >= MIN_SECRET_LENGTH

    def is_admin_configured(self):
        return self.is_token_signing_configured() and bool(self.app_settings.ADMIN_USERNAME) \
            and bool(self.app_settings.ADMIN_PASSWORD_HASH)

    def is_student_configured(self):
        return self.is_token_signing_configured() and bool(self.app_settings.STUDENT_ACCESS_CODE_HASH) \
            and bool(self.app_settings.STUDENT_PROJECT_ID)

    def _create_token(self, subject: str, role: str, expires_minutes: int, extra_claims: dict = None):
        token = create_access_token(
            subject=subject, role=role,
            secret_key=self.app_settings.JWT_SECRET_KEY,
            expires_minutes=expires_minutes,
            algorithm=self.app_settings.JWT_ALGORITHM,
            extra_claims=extra_claims,
        )
        return {"access_token": token, "token_type": "bearer", "role": role,
                "expires_in": int(expires_minutes) * 60}

    def authenticate_admin(self, username: str, password: str):
        """Returns token info, or None for wrong credentials / not configured."""
        if not self.is_admin_configured():
            return None
        username_ok = hmac.compare_digest((username or "").encode("utf-8"),
                                          self.app_settings.ADMIN_USERNAME.encode("utf-8"))
        password_ok = verify_password(password, self.app_settings.ADMIN_PASSWORD_HASH if username_ok else _DUMMY_HASH)
        if not (username_ok and password_ok):
            return None
        return self._create_token(subject=f"admin:{self.app_settings.ADMIN_USERNAME}", role=ROLE_ADMIN,
                                  expires_minutes=self.app_settings.ADMIN_TOKEN_EXPIRE_MINUTES)

    def authenticate_student(self, access_code: str, display_name: str = None):
        """Students share the course access code; every login gets its own anonymous session subject,
        so rate limits apply per session. The token is bound to STUDENT_PROJECT_ID."""
        if not self.is_student_configured():
            return None
        if not verify_password(access_code, self.app_settings.STUDENT_ACCESS_CODE_HASH):
            return None
        name = (display_name or "").strip()[:40] or None
        return self._create_token(
            subject=f"student:{uuid.uuid4().hex}", role=ROLE_STUDENT,
            expires_minutes=self.app_settings.STUDENT_TOKEN_EXPIRE_MINUTES,
            extra_claims={"project_id": self.app_settings.STUDENT_PROJECT_ID, "name": name},
        )

    def decode_token(self, token: str):
        """Raises jwt.ExpiredSignatureError / jwt.InvalidTokenError; returns the claims."""
        if not self.is_token_signing_configured():
            raise PermissionError("token signing is not configured")
        claims = decode_access_token(token, self.app_settings.JWT_SECRET_KEY, self.app_settings.JWT_ALGORITHM)
        if claims.get("role") not in (ROLE_ADMIN, ROLE_STUDENT):
            raise PermissionError("unknown role")
        return claims
