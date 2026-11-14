"""
Password hashing and signed access tokens.

- Passwords / access codes: scrypt (stdlib hashlib) with a random 16-byte salt per hash,
  stored as  scrypt:<n>:<r>:<p>:<salt_b64>:<hash_b64>   (no '$', so it is safe in .env files).
- Tokens: JWT signed with HS256 (PyJWT), always with exp / iat / sub / role / jti.

Generate values for .env (prompts, nothing is echoed or logged):
    python -m helpers.security hash      -> hash for ADMIN_PASSWORD_HASH / STUDENT_ACCESS_CODE_HASH
    python -m helpers.security secret    -> random JWT_SECRET_KEY
"""
import base64
import hashlib
import hmac
import secrets
import time
import uuid

import jwt

SCRYPT_N = 2 ** 14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 32
MIN_SECRET_LENGTH = 32

ROLE_ADMIN = "admin"
ROLE_STUDENT = "student"


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def hash_password(password: str) -> str:
    if not password:
        raise ValueError("password must not be empty")
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P,
                            dklen=SCRYPT_DKLEN)
    return f"scrypt:{SCRYPT_N}:{SCRYPT_R}:{SCRYPT_P}:{_b64e(salt)}:{_b64e(digest)}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Constant-time check. Returns False (never raises) for empty input or malformed hashes."""
    if not password or not stored_hash:
        return False
    try:
        algorithm, n, r, p, salt_b64, digest_b64 = stored_hash.strip().split(":")
        if algorithm != "scrypt":
            return False
        expected = _b64d(digest_b64)
        actual = hashlib.scrypt(password.encode("utf-8"), salt=_b64d(salt_b64), n=int(n), r=int(r), p=int(p),
                                dklen=len(expected))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


def create_access_token(subject: str, role: str, secret_key: str, expires_minutes: int,
                        algorithm: str = "HS256", extra_claims: dict = None) -> str:
    now = int(time.time())
    claims = {
        **(extra_claims or {}),
        "sub": subject,
        "role": role,
        "iat": now,
        "exp": now + int(expires_minutes) * 60,
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(claims, secret_key, algorithm=algorithm)


def decode_access_token(token: str, secret_key: str, algorithm: str = "HS256") -> dict:
    """Raises jwt.ExpiredSignatureError / jwt.InvalidTokenError on any problem."""
    return jwt.decode(token, secret_key, algorithms=[algorithm],
                      options={"require": ["exp", "iat", "sub", "role", "jti"]})


if __name__ == "__main__":
    import getpass
    import sys

    command = sys.argv[1] if len(sys.argv) > 1 else "hash"
    if command == "secret":
        print(secrets.token_urlsafe(48))
    elif command == "hash":
        first = getpass.getpass("Password / access code: ")
        if first != getpass.getpass("Repeat: "):
            sys.exit("Values do not match.")
        if len(first) < 8:
            sys.exit("Use at least 8 characters.")
        print(hash_password(first))
    else:
        sys.exit("usage: python -m helpers.security [hash|secret]")
