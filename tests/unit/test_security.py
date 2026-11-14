"""Password hashing, signed tokens and the rate limiter."""
import time

import jwt
import pytest

from conftest import TEST_ADMIN_PASSWORD, TEST_ADMIN_USERNAME, TEST_STUDENT_CODE, TEST_STUDENT_PROJECT
from controllers import AuthController
from helpers.rate_limit import RateLimiter
from helpers.security import (create_access_token, decode_access_token, hash_password, verify_password)

SECRET = "x" * 40


# --------------------------------------------------------------------------- hashing
def test_password_hash_is_salted_and_verifies():
    h1, h2 = hash_password("s3cret-pass"), hash_password("s3cret-pass")
    assert h1 != h2                                  # random salt
    assert "s3cret-pass" not in h1 and "$" not in h1  # no plaintext, .env-safe
    assert verify_password("s3cret-pass", h1) and verify_password("s3cret-pass", h2)
    assert not verify_password("wrong", h1)


@pytest.mark.parametrize("stored", ["", None, "plaintext", "md5:abc", "scrypt:1:2", "scrypt:16384:8:1:!!:!!"])
def test_verify_password_never_raises_on_malformed_hash(stored):
    assert verify_password("anything", stored) is False


def test_empty_password_rejected():
    with pytest.raises(ValueError):
        hash_password("")
    assert verify_password("", hash_password("abc12345")) is False


# --------------------------------------------------------------------------- tokens
def test_token_roundtrip_and_required_claims():
    token = create_access_token("admin:ta", "admin", SECRET, 5, extra_claims={"project_id": "c1"})
    claims = decode_access_token(token, SECRET)
    assert claims["sub"] == "admin:ta" and claims["role"] == "admin" and claims["project_id"] == "c1"
    assert claims["exp"] > time.time() and claims["jti"]


def test_expired_token_rejected():
    token = create_access_token("s", "student", SECRET, expires_minutes=-1)
    with pytest.raises(jwt.ExpiredSignatureError):
        decode_access_token(token, SECRET)


def test_tampered_or_wrong_key_token_rejected():
    token = create_access_token("s", "student", SECRET, 5)
    with pytest.raises(jwt.InvalidTokenError):
        decode_access_token(token, "y" * 40)
    header, payload, sig = token.split(".")
    forged = jwt.encode({"sub": "s", "role": "admin", "iat": 0, "exp": 9999999999, "jti": "x"}, "y" * 40)
    with pytest.raises(jwt.InvalidTokenError):
        decode_access_token(".".join([header, forged.split(".")[1], sig]), SECRET)


def test_alg_none_token_rejected():
    unsigned = jwt.encode({"sub": "s", "role": "admin", "iat": 0, "exp": 9999999999, "jti": "x"},
                          key=None, algorithm="none")
    with pytest.raises(jwt.InvalidTokenError):
        decode_access_token(unsigned, SECRET)


def test_token_missing_role_rejected():
    token = jwt.encode({"sub": "s", "iat": int(time.time()), "exp": int(time.time()) + 60, "jti": "x"}, SECRET)
    with pytest.raises(jwt.InvalidTokenError):
        decode_access_token(token, SECRET)


# --------------------------------------------------------------------------- AuthController
def test_admin_authentication():
    ctrl = AuthController()
    assert ctrl.is_admin_configured()
    token = ctrl.authenticate_admin(TEST_ADMIN_USERNAME, TEST_ADMIN_PASSWORD)
    assert token["role"] == "admin" and ctrl.decode_token(token["access_token"])["role"] == "admin"
    assert ctrl.authenticate_admin(TEST_ADMIN_USERNAME, "wrong") is None
    assert ctrl.authenticate_admin("someone", TEST_ADMIN_PASSWORD) is None


def test_student_authentication_binds_project_and_unique_sessions():
    ctrl = AuthController()
    t1 = ctrl.authenticate_student(TEST_STUDENT_CODE, "Mona")
    t2 = ctrl.authenticate_student(TEST_STUDENT_CODE)
    c1, c2 = ctrl.decode_token(t1["access_token"]), ctrl.decode_token(t2["access_token"])
    assert c1["role"] == "student" and c1["project_id"] == TEST_STUDENT_PROJECT and c1["name"] == "Mona"
    assert c1["sub"] != c2["sub"]
    assert ctrl.authenticate_student("wrong-code") is None


def test_not_configured_disables_login(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "too-short")
    ctrl = AuthController()
    assert not ctrl.is_token_signing_configured()
    assert ctrl.authenticate_admin(TEST_ADMIN_USERNAME, TEST_ADMIN_PASSWORD) is None
    with pytest.raises(PermissionError):
        ctrl.decode_token("anything")


def test_unknown_role_rejected():
    import os
    token = create_access_token("x", "superuser", os.environ["JWT_SECRET_KEY"], 5)
    with pytest.raises(PermissionError):
        AuthController().decode_token(token)


# --------------------------------------------------------------------------- rate limiter
def test_rate_limiter_blocks_and_recovers(monkeypatch):
    import helpers.rate_limit as rl
    now = [1000.0]
    monkeypatch.setattr(rl.time, "monotonic", lambda: now[0])
    limiter = RateLimiter()
    assert [limiter.hit("k", 3, 60)[0] for _ in range(3)] == [True, True, True]
    allowed, retry = limiter.hit("k", 3, 60)
    assert not allowed and 1 <= retry <= 61
    assert limiter.hit("other", 3, 60)[0]   # keys are independent
    now[0] += 61
    assert limiter.hit("k", 3, 60)[0]       # window passed


def test_rate_limit_zero_disables():
    limiter = RateLimiter()
    assert all(limiter.hit("k", 0, 60)[0] for _ in range(100))


@pytest.mark.parametrize("value", ["", "short-secret"])
def test_app_refuses_to_start_without_strong_jwt_secret(monkeypatch, value):
    import asyncio
    import main
    monkeypatch.setenv("JWT_SECRET_KEY", value)
    with pytest.raises(RuntimeError, match="JWT_SECRET_KEY"):
        asyncio.run(main.startup_span())
