"""
Authorization enforced by the BACKEND (endpoints called directly, no UI involved).
Real MongoDB (temp DB) + temp Qdrant + fake LLM/embeddings.
"""
import pytest

from conftest import (TEST_ADMIN_PASSWORD, TEST_ADMIN_USERNAME, TEST_STUDENT_CODE, TEST_STUDENT_PROJECT,
                      auth_headers, make_token)
from integration.test_api_flow import make_pdf, upload

pytestmark = pytest.mark.integration

P = TEST_STUDENT_PROJECT
ADMIN_ONLY = [
    ("post", f"/api/v1/data/process/{P}", {"json": {}}),
    ("post", f"/api/v1/nlp/index/push/{P}", {"json": {"do_reset": 1}}),
    ("get", f"/api/v1/nlp/index/info/{P}", {}),
    ("post", f"/api/v1/nlp/index/search/{P}", {"json": {"text": "loop"}}),
    ("get", f"/api/v1/data/assets/{P}", {}),
    ("delete", f"/api/v1/data/asset/{P}/abcdefghijkl_x.pdf", {}),
]


def call(client, method, url, headers=None, **kw):
    return getattr(client, method)(url, headers=headers, **kw)


@pytest.fixture
def anon(api_client):
    api_client.headers.pop("Authorization", None)
    return api_client


@pytest.fixture
def student_headers(api_client):
    r = api_client.post("/api/v1/auth/student/login", json={"access_code": TEST_STUDENT_CODE, "display_name": "Ali"})
    assert r.status_code == 200, r.text
    return auth_headers(r.json()["access_token"])


@pytest.fixture
def indexed_course(api_client, tmp_path):
    pdf = make_pdf(tmp_path / "Lecture 5.pdf", ["The do-while loop executes its body at least once."])
    assert upload(api_client, P, pdf, doc_type="lecture", course="CS101").status_code == 200
    assert api_client.post(f"/api/v1/data/process/{P}", json={"do_reset": 1}).status_code == 200
    assert api_client.post(f"/api/v1/nlp/index/push/{P}", json={"do_reset": 1}).status_code == 200
    return pdf


# --------------------------------------------------------------------------- authentication
def test_login_endpoints(anon):
    r = anon.post("/api/v1/auth/admin/login", json={"username": TEST_ADMIN_USERNAME, "password": TEST_ADMIN_PASSWORD})
    assert r.status_code == 200 and r.json()["role"] == "admin"
    me = anon.get("/api/v1/auth/me", headers=auth_headers(r.json()["access_token"]))
    assert me.json()["role"] == "admin"

    assert anon.post("/api/v1/auth/admin/login",
                     json={"username": TEST_ADMIN_USERNAME, "password": "nope"}).status_code == 401
    assert anon.post("/api/v1/auth/student/login", json={"access_code": "nope"}).status_code == 401
    r = anon.post("/api/v1/auth/student/login", json={"access_code": TEST_STUDENT_CODE})
    assert r.status_code == 200 and r.json()["project_id"] == P
    assert "password" not in r.text.lower() and TEST_STUDENT_CODE not in r.text


def test_student_cannot_use_admin_password_for_admin_and_vice_versa(anon):
    assert anon.post("/api/v1/auth/admin/login",
                     json={"username": TEST_ADMIN_USERNAME, "password": TEST_STUDENT_CODE}).status_code == 401
    assert anon.post("/api/v1/auth/student/login", json={"access_code": TEST_ADMIN_PASSWORD}).status_code == 401


@pytest.mark.parametrize("method,url,kw", ADMIN_ONLY + [
    ("post", f"/api/v1/nlp/index/answer/{P}", {"json": {"text": "loop"}}),
    ("get", "/api/v1/auth/me", {}),
])
def test_unauthenticated_requests_rejected(anon, method, url, kw):
    r = call(anon, method, url, **kw)
    assert r.status_code == 401
    assert r.json()["detail"]["signal"] == "authentication_required"


def test_upload_requires_auth(anon, tmp_path):
    pdf = make_pdf(tmp_path / "x.pdf", ["x"])
    assert upload(anon, P, pdf).status_code == 401


@pytest.mark.parametrize("token,signal", [
    ("not-a-jwt", "invalid_token"),
    ("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ4In0.bad", "invalid_token"),
    (None, "token_expired"),  # replaced by an expired token below
])
def test_invalid_and_expired_tokens_rejected(anon, token, signal):
    token = token or make_token("admin", expires_minutes=-1)
    r = anon.get(f"/api/v1/nlp/index/info/{P}", headers=auth_headers(token))
    assert r.status_code == 401 and r.json()["detail"]["signal"] == signal


def test_token_signed_with_other_key_rejected(anon):
    from helpers.security import create_access_token
    forged = create_access_token("admin:x", "admin", "z" * 48, 5)
    assert anon.get(f"/api/v1/nlp/index/info/{P}", headers=auth_headers(forged)).status_code == 401


# --------------------------------------------------------------------------- authorization
@pytest.mark.parametrize("method,url,kw", ADMIN_ONLY)
def test_students_cannot_call_admin_endpoints(api_client, student_headers, method, url, kw):
    r = call(api_client, method, url, headers=student_headers, **kw)
    assert r.status_code == 403 and r.json()["detail"]["signal"] == "forbidden"


def test_student_cannot_upload(api_client, student_headers, tmp_path):
    pdf = make_pdf(tmp_path / "student.pdf", ["malicious"])
    with open(pdf, "rb") as f:
        r = api_client.post(f"/api/v1/data/upload/{P}", headers=student_headers,
                            files={"file": ("student.pdf", f, "application/pdf")}, data={"doc_type": "lecture"})
    assert r.status_code == 403


def test_student_cannot_delete_course_material(api_client, student_headers, indexed_course):
    assets = api_client.get(f"/api/v1/data/assets/{P}").json()["assets"]
    fid = assets[0]["file_id"]
    r = api_client.delete(f"/api/v1/data/asset/{P}/{fid}", headers=student_headers)
    assert r.status_code == 403
    assert len(api_client.get(f"/api/v1/data/assets/{P}").json()["assets"]) == 1  # untouched


def test_student_can_ask_questions_and_get_sources(api_client, student_headers, indexed_course):
    api_client.fake_generation.answer = "At least once.\n[[USED_DOCUMENTS: 1]]"
    r = api_client.post(f"/api/v1/nlp/index/answer/{P}", headers=student_headers,
                        json={"text": "ما هي do-while loop؟", "chat_history": [
                            {"role": "user", "content": "loops"}, {"role": "assistant", "content": "ok"}]})
    assert r.status_code == 200, r.text
    assert r.json()["sources"]["cited"][0]["document_name"] == "Lecture 5.pdf"


def test_student_bound_to_course_project(api_client, student_headers):
    r = api_client.post("/api/v1/nlp/index/answer/otherproject", headers=student_headers, json={"text": "x"})
    assert r.status_code == 403


def test_admin_can_access_all_operations(api_client, indexed_course):
    assert api_client.get(f"/api/v1/nlp/index/info/{P}").status_code == 200
    assert api_client.post(f"/api/v1/nlp/index/search/{P}", json={"text": "loop"}).status_code == 200
    assert api_client.get(f"/api/v1/data/assets/{P}").status_code == 200
    api_client.fake_generation.answer = "x [[USED_DOCUMENTS: 1]]"
    assert api_client.post(f"/api/v1/nlp/index/answer/{P}", json={"text": "loop"}).status_code == 200


def test_request_size_limits(api_client, student_headers):
    assert api_client.post(f"/api/v1/nlp/index/answer/{P}", headers=student_headers,
                           json={"text": "x", "limit": 1000}).status_code == 422
    assert api_client.post(f"/api/v1/nlp/index/answer/{P}", headers=student_headers,
                           json={"text": "x" * 9000}).status_code == 422


# --------------------------------------------------------------------------- rate limiting
def test_login_brute_force_rate_limited(anon, monkeypatch):
    monkeypatch.setenv("LOGIN_RATE_LIMIT", "3")
    codes = [anon.post("/api/v1/auth/student/login", json={"access_code": "guess"}).status_code for _ in range(5)]
    assert codes[:3] == [401, 401, 401] and codes[3:] == [429, 429]
    r = anon.post("/api/v1/auth/admin/login", json={"username": TEST_ADMIN_USERNAME, "password": TEST_ADMIN_PASSWORD})
    assert r.status_code == 429  # same IP, correct password still blocked during the window
    assert int(r.headers["Retry-After"]) > 0
    assert r.json()["detail"]["signal"] == "rate_limit_exceeded"


def test_student_question_rate_limit(api_client, student_headers, indexed_course, monkeypatch):
    monkeypatch.setenv("STUDENT_ANSWER_RATE_LIMIT_PER_MINUTE", "2")
    api_client.fake_generation.answer = "ok [[USED_DOCUMENTS: none]]"
    codes = [api_client.post(f"/api/v1/nlp/index/answer/{P}", headers=student_headers,
                             json={"text": "loop"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    # admins are not limited
    assert api_client.post(f"/api/v1/nlp/index/answer/{P}", json={"text": "loop"}).status_code == 200


def test_rate_limit_is_per_student_session(api_client, indexed_course, monkeypatch):
    monkeypatch.setenv("STUDENT_ANSWER_RATE_LIMIT_PER_MINUTE", "1")
    api_client.fake_generation.answer = "ok [[USED_DOCUMENTS: none]]"
    tokens = [api_client.post("/api/v1/auth/student/login", json={"access_code": TEST_STUDENT_CODE}).json()["access_token"]
              for _ in range(2)]
    for t in tokens:
        assert api_client.post(f"/api/v1/nlp/index/answer/{P}", headers=auth_headers(t), json={"text": "q"}).status_code == 200
    assert api_client.post(f"/api/v1/nlp/index/answer/{P}", headers=auth_headers(tokens[0]),
                           json={"text": "q"}).status_code == 429
