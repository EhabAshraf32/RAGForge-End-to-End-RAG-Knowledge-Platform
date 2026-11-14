"""
Shared test configuration.

Isolation guarantees:
- cwd is moved to the repo root, so `Settings` does NOT read src/.env in unit/integration tests;
  only safe, non-secret dummy values are provided through environment variables.
- Files are written to a pytest tmp dir (BaseController paths are redirected), never to src/assets.
- Qdrant runs in a tmp dir. MongoDB integration tests use a throw-away database that is dropped afterwards.
- No test in tests/unit or tests/integration calls an external API (fake LLM / embedding clients).
"""
import hashlib
import math
import os
import re
import sys
import uuid
from types import SimpleNamespace

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
SRC_ENV_FILE = os.path.join(SRC_DIR, ".env")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
os.chdir(REPO_ROOT)  # no .env here -> Settings only sees the variables below

_SAFE_ENV = {
    "APP_NAME": "mini-rag-test",
    "APP_VERSION": "0.0-test",
    "FILE_ALLOWED_TYPES": '["text/plain","application/pdf"]',
    "FILE_MAX_SIZE": "10",
    "FILE_DEFAULT_CHUNK_SIZE": "512000",
    "MONGODB_URL": "mongodb://localhost:1",
    "MONGODB_DATABASE": "mini_rag_unit_tests",
    "GENERATION_BACKEND": "OPENAI",
    "EMBEDDING_BACKEND": "HUGGINGFACE",
    "INPUT_DAFAULT_MAX_CHARACTERS": "1024",
    "GENERATION_DAFAULT_MAX_TOKENS": "2000",
    "GENERATION_DAFAULT_TEMPERATURE": "0.1",
    "VECTOR_DB_BACKEND": "QDRANT",
    "VECTOR_DB_PATH": "qdrant_test_unused",
    "VECTOR_DB_DISTANCE_METHOD": "cosine",
}
for _k, _v in _SAFE_ENV.items():
    os.environ[_k] = _v

# test-only credentials (random signing key per run; nothing from src/.env)
import secrets as _secrets  # noqa: E402
from helpers.security import hash_password as _hash_password  # noqa: E402

TEST_ADMIN_USERNAME = "ta_admin"
TEST_ADMIN_PASSWORD = "admin-test-password-123"
TEST_STUDENT_CODE = "student-code-456"
TEST_STUDENT_PROJECT = "itest1"
os.environ.update({
    "JWT_SECRET_KEY": _secrets.token_urlsafe(48),
    "ADMIN_USERNAME": TEST_ADMIN_USERNAME,
    "ADMIN_PASSWORD_HASH": _hash_password(TEST_ADMIN_PASSWORD),
    "STUDENT_ACCESS_CODE_HASH": _hash_password(TEST_STUDENT_CODE),
    "STUDENT_PROJECT_ID": TEST_STUDENT_PROJECT,
})


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    from helpers.rate_limit import rate_limiter
    rate_limiter.reset()
    yield
    rate_limiter.reset()


def make_token(role="admin", expires_minutes=5, **claims):
    from helpers.security import create_access_token
    sub = "admin:ta_admin" if role == "admin" else "student:test"
    if role == "student":
        claims.setdefault("project_id", TEST_STUDENT_PROJECT)
    return create_access_token(sub, role, os.environ["JWT_SECRET_KEY"], expires_minutes, extra_claims=claims)


def auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


# --------------------------------------------------------------------------- fakes
class FakeEmbeddingClient:
    """Deterministic bag-of-words hashing embedder (no model download, no network).
    Similar texts share tokens -> similar vectors, so retrieval behaves meaningfully."""

    def __init__(self, embedding_size: int = 256):
        self.embedding_size = embedding_size
        self.calls = []

    def embed_text(self, text, document_type=None):
        self.calls.append(document_type)
        vec = [0.0] * self.embedding_size
        for tok in re.findall(r"\w+", (text or "").lower()):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % self.embedding_size] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class FakeGenerationClient:
    """Records the messages it receives and returns a canned answer (with citation marker)."""

    def __init__(self, answer="```c\nint x = 5;\n```\n```text\n5\n```\n[[USED_DOCUMENTS: 1]]"):
        from stores.llm.LLMEnums import OpenAIEnums
        self.enums = OpenAIEnums
        self.answer = answer
        self.received = []

    def construct_prompt(self, prompt, role):
        return {"role": role, "content": prompt}

    def generate_text(self, prompt, chat_history=None, max_output_tokens=None, temperature=None):
        messages = list(chat_history or []) + [self.construct_prompt(prompt, "user")]
        self.received.append(messages)
        return self.answer


@pytest.fixture
def fake_embedding():
    return FakeEmbeddingClient()


@pytest.fixture
def fake_generation():
    return FakeGenerationClient()


@pytest.fixture
def qdrant(tmp_path):
    from stores.vectordb.providers import QdrantDBProvider
    q = QdrantDBProvider(db_path=str(tmp_path / "qdrant"), distance_method="cosine")
    q.connect()
    yield q
    q.client.close()


@pytest.fixture
def isolated_files(tmp_path, monkeypatch):
    """Redirect BaseController's assets/files and assets/database dirs to a tmp dir."""
    from controllers.BaseController import BaseController
    original_init = BaseController.__init__
    files_dir = tmp_path / "files"
    db_dir = tmp_path / "database"
    files_dir.mkdir()
    db_dir.mkdir()

    def patched_init(self, *a, **kw):
        original_init(self, *a, **kw)
        self.files_dir = str(files_dir)
        self.database_dir = str(db_dir)

    monkeypatch.setattr(BaseController, "__init__", patched_init)
    return files_dir


def nlp_settings(**overrides):
    base = dict(OPENAI_API_KEY="k", OPENAI_API_URL="https://openrouter.ai/api/v1", COHERE_API_KEY=None,
                INPUT_DAFAULT_MAX_CHARACTERS=1024, GENERATION_DAFAULT_MAX_INPUT_CHARACTERS=20000,
                GENERATION_DAFAULT_MAX_TOKENS=2000, GENERATION_DAFAULT_TEMPERATURE=0.1,
                OPENAI_API_TIMEOUT=30)
    base.update(overrides)
    return SimpleNamespace(**base)


# --------------------------------------------------------------------------- MongoDB (integration)
def _test_mongo_url():
    url = os.environ.get("TEST_MONGODB_URL")
    if url:
        return url
    if os.path.exists(SRC_ENV_FILE):
        from dotenv import dotenv_values
        return dotenv_values(SRC_ENV_FILE).get("MONGODB_URL")
    return None


@pytest.fixture
def mongo_test_db():
    """A brand-new database on the local MongoDB, dropped after the test. Never touches existing DBs."""
    import pymongo
    url = _test_mongo_url()
    if not url:
        pytest.skip("No MongoDB URL configured (set TEST_MONGODB_URL)")
    sync_client = pymongo.MongoClient(url, serverSelectionTimeoutMS=3000)
    try:
        sync_client.admin.command("ping")
    except Exception:
        pytest.skip("MongoDB is not reachable")

    db_name = f"mini_rag_test_{uuid.uuid4().hex[:10]}"
    assert db_name not in sync_client.list_database_names()
    yield url, db_name
    sync_client.drop_database(db_name)   # only the database created above
    sync_client.close()


@pytest.fixture
def api_client(mongo_test_db, qdrant, isolated_files, fake_embedding, fake_generation, monkeypatch):
    """FastAPI TestClient wired to: temp Mongo DB, temp Qdrant, temp files dir, fake LLM + embeddings.
    The app's startup/shutdown handlers are disabled (they would connect to the real configured services)."""
    from fastapi.testclient import TestClient
    from motor.motor_asyncio import AsyncIOMotorClient
    from stores.llm.templatess.template_parser import TemplateParser
    import main

    url, db_name = mongo_test_db
    monkeypatch.setattr(main.app.router, "on_startup", [])
    monkeypatch.setattr(main.app.router, "on_shutdown", [])

    main.app.vectordb_client = qdrant
    main.app.embedding_client = fake_embedding
    main.app.generation_client = fake_generation
    main.app.template_parser = TemplateParser(language="en", default_language="en")

    # one TestClient context = one event loop for all requests (Motor binds to its loop)
    with TestClient(main.app) as client:
        mongo = AsyncIOMotorClient(url)
        main.app.mongo_conn = mongo
        main.app.db_client = mongo[db_name]
        client.fake_generation = fake_generation
        client.fake_embedding = fake_embedding
        client.qdrant = qdrant
        # existing flow tests act as the TA; security tests override / remove this header
        client.headers.update(auth_headers(make_token("admin")))
        yield client
        mongo.close()


# --------------------------------------------------------------------------- live HTTP backend (Streamlit tests)
STREAMLIT_DIR = os.path.join(REPO_ROOT, "streamlit_app")


@pytest.fixture
def live_backend(mongo_test_db, qdrant, isolated_files, fake_embedding, fake_generation, monkeypatch):
    """The real FastAPI app served by a real uvicorn server on a free localhost port (in this process),
    wired to a temp MongoDB database, temp Qdrant, temp files dir and FAKE LLM / embeddings.
    Streamlit AppTest scripts talk to it over HTTP exactly like in production."""
    import socket
    import threading
    import time as _time
    import uvicorn
    from motor.motor_asyncio import AsyncIOMotorClient
    from stores.llm.templatess.template_parser import TemplateParser
    import main

    url, db_name = mongo_test_db
    monkeypatch.setattr(main.app.router, "on_startup", [])
    monkeypatch.setattr(main.app.router, "on_shutdown", [])
    mongo = AsyncIOMotorClient(url)
    main.app.mongo_conn = mongo
    main.app.db_client = mongo[db_name]
    main.app.vectordb_client = qdrant
    main.app.embedding_client = fake_embedding
    main.app.generation_client = fake_generation
    main.app.template_parser = TemplateParser(language="en", default_language="en")

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(main.app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        _time.sleep(0.05)
    assert server.started, "uvicorn did not start"

    if STREAMLIT_DIR not in sys.path:
        sys.path.insert(0, STREAMLIT_DIR)
    import settings as frontend_settings
    base = f"http://127.0.0.1:{port}/api/v1"
    monkeypatch.setattr(frontend_settings, "API_BASE_URL", base)

    yield SimpleNamespace(base_url=base, fake_generation=fake_generation, qdrant=qdrant, files_dir=isolated_files)
    server.should_exit = True
    thread.join(timeout=10)
    mongo.close()


def pdf_bytes(pages):
    """A real PDF (one page per text) built with PyMuPDF."""
    import pymupdf
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_text((50, 72), text, fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


def app_texts(at):
    """All text-like elements of a Streamlit AppTest run, joined."""
    out = []
    for kind in ("markdown", "success", "error", "warning", "info", "caption"):
        out += [e.value for e in getattr(at, kind)]
    return "\n".join(str(t) for t in out)
