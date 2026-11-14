"""Hosted-deployment wiring: Cloudflare Workers AI embeddings, remote Qdrant, frontend identity headers,
and the Render blueprint (no secrets committed, free plan)."""
import os
import re
import sys

import pytest

from conftest import REPO_ROOT, STREAMLIT_DIR, nlp_settings
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.llm.providers import OpenAIProvider


def test_cloudflare_embeddings_reuse_openai_compatible_provider():
    cfg = nlp_settings(CLOUDFLARE_ACCOUNT_ID="acc123", CLOUDFLARE_API_TOKEN="cf-token")
    p = LLMProviderFactory(cfg).create("CLOUDFLARE")
    assert isinstance(p, OpenAIProvider)
    assert str(p.client.base_url).rstrip("/") == "https://api.cloudflare.com/client/v4/accounts/acc123/ai/v1"
    assert p.api_key == "cf-token"


def test_cloudflare_without_credentials_is_not_created():
    cfg = nlp_settings(CLOUDFLARE_ACCOUNT_ID=None, CLOUDFLARE_API_TOKEN=None)
    assert LLMProviderFactory(cfg).create("CLOUDFLARE") is None


def test_remote_qdrant_mode(monkeypatch):
    import stores.vectordb.providers  # noqa: F401  (the package re-exports the class under the module's name)
    mod = sys.modules["stores.vectordb.providers.QdrantDBProvider"]
    calls = {}

    class FakeClient:
        def __init__(self, **kw):
            calls["init"] = kw
            self.indexes = []
            self.collections = set()
        def collection_exists(self, collection_name):
            return collection_name in self.collections
        def create_collection(self, collection_name, vectors_config):
            self.collections.add(collection_name)
        def create_payload_index(self, collection_name, field_name, field_schema):
            self.indexes.append(field_name)

    monkeypatch.setattr(mod, "QdrantClient", FakeClient)
    q = mod.QdrantDBProvider(db_path="unused", distance_method="cosine",
                             url="https://x.cloud.qdrant.io:6333", api_key="qk")
    q.connect()
    assert calls["init"]["url"] == "https://x.cloud.qdrant.io:6333" and calls["init"]["api_key"] == "qk"
    assert "path" not in calls["init"]
    q.create_collection("c", embedding_size=1024)
    assert set(q.client.indexes) == {"metadata.doc_type", "metadata.file_id"}


def test_vectordb_factory_passes_qdrant_cloud_settings(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
    cfg = SimpleNamespace(VECTOR_DB_PATH=str(tmp_path / "q"), VECTOR_DB_DISTANCE_METHOD="cosine",
                          QDRANT_URL="https://x:6333", QDRANT_API_KEY="k")
    provider = VectorDBProviderFactory(cfg).create("QDRANT")
    assert provider.url == "https://x:6333" and provider.api_key == "k" and provider.is_remote()


def test_frontend_identity_headers():
    sys.path.insert(0, STREAMLIT_DIR)
    from api_client import BackendClient
    c = BackendClient("http://x/api/v1", "tok", client_ip="1.2.3.4", frontend_secret="s")
    assert c._headers() == {"Authorization": "Bearer tok", "X-Frontend-Secret": "s", "X-Client-IP": "1.2.3.4"}
    assert BackendClient("http://x", "tok")._headers() == {"Authorization": "Bearer tok"}


def test_render_blueprint_is_free_and_has_no_secret_values():
    text = open(os.path.join(REPO_ROOT, "render.yaml"), encoding="utf-8").read()
    assert "plan: free" in text
    secret_keys = ["MONGODB_URL", "OPENAI_API_KEY", "CLOUDFLARE_API_TOKEN", "QDRANT_API_KEY",
                   "ADMIN_PASSWORD_HASH", "STUDENT_ACCESS_CODE_HASH", "TRUSTED_FRONTEND_SECRET", "ADMIN_USERNAME"]
    for key in secret_keys:
        block = re.search(rf"- key: {key}\n\s+(\w+):\s*(.*)", text)
        assert block and block.group(1) == "sync" and block.group(2).strip() == "false", key
    assert re.search(r"- key: JWT_SECRET_KEY\n\s+generateValue: true", text)


def test_deploy_requirements_exclude_local_ml_stack():
    reqs = open(os.path.join(REPO_ROOT, "src", "requirements-deploy.txt"), encoding="utf-8").read().lower()
    for heavy in ("torch", "transformers", "sentence-transformers"):
        assert not re.search(rf"^{heavy}\b", reqs, re.M)
    full = open(os.path.join(REPO_ROOT, "src", "requirements.txt"), encoding="utf-8").read().lower()
    needed = {re.split(r"[<>=\[ ]", l)[0] for l in full.splitlines() if l and not l.startswith("#")}
    needed -= {"torch", "transformers", "sentence-transformers"}
    deploy = {re.split(r"[<>=\[ ]", l)[0] for l in reqs.splitlines() if l and not l.startswith("#")}
    assert needed <= deploy, needed - deploy
