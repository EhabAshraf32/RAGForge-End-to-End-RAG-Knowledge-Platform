"""Settings / configuration files (B11, B12 and new settings)."""
import os
import re

from conftest import REPO_ROOT, SRC_DIR


def test_settings_defaults_for_new_options():
    from helpers.config import get_settings
    s = get_settings()
    assert s.GENERATION_DAFAULT_MAX_INPUT_CHARACTERS == 20000
    assert s.OPENAI_API_TIMEOUT == 120
    assert s.RAG_HISTORY_MAX_MESSAGES == 6
    assert s.RAG_HISTORY_MAX_CHARACTERS == 3000
    # API keys are optional (Cohere must not be required)
    assert s.COHERE_API_KEY is None


def test_settings_reads_values_from_environment(monkeypatch):
    from helpers.config import get_settings
    monkeypatch.setenv("GENERATION_DAFAULT_MAX_INPUT_CHARACTERS", "999")
    monkeypatch.setenv("OPENAI_API_URL", "https://openrouter.ai/api/v1")
    s = get_settings()
    assert s.GENERATION_DAFAULT_MAX_INPUT_CHARACTERS == 999
    assert s.OPENAI_API_URL == "https://openrouter.ai/api/v1"


def _env_keys(path):
    keys = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        assert "=" in line, f"invalid line in {path}: {line!r}"
        key = line.split("=", 1)[0].strip()
        assert re.fullmatch(r"[A-Z_][A-Z0-9_]*", key), f"invalid key {key!r} in {path}"
        keys.append(key)
    return keys


def test_b12_env_example_is_valid_and_covers_settings():
    from helpers.config import Settings
    keys = _env_keys(os.path.join(SRC_DIR, "env.example"))
    assert "MONGODB_DATABASE" in keys and "MONOGODB_NAME" not in keys
    missing = set(Settings.model_fields) - set(keys)
    assert not missing, f"settings not documented in env.example: {missing}"


def test_b12_env_example_targets_openrouter_and_huggingface():
    content = open(os.path.join(SRC_DIR, "env.example"), encoding="utf-8").read()
    assert re.search(r'^GENERATION_BACKEND\s*=\s*"OPENAI"', content, re.M)
    assert re.search(r'^OPENAI_API_URL\s*=\s*"https://openrouter.ai/api/v1"', content, re.M)
    assert re.search(r'^EMBEDDING_BACKEND\s*=\s*"HUGGINGFACE"', content, re.M)
    assert not re.search(r'^(GENERATION|EMBEDDING)_BACKEND\s*=\s*"COHERE"', content, re.M)


def test_b12_docker_env_example_uses_key_value_format():
    keys = _env_keys(os.path.join(REPO_ROOT, "docker", ".env.example"))
    assert keys == ["MONGO_INITDB_ROOT_USERNAME", "MONGO_INITDB_ROOT_PASSWORD"]


def test_b11_requirements_declare_imported_packages():
    reqs = [l.split("#")[0].strip().lower() for l in open(os.path.join(SRC_DIR, "requirements.txt"))]
    reqs = [re.split(r"[<>=\[]", r)[0] for r in reqs if r]
    for pkg in ["langchain-community", "langchain-text-splitters", "pymupdf", "motor", "openai",
                "qdrant-client", "sentence-transformers", "transformers", "pyjwt"]:
        assert pkg in reqs, f"{pkg} missing from requirements.txt"
    assert "cohere" not in reqs, "cohere must be optional"
